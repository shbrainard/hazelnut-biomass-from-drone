from contextlib import ExitStack

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from scipy import ndimage

from crown_outlines.geometry import (
    choose_mask,
    clean_mask,
    polygon_from_mask,
    smooth_mask,
)
from crown_outlines.inputs import crop_photo, load_returns, reference_height
from crown_outlines.measurement import first_covering_chm, measure
from crown_outlines.pipeline import parser, run
from crown_outlines.segmentation import canopy_box, predict_masks


def raster(path, data, transform=None, nodata=None):
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=len(data),
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype,
        crs="EPSG:32616",
        transform=transform or from_origin(0, 4, 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


def test_crop_placement_offset_sign_and_boundless(tmp_path):
    rows, cols = np.mgrid[:12, :12]
    data = np.stack([cols * 10, rows * 10, cols * 10]).astype("uint8")
    path = raster(tmp_path / "photo.tif", data, from_origin(0, 12, 1, 1))
    with rasterio.open(path) as src:
        image, origin = crop_photo(src, 5, 7, src.crs, 2, 1)
        shifted, _ = crop_photo(src, 5, 7, src.crs, 2, 1, (2, -1))
        outside, _ = crop_photo(src, -10, -10, src.crs, 2, 1)
    assert origin == (4, 8, 1)
    np.testing.assert_array_equal(image[:, :, 0], [[45, 55], [45, 55]])
    np.testing.assert_array_equal(image[:, :, 1], [[45, 45], [55, 55]])
    np.testing.assert_array_equal(shifted[:, :, 0], image[:, :, 0] + 20)
    np.testing.assert_array_equal(shifted[:, :, 1], image[:, :, 1] + 10)
    assert not outside.any()


def test_canopy_box_planted_returns():
    cols, rows = np.meshgrid(np.arange(14, 19), np.arange(14, 18))
    x = (cols.ravel() + 0.5) * 0.1
    y = 3.2 - (rows.ravel() + 0.5) * 0.1
    h = np.full(len(x), 1.0)
    box, upper, threshold = canopy_box(
        x, y, h, (1.6, 1.6), (0, 3.2, 0.005), (640, 640), 2
    )
    np.testing.assert_array_equal(box, [260, 260, 400, 380])
    assert upper.sum() == 5 * 4 * 20 * 20
    assert threshold == 1
    box, upper, threshold = canopy_box(
        x, y, h * 0, (1.6, 1.6), (0, 3.2, 0.005), (640, 640), np.nan
    )
    assert box is None
    assert not upper.any()
    assert threshold == 0.25


@pytest.mark.parametrize(
    ("scores", "expected"),
    [([0.9, 0.75, 0.74], 1), ([0.7, 0.6, 0.5], 0), ([0.75, 0.8, 0.95], 2)],
)
def test_choice_and_fallback(scores, expected):
    masks = np.zeros((3, 10, 10), bool)
    masks[0, :8] = True
    masks[1, :5] = True
    masks[2, :2] = True
    assert choose_mask(masks, np.array(scores)) == expected


def test_clean_center_nearest_holes_and_empty():
    mask = np.zeros((20, 20), bool)
    mask[7:14, 7:14] = True
    mask[9:11, 9:11] = False
    mask[:3, :3] = True
    result = clean_mask(mask)
    assert result.sum() == 49
    mask[7:14, 7:14] = False
    assert clean_mask(mask).sum() == 9
    assert not clean_mask(np.zeros_like(mask)).any()


def test_smoothing_inlet_lobe_and_exact_disk():
    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:65] = True
    mask[35:65, 65:85] = True
    mask[20:40, 40] = False
    smoothed, status = smooth_mask(mask, 6)
    assert status == "smoothed"
    assert smoothed[30, 40]
    assert smoothed[50, 80]
    yy, xx = np.mgrid[-6:7, -6:7]
    disk = xx**2 + yy**2 <= 36
    padded = np.pad(clean_mask(mask), 14)
    expected = ndimage.binary_opening(
        ndimage.binary_closing(padded, structure=disk), structure=disk
    )[14:-14, 14:-14]
    np.testing.assert_array_equal(smoothed, clean_mask(expected))


def test_smoothing_border_tiny_and_disabled():
    mask = np.ones((40, 40), bool)
    smoothed, _ = smooth_mask(mask, 6)
    assert smoothed[0, 20]
    tiny = np.zeros((40, 40), bool)
    tiny[20, 20] = True
    retained, status = smooth_mask(tiny, 6)
    np.testing.assert_array_equal(retained, tiny)
    assert status == "erased_retained_original"
    unchanged, status = smooth_mask(mask, 0)
    np.testing.assert_array_equal(unchanged, mask)
    assert status == "disabled"
    with pytest.raises(ValueError, match="nonnegative integer"):
        smooth_mask(mask, -1)


def test_polygon_validity_and_georeferencing():
    mask = np.zeros((20, 20), bool)
    mask[4:14, 5:15] = True
    polygon, tolerance, iou = polygon_from_mask(mask, (100, 200, 0.1))
    assert polygon.is_valid
    assert polygon.bounds == pytest.approx((100.5, 198.6, 101.5, 199.6))
    assert polygon.area == pytest.approx(1)
    assert tolerance == 1
    assert iou == 1


def test_measurement_chm_p95_cellmax_and_visibility(tmp_path):
    data = np.array(
        [[1, 2, -1, 8], [3, -999, 4, 8], [9, 9, 9, 9], [9, 9, 9, 9]],
        dtype="float32",
    )
    path = raster(tmp_path / "chm.tif", data, nodata=-999)
    mask = np.zeros((40, 40), bool)
    mask[:20, :30] = True
    x = np.array([0.15, 0.16, 1.15, 1.25, 1.35, 1.45, 3.5])
    y = np.full(7, 3.85)
    heights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 6.1])
    with rasterio.open(path) as src:
        row = measure(
            mask, (0, 4, 0.1), x, y, heights, [src], (0.5, 3.5), src.crs, 0.575
        )
    assert row["mask_area_m2"] == pytest.approx(6)
    assert row["vol_outline_chm_m3"] == 10
    assert row["h_outline_cm"] == pytest.approx(59.5)
    assert row["h_outline_p95_cm"] == pytest.approx(57.5)
    assert row["vol_outline_cellmax_m3"] == pytest.approx(0.02)
    assert row["n_returns_outline"] == 6
    assert row["visible"]
    low = measure(
        mask,
        (0, 4, 0.1),
        x[:4],
        y[:4],
        heights[:4],
        [],
        (0.5, 3.5),
        "EPSG:32616",
    )
    assert np.isnan(low["h_outline_cm"])
    assert np.isnan(low["h_outline_p95_cm"])
    assert np.isnan(low["vol_outline_chm_m3"])
    assert not low["visible"]


@pytest.mark.parametrize(("height", "visible"), [(0.599, False), (0.6, True)])
def test_visible_threshold(height, visible):
    row = measure(
        np.ones((10, 10), bool),
        (0, 1, 0.1),
        np.full(5, 0.5),
        np.full(5, 0.5),
        np.full(5, height),
        [],
        (0.5, 0.5),
        "EPSG:32616",
    )
    assert row["visible"] is visible


def test_first_covering_chm_skips_nodata_and_respects_order(tmp_path):
    paths = [
        raster(
            tmp_path / f"chm{i}.tif",
            np.full((4, 4), value, dtype="float32"),
            nodata=-999,
        )
        for i, value in enumerate([-999, 2, 3])
    ]
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in paths]
        assert (
            first_covering_chm(sources, (0.5, 3.5), sources[0].crs)
            is sources[1]
        )
        assert (
            first_covering_chm(sources[::-1], (0.5, 3.5), sources[0].crs)
            is sources[2]
        )
        assert first_covering_chm(sources, (20, 20), sources[0].crs) is None


def test_reference_height_strict_lower_bound():
    h = np.array([0.1, 0.11, 1, 6, 6.1, np.nan, 5])
    x = np.array([0, 0, 0, 0, 0, 0, 1.01])
    assert reference_height(x, x * 0, h, (0, 0)) == np.percentile(h[1:4], 99)


class FakePredictor:
    """Supply deterministic masks while checking the prompt contract."""

    def set_image(self, image):
        """Store the crop for deterministic predictions."""
        self.image = image

    def predict(self, *, point_coords, point_labels, box, multimask_output):
        """Check the prompts and return three nested masks."""
        np.testing.assert_array_equal(point_coords, [[320, 320]])
        np.testing.assert_array_equal(point_labels, [1])
        assert box is None or box.shape == (4,)
        assert multimask_output
        masks = np.zeros((3, *self.image.shape[:2]), bool)
        masks[0, 200:440, 200:440] = True
        masks[1, 220:420, 220:420] = True
        masks[2, 240:400, 240:400] = True
        return masks, np.array([0.9, 0.8, 0.6]), None


def test_predictor_contract():
    masks, scores = predict_masks(
        FakePredictor(), np.zeros((640, 640, 3)), None
    )
    assert masks.shape == (3, 640, 640)
    assert choose_mask(masks, scores) == 1


def test_one_flight_outputs_and_ground_filter(tmp_path):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(32616))
    points = laspy.LasData(header)
    points.x = [2, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06]
    points.y = [2] * 7
    points.z = [1] * 5 + [5, 6]
    points.classification = [1] * 5 + [2, 7]
    cloud = tmp_path / "cloud.las"
    points.write(cloud)
    dtm = raster(tmp_path / "dtm.tif", np.zeros((4, 4), "float32"))
    photo = raster(tmp_path / "photo.tif", np.ones((3, 4, 4), "uint8"))
    chm = raster(tmp_path / "chm.tif", np.ones((4, 4), "float32"))
    plants = tmp_path / "plants.csv"
    pd.DataFrame({"id": ["001"], "x": [2], "y": [2]}).to_csv(
        plants, index=False
    )
    x, _, height = load_returns(cloud, dtm, np.array([[2, 2]]))
    assert len(x) == 5
    np.testing.assert_array_equal(height, np.ones(5))
    args = parser().parse_args(
        [
            "--orthophoto",
            str(photo),
            "--cloud",
            str(cloud),
            "--dtm",
            str(dtm),
            "--plants",
            str(plants),
            "--id-col",
            "id",
            "--plants-crs",
            "EPSG:32616",
            "--chm",
            str(chm),
            "--checkpoint",
            "unused",
            "--output",
            str(tmp_path / "out"),
            "--review",
            "--smoothing-radius",
            "0",
        ]
    )
    frame = run(args, predictor=FakePredictor())
    assert frame.plant_id.tolist() == ["001"]
    assert frame.n_returns_outline.tolist() == [5]
    assert frame.smoothing_status.tolist() == ["disabled"]
    assert (tmp_path / "out/review/000001.png").exists()
    saved = gpd.read_file(tmp_path / "out/outlines.gpkg")
    assert saved.crs == CRS.from_epsg(32616)
    assert saved.geometry.is_valid.all()
    assert len(pd.read_csv(tmp_path / "out/outlines.csv")) == 1
