"""Run one flight and write crown outlines and review images."""

import argparse
import logging
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

from .geometry import choose_mask, polygon_from_mask, smooth_mask
from .inputs import cloud_crs, crop_photo, load_returns, reference_height
from .measurement import measure
from .segmentation import canopy_box, load_predictor, predict_masks

LOG = logging.getLogger(__name__)


def read_plants(path, id_col, crs, xy_cols=("x", "y"), csv_crs=None):
    """Read unique plant points and transform them into the cloud CRS."""
    if Path(path).suffix.lower() == ".csv":
        if csv_crs is None:
            raise ValueError("CSV plants require --plants-crs")
        frame = pd.read_csv(path, dtype={id_col: str})
        plants = gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame[xy_cols[0]], frame[xy_cols[1]]),
            crs=csv_crs,
        )
    else:
        plants = gpd.read_file(path)
    if plants.empty or plants.crs is None:
        raise ValueError("Plants must be nonempty and declare a CRS")
    if (
        plants[id_col].isna().any()
        or plants[id_col].astype(str).duplicated().any()
    ):
        raise ValueError("Plant identifiers must be non-null and unique")
    if not (plants.geometry.geom_type == "Point").all():
        raise ValueError("Plants must contain only point geometries")
    plants = plants.to_crs(crs)
    if not np.isfinite(np.c_[plants.geometry.x, plants.geometry.y]).all():
        raise ValueError("Plant coordinates must be finite")
    plants[id_col] = plants[id_col].astype(str)
    return plants


def write_review(path, image, polygon, origin):
    """Draw the exported outline on the crop without a plotting dependency."""
    from PIL import Image, ImageDraw

    photo = Image.fromarray(image)
    if not polygon.is_empty:
        west, north, res = origin
        coords = [
            ((x - west) / res, (north - y) / res)
            for x, y in polygon.exterior.coords
        ]
        ImageDraw.Draw(photo).line(coords, fill=(0, 255, 255), width=2)
    photo.save(path)


def run(args, predictor=None):
    """Process one flight with an optional supplied predictor."""
    if args.smoothing_radius < 0 or args.visible_height < 0:
        raise ValueError(
            "Smoothing radius and visible height must be nonnegative"
        )
    if args.crop_size <= 0 or args.resolution <= 0:
        raise ValueError("Crop size and resolution must be positive")
    crs = cloud_crs(args.cloud)
    plants = read_plants(
        args.plants, args.id_col, crs, args.xy_cols, args.plants_crs
    )
    references = None
    if args.reference_heights:
        table = pd.read_csv(
            args.reference_heights, dtype={args.reference_id_col: str}
        )
        if table[args.reference_id_col].duplicated().any():
            raise ValueError("Reference height identifiers must be unique")
        references = table.set_index(args.reference_id_col)[args.height_col]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.review:
        (output / "review").mkdir(exist_ok=True)
    xy = np.c_[plants.geometry.x, plants.geometry.y]
    x, y, height = load_returns(args.cloud, args.dtm, xy)
    tree = cKDTree(np.c_[x, y])
    if predictor is None:
        predictor = load_predictor(args.checkpoint, args.device)
    rows, polygons = [], []
    with ExitStack() as stack:
        kwargs = (
            {}
            if args.overview_level is None
            else {"overview_level": args.overview_level}
        )
        photo = stack.enter_context(rasterio.open(args.orthophoto, **kwargs))
        chms = [stack.enter_context(rasterio.open(path)) for path in args.chm]
        for src in chms:
            if src.crs != crs:
                raise ValueError("CHM and point cloud must have the same CRS")
        for number, (_, plant) in enumerate(plants.iterrows(), 1):
            center = plant.geometry.x, plant.geometry.y
            idx = np.asarray(tree.query_ball_point(center, 1.7), dtype=int)
            px, py, ph = x[idx], y[idx], height[idx]
            p99 = reference_height(px, py, ph, center)
            if references is not None and plant[args.id_col] in references:
                p99 = float(references.loc[plant[args.id_col]])
            image, origin = crop_photo(
                photo,
                *center,
                crs,
                args.crop_size,
                args.resolution,
                args.offset,
            )
            box, _, threshold = canopy_box(
                px, py, ph, center, origin, image.shape[:2], p99
            )
            masks, scores = predict_masks(predictor, image, box)
            choice = choose_mask(masks, scores)
            mask, status = smooth_mask(masks[choice], args.smoothing_radius)
            polygon, tolerance, iou = polygon_from_mask(mask, origin)
            row = measure(
                mask,
                origin,
                px,
                py,
                ph,
                chms,
                center,
                crs,
                args.visible_height,
            )
            row.update(
                {
                    "plant_id": plant[args.id_col],
                    "mask_choice": choice,
                    "mask_score": float(scores[choice]),
                    "smoothing_status": status,
                    "smoothing_radius_px": args.smoothing_radius,
                    "box_threshold_m": threshold,
                    "reference_p99_m": p99,
                    "simplify_tolerance_px": tolerance,
                    "polygon_mask_iou": iou,
                    "polygon_area_m2": polygon.area,
                }
            )
            rows.append(row)
            polygons.append(polygon)
            if args.review:
                write_review(
                    output / "review" / f"{number:06d}.png",
                    image,
                    polygon,
                    origin,
                )
            LOG.info(
                "processed %d/%d: %s", number, len(plants), row["plant_id"]
            )
    frame = gpd.GeoDataFrame(rows, geometry=polygons, crs=crs)
    frame.to_file(output / "outlines.gpkg", layer="outlines", driver="GPKG")
    frame.drop(columns="geometry").to_csv(output / "outlines.csv", index=False)
    return frame


def parser():
    """Build the command-line interface for one flight."""
    ap = argparse.ArgumentParser(description=__doc__)
    for name in (
        "orthophoto",
        "cloud",
        "dtm",
        "plants",
        "id-col",
        "checkpoint",
        "output",
    ):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--chm", nargs="+", required=True)
    ap.add_argument("--overview-level", type=int)
    ap.add_argument(
        "--offset",
        nargs=2,
        type=float,
        default=(0.0, 0.0),
        metavar=("EAST", "NORTH"),
    )
    ap.add_argument("--plants-crs")
    ap.add_argument("--xy-cols", nargs=2, default=("x", "y"))
    ap.add_argument("--reference-heights")
    ap.add_argument("--reference-id-col", default="plant_number")
    ap.add_argument("--height-col", default="p99_1m")
    ap.add_argument("--crop-size", type=float, default=3.2)
    ap.add_argument("--resolution", type=float, default=0.005)
    ap.add_argument("--smoothing-radius", type=int, default=6)
    ap.add_argument("--visible-height", type=float, default=0.6)
    ap.add_argument("--device", choices=("cpu", "mps"))
    ap.add_argument("--review", action="store_true")
    return ap


def main(argv=None):
    """Parse arguments and run a flight."""
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args)
    return 0
