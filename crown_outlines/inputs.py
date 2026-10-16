"""Sample photos and load canopy returns in the point cloud frame."""

import laspy
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from plant_traits.per_plant_traits import height_above_ground, load_points


def cloud_crs(path):
    """Read and validate a projected point cloud CRS with meter units."""
    with laspy.open(path) as src:
        crs = src.header.parse_crs()
    if crs is None:
        raise ValueError("Point cloud must declare its CRS")
    crs = CRS.from_user_input(crs)
    if not crs.is_projected or any(
        axis.unit_conversion_factor != 1 for axis in crs.axis_info[:2]
    ):
        raise ValueError("Point cloud CRS must be projected in meters")
    return crs


def crop_photo(src, x, y, target_crs, size=3.2, res=0.005, offset=(0, 0)):
    """Sample a north-up square, adding offsets in the photo's own CRS."""
    n = int(round(size / res))
    if size <= 0 or res <= 0 or not np.isclose(n * res, size):
        raise ValueError("Crop size must be a positive multiple of resolution")
    half = size / 2
    xs = x - half + (np.arange(n) + 0.5) * res
    ys = y + half - (np.arange(n) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    transformer = Transformer.from_crs(target_crs, src.crs, always_xy=True)
    ex, ey = transformer.transform(gx.ravel(), gy.ravel())
    cols, rows = ~src.transform * (ex + offset[0], ey + offset[1])
    c0, r0 = int(np.floor(cols.min())) - 2, int(np.floor(rows.min())) - 2
    c1, r1 = int(np.ceil(cols.max())) + 3, int(np.ceil(rows.max())) + 3
    data = src.read(
        [1, 2, 3],
        window=Window(c0, r0, c1 - c0, r1 - r0),
        boundless=True,
        fill_value=0,
    )
    # Direct affine coordinates preserve the pilot sampling convention.
    out = np.stack(
        [
            map_coordinates(
                band.astype(np.float32),
                [rows - r0, cols - c0],
                order=1,
                mode="nearest",
            )
            for band in data
        ],
        axis=-1,
    )
    return out.reshape(n, n, 3).clip(0, 255).astype(np.uint8), (
        x - half,
        y + half,
        res,
    )


def load_returns(las_path, dtm_path, plants_xy, radius=1.7):
    """Load non-ground returns near plants using the existing trait helpers."""
    crs = cloud_crs(las_path)
    with rasterio.open(dtm_path) as dtm:
        if dtm.crs != crs:
            raise ValueError("DTM and point cloud must have the same CRS")
    x, y, z, classes, _ = load_points(las_path)
    tree = cKDTree(plants_xy)
    near = np.zeros(len(x), dtype=bool)
    for start in range(0, len(x), 2_000_000):
        end = start + 2_000_000
        distance, _ = tree.query(
            np.c_[x[start:end], y[start:end]], distance_upper_bound=radius
        )
        near[start:end] = np.isfinite(distance)
    near &= classes != 2
    x, y, z = x[near], y[near], z[near]
    height, _ = height_above_ground(x, y, z, dtm_path)
    return x, y, height


def reference_height(x, y, height, center):
    """Compute p99_1m with the trait module's strict lower height cutoff."""
    selected = (
        (np.hypot(x - center[0], y - center[1]) <= 1.0)
        & np.isfinite(height)
        & (height > 0.1)
        & (height <= 6.0)
    )
    return (
        float(np.percentile(height[selected], 99))
        if selected.any()
        else np.nan
    )
