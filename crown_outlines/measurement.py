"""Measure final mask footprints using canopy returns and CHM cell centers."""

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.windows import Window, from_bounds


def inside_mask(x, y, mask, origin):
    """Locate coordinates using the pilot's mask pixel indexing."""
    west, north, res = origin
    cols = ((x - west) / res).astype(int)
    rows = ((north - y) / res).astype(int)
    valid = (
        (cols >= 0)
        & (cols < mask.shape[1])
        & (rows >= 0)
        & (rows < mask.shape[0])
    )
    inside = np.zeros(np.shape(x), dtype=bool)
    inside[valid] = mask[rows[valid], cols[valid]]
    return inside


def first_covering_chm(sources, center, crs):
    """Select the first raster with a finite, unmasked cell at the plant."""
    for src in sources:
        transformer = Transformer.from_crs(crs, src.crs, always_xy=True)
        x, y = transformer.transform(*center)
        row, col = src.index(x, y)
        if 0 <= row < src.height and 0 <= col < src.width:
            value = src.read(1, window=Window(col, row, 1, 1), masked=True)
            if not np.ma.getmaskarray(value)[0, 0] and np.isfinite(
                value[0, 0]
            ):
                return src
    return None


def chm_volume(src, mask, origin, crs):
    """Sum nonnegative CHM heights at cell centers inside the final mask."""
    if src is None:
        return np.nan
    if src.crs != crs:
        raise ValueError("CHM and point cloud must have the same CRS")
    tr = src.transform
    if tr.b or tr.d or tr.a <= 0 or tr.e >= 0:
        raise ValueError("CHM must use a north-up grid")
    west, north, res = origin
    win = (
        from_bounds(
            west,
            north - mask.shape[0] * res,
            west + mask.shape[1] * res,
            north,
            tr,
        )
        .round_offsets()
        .round_lengths()
    )
    data = (
        src.read(1, window=win, boundless=True, masked=True)
        .astype(float)
        .filled(np.nan)
    )
    tr = src.window_transform(win)
    rows, cols = np.mgrid[: data.shape[0], : data.shape[1]]
    x = tr.c + (cols + 0.5) * tr.a
    y = tr.f + (rows + 0.5) * tr.e
    inside = inside_mask(x, y, mask, origin)
    return float(
        np.nansum(np.where(inside, np.clip(data, 0, None), 0))
        * abs(tr.a * tr.e)
    )


def measure(mask, origin, x, y, height, chms, center, crs, visible_height=0.6):
    """Measure area, canopy height, two volumes, and the visibility flag.

    Inputs are the extracted non-ground returns. The final raster footprint
    remains authoritative so disabling smoothing reproduces pilot metrics.
    """
    inside = inside_mask(x, y, mask, origin)
    canopy = inside & np.isfinite(height) & (height >= 0.1) & (height <= 6)
    count = int(canopy.sum())
    p99 = float(np.percentile(height[canopy], 99)) if count >= 5 else np.nan
    p95 = float(np.percentile(height[canopy], 95)) if count >= 5 else np.nan
    west, north, res = origin
    cell = np.floor((x[canopy] - west) / 0.1).astype(int) * 100000 + np.floor(
        (north - y[canopy]) / 0.1
    ).astype(int)
    cellmax = float(pd.Series(height[canopy]).groupby(cell).max().sum() * 0.01)
    src = first_covering_chm(chms, center, crs)
    return {
        "mask_area_m2": float(mask.sum() * res * res),
        "h_outline_cm": p99 * 100,
        "h_outline_p95_cm": p95 * 100,
        "vol_outline_chm_m3": chm_volume(src, mask, origin, crs),
        "vol_outline_cellmax_m3": cellmax,
        "n_returns_outline": count,
        "visible": bool(p95 >= visible_height),
        "chm_source": str(src.name) if src is not None else "",
    }
