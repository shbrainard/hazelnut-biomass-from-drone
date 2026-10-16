#!/usr/bin/env python
"""
Find planting rows and plants when no plant map exists.

Use a canopy height model to locate rows and individual plants.

The method assumes straight, parallel rows. It finds the row azimuth as the
direction that makes the canopy's across-row profile sharpest, locates each
row as a peak of that profile, then walks along each row and places a plant
at every peak of the along-row canopy profile that is at least --min-spacing
apart. Plants closer together than the CHM can separate (crowns touching at
narrow spacings) come out merged, and the output flags each row's plant
count so the merging is visible rather than silent.

Inputs
    --chm        canopy height model GeoTIFF (height above ground, meters)
    --extent     optional polygon file; only canopy inside it is used
Outputs
    --out        GeoPackage of plant points with row index, position along
                 the row, canopy height at the point, and a row layer with
                 the fitted centerlines
    --fig        optional overview PNG
"""

import argparse
import logging
import math
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import mask as rio_mask
from scipy import ndimage
from scipy.signal import find_peaks
from shapely.geometry import LineString, Point

LOG = logging.getLogger("detect_plants")


def load_chm(path, extent=None):
    """Read canopy heights, optionally clipped to a polygon extent."""
    with rasterio.open(path) as src:
        if extent:
            gdf = gpd.read_file(extent)
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            arr, transform = rio_mask.mask(
                src, list(gdf.geometry), crop=True, nodata=np.nan
            )
            arr = arr[0].astype(np.float64)
        else:
            arr = src.read(1).astype(np.float64)
            transform = src.transform
        if src.nodata is not None and not np.isnan(src.nodata):
            arr[arr == src.nodata] = np.nan
        return arr, transform, src.crs


def canopy_coords(chm, transform, min_height):
    """Return cell centers and heights above the canopy threshold."""
    rows, cols = np.where(np.isfinite(chm) & (chm > min_height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.asarray(xs), np.asarray(ys), chm[rows, cols]


def best_azimuth(x, y, bin_m, candidates, w=None):
    """Return the azimuth with the greatest across-row profile variance.

    Straight parallel rows maximize this variance; angles are radians
    measured from east.
    """
    best, best_var = None, -1.0
    for ang in candidates:
        across = -x * math.sin(ang) + y * math.cos(ang)
        hist, _ = np.histogram(
            across,
            bins=int((across.max() - across.min()) / bin_m) + 1,
            weights=w,
        )
        v = hist.var()
        if v > best_var:
            best, best_var = ang, v
    return best


def refine_azimuth(x, y, bin_m, w=None):
    """Refine the strongest coarse row azimuth on a finer angular grid."""
    coarse = best_azimuth(
        x, y, bin_m, np.deg2rad(np.arange(0, 180, 1.0)), w=w
    )
    fine = best_azimuth(
        x, y, bin_m, coarse + np.deg2rad(np.arange(-1.0, 1.0001, 0.05)), w=w
    )
    return fine


def main(argv=None):
    """Parse arguments and write detected rows and plant points."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--chm", required=True)
    ap.add_argument("--extent", default=None)
    ap.add_argument(
        "--min-height",
        type=float,
        default=0.3,
        help="canopy threshold for the profiles (m)",
    )
    ap.add_argument(
        "--row-spacing",
        type=float,
        required=True,
        help="expected distance between rows (m)",
    )
    ap.add_argument(
        "--min-spacing",
        type=float,
        required=True,
        help="smallest plant-to-plant distance to resolve (m)",
    )
    ap.add_argument(
        "--row-half-width",
        type=float,
        default=None,
        help="half the strip width sampled around each row "
        "centerline (default 0.3 * row spacing)",
    )
    ap.add_argument(
        "--azimuth",
        type=float,
        default=None,
        help="row azimuth in degrees from east, if known",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chm, transform, crs = load_chm(args.chm, args.extent)
    res = abs(transform.a)
    x, y, h = canopy_coords(chm, transform, args.min_height)
    LOG.info("%d canopy cells above %.2f m", len(x), args.min_height)
    if len(x) < 100:
        raise SystemExit("too little canopy above the threshold to find rows")

    ang = (
        math.radians(args.azimuth)
        if args.azimuth is not None
        else refine_azimuth(x, y, res, w=h)
    )
    LOG.info("row azimuth %.2f degrees from east", math.degrees(ang))
    ua = np.array([math.cos(ang), math.sin(ang)])  # along the row
    ub = np.array([-math.sin(ang), math.cos(ang)])  # across the rows
    x0, y0 = x.mean(), y.mean()
    along = (x - x0) * ua[0] + (y - y0) * ua[1]
    across = (x - x0) * ub[0] + (y - y0) * ub[1]

    # row centerlines: peaks of the across-row canopy profile, weighted by
    # height so a row of plants outweighs a strip of low vegetation of the
    # same width in the alley beside it
    edges = np.arange(across.min() - res, across.max() + 2 * res, res)
    prof = np.histogram(across, bins=edges, weights=h)[0].astype(float)
    sigma = max(1.0, 0.15 * args.row_spacing / res)
    # pad with empty bins so a row at the edge of the canopy extent is an
    # interior peak after smoothing; find_peaks cannot return the first or
    # last sample
    pad = int(3 * sigma) + 1
    prof = ndimage.gaussian_filter1d(np.pad(prof, pad), sigma=sigma)
    peaks, _ = find_peaks(
        prof,
        distance=max(1, int(0.6 * args.row_spacing / res)),
        height=prof.max() * 0.08,
    )
    peaks = peaks - pad
    peaks = peaks[(peaks >= 0) & (peaks < len(edges) - 1)]
    row_pos = 0.5 * (edges[peaks] + edges[peaks + 1])
    LOG.info(
        "%d rows found; spacing median %.2f m",
        len(row_pos),
        float(np.median(np.diff(row_pos)))
        if len(row_pos) > 1
        else float("nan"),
    )

    half = args.row_half_width or 0.3 * args.row_spacing
    plants = []
    lines = []
    for ri, rp in enumerate(row_pos):
        m = np.abs(across - rp) <= half
        if m.sum() < 10:
            continue
        a, hh = along[m], h[m]
        e2 = np.arange(a.min() - res, a.max() + 2 * res, res)
        # along-row profile weighted by height, so a tall crown dominates
        # the low grass around it
        p2 = np.histogram(a, bins=e2, weights=hh)[0]
        sig2 = max(1.0, 0.2 * args.min_spacing / res)
        pad2 = int(3 * sig2) + 1
        p2 = ndimage.gaussian_filter1d(np.pad(p2, pad2), sigma=sig2)
        pk, props = find_peaks(
            p2,
            distance=max(1, int(args.min_spacing / res)),
            height=p2.max() * 0.05,
            prominence=p2.max() * 0.03,
        )
        pk = pk - pad2
        pk = pk[(pk >= 0) & (pk < len(e2) - 1)]
        pos = 0.5 * (e2[pk] + e2[pk + 1])
        for pi, pa in enumerate(pos):
            px = x0 + pa * ua[0] + rp * ub[0]
            py = y0 + pa * ua[1] + rp * ub[1]
            plants.append(
                {
                    "row_idx": ri,
                    "plant_idx": pi,
                    "along_m": float(pa),
                    "across_m": float(rp),
                    "geometry": Point(px, py),
                }
            )
        p0 = (
            x0 + a.min() * ua[0] + rp * ub[0],
            y0 + a.min() * ua[1] + rp * ub[1],
        )
        p1 = (
            x0 + a.max() * ua[0] + rp * ub[0],
            y0 + a.max() * ua[1] + rp * ub[1],
        )
        lines.append(
            {
                "row_idx": ri,
                "across_m": float(rp),
                "n_plants": int(len(pos)),
                "length_m": float(a.max() - a.min()),
                "geometry": LineString([p0, p1]),
            }
        )

    pts = gpd.GeoDataFrame(plants, geometry="geometry", crs=crs)
    # canopy height at each detected point: max of the CHM within 0.3 m
    inv = ~transform
    hvals = []
    r_px = max(1, int(round(0.3 / res)))
    for g in pts.geometry:
        c, r = inv * (g.x, g.y)
        c, r = int(c), int(r)
        win = chm[
            max(0, r - r_px) : r + r_px + 1, max(0, c - r_px) : c + r_px + 1
        ]
        hvals.append(
            float(np.nanmax(win)) if np.isfinite(win).any() else np.nan
        )
    pts["chm_h"] = hvals
    rows_gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs=crs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pts.to_file(args.out, layer="plants", driver="GPKG")
    rows_gdf.to_file(args.out, layer="rows", driver="GPKG")
    LOG.info(
        "wrote %s: %d plants in %d rows", args.out, len(pts), len(rows_gdf)
    )
    LOG.info("plants per row: %s", rows_gdf.n_plants.tolist())

    if args.fig:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(9, 9 * chm.shape[0] / max(chm.shape[1], 1))
        )
        ext = rasterio.transform.array_bounds(
            chm.shape[0], chm.shape[1], transform
        )
        ax.imshow(
            np.clip(np.nan_to_num(chm), 0, 3),
            cmap="Greens",
            extent=[ext[0], ext[2], ext[1], ext[3]],
            origin="upper",
        )
        rows_gdf.plot(ax=ax, color="tab:blue", linewidth=0.6)
        pts.plot(ax=ax, color="red", markersize=3)
        ax.set_title(
            f"{len(pts)} plants in {len(rows_gdf)} rows, "
            f"azimuth {math.degrees(ang):.1f} deg"
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(args.fig, dpi=150)
    return 0


if __name__ == "__main__":
    sys.exit(main())
