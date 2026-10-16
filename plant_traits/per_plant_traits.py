#!/usr/bin/env python
"""
Measure per-plant canopy traits around known plant positions.

Use the returns of a ground-classified point cloud.

The watershed pipeline measures each bush on an IDW-smoothed canopy height
model and reports its maximum. On 1,683 hand-taped hazelnut bushes that
maximum under-read by 0.114 m with an RMSE of 0.210 m, while the 99th
percentile of canopy returns above the DTM read within 0.025 m with an RMSE
of 0.153 m. This module therefore measures height as a percentile of the
returns, never as a maximum, and measures size and shape on the returns too,
inside fixed footprints around each seed point and, when canopy polygons are
supplied, inside each polygon.

Inputs
    --las        ground-classified LAS/LAZ (ground = class 2; class 7 noise is
                 dropped)
    --dtm        DTM GeoTIFF; heights above ground are taken against this grid
    --seeds      one point per plant (shapefile, GeoPackage, or CSV with x/y)
    --segments   optional canopy polygons from the watershed stage (tree_id)
    --treetops   optional refined tree tops from the watershed stage; joins the
                 seeds to the segments on --id-col when both are given

Outputs one CSV row per seed. Column suffixes name the footprint:
    _1m   within --radius-height of the seed (height and presence)
    _ff   within --radius-footprint (size, kept small so neighbors never enter)
    _sh   within --radius-shape (shape; set below half the plant spacing)
    _seg  within the canopy polygon, when segments are supplied
"""

import argparse
import logging
import math
import os
import sys

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.spatial import ConvexHull, QhullError, cKDTree
from shapely.geometry import Point

LOG = logging.getLogger("per_plant_traits")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_points(las_path, drop_classes=(7,)):
    """Return x, y, z, classification, return_number arrays, noise dropped."""
    LOG.info("reading %s", las_path)
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    c = np.asarray(las.classification, dtype=np.uint8)
    rn = np.asarray(las.return_number, dtype=np.uint8)
    keep = ~np.isin(c, list(drop_classes))
    LOG.info(
        "%d points, %d kept after dropping classes %s",
        len(x),
        int(keep.sum()),
        drop_classes,
    )
    return x[keep], y[keep], z[keep], c[keep], rn[keep]


def height_above_ground(x, y, z, dtm_path):
    """Height of every point above the DTM cell it falls in (NaN off-grid)."""
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype(np.float64)
        if src.nodata is not None:
            dtm[dtm == src.nodata] = np.nan
        inv = ~src.transform
        crs = src.crs
    h, w = dtm.shape
    col = np.floor(inv.a * x + inv.b * y + inv.c).astype(np.int64)
    row = np.floor(inv.d * x + inv.e * y + inv.f).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    hag = np.full(len(x), np.nan)
    hag[inside] = z[inside] - dtm[row[inside], col[inside]]
    LOG.info(
        "%d of %d points inside the DTM grid, %d over nodata",
        int(inside.sum()),
        len(x),
        int(np.isnan(hag[inside]).sum()),
    )
    return hag, crs


def read_seeds(path, xy_cols=None, crs=None):
    """Seeds as a GeoDataFrame; CSVs need --xy-cols and a CRS."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if not xy_cols:
            raise SystemExit("--xy-cols is required for CSV seeds")
        xc, yc = xy_cols
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(a, b) for a, b in zip(df[xc], df[yc])],
            crs=crs,
        )
    else:
        gdf = gpd.read_file(path)
    return gdf


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def row_axes(seeds, row_col):
    """Return unit vectors along and across the rows.

    Each row's direction is the major axis of its seed points; the rows are
    averaged as axial data (angles doubled) so opposite orientations agree.
    Falls back to the major axis of all seeds when no row column is given.
    """
    groups = [g for _, g in seeds.groupby(row_col)] if row_col else [seeds]
    s = c = 0.0
    n_used = 0
    for g in groups:
        if len(g) < 3:
            continue
        xy = np.c_[g.geometry.x.values, g.geometry.y.values]
        xy = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        ang = math.atan2(vt[0, 1], vt[0, 0])
        s += math.sin(2 * ang)
        c += math.cos(2 * ang)
        n_used += 1
    if n_used == 0:
        raise SystemExit("no row with three or more seeds to fit an axis")
    ang = 0.5 * math.atan2(s, c)
    along = np.array([math.cos(ang), math.sin(ang)])
    across = np.array([-along[1], along[0]])
    LOG.info(
        "row axis azimuth %.2f degrees from east, from %d rows",
        math.degrees(ang),
        n_used,
    )
    return along, across


def cell_keys(x, y, cell):
    """Integer cell indices for 2-D binning."""
    return (
        np.floor(x / cell).astype(np.int64),
        np.floor(y / cell).astype(np.int64),
    )


def cell_max_volume(x, y, hag, cell):
    """Sum the highest return per occupied cell times the cell area.

    Integrate a max-Z CHM without interpolation or smoothing.
    """
    if len(x) == 0:
        return 0.0, 0.0
    ix, iy = cell_keys(x, y, cell)
    key = ix * 10_000_000 + iy
    uniq, inv = np.unique(key, return_inverse=True)
    cmax = np.full(len(uniq), -np.inf)
    np.maximum.at(cmax, inv, hag)
    return float(cmax.sum() * cell * cell), float(len(uniq) * cell * cell)


def voxel_stats(x, y, hag, cell):
    """Return filled voxel volume and filled / bounding-box porosity."""
    if len(x) == 0:
        return np.nan, np.nan, 0
    ix, iy = cell_keys(x, y, cell)
    iz = np.floor(hag / cell).astype(np.int64)
    key = (ix * 10_000_000 + iy) * 1_000 + iz
    filled = len(np.unique(key))
    nx = len(np.unique(ix))
    ny = len(np.unique(iy))
    nz = int(iz.max() - iz.min() + 1)
    bbox = max(nx * ny * nz, 1)
    return float(filled * cell**3), float(filled / bbox), filled


def hull_volume(x, y, hag):
    """Return the hull volume of canopy returns and ground projections.

    Ground projections give a flat crown with a gap underneath a base.
    """
    if len(x) < 4:
        return np.nan
    pts = np.c_[np.r_[x, x], np.r_[y, y], np.r_[hag, np.zeros(len(x))]]
    try:
        return float(ConvexHull(pts).volume)
    except QhullError:
        return np.nan


def shape_metrics(x, y, hag, along, across, cell):
    """Shape of one plant from its canopy returns, relative to the row.

    Returns a dict. Horizontal extents are the 2nd to 98th percentile range
    of the coordinates projected on the row axes, which ignores a stray
    return but keeps the crown edge.
    """
    out = {}
    if len(x) < 10:
        return out
    cx, cy = x.mean(), y.mean()
    dx, dy = x - cx, y - cy
    a = dx * along[0] + dy * along[1]
    b = dx * across[0] + dy * across[1]
    p = np.percentile
    out["depth_along"] = float(p(a, 98) - p(a, 2))
    out["width_across"] = float(p(b, 98) - p(b, 2))
    h99, h05 = float(p(hag, 99)), float(p(hag, 5))
    out["crown_depth"] = h99 - h05
    out["crown_base"] = h05
    horiz = 0.5 * (out["depth_along"] + out["width_across"])
    out["vert_ratio"] = h99 / horiz if horiz > 0 else np.nan
    # top third of the crown by height
    top_cut = h05 + 2.0 / 3.0 * (h99 - h05)
    out["top_frac"] = float((hag > top_cut).mean())
    # slice areas: occupied cells per height slice
    ix, iy = cell_keys(x, y, cell)
    iz = np.floor(hag / cell).astype(np.int64)
    key = ix * 10_000_000 + iy
    slices = {}
    for zi in np.unique(iz):
        slices[int(zi)] = len(np.unique(key[iz == zi]))
    zs = np.array(sorted(slices))
    areas = np.array([slices[int(k)] for k in zs], dtype=float)
    widest = zs[int(np.argmax(areas))]
    out["z_maxwidth_fr"] = (
        float(((widest + 0.5) * cell) / h99) if h99 > 0 else np.nan
    )
    nz = len(zs)
    if nz >= 3:
        third = max(nz // 3, 1)
        bottom = areas[:third].mean()
        top = areas[-third:].mean()
        out["taper_tb"] = (
            float(math.sqrt(top / bottom)) if bottom > 0 else np.nan
        )
    return out


# --------------------------------------------------------------------------
# per-plant measurement
# --------------------------------------------------------------------------
def percentile_block(hag, suffix):
    """Return height percentiles, maximum, and count with suffixed keys."""
    p = np.percentile
    if len(hag) == 0:
        return {
            f"p99{suffix}": np.nan,
            f"p95{suffix}": np.nan,
            f"max{suffix}": np.nan,
            f"n{suffix}": 0,
        }
    return {
        f"p99{suffix}": float(p(hag, 99)),
        f"p95{suffix}": float(p(hag, 95)),
        f"max{suffix}": float(hag.max()),
        f"n{suffix}": int(len(hag)),
    }


def measure_footprints(seeds, cx, cy, hag, first, along, across, args):
    """Fixed-footprint metrics for every seed."""
    tree = cKDTree(np.c_[cx, cy])
    rows = []
    r_h, r_f, r_s = (
        args.radius_height,
        args.radius_footprint,
        args.radius_shape,
    )
    r_max = max(r_h, r_f, r_s)
    for geom in seeds.geometry:
        sx, sy = geom.x, geom.y
        idx = np.asarray(
            tree.query_ball_point([sx, sy], r_max), dtype=np.int64
        )
        d = np.hypot(cx[idx] - sx, cy[idx] - sy)
        rec = {}
        # height and presence within r_h
        m = d <= r_h
        rec.update(percentile_block(hag[idx[m]], "_1m"))
        fr = hag[idx[m & first[idx]]]
        if len(fr) > 5:
            rec["canopy_top_1m"] = float(np.percentile(fr, 95))
        elif len(fr):
            rec["canopy_top_1m"] = float(fr.max())
        else:
            rec["canopy_top_1m"] = 0.0
        rec["n_first_1m"] = int(len(fr))
        rec["lidar_status"] = (
            "dead_missing"
            if rec["canopy_top_1m"] < args.dead_threshold
            else "alive"
        )
        # size within r_f
        m = d <= r_f
        v, a = cell_max_volume(cx[idx[m]], cy[idx[m]], hag[idx[m]], args.cell)
        rec["vol_cellmax_ff"], rec["area_ff"] = v, a
        rec["voxel_vol_ff"], rec["voxel_porosity_ff"], _ = voxel_stats(
            cx[idx[m]], cy[idx[m]], hag[idx[m]], args.cell
        )
        rec["hull_vol_ff"] = hull_volume(cx[idx[m]], cy[idx[m]], hag[idx[m]])
        rec.update(
            {
                f"{k}_ff": v
                for k, v in percentile_block(hag[idx[m]], "").items()
            }
        )
        # shape within r_s
        m = d <= r_s
        rec.update(
            {
                f"{k}_sh": v
                for k, v in shape_metrics(
                    cx[idx[m]],
                    cy[idx[m]],
                    hag[idx[m]],
                    along,
                    across,
                    args.cell,
                ).items()
            }
        )
        rec["n_sh"] = int(m.sum())
        rows.append(rec)
    return pd.DataFrame(rows, index=seeds.index)


def measure_segments(
    seeds, segments, seg_id, cx, cy, hag, along, across, dtm_path, args
):
    """Per-polygon metrics, assigned to seeds through seg_id."""
    with rasterio.open(dtm_path) as src:
        transform, shape = src.transform, (src.height, src.width)
    shapes = [
        (geom, int(i))
        for geom, i in zip(segments.geometry, segments[args.seg_id_col])
    ]
    grid = rasterize(
        shapes, out_shape=shape, transform=transform, fill=0, dtype="int32"
    )
    inv = ~transform
    col = np.floor(inv.a * cx + inv.b * cy + inv.c).astype(np.int64)
    row = np.floor(inv.d * cx + inv.e * cy + inv.f).astype(np.int64)
    ok = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
    label = np.zeros(len(cx), dtype=np.int32)
    label[ok] = grid[row[ok], col[ok]]
    order = np.argsort(label, kind="stable")
    lab_sorted = label[order]
    uniq, starts = np.unique(lab_sorted, return_index=True)
    ends = np.r_[starts[1:], len(lab_sorted)]
    by_label = {int(u): order[s:e] for u, s, e in zip(uniq, starts, ends) if u}
    rows = []
    for sid in seg_id:
        rec = {}
        if pd.isna(sid) or int(sid) not in by_label:
            rows.append(rec)
            continue
        idx = by_label[int(sid)]
        rec.update(percentile_block(hag[idx], "_seg"))
        v, a = cell_max_volume(cx[idx], cy[idx], hag[idx], args.cell)
        rec["vol_cellmax_seg"], rec["area_cells_seg"] = v, a
        rec["voxel_vol_seg"], rec["voxel_porosity_seg"], _ = voxel_stats(
            cx[idx], cy[idx], hag[idx], args.cell
        )
        rec["hull_vol_seg"] = hull_volume(cx[idx], cy[idx], hag[idx])
        rec.update(
            {
                f"{k}_seg": v
                for k, v in shape_metrics(
                    cx[idx], cy[idx], hag[idx], along, across, args.cell
                ).items()
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows, index=seeds.index)


def join_segments(seeds, segments, treetops, args):
    """Assign a segment identifier to each seed.

    Join refined tree tops on --id-col when given; otherwise use the
    containing polygon or nearest polygon within 1 m.
    """
    if treetops is not None:
        if args.id_col == args.seg_id_col:
            # the seeds are the refined tree tops themselves
            return seeds[args.id_col]
        id_col = args.id_col
        if id_col not in treetops.columns and id_col[:10] in treetops:
            # a shapefile truncates field names to ten characters
            id_col = id_col[:10]
        tt = treetops[[id_col, args.seg_id_col]].rename(
            columns={id_col: args.id_col}
        )
        tt[args.id_col] = tt[args.id_col].astype(str)
        if tt[args.id_col].str.endswith(".0").any():
            # a numeric id read back as float
            tt[args.id_col] = tt[args.id_col].str.replace(
                r"\.0$", "", regex=True
            )
        key = seeds[args.id_col].astype(str)
        m = key.map(
            tt.drop_duplicates(args.id_col).set_index(args.id_col)[
                args.seg_id_col
            ]
        )
        LOG.info(
            "%d of %d seeds matched a refined tree top on %s",
            int(m.notna().sum()),
            len(seeds),
            args.id_col,
        )
        return m
    j = gpd.sjoin_nearest(
        seeds[["geometry"]],
        segments[[args.seg_id_col, "geometry"]],
        how="left",
        max_distance=1.0,
    )
    j = j[~j.index.duplicated(keep="first")]
    LOG.info(
        "%d of %d seeds fall in or within 1 m of a segment",
        int(j[args.seg_id_col].notna().sum()),
        len(seeds),
    )
    return j[args.seg_id_col]


# --------------------------------------------------------------------------
def main(argv=None):
    """Parse arguments and write per-plant traits."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--las", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument(
        "--xy-cols",
        nargs=2,
        metavar=("X", "Y"),
        help="coordinate columns when --seeds is a CSV",
    )
    ap.add_argument(
        "--seeds-crs", default=None, help="CRS of CSV seeds, e.g. EPSG:32616"
    )
    ap.add_argument(
        "--id-col",
        default=None,
        help="identifier column carried to the output and used to "
        "join --treetops",
    )
    ap.add_argument(
        "--row-col",
        default=None,
        help="row column in the seeds, for the row axis",
    )
    ap.add_argument("--segments", default=None)
    ap.add_argument("--treetops", default=None)
    ap.add_argument("--seg-id-col", default="tree_id")
    ap.add_argument("--radius-height", type=float, default=1.0)
    ap.add_argument("--radius-footprint", type=float, default=0.55)
    ap.add_argument("--radius-shape", type=float, default=1.4)
    ap.add_argument("--cell", type=float, default=0.1)
    ap.add_argument("--min-hag", type=float, default=0.10)
    ap.add_argument("--max-hag", type=float, default=6.0)
    ap.add_argument("--dead-threshold", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    x, y, z, cls, rn = load_points(args.las)
    hag, dtm_crs = height_above_ground(x, y, z, args.dtm)
    canopy = (
        (cls != 2)
        & np.isfinite(hag)
        & (hag > args.min_hag)
        & (hag <= args.max_hag)
    )
    LOG.info(
        "%d canopy returns between %.2f and %.1f m",
        int(canopy.sum()),
        args.min_hag,
        args.max_hag,
    )
    cx, cy, ch = x[canopy], y[canopy], hag[canopy]
    first = rn[canopy] == 1
    del x, y, z, cls, rn, hag

    seeds = read_seeds(args.seeds, args.xy_cols, args.seeds_crs)
    if seeds.crs is not None and dtm_crs is not None and seeds.crs != dtm_crs:
        LOG.info("reprojecting seeds from %s to %s", seeds.crs, dtm_crs)
        seeds = seeds.to_crs(dtm_crs)
    along, across = row_axes(seeds, args.row_col)

    out = seeds.drop(columns="geometry").copy()
    out["seed_x"] = seeds.geometry.x.values
    out["seed_y"] = seeds.geometry.y.values
    fp = measure_footprints(seeds, cx, cy, ch, first, along, across, args)
    out = pd.concat([out, fp], axis=1)

    if args.segments:
        segments = gpd.read_file(args.segments)
        if segments.crs != seeds.crs:
            segments = segments.to_crs(seeds.crs)
        treetops = gpd.read_file(args.treetops) if args.treetops else None
        if treetops is not None and not args.id_col:
            raise SystemExit("--treetops needs --id-col to join on")
        seg_id = join_segments(seeds, segments, treetops, args)
        out["seg_id"] = seg_id.values
        sg = measure_segments(
            seeds,
            segments,
            seg_id.values,
            cx,
            cy,
            ch,
            along,
            across,
            args.dtm,
            args,
        )
        out = pd.concat([out, sg], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    LOG.info(
        "wrote %s (%d rows, %d columns)", args.out, len(out), out.shape[1]
    )
    status = out["lidar_status"].value_counts().to_dict()
    LOG.info("lidar_status: %s", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
