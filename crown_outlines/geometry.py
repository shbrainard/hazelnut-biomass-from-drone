"""Choose, clean, smooth, and polygonize crown masks."""

import numpy as np
from rasterio.features import rasterize, shapes
from scipy import ndimage
from shapely import make_valid
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, shape


def choose_mask(masks, scores, min_score=0.75):
    """Choose the smallest qualifying mask, or the highest scoring mask."""
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    eligible = np.flatnonzero(scores >= min_score)
    return (
        int(eligible[np.argmin(areas[eligible])])
        if len(eligible)
        else int(np.argmax(scores))
    )


def center_piece(mask):
    """Keep the four-connected piece holding or nearest the center pixel."""
    labels, count = ndimage.label(mask)
    if not count:
        return mask.copy()
    center = tuple(s // 2 for s in mask.shape)
    label = labels[center]
    if not label:
        indices = ndimage.distance_transform_edt(
            labels == 0, return_indices=True
        )[1]
        label = labels[tuple(indices[:, center[0], center[1]])]
    return labels == label


def clean_mask(mask):
    """Keep the center piece and fill its holes."""
    return ndimage.binary_fill_holes(center_piece(mask))


def smooth_mask(mask, radius=6):
    """Close then open with an integer disk, retaining erased tiny crowns."""
    if radius < 0 or int(radius) != radius:
        raise ValueError("Smoothing radius must be a nonnegative integer")
    original = clean_mask(mask)
    if radius == 0:
        return original, "disabled"
    if not original.any():
        return original, "empty"
    pad = 2 * radius + 2
    yy, xx = np.nonzero(original)
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    work = np.pad(original[y0:y1, x0:x1], pad)
    # EDT thresholds give exactly x*x + y*y <= radius*radius.
    work = ndimage.distance_transform_edt(~work) <= radius
    work = ndimage.distance_transform_edt(work) > radius
    work = ndimage.distance_transform_edt(work) > radius
    work = ndimage.distance_transform_edt(~work) <= radius
    full = np.zeros_like(original)
    full[y0:y1, x0:x1] = work[pad:-pad, pad:-pad]
    result = clean_mask(ndimage.binary_fill_holes(full))
    if not result.any():
        return original, "erased_retained_original"
    return result, "smoothed"


def largest_polygon(geom):
    """Repair geometry and retain its largest polygon."""
    geom = make_valid(geom) if not geom.is_valid else geom
    if geom.geom_type == "Polygon":
        return geom
    parts = []
    for child in getattr(geom, "geoms", []):
        poly = largest_polygon(child)
        if not poly.is_empty:
            parts.append(poly)
    return max(parts, key=lambda p: p.area) if parts else Polygon()


def polygon_from_mask(mask, origin):
    """Export pixel edges with the review packet's simplification checks.

    Return the georeferenced polygon, tolerance in pixels, and raster IoU.
    Measurements retain the final mask footprint, before vector approximation.
    """
    if not mask.any():
        return Polygon(), 0.0, 1.0
    parts = [
        shape(g)
        for g, value in shapes(mask.astype(np.uint8), mask=mask)
        if value == 1
    ]
    geom = largest_polygon(max(parts, key=lambda p: p.area))
    scale = np.array([mask.shape[1], mask.shape[0]])
    for tolerance in (1.0, 0.5, 0.25, 0.0):
        poly = geom.simplify(tolerance, preserve_topology=True)
        if poly.geom_type != "Polygon" or len(poly.interiors):
            continue
        coords = np.array(poly.exterior.coords[:-1])
        written = Polygon(np.round(coords / scale, 6) * scale)
        if not written.is_valid or not written.exterior.is_simple:
            continue
        back = rasterize([(written, 1)], out_shape=mask.shape).astype(bool)
        iou = np.count_nonzero(mask & back) / np.count_nonzero(mask | back)
        if iou >= 0.95:
            west, north, res = origin
            return (
                affine_transform(written, [res, 0, 0, -res, west, north]),
                tolerance,
                float(iou),
            )
    raise ValueError("Cannot export a valid polygon with raster IoU >= 0.95")
