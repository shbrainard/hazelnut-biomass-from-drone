# Photo crown outlines

`python -m crown_outlines` processes one flight: a photo mosaic, a classified
point cloud, a DTM, one or more CHMs, and plant points. It extracts a photo
crown around each planting position, then measures canopy returns and CHM
volume inside the final mask. It does not change the existing watershed or
plant-trait modules.

## Inputs and execution

Run from the repository root in an environment containing NumPy, SciPy,
pandas, rasterio, pyproj, geopandas, shapely, laspy, a LAZ reader when needed,
Pillow, torch, and segment_anything. Torch and segment_anything are imported
only when loading the predictor; the geometry, sampling, measurement, and
synthetic tests do not need them.

The official ViT-B checkpoint is
[`sam_vit_b_01ec64.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).
Obtain that file separately and pass its local path. The module never
downloads files.

```sh
python -m crown_outlines \
  --orthophoto /data/flight/rgb.tif \
  --cloud /data/flight/classified.laz \
  --dtm /data/flight/dtm.tif \
  --chm /data/flight/chm_a.tif /data/flight/chm_b.tif \
  --plants /data/plants.gpkg --id-col plant_number \
  --checkpoint /data/checkpoints/sam_vit_b_01ec64.pth \
  --output /data/flight/outlines --review
```

Plant points must have a CRS and unique, non-null identifiers. CSV points
are also accepted with `--plants-crs EPSG:32616` and, if necessary,
`--xy-cols easting northing`. Point layers are transformed into the point
cloud CRS. The point cloud must declare a projected CRS with meter units.
The DTM and north-up CHMs must use that same CRS; the orthophoto can use a
different CRS. Each CHM's valid cell at the plant point is checked in argument
order, and the first valid match supplies the entire outline's CHM volume.
No covering CHM produces NaN, not zero. Where two CHMs overlap at a plant,
list the plant's own CHM first, or run each CHM's plants separately.

An optional `--reference-heights /data/traits.csv` provides heights keyed by
`--reference-id-col` (default `plant_number`) in `--height-col` (default
`p99_1m`, meters). Missing identifiers fall back to computation. A supplied
NaN height yields the 0.25 m minimum box threshold, as in the pilot. Computed
p99 uses non-ground, finite returns within 1 m, with height strictly greater
than 0.10 m and at most 6 m, matching `plant_traits.per_plant_traits`.

## Method and defaults

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--crop-size` | 3.2 | Square width in meters |
| `--resolution` | 0.005 | Crop pixel width in meters |
| `--overview-level` | none | Rasterio overview index for the orthophoto |
| `--offset EAST NORTH` | 0 0 | Added offsets in the orthophoto's CRS grid |
| `--xy-cols` | x y | Coordinate columns for CSV plants |
| `--plants-crs` | none | Required CRS for CSV plants |
| `--reference-heights` | none | Optional reference height CSV |
| `--reference-id-col` | plant_number | Reference height join column |
| `--height-col` | p99_1m | Reference height column in meters |
| `--smoothing-radius` | 6 | Integer disk radius in pixels; 0 disables smoothing |
| `--visible-height` | 0.60 | Minimum outline p95 height in meters |
| `--device` | automatic | MPS when available, otherwise CPU; can force either |
| `--review` | off | Write PNG crops with exported polygon edges |

The crop grid is centered on each plant in the point cloud CRS, with rows
north to south. Defaults produce 640 by 640 RGB pixels. Sampling transforms
each target location to the orthophoto CRS, adds the east and north offsets,
and uses bilinear interpolation with a zero-filled boundless read. Positive
east moves the sampling location east in the source mosaic. The direct
inverse-affine sampling convention intentionally retains the pilot's
half-source-pixel phase; no extra half-pixel correction is applied.

The point loader and DTM height helper are imported from
`plant_traits.per_plant_traits`. Class 7 noise is dropped by that loader;
class 2 ground is removed, and each plant uses returns within 1.7 m. The
current helper reads the full cloud before spatial filtering, so large
clouds require enough memory for its arrays.

The upper-canopy prompt uses returns within 1.5 m, with heights between
`max(0.25, 0.5 * p99_1m)` and 6 m. Returns occupy 0.1 m crop-grid cells.
One binary closing is followed by the four-connected component holding or
nearest the center cell. Its bounds become a box padded by 20 crop pixels
and clipped to `[0, image_width - 1]`. There is no box when the footprint is
empty. Crop dimensions and resolution must divide into exact 0.1 m cells.

Segment Anything ViT-B receives the center point, a positive label, and the
optional box, with multiple outputs requested. Selection takes the smallest
mask with predicted score at least 0.75, falling back to the highest score.
Ties take the first candidate. Cleaning keeps the four-connected piece at
or nearest the center and fills holes.

Smoothing closes then opens with the exact integer disk
`x*x + y*y <= radius*radius`, with zero padding to avoid artificial border
erosion. It fills holes and keeps the center piece again. An erased tiny
mask retains its cleaned unsmoothed footprint and is flagged
`erased_retained_original`. Radius 0 returns the cleaned research mask
unchanged. Empty predictions remain empty and have no invented polygon.

Polygonization follows pixel edges, repairs geometry, and retains the
largest polygon. Topology-preserving simplification tries 1, 0.5, 0.25, then
0 pixels. As in the review export, normalized coordinates are rounded to
six decimals and the first valid, simple, hole-free polygon with raster
IoU at least 0.95 is used. Coordinates are then georeferenced to the cloud
CRS. The final mask remains the measurement footprint; the simplified vector
is an approximation recorded with its own area and raster IoU.

Inside the final mask, height measurements use returns from 0.10 through
6 m inclusive. Count is always reported; p99 and p95 are NaN with fewer than
five returns. CHM volume sums nonnegative valid cell heights times cell
area, including cells by their centers. Per-cell maximum-return volume
uses 0.1 m cells anchored to the crop's west and north edges. The return
pixel indexing and CHM window rounding preserve the research conventions.
`visible` is true when p95 is at least `--visible-height`; unknown p95 is
false and remains distinguishable by its missing height and return count.

## Outputs

- `outlines.gpkg`, layer `outlines`: one polygon record per input plant, in
  the cloud CRS. Empty predictions retain a record with empty geometry.
- `outlines.csv`: identical attributes without geometry.
- `review/000001.png`, etc., when requested: crops with cyan polygon edges,
  numbered in input order to avoid unsafe identifiers in filenames.

The input identifier is preserved as the string `plant_id`. Attributes are
`mask_choice` (zero-based), `mask_score`, `smoothing_status`,
`smoothing_radius_px`, `box_threshold_m`, `reference_p99_m`,
`simplify_tolerance_px`, `polygon_mask_iou`, `polygon_area_m2`,
`mask_area_m2`, `h_outline_cm` (p99), `h_outline_p95_cm`,
`vol_outline_chm_m3`, `vol_outline_cellmax_m3`, `n_returns_outline`,
`visible`, and `chm_source`. Height attributes are centimeters; areas are
square meters and volumes are cubic meters. Numeric measurements are saved
without research-report rounding. Existing output filenames are replaced
when the same output directory is used again.

## Evidence for the defaults

On a 74-plant Valley Farm pilot, CHM volume inside the photo outline was a
median 19 percent (2025) and 23 percent (2026) below the watershed-segment
volume, with rankings preserved: Spearman correlations were 0.994 and
0.985. Heights inside the outline moved a median 0.5 to 1 cm. Against 2025
tape heights, RMSE was 13.8 cm versus 13.9 cm on 72 plants.

Across all 1,709 plants in both years, median outline volume was 0.80 (2025)
and 0.79 (2026) times watershed-segment volume. Outline p95 was below 0.60 m
for 162 of 1,685 plants in 2025 and 103 of 1,689 in 2026.

The p95 >= 0.60 m rule retained 99 of 103 visible shrubs and none of four
grass crops in the checked sample. Below 0.60 m, nothing tested separated a
small hazel from a grass clump. The flag therefore identifies outlines for
human review; it is not a species classification.

With 3 cm smoothing on 3,418 outlines, 74.5 percent of changed pixels were
additions, filling inlets and gaps. Five tiny outlines erased by opening
were retained unsmoothed and flagged. These measurements motivate the
6-pixel default radius and its explicit fallback.

## Verification

Synthetic tests use planted images, rasters, returns, and a fake predictor;
no checkpoint is needed. From the repository root:

```sh
python -m pytest -q -p no:cacheprovider
```

Run against the inputs of the research run that first produced these
outlines, the module reproduced the mask choice, outline area, p99 and p95
heights, per-cell maximum-return volume and return count exactly on 60
plant-years, and the CHM volume on 59. The exception is a plant at a trial
boundary where two CHMs overlap and the first one listed was used. Run from
the raw photo and point cloud on six plant-years, the crops, prompt boxes
and chosen masks were identical to the research run's.
