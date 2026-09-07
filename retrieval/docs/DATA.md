# Data manifest — measured, do not re-derive

Everything here was measured by reconnaissance on 2026-09-07. A fresh session
should **trust this file and not repeat the survey** — it cost two long agent
runs. Re-measure only if the data directory changes.

**Data root:** `/home/omer/PycharmProjects/Dynamic-Terrain/data` — **READ-ONLY.**
It belongs to another project. Never write, move or convert in place.

---

## The index: 8 source scenes, all 10.0-10.5 cm/px

| AOI | Path (relative to data root) | W x H | Bands | CRS | proj px (m) | **true GSD** | Date | 448 tiles |
|---|---|---|---|---|---|---|---|---|
| leb | `leb/2022-10-29.tif` | 20179x9784 | 3 | 3857 | 0.125002 | **10.47 cm** | 2022-10-29 | 945 |
| leb | `leb/2025-06-06.tif` | 20179x9784 | 3 | 3857 | 0.125002 | **10.47 cm** | 2025-06-06 | 945 |
| AYOSH | `AYOSH/X693_Y3500.tif` | 10240x10240 | 3 | 32636 | 0.100000 | **10.00 cm** | unknown | 484 |
| AYOSH | `AYOSH/X693_Y3501.tif` | 10240x10240 | 3 | 32636 | 0.100000 | **10.00 cm** | unknown | 352* |
| gaza | `gaza/X625_Y3404.tif` | 10240x10240 | 3 | 32636 | 0.100000 | **10.00 cm** | unknown | 484 |
| gaza | `gaza/X625_Y3405.tif` | 10240x10240 | 3 | 32636 | 0.100000 | **10.00 cm** | unknown | 484 |
| gaza | `gaza/X625_Y3406.tif` | 10240x10240 | 3 | 32636 | 0.100000 | **10.00 cm** | unknown | 484 |
| — | `X605_Y3388.tif` | 10240x10240 | 3 | 32636 | 0.100000 | **10.00 cm** | unknown | 397* |

\* reduced by nodata: `X693_Y3501` is 25.1% zero-fill, `X605_Y3388` is 14.2%.

**Total ~4,146 valid tiles at 448 px.** With the `[448, 224, 112]` pyramid:
5,198 + 20,704 + 82,640 = **108,542** tiles.

### The GSD trap — read this before computing any ground distance

`gdalinfo` prints **0.125 m** for `leb`. That is wrong as a ground distance.
EPSG:3857 metres are inflated by `1/cos(latitude)`. At `leb`'s centre latitude
33.10707:

```
true_gsd = 0.125002 * cos(radians(33.10707)) = 0.125002 * 0.837651 = 0.10471 m
448 px tile = 448 * 0.10471 = 46.91 TRUE ground metres   (not 56.0)
```

The UTM scenes (EPSG:32636, k0=0.9996 at eastings 709-710 km) are within 0.04%
of unity — **no correction needed**, true GSD is 10.00 cm.

A 448 px tile therefore frames ~45-47 m in both groups. They differ by 4.7%, so
one tile means the same thing across all 8 scenes. They sit in **two different
CRSs**; only the geo-lookup layer needs to care.

### Effective resolution is coarser than the grid

Measured: gradient-growth ratios and downsample residuals (4x box-downsample
then nearest-neighbour re-upsample loses only 2.9% of variance on 2022, 5.3%
on 2025) both indicate the true optical resolution is **~35-45 cm** — this is
sub-metre satellite imagery resampled onto a fixed web-map grid. 2022 is
measurably softer than 2025.

**Consequence for query design.** A 4.5 m car is 43x17 px at 10.47 cm/px but
carries only ~10-13 independent samples along its length. Supportable:
presence and coarse class — `car`, `tents`, `debris`, `building`, `dirt road`.
**Not** supportable: fine attributes such as vehicle colour, model, or trim.
Do not promise those in the UI (spec U-5).

---

## Source vs derived — the rule is measured, not heuristic

**Every derived raster in this tree has <= 6 distinct values per band. Every
source raster has 187-256.** Classify by value distribution and band count,
**never by filename.**

On `leb`, exactly **2 of 31** rasters are source imagery. Outside `leb`, only
**12 of ~232**. Getting this wrong produces a plausible-looking index that is
silently meaningless.

---

## Excluded, with cause

| What | Cause |
|---|---|
| `sin/Sini_Oct_Det_2025.tif` | 65536x98304, 6.44 Gpx, **true GSD 415.7 cm** (4.08-4.23 m across the scene). A 448 tile spans 1.86 km; nothing man-made resolvable. Was **86% of the nominal tile budget**. Shares EPSG:3857 with `leb` at 40x different scale — a trap. No overviews. |
| `Teheran/` (66 source) + `iran/x1_y7.tif` | **50 cm/px**, EPSG:32639. A 448 tile spans 224 m — a city block. Vehicles not resolvable. Admissible later as a **separate coarse tier**, never in the same embedding space (25x the ground area per tile). |
| `gaza.tiff` (loose) | **Not aerial.** 1600x800, no CRS, no geotransform. A ground-level oblique press photograph of armour with the Gaza skyline. A web/UI asset. |
| `iran/Ax00_y06.tif`, `iran/Ax01_y06.tif` | md5-identical to `Teheran/`'s. |
| ~230 derived rasters | Binary masks, class rasters, fused CD output. See the rule above. |

---

## Evaluation ground truth — already on disk, no labelling needed

**Primary (spec E-1): `tile_cropped_x3308_y3674_z0.125.tif`** in the data root.
8192x8192, 1-band uint8 **palette**, EPSG:4326 at ~12.1 cm, PACKBITS,
strip-organised (8192x1), nodata=0. Covers lon 35.2289-35.2395, lat
33.1071-33.1162 — **the `leb` AOI**.

Carries a GDAL metadata tag **`ID_TO_LABEL_MAPPING` with 45 classes**, 31
actually present. Directly useful class names include `Car`, `House`,
`PavedRoad`, `DirtRoad`, `DirtRoadB`, `Pavement`, `Water`, `GreenGrassland`,
`BrickWall`, `Shadow`, plus orchard / field / batha / garigue / maquis
vegetation classes and limestone / dolomite / basalt / chalk terrain classes.
Class 0 (`Unclassified`) covers ~47%. Companion to
`All_Valid_Labels_45.txt` and `All_Valid_Labels_48.txt` in the data root.

**Secondary:** `leb/cls_leb_legend.txt` and `cls_leb_names.txt` — the 10-class
legend (road, sidewalk, building, rut, rock, wall, fence, tree/hedge,
grass/soil, **car**). `leb/ai_segmentation.gpkg` (3.9 MB vector). The `cdinf`
positive-detection shapefiles under `leb/` for building-to-debris and roads.

Use these to compute retrieval recall@k without any manual labelling: query a
class name, check the returned tiles against the label raster.

---

## I/O characteristics

| Scene set | Block layout | Overviews | Amplification for one 448^2 RGB read |
|---|---|---|---|
| `leb` (2 scenes) | **20179 x 1 strips**, uncompressed | none | **~45x** |
| AYOSH / gaza / X605 (6 scenes) | **tiled 256x256** | none | ~2.9x |

**`leb` is the anomaly, not the norm.** The six 0.1 m scenes are already
properly tiled and need no conversion. Build COG copies for `leb`'s two scenes
only — under `retrieval/index/cog/`, never in place.

---

## Dates: mostly absent (spec F-7 is narrowed because of this)

Only `leb` is a true multi-date pair over one footprint. **Five of the eight
indexed footprints carry no acquisition date in the file at all.** The two
non-indexed groups that do carry *acquisition windows*, not dates — and
`iran/x1_y7`'s window spans **2015-09-15 to 2024-04-14**, 8.6 years, so a
single mosaic may internally mix epochs.

Owner decision: **date filtering is implemented for `leb` only**, with the
manifest infrastructure built so other AOIs can be dated later. Scenes with no
known date carry `date: unknown` and are excluded from date-filtered queries
rather than silently included.

---

## Prior art in this repo — this is the second attempt

`/data/vlm_logs/` holds four JSON records from 2026-02-16 13:31-13:34 UTC: a
bbox-plus-text-prompt UI over a 10240^2 scene. Operator prompts were
**`"tents"`** and **`"find all tents in the frame"`**. Every record carries
**`"mock": true`**, the analysis paragraph is byte-identical across all four,
and detections are always `Vegetation@0.87 / Road@0.78 / Building@0.65` with
bboxes computed as fixed fractions of the window. **A stub endpoint — no real
inference ever ran.** No usable labels there.

What it does establish: the intended workload is exactly small-object text
retrieval on the 10 cm imagery, and the coordinate space (max 6145, 5319) is
consistent with `X605_Y3388.tif` — the tent camp. That scene is also the
canonical demo input in the sibling project's docs
(`Dynamic-Terrain/docs/PIPELINE_API_REFERENCE.md:43`,
`docs/ST_INFERENCE_PIPELINE.md:245`, `README.md:198`).

**Recommended demo AOI: `X605_Y3388.tif`** — a displaced-persons tent camp
among date palms, with individually identifiable cars. It exercises small-object
retrieval on the exact content the previous attempt was aiming at.

## Content notes per AOI

- **leb** — rural-to-peri-urban hill village, S Lebanon (Bint Jbeil district).
  Terraced olive groves, stone walls, flat-roofed concrete houses. The
  2022->2025 pair is a **destruction pair**: intact village core in 2022, the
  same pixels bright rubble in 2025.
- **AYOSH** — N West Bank, Jenin governorate. Olive groves in planted rows,
  terraced limestone, hilltop village, greenhouses. Vehicles resolvable.
- **gaza** — E/NE Gaza City (Shuja'iyya / Tuffah). Near-total urban
  destruction: rubble fields, pancaked slabs, bulldozed tracks. Few vehicles.
- **X605_Y3388** — coastal Khan Yunis / al-Mawasi. Dense tent/tarpaulin
  shelter camp among date palms; white, grey and blue tarps, sand tracks.
  Individual cars unambiguous.

All are nadir or near-nadir. All are 3-band uint8 RGB.
