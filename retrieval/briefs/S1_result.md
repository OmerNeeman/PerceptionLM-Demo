# S1 result — source classifier + inventory

**Status: `BLOCKED`**

Everything the brief asks for is written and **21/21 tests are green**. The
block is not code: it is the named blocker condition in the brief —

> Flag as a blocker rather than working around it if: the classifier does not
> return exactly the two expected `leb` rasters […] A disagreement with
> `docs/DATA.md` is a finding the PM must see.

The D-1 pixel rule, applied exactly as specified, returns **four** rasters
under `leb/`, not two. `spec.md` D-1 and `docs/DATA.md` both say "exactly 2 of
31". The **indexable set is still exactly the expected pair**, so the stage's
output is usable as-is — but the numeral in two documents is not what the rule
returns, and I did not tune the test to hide it. See **Finding 1**.

---

## Deliverables

| File | Lines | What |
|---|---|---|
| `retrieval/src/geo.py` | 217 | D-2: CRS-conditional ground-resolution maths |
| `retrieval/src/inventory.py` | 589 | D-1/D-3/D-4: classifier + inventory + JSON emit |
| `retrieval/tests/test_geo.py` | 140 | 12 tests |
| `retrieval/tests/test_inventory.py` | 347 | 9 tests |
| `retrieval/index/inventory.json` | 235,707 B | the artifact (gitignored) |

Nothing was written, moved, renamed or converted under
`/home/omer/PycharmProjects/Dynamic-Terrain/data`. S0's files were not opened
for writing.

## Inventory summary line

```
8 indexable, 169 derived_raster, 3 duplicate_of, 5 no_georeference,
69 resolution_regime_excluded, 16 too_few_bands, 270 rasters of 320 files scanned
```

```
indexable scenes (8):
  AYOSH/X693_Y3500.tif    10240x10240 3b EPSG:32636  10.00 cm/px  448px tile = 44.80 m
  AYOSH/X693_Y3501.tif    10240x10240 3b EPSG:32636  10.00 cm/px  448px tile = 44.80 m
  X605_Y3388.tif          10240x10240 3b EPSG:32636  10.00 cm/px  448px tile = 44.80 m
  gaza/X625_Y3404.tif     10240x10240 3b EPSG:32636  10.00 cm/px  448px tile = 44.80 m
  gaza/X625_Y3405.tif     10240x10240 3b EPSG:32636  10.00 cm/px  448px tile = 44.80 m
  gaza/X625_Y3406.tif     10240x10240 3b EPSG:32636  10.00 cm/px  448px tile = 44.80 m
  leb/2022-10-29.tif      20179x9784  3b EPSG:3857   10.47 cm/px  448px tile = 46.91 m
  leb/2025-06-06.tif      20179x9784  3b EPSG:3857   10.47 cm/px  448px tile = 46.91 m
```

Dimensions, band counts, CRSs and true GSDs all match `docs/DATA.md` exactly.

## The `leb` denominator — the actual counts

| Quantity | Count |
|---|---|
| files under `leb/` (any type) | **65** |
| of those, `.tif` / `.tiff` | **31** |
| readable rasters (GDAL opens them) | **32** — the 31 TIFFs + a 357x233 legend PNG |
| pass the D-1 pixel rule | **4** (3 of them TIFFs) |
| indexable source scenes | **2** |

**The denominator is 31, not 50.** `notes.md#intake`'s "2 of `leb`'s 50 TIFFs,
the other 48 derived" is wrong on both numbers; `spec.md` D-1 and
`docs/DATA.md` have the right denominator. The numerator **2** holds for the
indexable set and is asserted as such.

---

## QA checklist — RED then GREEN

Every command was run as
`env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python -m pytest …`
from `retrieval/`.

### RED 1 — tests written before any module exists

```
tests/test_geo.py:26: in <module>
    import geo
E   ModuleNotFoundError: No module named 'geo'
tests/test_inventory.py:22: in <module>
    import inventory
E   ModuleNotFoundError: No module named 'inventory'
2 errors in 0.24s
```

### D-2 · `test_gsd_correction` — RED on the raw geotransform

`geo.py` step 1: pixel size reported as ground distance (CLAUDE.md trap 3).

```
        gsd_cm = res.true_gsd_m * 100.0
>       assert gsd_cm == pytest.approx(10.47, abs=0.01), f"true GSD is {gsd_cm:.2f} cm/px"
E       AssertionError: true GSD is 12.50 cm/px
E       assert 12.500077882039701 == 10.47 ± 0.01
1 failed
```

The independent geodesic cross-check failed against it in the same run
(`0.10456 != 0.12500 ± 6.3e-04`), which is the measurement saying the same
thing without any Mercator arithmetic.

### D-2 (UTM) · `test_gsd_no_correction_for_utm` — RED on a globally applied cosine

`geo.py` step 2: `cos(lat)` applied to **every** projected CRS.

```
        gsd_cm = res.true_gsd_m * 100.0
>       assert gsd_cm == pytest.approx(10.00, abs=0.01), f"true GSD is {gsd_cm:.2f} cm/px"
E       AssertionError: true GSD is 8.54 cm/px
E       assert 8.539441552345613 == 10.0 ± 0.01
1 failed
```

and, in the same run:

```
>       assert abs(a - b) / max(a, b) < 0.05
E       assert (8.652066930952387 / 46.90876508546073) < 0.05
E        +  where 46.90876508546073 = max(46.90876508546073, 38.25669815450834)
```

38.26 m instead of 44.80 m for a 448 px UTM tile — exactly the ~8.4 cm
regression the brief predicted. Both directions of the trap are now pinned.

### D-1 · `test_source_classifier` — RED on a filename fast path

`inventory.py` step 1: `_decide_source` returned
`bool(re.match(r"^\d{4}-\d{2}-\d{2}$", Path(path).stem))`. Note that the
straightforward assertion (`indexable_paths(inv, subdir="leb")` == the pair)
**passed** — the filename rule happens to get `leb` right. Only the symlink
assertions caught it:

```
        for name in source_under_derived_names:
>           assert inventory.is_source_imagery(probe / name) is True, (
                f"{name} is real imagery reached through a derived-looking name; "
                "the classifier is reading the filename, not the pixels"
            )
E           AssertionError: cls_binary_ST48_no_softmax.tif is real imagery reached
E           through a derived-looking name; the classifier is reading the filename,
E           not the pixels
E           assert False is True
```

The probe (under `retrieval/index/symlink_probe/`, never in the data dir)
crosses the names over both ways:

| symlink name | target | truth |
|---|---|---|
| `cls_binary_ST48_no_softmax.tif` | `leb/2022-10-29.tif` | source |
| `2019-01-01_output_Roads_fused_mask.tiff` | `leb/2025-06-06.tif` | source |
| `2022-10-29.tif` | `leb/cache/2022-10-29_ST_output.tiff` (4-band softmax) | derived |
| `2025-06-06.tif` | `leb/cache/2025-06-06_debris_48_new_0706_binary.tiff` | derived |

This is the evidence that the symlink clause in the brief was load-bearing:
without it, a filename classifier passes the obvious test.

### GREEN — all 21

```
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[3857-mercator] PASSED
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[3395-mercator] PASSED
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[32636-projected] PASSED
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[32639-projected] PASSED
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[4326-geographic] PASSED
tests/test_geo.py::test_crs_kind_of_none_is_none PASSED
tests/test_geo.py::test_tile_ground_extent_is_gsd_times_pixels PASSED
tests/test_geo.py::test_gsd_correction PASSED
tests/test_geo.py::test_mercator_correction_agrees_with_an_independent_geodesic_measurement PASSED
tests/test_geo.py::test_gsd_no_correction_for_utm PASSED
tests/test_geo.py::test_utm_no_correction_agrees_with_an_independent_geodesic_measurement PASSED
tests/test_geo.py::test_the_two_regimes_frame_the_same_ground_area_within_5_percent PASSED
tests/test_inventory.py::test_source_classifier PASSED
tests/test_inventory.py::test_d1_pixel_rule_over_leb_finds_more_than_the_two_scenes PASSED
tests/test_inventory.py::test_leb_denominator_is_reported_as_a_fact PASSED
tests/test_inventory.py::test_classification_is_by_band_count_and_distinct_levels_only PASSED
tests/test_inventory.py::test_source_inventory_matches_manifest PASSED
tests/test_inventory.py::test_summary_line_states_counts_per_reason PASSED
tests/test_inventory.py::test_data_dir_readonly PASSED
tests/test_inventory.py::test_inventory_deterministic PASSED
tests/test_inventory.py::test_the_pixel_rule_has_a_wide_margin PASSED

======================== 21 passed, 1 warning in 36.57s ========================
```

### D-3 — verified independently of pytest, at the shell

```
$ find $DATA -printf '%p %s %T@\n' | sort | sha256sum     # before
098dd8ebbe1db3466896ec6eaf5bea9b07b1923b49ff9c8f01d37eb7687195f4  -
   … two full inventory runs …
$ find $DATA -printf '%p %s %T@\n' | sort | sha256sum     # after
098dd8ebbe1db3466896ec6eaf5bea9b07b1923b49ff9c8f01d37eb7687195f4  -
IDENTICAL

$ find $DATA -newermt '2026-09-07' -print
(nothing)
```

`test_data_dir_readonly` additionally asserts that every path in
`inventory.output_paths()` resolves inside `retrieval/index/`, that
`write_inventory` **raises** for a path outside it, and — statically — that no
`rasterio.open(...)` call in `geo.py` or `inventory.py` passes `"r+"`, `"w"`
or `"a"`.

### D-4 — byte-identical across runs

```
D-4 byte-identical: True
sizes: 235707 235707
digest run a: sha256:7cacf2fe1ca4c3141329d62a0d71068967914847ac2f7d62ed2f5a555045bf08
digest run b: sha256:7cacf2fe1ca4c3141329d62a0d71068967914847ac2f7d62ed2f5a555045bf08
```

Sampling strategy, recorded in the artifact:

```json
"sampling": {
  "strategy": "fixed_stride_window_grid", "grid": 8, "window_px": 64,
  "pixels_per_band": 262144, "seed": null,
  "note": "no RNG: window origins are round(i*(extent-win)/(grid-1))",
  "dedup_grid": 4, "dedup_window_px": 8
}
```

A fixed 8x8 grid of 64x64 windows, 262,144 sampled pixels per band. No RNG at
all, so there is no seed to get wrong. Nothing is read whole —
`sin/Sini_Oct_Det_2025.tif` is 6.44 Gpx and the full 320-file walk takes ~9 s.

### D-2 cross-check — the rule against an independent geodesic measurement

Not required by the brief; it is the strongest single piece of evidence that
the *conditional* is right, so it is in the artifact and in two tests. Each
per-pixel distance is also measured on the WGS84 ellipsoid via `pyproj.Geod`,
which knows nothing about Mercator:

| scene | CRS kind | correction | rule GSD | geodesic GSD |
|---|---|---|---|---|
| `leb/2022-10-29.tif` | mercator | 0.8376513 | 10.4707 cm | 10.4565 cm |
| `leb/2025-06-06.tif` | mercator | 0.8376513 | 10.4707 cm | 10.4565 cm |
| `AYOSH/X693_Y3500.tif` | projected | 1.0 | 10.0000 cm | 9.9986 cm |
| `X605_Y3388.tif` | projected | 1.0 | 10.0000 cm | 10.0022 cm |
| `gaza/X625_Y3404.tif` | projected | 1.0 | 10.0000 cm | 10.0016 cm |

Mercator: rule and measurement agree to 0.14%. UTM: to 0.02% — which is the
"within 0.04% of unity" claim in `docs/DATA.md`, measured rather than asserted.

---

## Findings — three disagreements with `docs/DATA.md` / `spec.md`

### Finding 1 (the blocker) — `leb` has 4 pixel-rule sources, not 2

`spec.md` D-1 and `docs/DATA.md` both state "exactly 2 of 31 rasters are
source imagery" on `leb`. The rule as written returns four:

| path | why it passes D-1 | why it is not indexable |
|---|---|---|
| `leb/2022-10-29.tif` | 3b, 238/245/248 distinct | — **indexed** |
| `leb/2025-06-06.tif` | 3b, source | — **indexed** |
| `leb/leb_crop_x4096_y3072_1024.tif` | 3b uint8, EPSG:3857, 236/246/255 distinct | `duplicate_of` `leb/2022-10-29.tif`, kind `contained_crop` |
| `leb/tmp_results/cls_leb_legend.png` | 4b PNG, many colours | `no_georeference` |

`leb_crop_…` is **byte-identical** to the 1024x1024 window of
`leb/2022-10-29.tif` at pixel offset (4096, 3072) — verified by direct pixel
comparison, `md5(window) == md5(crop) == e3b380b8e5c3`, and *not* identical to
the same window of `2025-06-06.tif` (`94173b61df3c`). It is real imagery over
ground already covered. The PNG is a 357x233 colour legend.

So the rule is sound, the index is right, and the documents' numeral is
wrong. Two ways to settle it, PM's call:

1. **Amend the docs.** `spec.md` D-1's *Expected* becomes "exactly 2 of 31
   `leb` rasters are indexable source scenes; the pixel rule alone also
   accepts a byte-identical crop of `2022-10-29.tif` and a legend PNG, both
   rejected downstream with specific causes." Nothing in `src/` changes.
2. **Narrow D-1 itself** to fold georeference and de-duplication into the
   definition of "source imagery". This changes the meaning of D-1 and would
   make `is_source_imagery` a compound predicate — I did **not** do this,
   because it touches `spec.md`.

I recommend (1): keeping "source pixels" and "indexable scene" as two separate
questions is what makes the five rejection reasons meaningful, and
`gaza.tiff`'s existence in the excluded table already relies on that split.

### Finding 2 — "every derived raster has <= 6 distinct values per band" is too strong

`docs/DATA.md` and `CLAUDE.md` both say so. Six rasters exceed it:

```
leb/cache/2022-10-29_ST_output.tiff              4b [13, 14, 15, 1]
leb/cache/2022-10-29_final_pipeline_output.tiff  4b [13, 16, 16, 1]
leb/cache/2025-06-06_ST_output.tiff              4b [14, 11, 11, 1]
leb/cache/2025-06-06_final_pipeline_output.tiff  4b [13, 13, 13, 1]
leb/tmp_results/2022-10-29seg_output.tiff        3b [7, 7, 8]
tile_cropped_x3308_y3674_z0.125.tif              1b [25]
```

The measured margin is therefore **25 (highest non-source band) vs 180 (lowest
source band)**, not 6 vs 187. The 64 threshold still sits comfortably in the
middle — 2.6x above the ceiling, 2.8x below the floor — so **D-1 needs no
change**, but the "<= 6" claim should be corrected to "<= 25" in both
documents. `test_the_pixel_rule_has_a_wide_margin` asserts the gap so a future
data change that closes it fails loudly.

(The source floor of 180 vs `DATA.md`'s "187-256" is a sampling difference,
not a contradiction — my grid samples 262,144 px/band, not the whole raster.)

### Finding 3 — "12 of ~232 rasters are real imagery" is wrong

`CLAUDE.md` and `docs/DATA.md` say 12 of ~232. Measured: **85 source-imagery
rasters of 270 readable rasters**, from 320 files.

```
Teheran 66 · iran 6 · leb 4 · gaza 3 · AYOSH 2 · <root> 2 · sin 2  = 85
```

The 12 appears to have excluded `Teheran/`'s 66 (which `DATA.md`'s own
excluded-with-cause table calls "66 source"), the three `*_seg_vis.png`s and
`sin/sin_min.tiff`. The raster denominator is 270, not 232 (266 TIFFs + 4
PNGs). None of this changes the indexable 8; it matters only because "12 of
232" is the figure a future worker would sanity-check against.

Also, `sin/sin_min.tiff` (408.3 cm true GSD) is source imagery in the same
excluded regime as `sin/Sini_Oct_Det_2025.tif` (415.7 cm) but is not in
`DATA.md`'s excluded table. Worth one line there.

---

## Rejection reasons — the five, all in use and distinguishable

| reason | count | example |
|---|---|---|
| `derived_raster` | 169 | `leb/tmp_results/2025-06-06_output_Trees_ST48_no_softmax.tif` — 4 bands, `[2,2,2,2]` |
| `too_few_bands` | 16 | `tile_cropped_x3308_y3674_z0.125.tif` — 1-band palette |
| `no_georeference` | 5 | `gaza.tiff` — real 3-band RGB, no CRS, identity transform |
| `resolution_regime_excluded` | 69 | `sin/…` 415.7 cm, `Teheran/…` + `iran/x1_y7.tif` 50.0 cm |
| `duplicate_of` | 3 | `iran/Ax0{0,1}_y06.tif` (identical footprint), `leb/leb_crop_…` (contained crop) |

`gaza.tiff` and a binary mask are kept apart exactly as the brief requires:
`no_georeference` vs `derived_raster`, asserted separately in
`test_source_inventory_matches_manifest`.

Precedence, applied in this order: not source (→ `too_few_bands` /
`derived_raster`) → `no_georeference` → `duplicate_of` →
`resolution_regime_excluded`. `duplicate_of` sits above the regime check so
`iran/Ax0*_y06.tif` reports the cause `DATA.md`'s table gives them
(md5-identical to Teheran's) rather than the resolution regime they also fail.

---

## Unspecified decisions I made

1. **The resolution regime is a constant, not a per-file list.**
   `INDEXABLE_GSD_RANGE_M = (0.05, 0.15)` on **true** ground GSD. The 8 scenes
   are 0.1000-0.1047; the nearest excluded value is 0.50 (3.3x outside) and
   then 4.08-4.16. Nothing sits near a boundary.
2. **No extension filter in the walk.** Every regular file is handed to GDAL,
   because filtering on `.tif` is a filename pre-filter and the brief forbids
   those "not as a pre-filter". Consequence: four PNGs enter the inventory as
   readable rasters and are rejected as `no_georeference`. The 50 files GDAL
   has no raster driver for are listed under `non_rasters`, not silently
   dropped.
3. **Duplicate detection is footprint containment + pixel verification**, not
   file hashing. Same CRS, same pixel size, same band count, integer pixel
   offset, smaller footprint inside the larger — then a 4x4 grid of 8x8
   windows compared exactly. Containment alone would be catastrophic here:
   `leb/2022-10-29.tif` and `leb/2025-06-06.tif` have **identical bounds**, so
   only the content check keeps the multi-date pair from collapsing into one
   scene. This is also what catches the contained crop, which no file hash
   would.
4. **Identical footprints tie-break on lexicographic path** so the canonical
   scene is stable across runs (D-4). That makes `Teheran/Ax00_y06.tif` the
   canonical and the `iran/` copies the duplicates — which matches
   `DATA.md`'s table.
5. **A geodesic cross-check field** (`geodesic_gsd_cm`) is computed per raster
   and asserted in two tests. It does not feed the reported GSD; the
   conditional Mercator rule does, exactly as the brief specifies.
6. **No wall-clock timestamp in `inventory.json`** — it would break the D-4
   byte-identity requirement. Run time goes to the log; the artifact carries
   `data_tree_digest` instead.
7. **`rasterio._env` is quietened to WARNING in `main()` only.** GDAL emits one
   INFO line per sidecar it cannot open — 50 of them, which buried the summary
   line the brief says a human reads first. The facts stay in `non_rasters`.
8. **`pyproj` is used only for `Geod`.** In this env pyproj cannot open the
   PROJ database (`CRSError: no database context specified` on
   `pyproj.CRS.from_user_input("EPSG:4326")`), so all CRS work goes through
   rasterio/GDAL and only ellipsoidal distance through `pyproj.Geod`, which
   needs no database. Worth adding to `CLAUDE.md`'s environment notes — a
   later stage that reaches for `pyproj.CRS` or `pyproj.Transformer` will hit
   this.

## Deviations from the brief

- **`test_source_classifier` asserts the indexable set**, not the raw
  pixel-rule set, because the raw set has four members (Finding 1). The
  pixel-rule set is asserted in full in its own test with the discrepancy
  documented in the docstring. I did not weaken either assertion, and I did
  not tune anything to whatever I happened to find.
- Test names: `test_gsd_correction`, `test_gsd_no_correction_for_utm`,
  `test_source_classifier`, `test_source_inventory_matches_manifest`,
  `test_data_dir_readonly` are exactly as briefed. D-4's is
  `test_inventory_deterministic` per the brief's acceptance section (`spec.md`
  §tests calls it `test_tile_id_stable`, which belongs to a later stage —
  there are no tiles yet).

## Blockers / questions for the PM

1. **Finding 1 needs a ruling** before this stage is gated. Option (1) is a
   doc edit and no code change; option (2) changes D-1's meaning.
2. **Findings 2 and 3 are doc corrections** with no code impact: "<= 6
   distinct" → "<= 25", and "12 of ~232 real imagery" → "85 of 270".
3. `notes.md#intake`'s "2 of `leb`'s 50 TIFFs, the other 48 derived" is wrong
   on both figures — it is 31 TIFFs, 28 of which fail the D-1 pixel rule.
4. No raster in the tree failed to open (`"unreadable": []`), so that blocker
   condition did not trigger.
