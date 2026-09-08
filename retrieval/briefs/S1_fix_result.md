# S1-fix — handback

**Status: DONE**

All four fixes complete. All 21 pre-existing tests green, plus 6 new tests
(1 for Fix 2, 2 for Fix 3, 3 for Fix 1) = 27 in `test_geo.py` +
`test_inventory.py` together, 39 including S0's untouched embedder tests.
Zero blockers hit: the Fix 3 geodesy guard did not fire on any of the eight
real indexable scenes, and the Fix 1 dedup fix cost ~0.1s, not "unacceptably
slow."

Order executed per addendum: Fix 2, Fix 3, Fix 4, Fix 1.

Baseline confirmed before any change: `pytest retrieval/tests/` -> **33
passed** (21 geo+inventory tests this brief covers, plus 12 S0 embedder
tests untouched), 111.09s.

---

## Fix 2 — geographic-CRS branch has no test

**DONE.** Added `test_ground_resolution_geographic` to `tests/test_geo.py`,
covering the two real EPSG:4326 rasters named in the brief:
`tile_cropped_x3308_y3674_z0.125.tif` (spec E-1's ground-truth label raster)
and `sin/sin_min.tiff`.

The test passes against the current (correct) code trivially — the code
itself is fine, only unguarded — so per the brief I reproduced the surviving
mutation myself (`src/geo.py:193`, `gsd_x, gsd_y = gx, gy` -> `px_x, px_y`)
to get a real RED:

**RED** (mutation applied):
```
> assert gsd_cm == pytest.approx(12.17, abs=0.05), f"true GSD is {gsd_cm:.5f} cm/px"
E       AssertionError: true GSD is 0.00012 cm/px
E       assert 0.0001200000000000246 == 12.17 ± 0.05
1 failed, 1 warning in 0.17s
```
(0.00012 cm vs expected 12.17 cm — the ~111,000x error the brief predicted,
confirmed.)

**GREEN** (mutation reverted, `gsd_x, gsd_y = gx, gy` restored):
```
../tests/test_geo.py::test_ground_resolution_geographic PASSED   [100%]
13 passed, 1 warning in 0.18s
```
Full `test_geo.py` run: 13/13 passed (12 pre-existing + 1 new).

No source change was needed for Fix 2 — the defect was the missing test,
not the code, exactly as the brief states ("the code is currently correct;
it is simply unguarded").

---

## Fix 3 — geodesy guard

**DONE.**

**Tolerance chosen: 1.5% (`GSD_GUARD_TOLERANCE = 0.015` in `src/geo.py`).**
Reasoning: measured real-data agreement is within 0.17% (Web Mercator, 3 UTM
zones, geographic); measured failure cases diverge 24-29%. 1.5% sits in the
brief's suggested 1-2% band: ~9x above the noise floor and ~16x below the
smallest known failure, so normal float/geodesic noise cannot trip it and the
known bad cases trip it with a large margin.

**Source change** (`src/geo.py`): added `GSD_GUARD_TOLERANCE`, a
`GsdGuardError(ValueError)`, and — inside `ground_resolution`, in the
non-geographic branch, right after computing `gsd_x, gsd_y` — a cross-check:
`rel_err = abs(rule - geodesic) / geodesic`; raises `GsdGuardError` naming
the CRS, both figures in cm, and the relative error when `rel_err >
GSD_GUARD_TOLERANCE`. Geographic is exempt (there the geodesic value already
*is* the reported GSD, so it agrees trivially). No silent substitution
anywhere — the exception is the only path.

**Fixture:** an in-memory (no disk write) synthetic 3-band raster,
EPSG:3994 (Mercator 41, `lat_ts=-41`), pixel size 8.778 cm, centred at the
equator near `lon_0=100`. Verified interactively it reproduces the brief's
reported numbers closely: rule 8.778 cm vs geodesy 11.614 cm (brief: 8.7780
vs 11.6232), relative error 24.4% (brief: -24.5%).

**RED** (guard code temporarily commented out to reproduce pre-fix
behaviour, restored immediately after from a backup copy):
```
../tests/test_geo.py::test_gsd_guard_fires_on_secant_mercator FAILED
...
E               Failed: DID NOT RAISE GsdGuardError
1 failed, 1 passed, 1 warning in 0.22s
```
(the second test, `test_gsd_guard_does_not_fire_on_the_eight_indexable_scenes`,
passed in both the RED and GREEN runs since it has nothing to catch pre-fix —
included for completeness of the real-data non-regression check.)

**GREEN** (guard restored):
```
../tests/test_geo.py::test_gsd_guard_fires_on_secant_mercator PASSED
../tests/test_geo.py::test_gsd_guard_does_not_fire_on_the_eight_indexable_scenes PASSED
15 passed, 1 warning in 0.19s
```

`test_gsd_guard_does_not_fire_on_the_eight_indexable_scenes` calls
`ground_resolution_for` on all eight real indexable scenes from docs/DATA.md
(both `leb` dates, both `AYOSH`, all three `gaza`, and `X605_Y3388`) and
asserts none raises. **The guard did not fire on any of the eight real
scenes** — no blocker to report here.

---

## Fix 4 — three trivial corrections

**DONE.** No RED/GREEN pair required for these (brief only mandates
test-first for Fixes 1-3); each is verified by inspection / a quick probe
below.

1. **`_bounds` (`src/inventory.py:333`) deleted.** Confirmed dead
   (`grep -rn "_bounds" src/ tests/` -> only its own definition, no callers)
   and confirmed wrong (anchored the footprint at `(0,0)` instead of the
   transform origin). Deleted rather than fixed, since nothing needs it and
   duplicate detection already uses the transform directly in
   `_same_pixels`.
2. **Module docstring** (`src/inventory.py:41`): "32,768 pixels per band" ->
   "262,144 pixels per band" (8x8 grid of 64x64 windows = 64 windows x
   4,096 px = 262,144, matching `sampling.pixels_per_band` already written
   into the artifact).
3. **`briefs/S1_result.md`** line 265: the crop's distinct-count line
   (236/246/255) is labelled full-raster, with the artifact's sampled value
   (231/242/249) added alongside. Verified both numbers directly before
   editing:
   ```
   full-raster distinct per band 1-3: [236, 246, 255]
   sampled distinct_per_band:         [231, 242, 249]
   ```
   (full via a whole-raster `np.unique` per band; sampled via
   `inventory.probe_raster` on the same file.) Both were already true, of
   different things; the doc now says which is which.

---

## Fix 1 — duplicate content check verifies 0.0005% of the raster

**DONE.**

### Fixtures, validated as adversarial BEFORE trusting them (the mandatory
### extra step from the ADDENDUM)

Two pairs, both generated fresh under `retrieval/index/adv_fix1/` every test
run (no dependency on `retrieval/index/adv/G1_scene_alpha.tif` /
`G2_scene_beta.tif`, which were only used to understand the defect, never
read by any test), built with a guaranteed construction rather than luck: a
random uint8 base array is `G1`; `G2 = G1 XOR 137` everywhere (nonzero XOR
guarantees every pixel differs), then the target windows are copied back
from `G1` into `G2` so those specific windows agree exactly.

- **Pair 1** (`adversarial_dedup_grid_pair`): agrees only on the historical
  dedup grid — `inventory.sample_windows(1024, 1024, DEDUP_GRID=4,
  DEDUP_WINDOW_PX=8)`, 1,024 px/band.
- **Pair 2** (`adversarial_full_sample_grid_pair`): agrees on the historical
  dedup grid **union** the entire primary sampling grid (`SAMPLE_GRID=8,
  SAMPLE_WINDOW_PX=64`, 262,144 px/band, ~25% of the raster) — the union
  matters: it must defeat the code as deployed today (which only ever
  sampled the dedup grid) **and** anticipate the lazy fix of merely
  enlarging the sample to the primary grid, per the ADDENDUM's "keep the
  first attempt's one good idea."

Both pairs have a permanent **fixture self-check** test
(`test_adversarial_*_pair_is_genuinely_adversarial`) that reopens the
written files from disk (not the in-memory arrays) and asserts pixel
equality at every one of the generating windows and inequality over >50% of
the raster overall. Since both tests call `inventory.sample_windows` with
the live `DEDUP_GRID`/`DEDUP_WINDOW_PX`/`SAMPLE_GRID`/`SAMPLE_WINDOW_PX`
constants rather than hardcoded numbers, the fixtures stay adversarial to
whatever the real sampling geometry is even if those constants change later.

### Mandatory validation: fixtures defeat the CURRENT, unfixed code (RED,
### addendum step 2)

```
dedup-grid pair: differ in 3,142,656 of 3,145,728 values (99.9%)
  inventory._same_pixels(G2, G1) -> True   <-- expect True on unfixed code
dedup UNION full-sample-grid pair: differ in 2,356,992 of 3,145,728 values (74.9%)
  inventory._same_pixels(G2, G1) -> True   <-- expect True on unfixed code
RED confirmed: both fixtures defeat the current, unfixed _same_pixels.
```
Both pairs genuinely reproduce the defect (matching the format of the PM's
own confirmation against the reviewer's fixtures: `differ in 3,130,471 of
3,145,728 values (99.5%)`, `_same_pixels(G2, G1) -> True`). Only after this
did I touch `src/inventory.py`.

**Note on a false start:** my first version of Pair 2 agreed only at the
primary-sampling-grid windows (not the dedup grid too), and on validation it
returned `False` against the unfixed code — i.e. it was *not* adversarial,
for exactly the reason the addendum warns about: the deployed
`_same_pixels` samples the dedup grid, not the primary grid, so a pair that
only matches the primary grid can miss the dedup grid entirely and get
caught by pure chance. Caught by the validation step itself, before any
test was trusted; fixed by taking the union (documented above).

### RED — the permanent regression test, before the fix

```
../tests/test_inventory.py::test_same_pixels_rejects_adversarial_agreement FAILED
...
>       assert inventory._same_pixels(g2, g1) is False, "dedup-grid-only agreement was accepted as identical"
E       AssertionError: dedup-grid-only agreement was accepted as identical
E       assert True is False
1 failed, 1 warning in 0.22s
```

### The fix (`src/inventory.py`)

- `_same_pixels`: unchanged cheap pre-filter (band count, integer pixel
  offset, containment bounds) — then a **complete streamed comparison** of
  the overlapping region in `DEDUP_STRIP_ROWS = 512`-row strips via
  `rasterio` windowed reads, `np.array_equal` per strip, returning `False`
  on the first mismatching strip. Neither raster is ever read whole.
- Old `DEDUP_GRID`/`DEDUP_WINDOW_PX`-based sampling in `_same_pixels`
  removed; the two constants are **kept** (no longer used for comparison)
  purely so the regression test can reconstruct "the historical dedup grid"
  by name instead of hardcoding `4`/`8` as magic numbers — commented to say
  so.
- Artifact schema: `sampling.dedup_grid` / `sampling.dedup_window_px` (no
  longer meaningful — nothing is sampled for dedup any more) replaced by
  `sampling.dedup_strategy = "full_overlap_streamed_early_exit"` and
  `sampling.dedup_strip_rows = 512`. **`schema_version` bumped 1 -> 2** to
  flag this, per "if your changes alter the schema, say so."
- Also deleted the dead/wrong `_bounds` helper (Fix 4) and fixed the module
  docstring's "32,768 pixels per band" -> "262,144" (Fix 4) in the same
  file — see Fix 4 above; unrelated to the dedup logic itself.

### GREEN — the permanent regression test, after the fix

```
../tests/test_inventory.py::test_same_pixels_rejects_adversarial_agreement PASSED
../tests/test_inventory.py::test_adversarial_dedup_grid_pair_is_genuinely_adversarial PASSED
../tests/test_inventory.py::test_adversarial_full_sample_grid_pair_is_genuinely_adversarial PASSED
3 passed, 1 warning in 0.24s
```

### `leb` pair and contained crop — still green (unchanged assertions)

`test_source_inventory_matches_manifest` (both `leb` dates indexed
separately, i.e. **not merged**) and
`test_d1_pixel_rule_over_leb_finds_more_than_the_two_scenes` (the 1024x1024
crop still caught as `duplicate_of leb/2022-10-29.tif`,
`duplicate_kind=contained_crop`) both pass unmodified after the fix — see
the full run below. The real-world containment/dedup behaviour this stage
depends on is unchanged; only the false-negative on adversarial content is
closed.

### Wall time — before / after

```
BEFORE Fix1 wall time: 8.98 s   (unfixed _same_pixels, sampled)
AFTER  Fix1 wall time: 9.08 s   (full-overlap streamed comparison)
```
Same `summary` both times: `8 indexable, 169 derived_raster, 3 duplicate_of,
5 no_georeference, 69 resolution_regime_excluded, 16 too_few_bands, 270
rasters of 320 files scanned` — the fix changes zero classifications, only
the thoroughness of the check. **Cost: ~0.1 s, within run-to-run noise.**
This confirms the brief's expectation: genuinely different scenes diverge
almost immediately, so the early-exit streamed comparison essentially never
reads more than the first strip or two in practice on this data.

### Determinism / byte-reproducibility — schema changed, still reproducible

The inventory schema **did change** (Fix 1 requires it — see above), so the
artifact is no longer byte-identical to the pre-fix `235,707 B`. Two fresh
runs **after** the fix agree byte-for-byte with each other:
```
run_a bytes: 235747
run_b bytes: 235747
identical: True
canonical inventory.json bytes: 235747
```
New size **235,747 B** (was 235,707 B; +40 B from
`"dedup_grid": 4, "dedup_window_px": 8` -> `"dedup_strategy":
"full_overlap_streamed_early_exit", "dedup_strip_rows": 512`, plus
`schema_version` 1 -> 2). `test_inventory_deterministic` (unchanged
assertion) also passed in the full run below, confirming two in-process
runs produce byte-identical JSON.

---

## Full final test run — 21 pre-existing + 6 new = 27 (`test_geo.py` +
## `test_inventory.py`)

```
../tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[3857-mercator] PASSED
../tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[3395-mercator] PASSED
../tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[32636-projected] PASSED
../tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[32639-projected] PASSED
../tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[4326-geographic] PASSED
../tests/test_geo.py::test_crs_kind_of_none_is_none PASSED
../tests/test_geo.py::test_tile_ground_extent_is_gsd_times_pixels PASSED
../tests/test_geo.py::test_gsd_correction PASSED
../tests/test_geo.py::test_mercator_correction_agrees_with_an_independent_geodesic_measurement PASSED
../tests/test_geo.py::test_gsd_no_correction_for_utm PASSED
../tests/test_geo.py::test_utm_no_correction_agrees_with_an_independent_geodesic_measurement PASSED
../tests/test_geo.py::test_the_two_regimes_frame_the_same_ground_area_within_5_percent PASSED
../tests/test_geo.py::test_ground_resolution_geographic PASSED                       [NEW - Fix 2]
../tests/test_geo.py::test_gsd_guard_fires_on_secant_mercator PASSED                  [NEW - Fix 3]
../tests/test_geo.py::test_gsd_guard_does_not_fire_on_the_eight_indexable_scenes PASSED [NEW - Fix 3]
../tests/test_inventory.py::test_source_classifier PASSED
../tests/test_inventory.py::test_d1_pixel_rule_over_leb_finds_more_than_the_two_scenes PASSED
../tests/test_inventory.py::test_leb_denominator_is_reported_as_a_fact PASSED
../tests/test_inventory.py::test_classification_is_by_band_count_and_distinct_levels_only PASSED
../tests/test_inventory.py::test_source_inventory_matches_manifest PASSED
../tests/test_inventory.py::test_summary_line_states_counts_per_reason PASSED
../tests/test_inventory.py::test_adversarial_dedup_grid_pair_is_genuinely_adversarial PASSED     [NEW - Fix 1]
../tests/test_inventory.py::test_adversarial_full_sample_grid_pair_is_genuinely_adversarial PASSED [NEW - Fix 1]
../tests/test_inventory.py::test_same_pixels_rejects_adversarial_agreement PASSED                [NEW - Fix 1]
../tests/test_inventory.py::test_data_dir_readonly PASSED
../tests/test_inventory.py::test_inventory_deterministic PASSED
../tests/test_inventory.py::test_the_pixel_rule_has_a_wide_margin PASSED

27 passed, 1 warning in 36.80s
```

Also reran the whole `retrieval/tests/` directory (including S0's untouched
`test_embedders.py`) as a final sanity check: **39 passed, 1 warning in
109.93s**. The 1 warning both times is the pre-existing
`pyproj unable to set PROJ database path` notice (CLAUDE.md's known pyproj
limitation, unrelated to this fix).

---

## Deviations / unspecified decisions

- **Fixture directory**: adversarial fixtures live under
  `retrieval/index/adv_fix1/` (gitignored, mirroring `retrieval/index/adv/`
  the reviewer used) — not specified by name in the brief, chosen to keep
  them clearly separate from the reviewer's own fixtures.
- **`schema_version` bump 1 -> 2** for the inventory artifact, and the
  `+40 B` size change — not explicitly required by the brief but flagged by
  its own instruction ("if your changes alter the schema, say so").
- **Kept `DEDUP_GRID`/`DEDUP_WINDOW_PX` as unused-in-production constants**
  in `src/inventory.py` (commented as historical), rather than deleting
  them or hardcoding `4`/`8` in the test — a judgment call to avoid magic
  numbers in the regression test while being explicit that they no longer
  drive any comparison.
- **`GsdGuardError` is a new public name** in `geo.py` (subclass of
  `ValueError`) — not specified by the brief, added so the guard's failure
  mode is distinguishable from other `ValueError`s if a later stage wants to
  catch it specifically.
- `_bounds` was deleted rather than fixed, per the brief's "delete it, or
  fix it and use it" — nothing in the tree calls it (verified with grep), so
  deletion was the smaller change.

## Blockers

None. Both of the brief's explicit stop-and-flag conditions were checked
and came back clean: the Fix 1 dedup fix did not make the inventory run
"unacceptably slow" (+0.1s), and the Fix 3 geodesy guard did not fire on any
of the eight real indexable scenes.
