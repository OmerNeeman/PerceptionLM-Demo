# Brief — S1-fix: close the two findings that later stages depend on

**Dispatch:** subagent. **Send-back on S1**, not a new stage.
**Proves:** D-1, D-2, D-4 remain green, and the latent defects under F-7 and
E-1 are closed.
**WIP:** this is S1's single permitted re-dispatch. A second send-back on S1
stops and escalates.

## Why you are being sent back

S1's spec criteria all pass and its classifier is genuinely
filename-independent — an adversarial reviewer attacked it with 20 constructed
fixtures and 16 code mutations, and could not make it read a filename. That
part of the work is good and you should not touch it.

But the reviewer **broke two things**, and both sit directly underneath work
that is already planned. Neither is a spec failure; both are latent defects
that become live defects in two stages' time. Fixing them now is far cheaper
than discovering them through F-7 or E-1.

## Fix 1 — the duplicate content check verifies 0.0005% of the raster

`_same_pixels` (`src/inventory.py:338`) compares a 4x4 grid of 8x8 windows —
**1,024 pixels per band**, which on `leb` is **0.000519%** of the image. Two
rasters agreeing at those 16 windows are declared identical regardless of
everything else.

The reviewer built `retrieval/index/adv/G1_scene_alpha.tif` and
`G2_scene_beta.tif`: two 1024x1024 3-band EPSG:32636 rasters with the same
transform that differ in **3,130,471 of 3,145,728 values (99.5%)**, contrived
to agree only at the 16 sampled windows. Result:

```
_same_pixels(G2, G1) -> True
G2_scene_beta.tif  is_indexable=False  reason=duplicate_of
```

**A genuinely distinct scene is silently dropped from the index.** The
real-world trigger is not exotic: two same-footprint scenes with large uniform
nodata or saturated regions covering the sample points. This tree already
contains scenes that are 25.1% and 14.2% zero-fill.

**What must be true when you are done:**

- The containment/duplicate decision rests on a **complete** comparison of the
  overlapping region, not a sample of it.
- Keep footprint containment as the **cheap pre-filter** — it is correct and it
  keeps the cost down. Only pairs that survive it get compared.
- Compare **streamed in chunks with early exit on first mismatch**, so the
  common case (two genuinely different scenes) stays fast: they diverge almost
  immediately. Do not load whole rasters into memory.
- **The reviewer's `G1`/`G2` pair becomes a regression test.** Rebuild that
  fixture yourself under `retrieval/index/` (do not depend on the reviewer's
  directory persisting) and assert `_same_pixels` now returns `False`.
- `leb/2022-10-29.tif` and `2025-06-06.tif` must still **not** be merged, and
  `leb_crop_x4096_y3072_1024.tif` must still be caught as a contained
  duplicate of `2022-10-29.tif`. Both are already tested; keep them green.
- Report the wall-time cost of a full inventory run before and after. If it
  grows more than a few seconds, say so with the number.

## Fix 2 — the geographic-CRS branch has no test at all

Of 16 mutations the reviewer made, **exactly one survived all 21 tests**:
replacing `gsd_x, gsd_y = gx, gy` with `px_x, px_y` at `src/geo.py:193` — i.e.
**reporting degrees as metres** — still gave `21 passed`. `test_geo.py` asserts
only `crs_kind(4326) == "geographic"`; nothing ever calls
`ground_resolution` on a geographic raster.

Two EPSG:4326 rasters already flow through that branch:

```
sin/sin_min.tiff                     759x1095   4b   408.33 cm  (excluded regime)
tile_cropped_x3308_y3674_z0.125.tif  8192x8192  1b    12.17 cm  (too_few_bands)
```

The second is **spec E-1's primary ground-truth label raster**. Under the
surviving mutant its GSD reads `0.00011 cm` instead of `12.17 cm` — a factor of
~111,000 — and nothing notices. The code is currently *correct*; it is simply
unguarded, and E-1 is the stage that will lean on it.

**What must be true when you are done:** a test calls `ground_resolution` on a
real geographic raster and asserts the true GSD in **metres**
(`tile_cropped_x3308_y3674_z0.125.tif` -> **12.17 cm +/- 0.05**, and
`sin/sin_min.tiff` -> **408.33 cm +/- 0.5**). It must fail if degrees are
returned as metres. Verify by making that exact mutation yourself, watching the
new test fail, then reverting.

## Fix 3 — use the geodesy you already compute as a guard

`geo.py` already computes `geodesic_gsd_m` for every raster, and the reviewer
found it agreed with independent ellipsoidal measurement in **100% of probes,
to all printed digits**. It is then **discarded** — never used as a check.

Meanwhile the Mercator correction is over-general. `_scale_correction` applies
`cos(lat)` to anything classified `mercator`, which is right only when the
standard parallel is the equator. For `lat_ts != 0` the factor is
`cos(lat)/cos(lat_ts)`:

```
EPSG:3994  Mercator 41 (lat_ts=-41)   rule 8.7780 cm   geodesy 11.6232 cm   -24.5%
+proj=merc +lat_ts=45                 rule 7.0275 cm   geodesy  9.9386 cm   -29.3%
EPSG:29873 Hotine Oblique Mercator    classified mercator, gets a cosine it should not
```

Nothing in the tree uses these, and `spec.md` D-2 states the same
over-general rule — so the code faithfully implements the spec and this is not
a compliance failure. But the fix is nearly free and it catches **every** case
above, including the oblique-Mercator misclassification, without your having
to enumerate projections.

**What must be true when you are done:**

- `ground_resolution` cross-checks its analytic result against
  `geodesic_gsd_m` and **fails loudly** when they disagree beyond a stated
  tolerance. Choose the tolerance from the measured evidence: real agreement is
  within **0.17%** across Web Mercator, three UTM zones, and geographic; the
  failure cases diverge by **24-29%**. Anything in the 1-2% region separates
  them cleanly. State the number you picked and why.
- The error names both figures and the CRS, so a human sees
  `rule 8.78 cm vs geodesy 11.62 cm` rather than a bare exception.
- Add tests for **EPSG:3994** (or `+proj=merc +lat_ts=45`) asserting the guard
  fires, and confirm all eight indexable scenes still pass it.
- Do **not** silently substitute the geodesic value for the analytic one. A
  disagreement means an unhandled projection; that is a fact the PM must see,
  not something to paper over. Fail, do not fall back.

## Fix 4 — three trivial corrections

- `_bounds` (`src/inventory.py:333`) is **dead code and also wrong** — it
  anchors the footprint at `(0,0)` instead of the transform origin. Delete it,
  or fix it and use it. Do not leave a wrong helper for the next stage to find.
- Module docstring (`src/inventory.py:41`) says "32,768 pixels per band"; the
  actual and recorded value is **262,144**.
- `briefs/S1_result.md` quotes the crop's distinct counts as `236/246/255`,
  which are the **full-raster** values; the artifact records the sampled
  `[231,242,249]`. Both are true of different things — label which is which,
  so report and artifact stop disagreeing.

## Explicitly NOT in scope

Do not touch these. They are recorded and decided:

- **The classifier's filename independence.** It survived a determined attack.
  Leave it alone.
- **The `> 64` threshold.** Dense re-measurement puts the real margin at
  **20 -> 64 -> 183**: 3.2x above the true ceiling of 3-band derived rasters
  and 2.9x below the source floor. It is correct.
- **D-1's *Expected* clause** (the "2 of 31" question). That is a spec
  amendment awaiting the owner. Not yours.
- **Sub-pixel-offset duplicates** and **D-1's inability to distinguish
  segmentation visualisations from imagery.** Both are going to the tech-debt
  ledger with reasons. Not yours.
- The `test_source_classifier` split across two tests. The reviewer judged the
  pair jointly **stronger** than the briefed single assertion. Keep it.

## Acceptance criteria — test-first, paste RED then GREEN

For each of Fixes 1, 2 and 3: **write the test, watch it fail against the
current code, then fix.** Fix 2 in particular requires you to reproduce the
surviving mutation yourself so the RED is real — a test written after the code
is already correct proves nothing.

- `test_same_pixels_rejects_adversarial_agreement` — the rebuilt `G1`/`G2`
  pair is **not** merged.
- `test_leb_pair_not_merged` and the contained-crop test — still green.
- `test_ground_resolution_geographic` — 12.17 cm and 408.33 cm in metres;
  fails if degrees are returned.
- `test_gsd_guard_fires_on_secant_mercator` — EPSG:3994 raises, naming both
  figures.
- All **21** existing tests still green. Report the full run.

## Constraints

```bash
env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python
```

**Your Bash tool does not persist env vars between calls — prefix every python
and pytest invocation inline, every time.**

- **`/home/omer/PycharmProjects/Dynamic-Terrain/data` is READ-ONLY.** Every
  fixture you build goes under `retrieval/index/` (gitignored).
- `pyproj.CRS` and `pyproj.Transformer` are broken in this env (no PROJ
  database). Route CRS work through **rasterio/GDAL**; `pyproj.Geod` works.
- Do not edit S0's files (`embedders.py`, `calibrate.py`, `test_embedders.py`,
  `conftest.py`, `_text_embed_probe.py`) or any of the four project docs.
- Determinism: the inventory must stay byte-reproducible. It is currently
  **235,707 B** and identical across runs; if your changes alter the schema,
  say so and confirm two fresh runs still match each other.

## Report back

Status, the QA checklist with **pasted RED-then-GREEN per fix**, the full
21+ test run, the inventory wall-time before/after, the tolerance you chose for
Fix 3 and why, deviations, unspecified decisions, blockers.

**Flag as a blocker rather than working around it if:** a full-overlap
comparison makes the inventory run unacceptably slow (report the number and
stop); the geodesy guard fires on any of the eight indexable scenes (that would
mean the analytic rule is wrong for real data, which is a finding, not a
tolerance to widen); or fixing dedup breaks the `leb` multi-date pair.

Do not widen a tolerance to make a test pass. If the guard fires on real data,
that is evidence, and the PM needs to see it.

---

# ADDENDUM — 2026-09-07, after the first attempt died mid-Fix-1

The first attempt was killed by a spend limit. It had written ~132 lines of
Fix 1 tests and **no source change**. The PM reverted them. Read why, because
it is the trap in this fix:

**Its tests passed against the unfixed code.** 24 passed, `src/` untouched. A
test that asserts `_same_pixels(...) is False` and passes *before* the fix
exists has proved nothing — its fixtures were not actually adversarial. The
agent never checked that its generated pair defeated the current
implementation, so it produced false assurance, which is worse than no test.

The PM then verified the defect is genuinely live, using the reviewer's own
fixtures:

```
reviewer G1 vs G2: differ in 3,130,471 of 3,145,728 values (99.5%)
inventory._same_pixels(G2, G1) -> True        <-- the defect, reproduced
```

## Therefore, a mandatory extra step for Fix 1

**Validate your fixture before you trust your test.** The order is:

1. Generate your adversarial pair.
2. Run `_same_pixels` on it against the **current, unfixed** code and **assert
   it returns `True`.** Paste that output. *This is your RED.* If it returns
   `False`, your fixture is not adversarial and your test is worthless — go
   back to step 1.
3. Only then fix `_same_pixels`.
4. Re-run: it must now return `False`. That is your GREEN.

Encode step 2 permanently as a **fixture self-check**: a test that asserts the
generated pair agrees at every window of the historical sampled geometry. That
way the fixture cannot silently stop being adversarial when someone later
changes the sampling constants.

**The reviewer's fixtures are on disk right now** at
`retrieval/index/adv/G1_scene_alpha.tif` and `G2_scene_beta.tif`. Use them to
confirm your understanding of the defect. Do **not** make your test depend on
them — `retrieval/index/` is gitignored, so they will not survive a clone.
Your test must generate its own pair deterministically.

## Keep the first attempt's one good idea

It built **two** adversarial pairs at different scales, so the test could not
be satisfied by merely *enlarging* the sample:

- one pair agreeing on the historical dedup grid (4x4 windows of 8x8 px);
- one pair agreeing on the **entire primary sampling grid** (8x8 windows of
  64x64 px = 262,144 px per band, ~25% of the raster).

That is the right instinct and it anticipates the lazy fix. Keep it — but this
time **prove both pairs defeat the unfixed code** before you fix anything.
Only a complete comparison of the overlap rejects both.

## Order of work — changed

Do **Fix 2, Fix 3 and Fix 4 first**, then Fix 1. They are smaller and
independent, so a spend interruption cannot cost you all of them again. Commit
nothing; the PM handles git.
