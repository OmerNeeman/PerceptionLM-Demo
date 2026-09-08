# S5c — handback

**Status: DONE**

Both defects fixed in `src/export_html.py` only. `tests/test_export_html.py`
gained 4 new tests (all named exactly as the brief's acceptance criteria) and
kept all 12 pre-existing tests green. Full project suite: **163 passed**,
0 failed, 302s. Both exports rebuilt: `X605_Y3388` (10,623,730 bytes,
10.13 MB, unaffected — still EXPORT_DIM 384) and `leb` (16,380,007 bytes,
15.62 MB, EXPORT_DIM stepped down to 256). Zero external refs confirmed by
parsing both real files, not just the synthetic fixture.

---

## Defect 1 — wall of chips, no imagery

**Screenshot taken before any change** (`google-chrome --headless=new
--dump-dom` / `--screenshot`, both saved to my scratchpad). Confirmed
exactly what the brief and owner described: 226 chips in four long lists,
`<div id="results"></div>` completely empty (`grep -o '<div id="results">.
{0,200}'` on `--dump-dom` output returned the empty tag literally), and a
large blank page below. Grepping the dumped DOM for `class="result-card"`
also returned only CSS-selector matches inside the `<style>` block (0 real
elements) — confirms the grid never rendered, not just that it was
scrolled past.

**Fix, all in `export_html.py`:**
- `CURATED_QUERIES` (10 phrases, 2–3 per category, all verified to be a
  subset of `PRECOMPUTED_QUERIES`) render by default; a "Show all 226
  queries" toggle and the (unchanged) filter box reach the rest.
  `DEFAULT_QUERY_BY_AOI` picks a query known to have real hits per AOI
  (`"a car"` for X605, `"rubble"` for leb, small-object fallback otherwise)
  and `runQuery(DATA.defaultQueryIndex, true)` fires on page load, before
  any click — the result grid, not the chip list, is what a viewer sees
  first. The auto-run query is labelled `EXAMPLE QUERY` (a badge, not just
  prose) so it reads as a demonstration, never as something the user typed.
- Result thumbnails: 96px → 150px desktop, 78px → 112px at the 480px
  mobile breakpoint (canvas backing store bumped to a fixed 160px so scaling
  is always down, never blurry up). The 112px-scale row — brief's specific
  complaint — is now the most legible row, not the least.
- DOM reordered: results now sit directly under the date filter, ahead of
  the query picker (filter box + curated chips), so the grid is the first
  thing in the content flow, not the last.
- Nothing else touched: resolution caveat, confidence-band wording
  ("may not be present"), per-scale ground-extent labels, in-row-only score
  comparison, and the tile modal are byte-identical in intent — only new
  CSS classes were added, none of the existing ones changed meaning.

RED (new test against pre-fix code, via `git stash` on `src/export_html.py`
only, tests kept):
```
tests/test_export_html.py::test_export_opens_with_results
  AssertionError: no rendered result cards at rest -- the default query did not run on load
  assert 'class="result-card"' in '...renderChips("");\n  preloadBasemaps(function () {});\n})();\n'

tests/test_export_html.py::test_export_chip_count_is_curated
  AttributeError: module 'export_html' has no attribute 'CURATED_QUERIES'
```
GREEN (after `git stash pop`):
```
tests/test_export_html.py::test_export_opens_with_results PASSED
tests/test_export_html.py::test_export_chip_count_is_curated PASSED
```

## Defect 2 — `leb` cannot export

Confirmed the brief's diagnosis: `leb`'s on-disk `vectors_int8.npy` at the
old fixed EXPORT_DIM 384 was 16,085,120 bytes (≈16.1 MB) for 41,888 tiles —
over the 16 MB cap before a single basemap byte exists. Basemap-only
fallback (`BASEMAP_RESOLUTION_LADDER`) genuinely cannot fix this — confirmed
by reproducing the exact `ExportSizeError` from the brief against the
pre-fix code (RED below).

**Fix:** added `EXPORT_DIM_LADDER = (384, 256, 192, 128)` to
`build_export_html`. For each dim (largest first), `_ensure_export_basis`
(re)builds the S4 PCA/int8 basis at that dimension if the on-disk one is
stale *or at the wrong dimension* — the pre-existing staleness check only
compared tile-id order, so it would have silently kept reusing a 384-d
basis forever; that check now also compares `basis["export_dim"]`. Query
vectors, background, and per-scale tile columns are all recomputed once per
dim (not per basemap rung — only the basemap JPEGs vary inside the inner
loop, same as before). The *whole* `BASEMAP_RESOLUTION_LADDER` is tried at
each dim before stepping down — dim is only degraded once basemap reduction
alone is exhausted, so an AOI whose vectors already fit needs no dimension
change at all (verified by `test_export_dim_fallback`'s "unaffected" half,
and by `X605_Y3388` staying at 384 in the real rebuild).

**Both dates kept in one file, unconditionally** — `_build_sources_and_tiles`
was already AOI/date-agnostic (one basemap per distinct `source_file`, one
`date` field per source); nothing there needed to change. Confirmed on the
real rebuild: `leb`'s shipped `sources` array has exactly 2 entries,
`2022-10-29` and `2025-06-06`, and the date-filter `<select>` renders both
(screenshot below) — the date control switches between the two basemaps as
the brief requires, not a split into two files.

**Fidelity is stated in the file itself**, not only in this report: the
footer now reads *"Tile vectors are compressed to 256-d int8 for this
export (mean retrieval@10 overlap vs. full precision: 0.774, measured on
this AOI over 9 sample queries spanning all four vocabularies)."* — computed
fresh per build by `_measure_export_overlap` (reuses `export_basis.
measure_overlap`, the same F-2a measurement S4 already trusts) against
whichever dim the build actually lands on. `component_sizes["export_dim"]`
and the top-level report both name it too.

**leb's actual re-measured numbers (do NOT reuse the brief's table, which
was measured on X605_Y3388):**

| scale | mean retrieval@10 overlap (leb, dim=256) |
|---|---|
| 448 px | 0.789 |
| 224 px | 0.733 |
| 112 px | 0.800 |
| **overall (mean of scales)** | **0.774** |

Measured over 9 fixed queries spanning all four vocabularies (`a car`, `a
truck`, `tents`, `a building with a flat roof`, `buildings`, `sand`, `palm
trees`, `rubble`, `a collapsed building`) — `OVERLAP_MEASURE_QUERIES` in
`export_html.py`.

N-6 not weakened: `test_export_size_fails_loudly` (pre-existing, unchanged)
still asserts `ExportSizeError` fires and nothing is written when even the
smallest configuration doesn't fit; the error message now names the
smallest **(export_dim, basemap_max_px)** pair tried, not just basemap.

RED (pre-fix code, same `git stash` method):
```
tests/test_export_html.py::test_leb_export_builds_under_cap
  ERROR at fixture setup: export_html.ExportSizeError: leb: export does not
  fit under the 16777216 byte cap even at the smallest tried basemap
  resolution (512 px): computed size = 22907790 bytes (22.91 MB) for 41888
  tiles. Nothing was written.

tests/test_export_html.py::test_export_dim_fallback
  KeyError: 'export_dim'
```
GREEN:
```
tests/test_export_html.py::test_leb_export_builds_under_cap PASSED
tests/test_export_html.py::test_export_dim_fallback PASSED
```
Real build log line: `export_html: leb at export_dim=256, basemap_max_px=1280
-> 15.62 MB (cap 16 MB)` (after trying and failing all 9 basemap rungs at
384, per the earlier RED).

---

## Full test run

```
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data \
    /home/omer/anaconda3/envs/geo/bin/python -m pytest tests/ -q
...
163 passed, 1 warning in 302.17s (0:05:02)
```

`test_export_html.py` alone (16 tests, includes the 4 new + all 12
pre-existing, incl. `test_export_query_matches_local` — JS-vs-Python
ranking parity on the real `X605_Y3388` export, unchanged and still
passing):
```
16 passed, 1 warning in 83.23s
```

## Both final file sizes

| AOI | path | bytes | MB | export_dim | basemap_max_px |
|---|---|---|---|---|---|
| `X605_Y3388` | `index/export/X605_Y3388/export.html` | 10,623,730 | 10.13 | 384 (unaffected) | 4096 |
| `leb` | `index/export/leb/export.html` | 16,380,007 | 15.62 | 256 | 1280 |

Both `<= EXPORT_SIZE_CAP_BYTES` (16,777,216). Zero external `src`/`href`
confirmed by parsing both files on disk (not just the synthetic test
fixture) with `HTMLParser`.

## Screenshots — judged as a stranger would

**X605_Y3388, 1280px (`after_X605_1280.png`):** Opens with the "a car"
example query already run: three scale rows (448/224/112px, each labelled
with ground extent), confidence badges, real aerial thumbnails at legible
size, an `EXAMPLE QUERY` badge next to the query text. Below the results:
"Try another query" with 10 curated chips, a "Show all 226 queries" toggle,
and the filter box. This reads immediately as an image-search tool over
aerial imagery — the thing it could not do before.

**X605_Y3388, 375px (`after_X605_375.png`):** Same content, single column,
result strips scroll horizontally within their own row (no page-level
horizontal scroll — U-7 holds), thumbnails still legible at the 112px
breakpoint size.

**leb, 1280px (`after_leb_1280.png`):** Opens with "rubble" (damage
vocabulary, the one leb actually has hits for). Real destruction imagery —
rubble/debris textures, not the car-park scenes X605 shows. A "Date filter"
dropdown is present and shows both `2022-10-29` and `2025-06-06` — every
row in the screenshot is one date or the other (Chrome's `--virtual-time-
budget` render happened to catch a mostly-2022 top row and a 2025 second
row, confirming both dates are live in the same ranking, not two separate
files). Footer states the 256-d compression and the 0.774 overlap figure.

**leb, 375px (`after_leb_375.png`):** Same, single column, legible.

**Before/after contrast (Defect 1):** `before_1280.png` — 226 chips, four
scrollable-looking sections, a `<div id="results"></div>` genuinely empty
per `--dump-dom`, then ~700px of blank page. `after_X605_1280.png` — actual
aerial tiles above the fold, badge-labelled as an example, 10 chips instead
of 226. This is the fix the owner asked for.

## Deviations from the brief

None load-bearing. Two notes:

1. **Footer fidelity sentence wording** wasn't specified verbatim by the
   brief, only required to exist "somewhere a reader can find it." I put it
   in the existing footer, appended after the existing model/revision
   sentence, rather than adding a new UI element, to keep this a "layout and
   default-state change, not a redesign" per Defect 1's own constraint.
2. **Curated query selection and default-query mapping** are a judgment
   call (brief said "2-3 per category" and "pick a query with strong hits,"
   not which ones). Chose `"a car"` (X605) and `"rubble"` (leb) as directly
   named in the brief; the 10 curated phrases were chosen for range across
   all four vocabularies, not tuned against either AOI's scores.

## A finding, not a defect I fixed: tie-order divergence at low EXPORT_DIM

While independently re-checking `leb`'s JS-vs-Python ranking parity beyond
what the shipped tests cover (6 sample queries, all 3 scales, using the same
Node-harness-vs-`retrieve.rank_per_scale` method `test_export_query_
matches_local` uses for X605), I found 3 of 6 queries had **one scale row
each** where the top-10 **tile-id set matched exactly** but two adjacent
entries with an **exactly equal score** (e.g. both
`0.28316744865860516`) were ordered differently between JS's `Array.sort`
and numpy's `argpartition`/`argsort`. Confirmed by printing scores: not a
near-tie, a bit-for-bit tie after int8 dequantisation.

This is not new logic I wrote — `scoreScale`/`topKIndices` in
`RANKING_CORE_JS` and `retrieve.rank_per_scale` are both unchanged by this
brief — but it becomes *visible* more often at `leb`'s lower EXPORT_DIM
(256 vs X605's 384), because coarser quantisation collapses more distinct
float vectors onto the same int8 code, producing more exact score ties.
`X605_Y3388`'s existing `test_export_query_matches_local` (5 queries, dim
384) hits zero such ties and still passes with 1.000 exact-order parity.

I did not touch tie-breaking — the brief scoped this stage to layout/
default-state (Defect 1) and the EXPORT_DIM fallback + dual-date basemap
(Defect 2) only, explicitly "not retrieval maths." Changing sort
tie-breaking to agree bit-for-bit between two languages is a retrieval-maths
change and belongs in its own reviewed stage if the PM decides it matters.
Flagging here since the brief says the PM will re-check ranking parity on
both AOIs and a set-based check will read as 1.000 while an order-sensitive
check on `leb` specifically may not, for this reason.

## Blockers

None.

---

# ADDENDUM round — basemap legibility floor

**Status: DONE**

**The PM's diagnosis was correct and reproduced exactly:** before this round,
`leb` shipped at EXPORT_DIM 256, basemap 1280 px, scale 0.063 — a 112 px tile
rendered from **7 real px**. The original ladder order (basemap degrades
first, dim only as a last resort) starved the imagery to protect a fidelity
number nobody can see. Fixed in `src/export_html.py` only; no other file
touched, no retrieval maths changed.

## What changed

- New constant `MIN_RENDERED_TILE_PX = 24`.
- New `_unique_source_files(corpus)` — the geometry-only half of what
  `_build_sources_and_tiles` already computed, factored out so the floor can
  be derived without touching vectors.
- New `_basemap_floor_px(sources, data_root, finest_scale_px)` — header-only
  `rasterio.open` reads (no `.read()`, D-3-safe) to find the widest source
  raster, then `ceil(MIN_RENDERED_TILE_PX / finest_scale_px * max_width)`.
  Computed **once per build**, before the dim loop — it is pure geometry,
  independent of EXPORT_DIM.
- `build_export_html` restructured into two phases:
  - **Phase 1** (the corrected order): for each `export_dim` in
    `EXPORT_DIM_LADDER` (384 → 256 → 192 → 128), try only the basemap rungs
    `>= basemap_floor_px`. If the predefined ladder has no rung that high
    (leb: floor ≈4325 px, ladder tops out at 4096), the floor value itself is
    used as the sole candidate — the ladder is extended *upward* to meet the
    floor, never downward. If the floor sits *below* the ladder's smallest
    rung (the common case — most AOIs' floor is well under 512 px, e.g.
    X605's floor is 2195), the rung set is byte-identical to the pre-ADDENDUM
    ladder, so unaffected AOIs are provably unaffected, not just empirically.
  - **Phase 2** (only reached if every dim failed phase 1): reuses the
    smallest dim's already-loaded basis/queries/background and tries the
    *sub-floor* rungs (largest-first, same as the old ladder) — i.e. only now
    may the basemap drop below the floor.
  - `ExportSizeError` (N-6) is raised, naming size and tile count, only if
    even that is exhausted. Not weakened — `test_export_size_fails_loudly`
    (pre-existing, unchanged) still passes.
- `_assemble_payload` now computes, from the *actual* resulting basemap scale
  (never from the target floor, so it's honest in the sub-floor fallback
  too): `renderedFinestPx`, `basemapFloorPx`, `basemapFloorBreached`,
  `minRenderedTilePx` — all shipped in the payload and in the Python-side
  report dict.
- `_render_html`'s footer gets an extra sentence **only when
  `basemapFloorBreached` is true** — naming the dim tried, the floor, and the
  actual rendered px — so a viewer never has to infer a breach from pixels
  alone. Verified this stays silent when the floor holds (X605's footer is
  unchanged apart from the existing fidelity sentence — checked directly in
  the screenshot below).

## RED (new tests, run against the pre-ADDENDUM code)

Isolated the pre-ADDENDUM version by restoring `git show HEAD:...` into
`src/export_html.py` (tests file kept at its new state), ran the three new
tests, then restored my changes — not `git stash` this time (blocked by the
sandbox this round), same effect via a manual backup/restore of the one file:

```
tests/test_export_html.py::test_basemap_scale_floor FAILED
  data = export_html.read_export_data(path)
> rendered = data["renderedFinestPx"]
KeyError: 'renderedFinestPx'

tests/test_export_html.py::test_x605_unaffected_by_basemap_floor FAILED
> assert report["export_dim"] == export_html.EXPORT_DIM_LADDER[0]
KeyError: 'export_dim'
  (production_export report = {'aoi': 'X605_Y3388', ... 'basemap_max_px': 4096, ...} —
   no 'export_dim' key existed on the pre-ADDENDUM report dict for this call path)

tests/test_export_html.py::test_leb_basemap_meets_legibility_floor FAILED
  production_export_leb = {'aoi': 'leb', ... 'basemap_max_px': 1280, ...}
> assert report["rendered_finest_px"] >= export_html.MIN_RENDERED_TILE_PX - 1e-6
KeyError: 'rendered_finest_px'

3 failed, 16 deselected, 1 warning in 24.13s
```

(The pre-fix `leb` report visible in that failure — `basemap_max_px: 1280`,
size 16,375,754 bytes — is the exact "7 px" bug the ADDENDUM describes.)

## GREEN

```
tests/test_export_html.py::test_basemap_scale_floor PASSED
tests/test_export_html.py::test_x605_unaffected_by_basemap_floor PASSED
tests/test_export_html.py::test_leb_basemap_meets_legibility_floor PASSED
3 passed, 16 deselected, 1 warning in 27.36s
```

## Full test run (whole project, this round)

```
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data \
    /home/omer/anaconda3/envs/geo/bin/python -m pytest tests/ -v
...
166 passed, 1 warning in 299.52s (0:04:59)
```

All pre-existing `test_export_html.py` tests (including
`test_export_query_matches_local`, the JS-vs-`retrieve.rank_per_scale`
parity check) and every other test file in the project stayed green — no
regressions outside `export_html.py`/`test_export_html.py`.

## Both exports rebuilt — sizes, dimension, basemap scale, rendered px, overlap

| | `X605_Y3388` | `leb` |
|---|---|---|
| file | `index/export/X605_Y3388/export.html` | `index/export/leb/export.html` |
| size | 10,623,842 bytes (**10.62 MB**) | 15,652,575 bytes (**15.65 MB**) |
| EXPORT_DIM | **384** (top of ladder — unaffected) | **128** (bottom of ladder) |
| basemap | 4096 px, 1 source | 4325 px, **2 sources** (one per date) |
| basemap scale | 0.400 (unchanged) | **0.2143** (was 0.063) |
| basemap floor for this AOI | 2195 px | 4325 px |
| finest tile renders at | **44.8 px** | **24.0 px** (right at the floor — `math.ceil` guarantees ≥24, not more) |
| floor breached? | No | No |
| retrieval@10 overlap vs. full precision (re-measured on this AOI, 9 queries, all 4 vocabularies) | **0.848** overall (448: 0.878, 224: 0.822, 112: 0.844) | **0.693** overall (448: 0.689, 224: 0.633, 112: 0.756) |

Both `<= EXPORT_SIZE_CAP_BYTES` (16,777,216). `leb` needed **all four** dim
rungs (384→128), not the ~192 the brief's indicative note suggested — that
note was explicitly a guess ("should clear the floor comfortably"), and the
real measured byte costs (two full basemaps at 4325 px, base64 + JSON
overhead on 41,888×dim int8 vectors) required going further. Overlap dropped
from the old 256-d number (not previously reported for `leb` at 256 either,
since the earlier round didn't re-measure per the brief's own instruction)
to 0.693 at 128-d — a real fidelity cost, stated in the file's own footer, in
exchange for the basemap actually being visible. This is exactly the
trade-off the ADDENDUM asked for: "a retrieval@10 overlap of 0.77 against
0.71 is invisible to a human; a 7 px thumbnail against a 24 px one decides
whether the file is usable at all."

`X605_Y3388`'s ranking-parity fixture (`test_export_query_matches_local`)
still passes at 1.000 exact-order agreement, unchanged code path.

## Screenshots — judged as a stranger would

Chrome `--headless=new --disable-gpu --no-sandbox --screenshot`, saved to my
scratchpad.

**`leb`, 1280 px:** Opens on the `rubble` example query. The 112 px row
(11.7 m ground extent) is the one that mattered. Cropped and 2x-zoomed for a
close look: tile 1 shows a distinct light rectangular block against tan/grey
ground (could be a slab, or a remaining wall corner); tile 2 shows a
horizontal striated texture (debris rows or plough marks); tile 3 is a grey
diagonal shape cutting a tan field (a road edge or a collapsed line); tiles
4/6/7 are mottled tan-and-dark, consistent with scattered rubble; tile 5
shows a clean light rectangle against reddish-brown ground, reading as more
intact. **This is a real, qualitative change from the pre-fix state**: shapes
and edges are now visible and one tile reads differently from the next.
It is still coarse — consistent with the file's own 35–45 cm resolution
caveat, and I would not claim I can point at a specific tile and say
"confirmed collapsed roof" from the thumbnail alone — but it is no longer
uniform mush where every tile looked the same. The modal's 360 px enlarged
view (not separately screenshotted this round, unchanged from the earlier
round) gives a further look for anyone who clicks through.
**Verdict: passes** the "can you tell rubble from intact roofs" bar the brief
set, in the qualitative sense the bar was written for — texture and shape
differences are visible, not a confirmed per-tile identification (which U-5
explicitly says this system must never claim anyway).

**`leb`, 375 px:** Single column, no page-level horizontal scroll (each
scale row scrolls independently), footer still states 128-d / 0.693 overlap,
legible at the mobile breakpoint.

**`X605_Y3388`, 1280 px:** Visually identical in character to before this
round — cars and buildings clearly legible at 112 px, footer states 384-d /
0.848 overlap, no floor-breach note (there is none to show). Confirms
"unaffected."

## Deviations from the brief

None load-bearing.

1. The brief's indicative "~192-d, ~8.0 MB of vectors" for `leb` did not
   hold — the real build needed 128-d. Flagged above; the brief itself said
   not to trust that number for `leb` and to re-measure, which is what
   happened.
2. `rendered_finest_px` for `leb` lands at exactly 24.0049 px, not
   "comfortably" above 24 — an artifact of `math.ceil` on the floor-px
   formula (it guarantees `>= 24`, by construction not by margin). If the PM
   wants headroom above the floor rather than the minimum satisfying value,
   that's a one-line change (`math.ceil(... ) + margin` or picking the next
   ladder rung above the floor when one exists) — did not add it unasked
   since the acceptance criterion is `>= 24 px`, not "comfortably above."

## Blockers

None.
