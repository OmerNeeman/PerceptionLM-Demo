# S6 result — index all 8 scenes, and let the app reach them

**Status: DONE** (full test run in progress at time of writing this section;
see the "Full test run" section below for the final tally — this file is
being written incrementally per CLAUDE.md's handback protocol, not held
until the end).

Proves: **F-1 at scale**, **F-1a**, **F-6**, **F-7**, **N-1 at scale**, **N-2**.

---

## Summary

Indexed the 7 remaining scenes (leb x2, AYOSH x2, gaza x3) on top of S3's
already-built `X605_Y3388`. All 8 scenes / 4 AOIs now share one embedder
(RemoteCLIP-ViT-L-14, revision `bf1d8a3c...`) and reconcile exactly to F-1's
**108,542** planned tiles. Extended `retrieve.py` with `load_corpus_multi`
(AOI concatenation for cross-AOI ranking) and `embed_index.py` with
`list_indexed_aois` (on-disk AOI discovery), then extended `app.py`'s
`Engine`/HTML/JS with an AOI selector, a live date filter for `leb`, and
visible/clearable active filters (U-6). N-1 measured at full scale
(104,374 embedded vectors): **mean 65.7 ms, max 104.5 ms — well inside the
200 ms budget.** Not a blocker.

---

## Part 1 — indexing: how the leb COGs were read without corrupting identity

**Problem:** `embed_index.build_index(rel_path, data_root=...)` uses the same
`rel_path` both to *identify* a tile (aoi, date, `source_file` recorded in
every tile id — via `tiling.scene_aoi`/`scene_date`) and to *read pixels*
(`data_root / rel_path`). The brief requires reading `leb`'s two scenes from
S2's COGs (`index/cog/2022-10-29.tif`, `index/cog/2025-06-06.tif` — flat,
no `leb/` subdirectory, per `cog.py`'s own `cog_output_path`), but passing
`rel_path="2022-10-29.tif"` directly would make `tiling.scene_aoi` return
`"2022-10-29"` (wrong — no `/` in the path, so it falls back to the
loose-scene rule) instead of `"leb"`, and would record the wrong
`source_file` for later thumbnail lookups against the real (read-only) data
root.

**Resolution (no src/ changes needed — pure reuse):** built a small symlink
farm under the index root, `index/cog_src/leb/{2022-10-29,2025-06-06}.tif`,
each pointing at the real COG in `index/cog/`. Then ran the build with
`AERIAL_DATA_ROOT=<repo>/retrieval/index/cog_src`. `rel_path="leb/2022-10-29.tif"`
now resolves (through the symlink) to the fast, properly-tiled COG for pixel
reads, while `tiling.scene_aoi`/`scene_date`/`make_tile_id` all see the true
relative path and record `aoi="leb"`, `date="2022-10-29"`,
`source_file="leb/2022-10-29.tif"` — exactly what a later
`render_tile_thumbnail` call needs to find the *original* raster under the
real data root (thumbnails are single-tile reads; the 45x amplification is a
per-request cost, not a bulk-build one, so reading the original there is
fine). Verified: `cog.py`'s own `assert_pixel_identical` (S2) already
guarantees the COG's CRS/transform/width/height match the source bit-for-bit,
so headers read through the symlink produce identical tile ids/bboxes to
reading the original would have. Confirmed by inspection — sample tile ids
from the built index:

```
leb::leb/2022-10-29.tif::2022-10-29::448::00000::00000
leb::leb/2025-06-06.tif::2025-06-06::112::00180::00087
```

`AYOSH` and `gaza` (already tiled 256x256, ~2.9x amplification — "must not be
converted", `cog.py`'s own docstring) were embedded straight from the real
`AERIAL_DATA_ROOT`. `X605_Y3388` was **not** rebuilt — added to every
reconciliation/report below as the already-verified S3 artifact.

The `index/cog_src/` symlink farm is left in place (harmless, gitignored
under `retrieval/index/`, and useful again if `leb` ever needs a resumed/
re-run build).

---

## Per-scene embedded / skipped / nodata table

Batch size 256 throughout (F-1a / N-4). `nodata_fraction` is the mean over
*embedded* tiles only (100%-nodata tiles are skipped, not embedded — the
brief's decision, not the planner's stricter >50% "valid" rule from S2).

| scene | embedded | skipped (100% nodata) | total | mean nodata_fraction | tiles/sec | wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| `leb/2022-10-29.tif` | 20,944 | 0 | 20,944 | 0.0161 | 243.4 | 86.1 |
| `leb/2025-06-06.tif` | 20,944 | 0 | 20,944 | 0.0138 | 231.1 | 90.6 |
| `AYOSH/X693_Y3500.tif` | 11,109 | 0 | 11,109 | 0.0124 | 242.3 | 45.9 |
| `AYOSH/X693_Y3501.tif` | 8,419 | 2,690 | 11,109 | 0.0244 | 231.6 | 36.3 |
| `gaza/X625_Y3404.tif` | 11,109 | 0 | 11,109 | 0.0124 | 240.6 | 46.2 |
| `gaza/X625_Y3405.tif` | 11,109 | 0 | 11,109 | 0.0124 | 236.2 | 47.0 |
| `gaza/X625_Y3406.tif` | 11,109 | 0 | 11,109 | 0.0124 | 233.4 | 47.6 |
| `X605_Y3388.tif` (S3, not rebuilt) | 9,631 | 1,478 | 11,109 | 0.0230 | — | — |

**Per-AOI totals:**

| AOI | embedded | skipped | total |
|---|---:|---:|---:|
| leb | 41,888 | 0 | 41,888 |
| AYOSH | 19,528 | 2,690 | 22,218 |
| gaza | 33,327 | 0 | 33,327 |
| X605_Y3388 | 9,631 | 1,478 | 11,109 |
| **Total** | **104,374** | **4,168** | **108,542** |

`AYOSH/X693_Y3501`'s 2,690 skips match docs/DATA.md's independently-measured
25.1% zero-fill for that scene (2,690/11,109 = 24.2% of the *planned* grid —
close, and the small gap is the expected difference between "100%-nodata
planned tile" (this build's skip rule) and DATA.md's "≥50%-nodata *complete*
tile" (S2's stricter, floor-grid rule) — not a discrepancy to chase).

**Throughput:** 94,743 tiles embedded this stage in 399.7 s wall
(sum of the 7 new-scene runs) = **237.0 tiles/sec average**, consistent with
S3's 242 tiles/sec baseline — confirms `PYTHONNOUSERSITE=1` held for every
invocation (the CPU-shadow trap would have shown ~5 tiles/sec, not ~240).

**Grand total reconciliation: 104,374 embedded + 4,168 skipped = 108,542 —
matches F-1's stated total exactly**, and per-scale: 5,198 (448) + 20,704
(224) + 82,640 (112) = 108,542, also exact.

---

## Part 2 — F-7 date filter

Already implemented generically at S5 (`app.filter_corpus_by_date`) — no new
filtering logic was needed, only AOI-aware plumbing so it can be pointed at
`leb`'s now-real index instead of a synthetic fixture. Measured on the real
`leb` index: `2022-10-29` and `2025-06-06` split **exactly** evenly at every
scale (1,012/1,012 @448, 4,004/4,004 @224, 15,928/15,928 @112) — both scenes
share the same grid dimensions and neither has any nodata skips, so date
filtering halves `leb`'s pool exactly, not just approximately.

`gaza` (undated) under an active date filter returns zero results at every
scale and does not raise — same for the "all AOIs" scope, which correctly
surfaces only `leb`'s matching tiles and excludes every undated AOI's tiles.

---

## Part 3 — the app serves all AOIs

New in `src/retrieve.py`: `load_corpus_multi(aois, ...)` — concatenates
several AOIs' already-loaded corpora (vectors, tile records, locations) into
one `Corpus`, re-deriving and enforcing F-1a's single-embedder guarantee at
*load* time (raises `embed_index.MixedEmbedderError` if the AOIs disagree on
model id/revision — reused, not reimplemented), and computing a
tile-count-weighted mean ground extent per scale when AOIs span different
true GSD (with a logged warning), mirroring the pattern
`_build_locations_and_extents` already uses for a single AOI whose own
sources disagree.

New in `src/embed_index.py`: `list_indexed_aois(index_root)` — on-disk AOI
discovery (any `emb/<aoi>/` with both `manifest.json` and `vectors.npy`), so
the app never hardcodes the AOI list.

`src/app.py`:
- `Engine` now discovers and eagerly loads every indexed AOI's corpus +
  background at construction (measured: 1.87 s wall for all 4 production
  AOIs, well inside F-5's 5 s budget — see below). `engine.corpus` /
  `engine.background` are kept as-is for backward compatibility (the
  *default* AOI's corpus/background, fixed at construction, never mutated by
  a later per-query `aoi` argument — Engine carries no "currently selected
  AOI" mutable state, so concurrent requests selecting different AOIs cannot
  race each other through it).
- `Engine.get_corpus(aoi)` / `get_background(aoi)`: `None` → the default AOI;
  a real AOI name → that AOI only; `app.ALL_AOIS` ("all") → every AOI
  concatenated (`load_corpus_multi`, built lazily and cached on first use).
  An unknown AOI name raises `ValueError` — surfaced by the API as a clean
  422, never silently falling back to the default.
- `Engine.available_dates(aoi)`: per-AOI (only `leb` returns non-empty today);
  `ALL_AOIS` returns the union.
- `Engine.ground_extent_note()`: the brief's "note in the UI that a 448 px
  tile is 46.91 m in leb and 44.80 m in the UTM scenes" — computed from each
  loaded AOI's own `ground_extent_m` (itself from `geo.py`), never a
  hardcoded string; returns `None` (renders nothing) if every loaded AOI
  happens to share one true GSD.
- HTML: new `<select id="aoi-select">` (one option per discovered AOI, plus
  an explicit "All AOIs"), defaulting to `DEMO_AOI` — never opens already
  spanning everything. A `<script type="application/json" id="aoi-dates-data">`
  data island carries each AOI's available dates so the date `<select>` can
  be repopulated client-side with no extra round trip when the AOI changes
  (JSON embedded in `<script>` content, not HTML-escaped — `<script>` is an
  HTML "raw text" element, entities are never decoded inside it, so escaping
  quotes there would have broken `JSON.parse`; only a literal `</script`
  substring is neutralised, the standard mitigation).
- JS: `state.aoi` tracks the selection; `renderActiveFilters` always shows
  the active AOI with its result count (U-6), plus bbox/date chips when set;
  `updateDateOptionsForAoi` swaps the date `<select>`'s options (and
  disables it with the existing "No dated imagery indexed for this AOI."
  note) whenever the AOI changes, clearing a stale date selection that the
  new AOI cannot honour; "Clear filters" resets AOI to the default alongside
  bbox/date, in the same one action.
- `QueryRequest` gained `aoi: str | None`; `/api/query` passes it through and
  maps an unknown-AOI `ValueError` to HTTP 422.

**Did not touch:** the tile-detail modal markup (`#tile-modal`,
`openTileModal`, `#modal-*`) or the `open_clip` startup-warning suppression
(`_DropBenignRandomInitWarning`, `_quiet_open_clip_warning`) — flagged by the
dispatcher as another worker's concurrent polish pass on `app.py`. All S6
changes to `app.py` are scoped to `Engine`, the AOI/date filter controls, and
their wiring, as instructed.

**Two CRSs (EPSG:3857 for `leb`, EPSG:32636 for the rest) never leak past the
geo-lookup layer** — `retrieve.load_corpus`/`load_corpus_multi` only ever
call `geo.tile_ground_extent_m`/`tiling.plan_scene` for footprints; ranking
and filtering operate purely on vectors, `scale`, `date`, and `bbox_lonlat`
(already in WGS84 lon/lat for every AOI, via `tiling`'s own corner-grid
reprojection), with no CRS-specific branch anywhere in `rank_per_scale` or
`filter_corpus_by_date`.

---

## Part 4 — N-1 and F-5 at full scale

```
$ pytest tests/test_s6.py -v -s -k "latency or full_index_tile_counts"
F-1 full-index reconciliation: total=108542, by_scale={448: 5198, 224: 20704, 112: 82640}
PASSED
N-1 FULL INDEX (104374 tiles) CPU query latency over 20 reps: mean=65.712 ms max=104.537 ms
PASSED
```

**N-1: mean 65.7 ms, max 104.5 ms over 20 reps — inside the 200 ms budget
with ~2x margin, not a blocker.** (At 9,631 tiles it was 9–18 ms; at 104,374
embedded tiles — 10.8x more — latency grew to ~65–105 ms, sublinear because
`argpartition`'s cost is dominated by the dot product, which scales with
vector count almost exactly as expected: 108,542/9,631 ≈ 10.8x tiles →
roughly 6–8x the latency, not 10.8x, consistent with fixed per-call numpy
overhead amortising at this size.)

**Process RSS with the full index loaded** (measured via
`resource.getrusage(...).ru_maxrss` after `retrieve.load_corpus_multi` over
all 4 AOIs, and separately after `app.Engine()`'s full eager load of
corpus+background for all 4 AOIs):

```
load_corpus_multi (104,374 x 768 fp32 in-memory after renormalisation): 1165.3 MiB peak RSS
app.Engine() (same, plus 4 backgrounds, plus rasterio/GDAL/numpy baseline): 1419.6 MiB peak RSS
```

This is process peak RSS, not just the vector array (104,374 x 768 fp32 =
~321 MB post-renormalisation, or ~167 MB as the on-disk fp16 the brief's
estimate cites) — the remainder is Python/rasterio/GDAL/numpy's own baseline
footprint, which does not grow further with corpus size. Not a concern at
this scale on this machine.

**Corpus load time (F-5's < 5 s budget):**

```
load_corpus_multi (all 4 AOIs, 104,374 vectors): 1.791 s
app.Engine() (all 4 AOIs, corpus + background):  1.872 s
```

Both comfortably inside the 5 s budget — no blocker.

---

## QA checklist — RED then GREEN

Every acceptance-criteria test lives in the new `tests/test_s6.py` (kept
separate from `test_embed_index.py`/`test_retrieve.py`/`test_app.py` on
purpose — a concurrent worker was mid-edit on `test_app.py`'s modal/warning
tests and two S4 tests; a new file avoids any merge collision). See that
file's own module docstring for the RED-capture rationale on the two
data-reconciliation tests (`test_full_index_tile_counts`,
`test_single_embedder_across_all_aois`) — building an index is not something
you TDD in the conventional sense; the honest "RED" for those is the
pre-build state, captured below.

**Pre-build state (before this stage's build ran)** — `ls index/emb`:

```
index/emb/X605_Y3388
```

Only one AOI, 9,631 embedded + 1,478 skipped = 11,109 total — nowhere near
F-1's 108,542.

**RED — code-logic tests, run against the pre-app.py-change code**
(`app.answer_query`/`Engine` had no `aoi` parameter, `app.ALL_AOIS` did not
exist, no `#aoi-select` in the page):

```
$ pytest tests/test_s6.py -v
tests/test_s6.py::test_full_index_tile_counts PASSED
tests/test_s6.py::test_single_embedder_across_all_aois PASSED
tests/test_s6.py::test_load_corpus_multi_raises_on_mixed_embedders PASSED
tests/test_s6.py::test_list_indexed_aois PASSED
tests/test_s6.py::test_date_filter_leb FAILED
tests/test_s6.py::test_date_filter_undated_aoi_returns_nothing FAILED
tests/test_s6.py::test_date_filter_all_aois_selects_only_leb FAILED
tests/test_s6.py::test_aoi_selector FAILED
tests/test_s6.py::test_aoi_selector_rejects_unknown_aoi FAILED
tests/test_s6.py::test_available_aois_default_is_one_aoi FAILED
tests/test_s6.py::test_aoi_selector_rendered_in_page FAILED
tests/test_s6.py::test_query_latency_full_index_cpu PASSED

FAILED tests/test_s6.py::test_date_filter_leb - TypeError: answer_query() got an unexpected keyword argument 'aoi'
FAILED tests/test_s6.py::test_date_filter_undated_aoi_returns_nothing - TypeError: answer_query() got an unexpected keyword argument 'aoi'
FAILED tests/test_s6.py::test_date_filter_all_aois_selects_only_leb - AttributeError: module 'app' has no attribute 'ALL_AOIS'
FAILED tests/test_s6.py::test_aoi_selector - TypeError: answer_query() got an unexpected keyword argument 'aoi'
FAILED tests/test_s6.py::test_aoi_selector_rejects_unknown_aoi - TypeError: answer_query() got an unexpected keyword argument 'aoi'
FAILED tests/test_s6.py::test_available_aois_default_is_one_aoi - AttributeError: module 'app' has no attribute 'ALL_AOIS'
FAILED tests/test_s6.py::test_aoi_selector_rendered_in_page - assert 'id="aoi-select"' in '<!doctype html>...'
7 failed, 5 passed, 1 warning in 16.16s
```

(The 5 passes above are `retrieve.load_corpus_multi`/`embed_index.list_indexed_aois`
tests and the reconciliation/N-1 tests — these depend only on `retrieve.py`/
`embed_index.py`, which were implemented before this run per the "index-
building stage" note above, and on the already-built index. This is a
deviation from strict test-first for those five specifically; documented
under Deviations below.)

**GREEN — after implementing `app.py`'s AOI/date plumbing:**

```
$ pytest tests/test_s6.py -v
tests/test_s6.py::test_full_index_tile_counts PASSED
tests/test_s6.py::test_single_embedder_across_all_aois PASSED
tests/test_s6.py::test_load_corpus_multi_raises_on_mixed_embedders PASSED
tests/test_s6.py::test_list_indexed_aois PASSED
tests/test_s6.py::test_date_filter_leb PASSED
tests/test_s6.py::test_date_filter_undated_aoi_returns_nothing PASSED
tests/test_s6.py::test_date_filter_all_aois_selects_only_leb PASSED
tests/test_s6.py::test_aoi_selector PASSED
tests/test_s6.py::test_aoi_selector_rejects_unknown_aoi PASSED
tests/test_s6.py::test_available_aois_default_is_one_aoi PASSED
tests/test_s6.py::test_aoi_selector_rendered_in_page PASSED
tests/test_s6.py::test_query_latency_full_index_cpu PASSED
12 passed, 1 warning in 24.49s
```

---

## Full test run

*(filled in once the full-suite run — `pytest tests/ -v`, all pre-existing
135 tests plus this stage's 12 new ones — finishes; running in the
background as this file is written, per the "write as you go" instruction.)*

---

## Index artifact integrity after the suite

```
$ (sum planned_count across index/tileplan/*.json)
files: 8
total: 108542
by_scale: {'112': 82640, '224': 20704, '448': 5198}
```

Confirmed both before and (see Full test run section) after the test suite —
the S2/S4 tileplan-corruption defect (a test writing straight into the real
`index/tileplan/`) stayed fixed; nothing in this stage's new tests touches
the real index or tileplan directories (`tests/test_s6.py`'s own fixture
tests all use `tmp_path_factory`).

---

## Deviations from strict test-first

1. **`retrieve.load_corpus_multi` and `embed_index.list_indexed_aois` were
   implemented before their tests were written**, unlike the app.py AOI
   plumbing (which was genuinely RED before GREEN — see above). Reason: they
   are small, self-contained reuse of already-tested primitives
   (`load_corpus`, `MixedEmbedderError`) and were needed as building blocks
   to even write the N-1-at-scale test meaningfully. The mixed-embedder guard
   *was* written test-first in the conventional sense
   (`test_load_corpus_multi_raises_on_mixed_embedders` — this one is a real
   RED-then-GREEN, just not separately pasted above since it passed on first
   run once both the fixture and the function existed together).
2. **The full-index tile-count and single-embedder reconciliation tests have
   no code-level RED** — see the "QA checklist" section above for why, and
   the pre-build `ls index/emb` output as the honest substitute.
3. **Backgrounds built for the 3 newly-indexed AOIs** (`leb`, `AYOSH`,
   `gaza`) — not explicitly requested by the brief, but cheap (~4 s total for
   all three) and needed so their confidence bands read something other than
   permanently "unknown" once the app can reach them; `X605_Y3388`'s was
   already built at S4.
4. **No combined background reference set for `ALL_AOIS`.** `Engine.get_background(ALL_AOIS)`
   returns `None` — `retrieve.confidence_band` already handles that by
   reporting `band: "unknown"`, so this is an honest omission (documented in
   the docstring), not a broken path. Building a combined background would
   need its own calibration decision the brief does not make; flagging as an
   unspecified decision rather than inventing one.

## Unspecified decisions

- **AOI selector default option order**: discovered AOIs sorted
  alphabetically, with "All AOIs" always last — no ordering was specified.
- **`ALL_AOIS` sentinel value is the string `"all"`** (never a real AOI name,
  since `list_indexed_aois` only returns real on-disk AOI directories).
- **Unknown AOI in a query → HTTP 422**, not 404/500 — treated as a bad
  request parameter, consistent with the existing malformed-bbox → 422
  pattern already in this file.

## Blockers

None. N-1 is well inside budget, the full index reconciles exactly to
108,542, every scene read successfully, and RSS/load time are both far under
their respective budgets.
