# Result — S11b: model selector + side-by-side comparison

**Status: DONE**

Depends on S11a (both indexes built, verified, model-scoped under
`index/emb/<model_id>/<aoi>/`). Neither index was rebuilt. Both were treated
strictly read-only throughout this stage.

## Summary

- A **model selector** (`<select id="model-select">`) sits beside the AOI
  and date filters. Switching it re-runs the current query against that
  model's own index. Default stays **RemoteCLIP-ViT-L-14** (`COMPARE_MODELS[0]`).
- A **compare view** (`<input id="compare-toggle">` + `POST /api/compare`):
  one query, two columns per scale row, each headed
  `<model id> · <dim>-d` (`RemoteCLIP-ViT-L-14 · 768-d`,
  `PE-Core-L14-336 · 1024-d`), each with its own within-model confidence
  band. Nothing is ever merged, interleaved, or jointly ranked across models
  — see "How F-1a is actually enforced" below.
- U-6: the active model (or, in compare mode, both models) is shown with the
  result count it produced, via the same `#active-filters` chip row the AOI/
  bbox/date filters already use, and `Clear filters` resets model + compare
  state in the same one action it already resets AOI/bbox/date.
- Both indexes are held **eagerly** in `Engine.__init__` (not lazily) —
  measured RSS made this the right call; see "Memory" below. Embedders
  (the GPU-resident, ~7 s-to-load part) remain **lazy per model**, unchanged
  from pre-S11b behaviour for the default model.

## Files touched

- `retrieval/src/app.py` — `COMPARE_MODELS`, `model_label()`, `Engine`
  (multi-model corpora/backgrounds/embedders, `run_query(model_id=...)`,
  new `run_compare()`), `answer_query(model_id=...)`, new `answer_compare()`,
  model selector + compare toggle in `render_index_html`, compare-view
  rendering in `_JS`/`_CSS`, new `POST /api/compare` route.
- `retrieval/src/retrieve.py` — `load_corpus`, `load_corpus_multi`,
  `background_path`, `build_background`, `load_background` all grew an
  optional `model_id` parameter (default `embed_index.DEFAULT_MODEL_ID`),
  mirroring the pattern S11a already established on `embed_index.py`. No
  other module needed changes; `embed_index.py`/`config.py` untouched.
- `retrieval/tests/test_s11b.py` — new, 17 tests.
- `retrieval/index/emb/PE-Core-L14-336/{AYOSH,X605_Y3388,gaza,leb}/background.json`
  — new production artifact (see "PE-Core background" below). **Not** under
  `index/export/`.

## How F-1a is actually enforced (not just documented)

- `Engine.run_compare` calls `run_query` **once per model**, independently,
  and keeps each model's result dict in its own bucket of a `{model_id: ...}`
  dict. No line anywhere concatenates, sorts, or compares two models'
  `results`/`score` values against each other.
- `answer_compare` enriches each model's rankings independently (reuses
  `enrich_result` per model, per scale) and never introduces a top-level
  `rankings`/`results` key alongside the per-model ones.
- The compare-view JS (`renderCompareResults`) builds each column from that
  model's own `rankings[scale]` only; the two columns are two independent
  DOM subtrees under one `.compare-columns` grid, never re-sorted against
  each other.
- `test_no_cross_model_merging` and `test_run_compare_never_touches_other_models_scores`
  (`tests/test_s11b.py`) prove this structurally: every displayed score is
  reproduced by a direct dot product against *that model's own* stored
  vectors, and a broken embedder on one model (monkeypatched to raise
  `UnitNormError`) is confirmed to leave the other model's already-computed
  result untouched.
- No "winner" copy anywhere in the page (`test_never_shows_a_winner_computed_from_raw_scores`
  checks the rendered HTML and greps `_JS` for any `models[...].score`
  comparison operator — there is none).

## RED then GREEN

Full RED run (17 new tests against the pre-S11b `app.py`/`retrieve.py`,
restored from `git show HEAD:...` for the duration of the run, then put
back): every new capability failed as expected —
`AttributeError: module 'app' has no attribute 'COMPARE_MODELS'`,
`TypeError: build_background() got an unexpected keyword argument 'model_id'`,
`TypeError: background_path() got an unexpected keyword argument 'model_id'`,
plus two content-assertion failures (`'id="compare-toggle"' in html`,
`"only meaningful within one model" in html`) against the un-S11b'd page:

```
FAILED tests/test_s11b.py::test_compare_view_rendered_in_page_with_model_and_dim_headers
FAILED tests/test_s11b.py::test_background_path_for_non_default_model_never_under_export
ERROR tests/test_s11b.py::test_model_selector_switches_index - TypeError: bui...
ERROR tests/test_s11b.py::test_default_model_is_remoteclip - TypeError: build...
ERROR tests/test_s11b.py::test_no_cross_model_merging - TypeError: build_back...
ERROR tests/test_s11b.py::test_run_compare_never_touches_other_models_scores
ERROR tests/test_s11b.py::test_compare_view_same_query_both_models - TypeErro...
ERROR tests/test_s11b.py::test_query_latency_per_model - TypeError: load_corp...
ERROR tests/test_s11b.py::test_http_compare_endpoint_ok_no_stack_trace - Type...
ERROR tests/test_s11b.py::test_http_switch_model_mid_query_no_stack_trace - T...
ERROR tests/test_s11b.py::test_http_compare_known_absent_query_no_stack_trace
ERROR tests/test_s11b.py::test_http_unknown_model_id_clean_422 - TypeError: b...
ERROR tests/test_s11b.py::test_http_model_with_missing_index_for_aoi_single_query_clean_422
ERROR tests/test_s11b.py::test_http_model_with_missing_index_for_aoi_compare_degrades_gracefully
ERROR tests/test_s11b.py::test_engine_construction_does_not_crash_with_missing_second_model_index
ERROR tests/test_s11b.py::test_http_compare_switch_aoi_no_stack_trace - TypeE...
2 failed, 1 passed, 1 warning, 14 errors in 15.12s
```

(Full traceback captured; representative excerpt above. The 1 passing test
out of 17 was `test_never_shows_a_winner_computed_from_raw_scores` — a
purely static content check for "winner"-style copy, which trivially holds
on the pre-S11b page too since that page has no compare view to make such a
claim about. It is not one of the brief's four named acceptance criteria.
Every one of those four — `test_model_selector_switches_index`,
`test_no_cross_model_merging`, `test_compare_view_same_query_both_models`,
`test_query_latency_per_model` — is in the ERROR list above, i.e. genuinely
RED.)

GREEN (same 17 tests, S11b implementation restored), with `-s` for the N-1
latency print:

```
Missing keys for loading model: []
Unexpected keys for loading model: []
Missing keys for loading model: []
Unexpected keys for loading model: []
.......
N-1 per-model FULL INDEX (104374 tiles) CPU query latency, 20 reps:
  RemoteCLIP-ViT-L-14 (768-d):  mean=64.916 ms max=100.310 ms
  PE-Core-L14-336 (1024-d): mean=87.898 ms max=119.247 ms
  compare total (search-only, sequential): 152.814 ms mean
.....Missing keys for loading model: []
Unexpected keys for loading model: []
.....
17 passed, 1 warning in 40.98s
```

Two bugs the RED/GREEN cycle itself caught mid-implementation (fixed before
the GREEN above, not after — no test was weakened to pass):

1. `retrieve.background_path`'s wording described the pre-S11b path only;
   `test_background_path_for_non_default_model_never_under_export` forced
   writing the actual model-scoped fallback (`embed_index.emb_dir`) rather
   than leaving it undefined for a second model.
2. My first `run_compare` caught `ValueError` around *every* `run_query`
   call unconditionally, which silently downgraded a genuinely unknown AOI
   (typo) into "both models unavailable" (200 OK) instead of the clean 422 a
   single-model query already gives for the same typo
   (`test_http_compare_switch_aoi_no_stack_trace` caught this). Fixed by
   validating the AOI against the union of every model's known AOIs
   (`Engine._known_aois`) *before* the per-model loop — a real AOI that one
   model simply hasn't been indexed for still degrades gracefully per
   column; a fake AOI still raises loudly, exactly like every other filter
   in this app ("a filter that silently does nothing is worse than one that
   fails loudly" — already this codebase's own stated principle).

## Full test run

All pre-existing tests stay green, plus the 17 new ones — **188 passed**
(171 + 17), confirmed in two independent full runs of the whole `tests/`
directory: one immediately after implementation (349.81 s), and one final
confirmation run after all manual/live-server testing below, including the
screenshot session (343.33 s):

```
........................................................................ [ 38%]
........................................................................ [ 76%]
............................................                             [100%]
188 passed, 1 warning in 343.33s (0:05:43)
```

`index/tileplan/*.json` reconciliation, checked after the full suite:
**108,542** tiles across three scales (`{'112': 82640, '224': 20704, '448': 5198}`)
— unchanged, matches `EXPECTED_PLANNED_BY_SCALE` in `test_s6.py`.

## Memory — peak RSS with both indexes loaded

Measured directly (`resource.getrusage(...).ru_maxrss`, cross-checked with
`/usr/bin/time -v`, fresh process each time):

| Configuration | Peak RSS |
|---|---|
| Baseline (imports only, before `Engine()`) | ~395 MiB |
| `Engine(models=(RemoteCLIP,))` — one index, corpora only | 1,017 MiB |
| `Engine(models=COMPARE_MODELS)` — **both indexes, corpora only** | **1,574 MiB** (1,611,636 KB via `/usr/bin/time -v`, cross-checks the in-process figure) |
| Both indexes + a real `run_compare()` call (both GPU embedders loaded and warmed) | **~7.47 GiB** (7,652,128 KB) |

**Decision: held both eagerly**, not lazily. 1.57 GiB for both corpora is in
the same range the brief's own prior measurement treated as acceptable for
one index (1,419.6 MiB), and this is a local, single-user demo on a machine
with 2×24 GiB GPUs — nothing suggests 1.57 GiB of host RAM for two vector
corpora is unreasonable. Embedders stay lazy per model exactly as before, so
opening the app (no query yet) never pays the ~7.5 GiB figure; that number
only appears once a session has actually queried *both* models (two
GPU-resident vision-language models loaded simultaneously — the unavoidable
cost of the compare feature actually working, not a cost of "holding both
indexes"). Flagging this clearly since it is a materially larger number than
the brief's own 1,468 MB figure, which — cross-checked against my
corpora-only number — was almost certainly measuring the same thing I did
(corpora only, no embedders), not a full compare session.

## N-1 — query latency per model, and for a compare query

Measured against the real production index (both models, 104,374 tiles
each — confirmed identical tile count across models), `rank_per_scale` only
(the thing N-1 times — GPU embedding is excluded, matching `test_s6.py`'s
own definition), 20 reps each, CPU:

| Model | Dim | Mean | Max |
|---|---|---|---|
| RemoteCLIP-ViT-L-14 | 768 | 58.7–64.9 ms | 77.4–100.3 ms |
| PE-Core-L14-336 | 1024 | 73.2–87.9 ms | 102.9–119.2 ms |
| **Compare total** (both, sequential search-only) | — | 131.9–152.8 ms | — |

(Two independent measurement runs gave the two ends of each range; both
runs are comfortably under the 200 ms budget, individually and combined —
1024-d costing roughly 25–35% more than 768-d, in line with the brief's
~33% arithmetic-cost estimate.)

**Both models individually clear N-1's 200 ms budget with wide margin, and
the compare view's combined search-only total (~132–153 ms) also clears it**
— no approximate index was needed or considered (spec §7 excludes it
regardless).

Live, warm end-to-end numbers from the running app (single default AOI,
9,631 tiles, both embedders already loaded — i.e. what a user actually
experiences on the second-or-later query): RemoteCLIP embed+search 31.6 ms
total, PE-Core embed+search 31.1 ms total, measured via a real
`/api/compare` call.

## Abuse-case walkthrough (used as an impatient user)

All exercised twice: once as pytest HTTP-layer tests
(`tests/test_s11b.py`, `TestClient`), once again against the **real running
app** (`uvicorn` on 127.0.0.1:8420, real production index, real GPU
embedders) via `curl`.

1. **Switch models mid-query.** `RemoteCLIP -> PE-Core -> RemoteCLIP` in
   sequence, same query text. Each switch re-ran cleanly; `model_id` in the
   response always matched the request; no stack trace.
2. **Switch AOI while in compare view.** `/api/compare` with `aoi=leb`
   (the one dated AOI) returned both models' full per-scale rankings,
   correct dates (`2022-10-29`/`2025-06-06`), no crash. A follow-up with a
   bogus AOI (`not_a_real_aoi`) correctly returned a clean `422` (see the
   RED/GREEN section above for the bug this caught).
3. **Compare on a known-absent query** (`"a nuclear submarine"`): both
   columns answered, neither raised, neither claimed "is not present" —
   and, tellingly, **the two models' confidence bands disagreed** (RemoteCLIP
   read "medium" / inconclusive, PE-Core read "high" / stands-out) for the
   *same* absent query. This is exactly the F-1a point made concrete: two
   models' raw signals for the identical input are not just different in
   scale, they can point different directions, which is precisely why nothing
   here ever collapses them into one verdict.
4. **Model/AOI combination with no index.** Built a fixture
   (`single_model_engine`) where PE-Core-L14-336 has **no** on-disk index
   for the AOI at all (not a hypothetical — this is what a real index_root
   looks like before a second model has been built for a newly-added AOI).
   `Engine()` construction itself did not crash (S11b's discovery is
   independent and forgiving per model). A single-model `/api/query` for the
   missing (model, AOI) pair returned a clean `422` naming the AOIs that
   model *does* have. A `/api/compare` over the same AOI returned `200`:
   RemoteCLIP's column fully populated, PE-Core's column
   `{"available": false, "model_label": "PE-Core-L14-336 · 1024-d", ...}` —
   one model's gap never took down the other's genuinely-available answer.
5. **Unknown model id** (`"not-a-real-model"`): clean `422`, message names
   the two real model ids.

No abuse case produced a stack trace, an unhandled 500, or a blank/broken
page at any point.

## Screenshots

`retrieval/briefs/S11b_screenshots/`:

- **`compare_view.png`** — query "tents", compare mode on, AOI
  `X605_Y3388`. Shows: the two-column layout per scale row (448/224/112px),
  each column headed `RemoteCLIP-ViT-L-14 · 768-d` / `PE-Core-L14-336 ·
  1024-d`, each with its own independent confidence band ("high (100th
  pct.)" for both, here — a real, present query), the status line reporting
  per-model timing (`RemoteCLIP-ViT-L-14 · 768-d: 26 ms · PE-Core-L14-336 ·
  1024-d: 28 ms`) with no combined/joint number, the `#active-filters` chip
  row showing each model's own result count (U-6), and both the cross-row
  and cross-model caveats in the notes section. **Visually**, the two
  columns' tile thumbnails are different tiles at every scale — a direct,
  visible confirmation of the brief's measured "0/5 top-5 agreement" claim,
  not just a number in a doc.
- **`single_model_selector.png`** — same session, compare unchecked, model
  switched to PE-Core-L14-336 via the selector. Shows the single-column
  layout unchanged from pre-S11b, the model selector reflecting the new
  selection, the status line (`Query "tents" [PE-Core-L14-336 · 1024-d] —
  29 ms`) and the active-filter chip (`Model: PE-Core-L14-336 · 1024-d — 30
  result(s)`).

Both captured against the real running app (`uvicorn`, real production
index) via headless Chrome driven over the DevTools Protocol (the DOM
interaction — checking the compare toggle, changing the model select, and
waiting for the resulting fetch to resolve — needed real click/change-event
simulation, which the documented static
`--screenshot=OUT.png` flag alone cannot do; a CDP session was scripted
instead, screenshot capability unchanged).

## PE-Core background (why it exists, and where)

The brief flagged that confidence bands must be **within-model**. For that
to mean anything for PE-Core in the *real* app (not just in a test
fixture), PE-Core needs its own background reference set — and it did not
exist before this stage (only RemoteCLIP's, under `index/export/<aoi>/`,
which the brief puts out of scope for a second model's artifacts).

Built for all 4 production AOIs (`AYOSH`, `X605_Y3388`, `gaza`, `leb`),
using the already-built PE-Core index (**no re-embedding of tiles** — this
is 30 text queries through PE-Core's text tower against vectors that
already exist on disk, a few seconds of GPU time per AOI, not an index
rebuild):

```
AYOSH built in 1.09 s -> index/emb/PE-Core-L14-336/AYOSH/background.json
X605_Y3388 built in 0.45 s -> index/emb/PE-Core-L14-336/X605_Y3388/background.json
gaza built in 1.54 s -> index/emb/PE-Core-L14-336/gaza/background.json
leb built in 1.84 s -> index/emb/PE-Core-L14-336/leb/background.json
```

`retrieve.background_path`/`build_background`/`load_background` all grew an
optional `model_id` parameter for this (default `DEFAULT_MODEL_ID`, so every
pre-S11b call site is unaffected). For the default model the path is
unchanged (`index/export/<aoi>/background.json`); for any other model it
resolves under that model's own `emb_dir` instead
(`index/emb/<model_id>/<aoi>/background.json`) — **never** under
`index/export/`, confirmed both by inspection (`index/export/*/background.json`
timestamps are untouched by this stage except where `test_retrieve.py`'s
own pre-existing fixture re-writes the RemoteCLIP one idempotently, which
predates S11b) and by `test_background_path_for_non_default_model_never_under_export`.

## Deviations from the brief

1. **Both indexes held eagerly**, not lazily — see "Memory" above for the
   measurement that justified this instead of the brief's fallback.
2. **`Engine.embedder` kept as a backward-compat property** (was a plain
   attribute) proxying `self.embedders[self.default_model]`, so the one
   pre-existing test that reads it directly
   (`test_app.py::test_first_query_latency_reported`,
   `assert fresh.embedder is None`) needed no change.
3. **Built PE-Core backgrounds for all 4 production AOIs** (see above) —
   not explicitly asked for, but without it every PE-Core confidence band in
   the real app would read "unknown" forever, which is a worse UX outcome
   than the brief's own U-3/F-1a intent.
4. **`run_compare` validates the AOI up front** (raises `ValueError` ->
   clean 422 for a genuinely unknown AOI) before falling into the
   per-model try/except that handles "this specific model lacks this
   (real) AOI" gracefully — not spelled out in the brief, but necessary to
   keep the single-model and compare-view AOI-error behaviour consistent
   (see bug #2 under RED/GREEN).

## Blockers

None.
