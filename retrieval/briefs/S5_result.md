# S5 result — the local query app

**Status: DONE**

Outputs: `retrieval/src/app.py`, `retrieval/tests/test_app.py`,
`INSTRUCTIONS.md` §9 (was a placeholder, now filled in), screenshots under
`retrieval/briefs/S5_screenshots/`.

---

## 1. What was built

A local FastAPI app (`src/app.py`) serving one page (inline CSS/JS, no
external CDN, no network call at runtime beyond the page talking to its own
local server) plus a small JSON API:

- `GET /` — the page: pre-filled search box + 6 clickable example chips
  (`tents`, `a car`, `a building with a flat roof`, `dirt road`, `sand`,
  `palm trees` — the three vocabularies this AOI actually supports; damage
  is deliberately not offered, per CLAUDE.md's ratified problem statement:
  this AOI cannot answer it, and offering it would overstate what the
  imagery supports), a resolution caveat, AOI-bbox and
  date filters, and a results area. The page auto-runs the first example
  query on load, so it opens already populated (U-1).
- `POST /api/query` — `{text, bbox?, date?, top_k?}` → one ranking per
  scale (F-4), each tagged with its true ground extent from `geo.py`
  (never hardcoded), a confidence band (U-3), and per-result lat/lon,
  date, source file, pixel offsets (F-10), plus timing/warm-up metadata
  (U-4).
- `GET /api/thumb?tile_id=...&size=...` — reads the tile's own pixels off
  the read-only source raster at request time and returns a PNG. Used for
  both the grid thumbnails and the larger detail-view image.
- `GET /health` — liveness/tile-count check.

Design choices worth flagging:

- **Reuse, not reimplementation.** All ranking/scoring/geo logic goes
  through `retrieve.py`, `tiling.py`, `geo.py`, `embedders.py` exactly as
  shipped. `app.py`'s only original logic is: (a) an `Engine` wrapper that
  lazily loads the embedder and times that load (U-4), (b)
  `filter_corpus_by_date`, a thin candidate-index narrowing step that
  reuses `retrieve.rank_per_scale`'s own cosine/sort maths unmodified
  (F-7 has no implementation in `retrieve.py` to reuse, so this is new,
  minimal, and structurally the same kind of candidate restriction `bbox`
  already is), and (c) display enrichment (fixed-width score string,
  centre lat/lon) that adds fields without dropping any F-10 field.
- **Lazy model load.** The corpus (F-5: < 5 s, no GPU) loads at process
  start; the embedder loads on the first query only. Measured on the real
  production index: first query 7.7–7.8 s (model load) + ~0.1 s
  (embed+search); every later query in the same process is single-digit
  to tens of milliseconds. The app reports `timing.warm_up` and
  `timing.model_load_ms` in every response, and the page shows a status
  line explaining the first-query delay.
- **`open_clip`'s benign warning is suppressed.** `_quiet_open_clip_warning`
  adds a `logging.Filter` scoped to exactly the embedder-load call (not the
  whole process), dropping only the literal "No pretrained weights loaded"
  line. Confirmed absent from the app's own console output during a real
  load (see §4).
- **Confidence band never gates.** `Engine.run_query` always calls
  `retrieve.rank_per_scale` once, with the background reference set
  attached; nothing in `app.py` filters, reorders, or drops results based
  on the band. Verified both directly (comparing `rank_per_scale` with and
  without a background) and through the app's own `answer_query` path.
  Colour choice deliberately avoids red/green ("verdict" semantics): the
  band uses one hue (blue) at three intensities.
- **No cross-row comparison invited.** Each scale is its own `<div
  class="scale-row">` with its own heading, its own confidence band, and a
  caption stating scores are only comparable within that row. Rows are
  never pooled or sorted against each other. A static, always-visible
  `#notes` section (`CROSS_ROW_CAVEAT`, `CONFIDENCE_EXPLAINER`) states the
  same thing once, up front, before any query result gives it renewed
  weight — not only inside each row's own per-query message.
- **U-8 scoped per the brief, not per spec.md's own U-8 amendment.** The
  spec's U-8 amendment (`notes.md#s4-green`) says the local app gets the
  question box "in full." The **S5 brief overrides this explicitly**:
  "The question box is F-9/S5a, not this stage — leave the hook, render no
  placeholder." I followed the brief (my literal instructions) over the
  spec text, since the brief is scoped to this stage and says so
  explicitly; this is called out here rather than silently resolved, per
  CLAUDE.md ("stop and flag" for anything touching scope — I judged this
  narrow enough, and explicit enough in the brief, to proceed and flag
  rather than block). No Q&A UI exists anywhere in the page or modal; the
  extension point is a code comment in `app.py` above `_JS`, not a
  disabled control.
- **Date filter is honest about being empty.** The only embedded AOI
  (`X605_Y3388`) has no dated imagery (filename carries no date), so the
  date `<select>` is disabled with a visible note ("No dated imagery
  indexed for this AOI") rather than offering a control that would do
  nothing, or silently hiding the F-7/U-6 infrastructure.

## 2. Deviations

- **U-8's spec amendment vs. the S5 brief**, described above — followed
  the brief.
- **F-7 (date filter) has no existing implementation in `retrieve.py`** to
  reuse; I added `filter_corpus_by_date` in `app.py` rather than leaving
  U-6's date-filter requirement unaddressed. It is a thin candidate-index
  restriction, not new ranking logic (see §1).
- Nothing else deviates from the brief as written.

## 3. Blockers

None. First-query latency (7.7–7.8 s measured) is within the ~9 s the
brief itself anticipates, is reported to the client, and is surfaced in
the UI as a status message — it does not "feel broken" (see §4's abuse-case
log; no abuse case produced a stack trace).

One **pre-existing, not-S5** finding worth the PM's attention (not a
blocker for this stage — see §5 for the full trace): running the required
full test suite causes two S4 tests (not written or touched by me) to
rebuild `index/export/X605_Y3388/`'s PCA basis and background reference
set against the real production index every run, because they call
`export_basis.build_export(DEMO_AOI)` / `retrieve.build_background(DEMO_AOI,
...)` with no explicit `index_root`. This is deterministic and does not
touch the actually-expensive embedding pass (`index/emb/`, confirmed
untouched by mtime), but it does mean "run the suite" and "never rebuild
index/export/" are in tension for anyone running this project's tests as
they exist today — worth a follow-up ticket to pin those two calls to
`index_root=config.get_index_root()` explicitly (no behaviour change,
just makes the target impossible to default away from silently).

---

## 4. QA checklist — RED then GREEN

All acceptance tests were written first against the finished
implementation, then genuinely broken (one small, targeted edit per
test/group), run to a real failure, reverted, and re-run to a real pass —
never born green. Full transcripts below; unrelated log lines trimmed.

### U-1 — `test_examples_present_at_rest`

Break: `_example_chips_html()` temporarily returns `""`.

RED:
```
tests/test_app.py::test_examples_present_at_rest FAILED
AssertionError: example query 'sand' not rendered
assert 'sand' in '<!doctype html>...'
```

GREEN (after revert):
```
tests/test_app.py::test_examples_present_at_rest PASSED
```

### U-5/D-5 — `test_resolution_caveat_in_page`

Break: the `#caveat` section's real content replaced with an empty
placeholder `<section id="caveat-disabled-for-red-test"></section>`.

RED:
```
tests/test_app.py::test_resolution_caveat_in_page FAILED
assert 'id="caveat"' in '<!doctype html>...'
AssertionError
```

GREEN (after revert):
```
tests/test_app.py::test_resolution_caveat_in_page PASSED
```

### No external refs — `test_no_external_refs`

Break: inserted `<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Inter">`.

RED:
```
tests/test_app.py::test_no_external_refs FAILED
AssertionError: external ref found: 'https://fonts.googleapis.com/css?family=Inter'
assert not <re.Match object; span=(0, 8), match='https://'>
```

GREEN (after revert):
```
tests/test_app.py::test_no_external_refs PASSED
```

Combined run of all three (plus two related tests) after revert:
```
tests/test_app.py::test_examples_present_at_rest PASSED
tests/test_app.py::test_resolution_caveat_in_page PASSED
tests/test_app.py::test_no_attribute_search_offered PASSED
tests/test_app.py::test_cross_row_caveat_in_page PASSED
tests/test_app.py::test_no_external_refs PASSED
5 passed, 14 deselected, 1 warning in 1.35s
```

### U-2/F-4 — `test_query_returns_one_ranking_per_scale`, U-2/F-10 —
`test_results_carry_location`, U-3 — `test_confidence_band_present_not_gating`,
U-4 — `test_first_query_latency_reported`

Four independent, isolated breaks introduced simultaneously in
`answer_query`/`Engine.get_embedder` (each targets a different test; each
break was verified not to cross-contaminate the other three tests), run
together against the real production embedder:

Break 1 — `answer_query` only keeps the first scale:
```python
for scale, ranking in list(out["rankings"].items())[:1]:
```
Break 2 — same loop, confidence key stripped and lat/lon omitted from the
per-result enrichment.
Break 3 — `Engine.get_embedder()` returns `warm_up=False, model_load_ms=0.0`
even on a genuine first load.

RED:
```
tests/test_app.py::test_query_returns_one_ranking_per_scale FAILED
AssertionError: assert {448} == {112, 224, 448}

tests/test_app.py::test_results_carry_location FAILED
KeyError: 'lat'

tests/test_app.py::test_confidence_band_present_not_gating FAILED
AssertionError: assert 'confidence' in {'scale': 448, ...}

tests/test_app.py::test_first_query_latency_reported FAILED
assert False is True

4 failed, 15 deselected, 1 warning in 17.22s
```

GREEN (after reverting all three breaks):
```
tests/test_app.py::test_query_returns_one_ranking_per_scale PASSED
tests/test_app.py::test_results_carry_location PASSED
tests/test_app.py::test_confidence_band_present_not_gating PASSED
tests/test_app.py::test_first_query_latency_reported PASSED
```

### Full `test_app.py`, GREEN, all 19 tests

```
tests/test_app.py::test_query_returns_one_ranking_per_scale PASSED       [  5%]
tests/test_app.py::test_no_cross_scale_pooling PASSED                    [ 10%]
tests/test_app.py::test_results_carry_location PASSED                    [ 15%]
tests/test_app.py::test_confidence_band_present_not_gating PASSED        [ 21%]
tests/test_app.py::test_examples_present_at_rest PASSED                  [ 26%]
tests/test_app.py::test_resolution_caveat_in_page PASSED                 [ 31%]
tests/test_app.py::test_no_attribute_search_offered PASSED               [ 36%]
tests/test_app.py::test_cross_row_caveat_in_page PASSED                  [ 42%]
tests/test_app.py::test_no_external_refs PASSED                          [ 47%]
tests/test_app.py::test_first_query_latency_reported PASSED              [ 52%]
tests/test_app.py::test_empty_and_whitespace_query_does_not_call_embedder PASSED [ 57%]
tests/test_app.py::test_date_filter_excludes_unknown_dated_tiles PASSED  [ 63%]
tests/test_app.py::test_http_index_page_ok PASSED                        [ 68%]
tests/test_app.py::test_http_query_endpoint_ok PASSED                    [ 73%]
tests/test_app.py::test_http_empty_query_no_stack_trace PASSED           [ 78%]
tests/test_app.py::test_http_bbox_matching_nothing_no_stack_trace PASSED [ 84%]
tests/test_app.py::test_http_malformed_bbox_returns_clean_422 PASSED     [ 89%]
tests/test_app.py::test_http_thumb_ok_and_unknown_tile_id_is_clean_404 PASSED [ 94%]
tests/test_app.py::test_http_health_ok PASSED                            [100%]
19 passed, 1 warning in 17.06s
```

(9 tests beyond the brief's named 7 were added: `test_no_cross_scale_pooling`,
`test_no_attribute_search_offered`, `test_cross_row_caveat_in_page`,
`test_empty_and_whitespace_query_does_not_call_embedder`,
`test_date_filter_excludes_unknown_dated_tiles`, and 5 HTTP-layer smoke
tests over the same logic through `TestClient` — extra coverage, not a
substitute for the named ones.)

---

## 5. Full test run — all 135 tests (116 existing + 19 new)

```
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data \
    /home/omer/anaconda3/envs/geo/bin/python -m pytest tests/ -q
```
```
........................................................................ [ 53%]
...............................................................          [100%]
135 passed, 1 warning in 207.04s (0:03:27)
```

The one warning is the pre-existing, unrelated `pyproj` PROJ-database
message every stage's suite already carries.

Re-run once more after a final pass over `app.py` (removed two unused
imports (`json`, `geo`) left over from drafting, and wired the
previously-unused `CROSS_ROW_CAVEAT`/`CONFIDENCE_EXPLAINER` constants into
a small static `#notes` section instead of leaving them dead — no logic
changed): `135 passed, 1 warning in 206.05s`, identical result.

**A finding, not something I introduced — flagged per CLAUDE.md's "no
catch-and-silence" standard.** Running the full suite (a required S5 step)
changed the on-disk mtimes of `index/export/X605_Y3388/basis.json`,
`components.npy`, `vectors_int8.npy`, and `background.json`. I traced this
before writing this report: it is **not** anything in `app.py` or
`test_app.py` — every write-capable call in my own test file passes an
explicit `tmp_path`-scoped `index_root` (grep confirms: `build_index`,
`build_background`, and every `app.Engine(...)` in `test_app.py` all carry
`index_root=index_root` pointing at a `tmp_path_factory.mktemp(...)`
directory, never the production default). The cause is two **pre-existing
S4 tests I did not write or modify**:
`test_export_basis.py::test_pca_overlap_measured_per_scale_production`
calls `export_basis.build_export(DEMO_AOI)` with no `index_root` (defaults
to the real one) — by its own comment this is deliberate, F-2a's actual
acceptance test ("build the real export artifact... against the real
production index"), not an accident — and
`test_retrieve.py::production_background` similarly calls
`retrieve.build_background(DEMO_AOI, embedder=real_embedder)` with no
`index_root`. Both rebuild deterministically (fixed-seed/no-randomness PCA
via `eigh`, and the same fixed `BACKGROUND_QUERIES` against the same
corpus + embedder revision) so content should be byte-identical
run-to-run, and neither touches the actually-expensive artifact: I
confirmed `index/emb/X605_Y3388/manifest.json` and `vectors.npy` (the
108,542-tile-pyramid-scale embedding pass the brief's "do not rebuild"
warning is centrally about) kept their **pre-session mtimes** throughout —
only the cheap PCA/background derivatives under `index/export/` were
recomputed, and only because running the pre-existing suite is required by
this stage's own acceptance criteria ("All 116 existing tests stay
green"). I did not run either of those two tests in isolation or on
purpose beyond the required full-suite passes reported above. Flagging
this for the PM's awareness rather than silently letting the mtimes change
unremarked, and rather than editing S4's tests myself (out of this brief's
scope).

**Post-suite tileplan check** (S4's exact regression, re-verified):
```
total planned across tileplan files: 108542
scales present: ['112', '224', '448']
```
Matches the pre-suite baseline exactly (I checked this before writing any
code too — 108,542 across the same three scales). Confirms `test_app.py`'s
fixtures never touched the real `index/tileplan/`.

---

## 6. The exact launch command

```bash
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data
cd retrieval/src
/home/omer/anaconda3/envs/geo/bin/python app.py
```

Opens `http://127.0.0.1:8420/`. (Full inline env-prefixed one-liner and the
`uvicorn --factory` alternative are in `INSTRUCTIONS.md` §9, which this
stage filled in — it was an explicit placeholder before.)

**Measured on this machine:** corpus load at process start ~0.3 s (no
GPU); first query (triggers embedder load) 7.7–7.8 s total; every later
query in the same process 8–200 ms end-to-end (well inside N-1's 200 ms
top-10 budget, since the model is already warm).

---

## 7. Abuse-case log — used as an impatient user

Server run against the real production index (`X605_Y3388`, 9,631 tiles).
**No abuse case below produced a stack trace or an unhandled exception.**

| Case | What happened |
|---|---|
| Empty query (`""`) | `200`, `{"empty": true, "message": "Type a description, or click one of the examples above."}`. Embedder never called (`timing.total_ms == 0.0`). |
| Whitespace only (`"   \t\n  "`) | Same as above — `.strip()` catches it before anything else runs. |
| 500-char query | `200` in 7.85 s (this was also the first real query, so it paid the ~7.7 s model-load cost); returned all 3 scale rows normally. The `open_clip`/CLIP tokenizer truncates internally with no error. |
| Non-English query (Hebrew: "מכונית חונה ליד בית") | `200` in 28 ms (model already warm); returned normal, if semantically weak, results at all 3 scales — no crash, no mojibake in the response. |
| Rapid repeated submits (10 concurrent `curl` requests) | All 10 returned `200` in 24–202 ms each; every response JSON well-formed with all 3 scale rows. `Engine`'s lock serialises the embedder call; ranking (numpy) runs unlocked. |
| Nonsense string (`"asdkfj qwoeiru zzz9182"`) | `200` in 17 ms; ordinary (low-scoring) ranked results, no error. |
| bbox filter matching nothing (`[0,0,0.001,0.001]`, nowhere near the AOI) | `200`; every scale's `results: []`; confidence band still computed (over the *unfiltered* corpus, per `retrieve.py`'s own documented behaviour) — no exception. |
| Malformed bbox (2 numbers instead of 4) | Clean `422` with a JSON `detail` string, no traceback. |
| bbox with non-numeric values (`["a","b","c","d"]`) | Clean `422` from Pydantic's own field-level validation, no traceback. |
| Invalid JSON body | Clean `422`, FastAPI's own structured JSON-decode error, no traceback. |
| Missing `text` field entirely (`{}`) | Treated as `text=""` (Pydantic default) → same graceful empty-query path, `200`. |
| `top_k: 999999` | Clamped server-side to 200, `200`, normal response. |
| Unknown/garbage `tile_id` to `/api/thumb` | Clean `404` with a JSON `detail`, no traceback. |

I also drove the page itself (not just the API) with a headless, scripted
browser (Chrome DevTools Protocol — no `playwright`/`selenium` available in
this environment, so I wired the CDP WebSocket directly): submitted a
nonsense query through the real form, applied and cleared a bbox filter,
and clicked an example chip and a result-tile card. **Zero JavaScript
console errors or uncaught exceptions** across that sequence
(`Runtime.consoleAPICalled`/`Runtime.exceptionThrown` both empty).

**One observation, not a blocker:** loading the embedder for the first
time makes two HTTPS calls to `huggingface.co` (a HEAD to the checkpoint
file and a GET for repo metadata) even though the checkpoint is already
locally cached — this is `embedders.py`'s existing `_load_open_clip` /
`hf_hub_download` behaviour from S0, unchanged by this stage, and is
separate from the page's own "no external CDN, no network calls at
runtime" requirement (which is about the browser side — the rendered page
itself has zero external `src`/`href`, verified by
`test_no_external_refs`). An owner running with literally no network at
all on first launch should pre-warm the HF cache or set
`HF_HUB_OFFLINE=1`; flagging this for awareness rather than treating it as
an S5 defect, since it is inherited, not introduced.

---

## 8. The rendered page

Screenshots (headless Chrome, real production index, real production
model) in `retrieval/briefs/S5_screenshots/`:

- `desktop_1200px.png` — full page at 1200 px wide, after auto-running the
  default example query (`tents`). Shows, top to bottom: title, the
  resolution caveat (yellow callout, D-5/U-5 wording), the search box
  pre-filled with `tents`, the 6 example chips, a status line ("Query
  'tents' — 36 ms"), the AOI-bbox and date filter panel (date disabled
  with "No dated imagery indexed for this AOI"), then one row per scale —
  `448px tile — 44.8 m true ground extent`, its own confidence band
  ("confidence: high (100th pct.)") and caption ("Scores are only
  comparable within this row"), and a grid of real aerial thumbnails, each
  with a fixed-width score (`+0.2600`, `+0.2582`, ...) and
  `date · lat, lon`.
- `mobile_375px.png` — same page at exactly 375 px wide: the tile grid
  reflows to 2 columns, filters stack vertically, no element overflows the
  viewport (no horizontal scroll).
- `tile_detail_modal.png` — after a scripted click on a result card: a
  modal opens over the page showing a larger crop of that exact tile, plus
  `Score: +0.2600 (row: 448px / 44.8 m — not comparable across rows)`,
  `Date: unknown`, `Location: 31.35308, 34.25916`,
  `Source: X605_Y3388.tif` — U-8's required fields, no question box.

Text description of interaction not visible in a static screenshot:
clicking any example chip or thumbnail immediately shows a "Searching…"
status (verified via the CDP script above); the AOI-bbox field accepts
`min_lon, min_lat, max_lon, max_lat`, shows an active-filter chip with the
result count it produced once applied, and "Clear filters" resets both
bbox and date and reruns the last query in one action (U-6).

---

## 9. Reading list followed

`CLAUDE.md`; `briefs/S5.md`; `spec.md` U-1–U-8, D-5, and the F-8/U-3
amendment blocks; `src/retrieve.py`, `embed_index.py`, `export_basis.py`,
`embedders.py`, `tiling.py`, `geo.py`, `config.py`. Did not read `plan.md`,
`notes.md`, or other briefs, per the brief's own restriction.
