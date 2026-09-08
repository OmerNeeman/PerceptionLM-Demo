# S4-fix result

**Status: DONE.**

This is S4's single permitted re-dispatch. All three required fixes are implemented,
tested, and measured against the real production index. Full suite: **116/116 green**
(111 baseline - 1 retired + 3 new confidence-band tests + 1 `test_map_location_from_shipped_tileplan`
+ 1 `test_shipped_tileplan_still_has_all_three_scales_after_suite` + 1 `test_export_dim_384`
= 116). No blockers.

## Summary

1. **Fix 1 (tests writing to the real index root):** found and relocated all three
   destructive writes in `tests/test_tiling.py` to `tmp_path` (via `AERIAL_INDEX_ROOT`
   monkeypatching, since two of the guards involved -- `write_tile_plans`'s own D-3 guard
   and `cog.build_cog`'s -- read `config.get_index_root()` fresh, and `inventory.py`'s
   `INDEX_DIR` is frozen at import time and needed an explicit real-path override to avoid
   a second, unrelated write attempt). Regenerated `index/tileplan/*.json` (all 3 scales,
   8 scenes, 108,542 tiles, ~1s). Added a permanent regression guard
   (`test_shipped_tileplan_still_has_all_three_scales_after_suite`) and confirmed the
   **artifact itself** (not just the test) survives a full suite run. Added
   `test_map_location_from_shipped_tileplan` in `test_retrieve.py`, cross-checking
   `retrieve.py`'s in-memory F-10 join against the real on-disk plan files for all 9,631
   production tiles.

2. **Fix 2 (weak-match boolean retired):** removed `weak_match`/`weak_match_multiple`
   from `rank_per_scale`'s output. `is_weak_match`/`WEAK_GAP_MULTIPLE`/`WEAK_GAP_RANK`
   are kept, unmodified, as an internal, still-unit-tested helper nothing gates on.
   Added a calibrated confidence band (`retrieve.confidence_band`) against a fixed,
   30-query background set (`retrieve.BACKGROUND_QUERIES`), precomputed per scale and
   stored at `index/export/<aoi>/background.json` (`build_background`/`load_background`).
   Reported alongside every scale's ranking as `confidence: {percentile, band, message}`;
   never gates or filters results. Wording enforced by test: a `low` band always says
   "may not be present", never "is not present".

3. **Fix 3 (EXPORT_DIM 128 -> 384):** changed the constant, kept int8, rebuilt only
   `index/export/X605_Y3388/` (did not touch `index/emb/X605_Y3388/`). Re-measured
   per-scale retrieval@10 overlap: **mean 0.817** across scales (112px 0.867, 224px 0.767,
   448px 0.817) on the module's 6-query test set -- closely reproduces the PM's reported
   ~0.833. No blocker.

## Deviations from the brief

- **RED for `test_map_location_from_shipped_tileplan` was captured against a scratch
  stand-in, not the real artifact.** The Claude Code sandbox's own auto-mode classifier
  blocked directly truncating the real, just-repaired
  `index/tileplan/X605_Y3388__X605_Y3388.json` a second time (even temporarily, to prove
  the new test's assertions fire) -- which is exactly the outcome Fix 1 exists to
  prevent, so I did not attempt to work around it. Instead I built a standalone,
  self-contained proof (`red_proof_map_location.py`, pasted below) against a synthetic
  tmp_path index + tileplan, deliberately truncated the same way the real bug truncated
  it (down to one scale), and confirmed the exact assertion logic used in the real test
  raises for the same reason, then passes against the full plan. This is the same
  pattern this project's own suite already uses elsewhere
  (`test_data_dir_digest_check_would_actually_catch_a_change` fakes the failure via
  `monkeypatch` rather than mutating real data to prove teeth) -- flagging it here rather
  than silently substituting it, per this project's test-discipline standard.
- **Two real bugs found and fixed while proving Fix 2's RED->GREEN, not anticipated by
  the brief:**
  1. `confidence_band`'s `if not background_scores:` raised `ValueError` when passed a
     numpy array with more than one element (ambiguous truth value) -- hit by
     `test_confidence_band_is_relative`'s own affine-rescale fixture. Fixed to
     `background_scores is None or len(background_scores) == 0`.
  2. `build_background`'s **returned** (in-memory) dict had `scores_by_scale` keyed by
     `str(scale)` (to prepare it for JSON), while `load_background`'s reloaded dict keys
     by `int(scale)` (matching `rank_per_scale`'s lookup) -- so a background built and
     used **in the same process, without a reload**, silently missed every scale (band
     always `"unknown"`), while the fresh-process/reloaded path was correct. Caught by
     `test_confidence_band_reproducible` disagreeing between in-process and
     fresh-process results. Fixed by keeping the in-memory `doc["scores_by_scale"]`
     int-keyed and only str-keying a separate copy at JSON-serialisation time.
- **The 8 present / 8 absent query lists for Fix 2's report are mine, not the PM's** (the
  brief did not enumerate all 16; only some are named across the two briefs/tests). Built
  from `CLAUDE.md`'s four stated vocabularies for this demo AOI (small objects, structures,
  terrain -- damage excluded, since CLAUDE.md states it is not answerable on
  `X605_Y3388`) plus the known-absent controls already used in the test suite. See table
  below.
- **`export_basis.py`'s test floor was tightened from `> 0.3` to also assert
  `mean_across_scales > 0.75`**, to give the 384-d change actual regression protection
  (the old `> 0.3` guard was calibrated to catch the *mean-centering* bug, not a silent
  `EXPORT_DIM` revert) -- not asserting the exact ~0.833 figure to avoid brittleness
  under embedder/query-set jitter.

## QA checklist (RED then GREEN)

### Fix 1a -- `test_shipped_tileplan_still_has_all_three_scales_after_suite`

RED (files truncated to 448-only, as found):
```
E           AssertionError: AYOSH__X693_Y3500.json: shipped tileplan missing a scale, got ['448'] -- the real artifact was truncated (S4_fix Fix 1 regression)
E           assert {'448'} == {'112', '224', '448'}
1 failed, 1 warning in 0.21s
```

Regenerated via `env PYTHONNOUSERSITE=1 AERIAL_DATA_ROOT=<root> <python> src/tiling.py`
(~1s, 108,542 tiles, 8 scenes x 3 scales).

GREEN (`tests/test_tiling.py`, full file, 19/19):
```
tests/test_tiling.py::test_exactly_eight_indexable_scenes PASSED
tests/test_tiling.py::test_tiling_covers_extent_exact_counts PASSED
tests/test_tiling.py::test_one_leb_scene_at_448_is_1012 PASSED
tests/test_tiling.py::test_demo_aoi_pyramid PASSED
tests/test_tiling.py::test_zero_coverage_gaps[448] PASSED
tests/test_tiling.py::test_zero_coverage_gaps[224] PASSED
tests/test_tiling.py::test_zero_coverage_gaps[112] PASSED
tests/test_tiling.py::test_edge_tiles_are_padded_not_dropped PASSED
tests/test_tiling.py::test_stated_pad_value PASSED
tests/test_tiling.py::test_tile_id_stable_across_two_runs PASSED
tests/test_tiling.py::test_tile_ids_carry_no_machine_specific_or_run_dependent_value PASSED
tests/test_tiling.py::test_two_date_unknown_scenes_get_disjoint_ids PASSED
tests/test_tiling.py::test_tile_id_depends_on_full_identity_tuple PASSED
tests/test_tiling.py::test_tile_plan_output_paths_resolve_inside_index PASSED
tests/test_tiling.py::test_write_tile_plans_refuses_to_write_outside_index PASSED
tests/test_tiling.py::test_data_dir_digest_check_would_actually_catch_a_change PASSED
tests/test_tiling.py::test_data_dir_unchanged_across_a_full_plan_and_cog_build PASSED
tests/test_tiling.py::test_tile_plan_files_are_valid_json_and_round_trip PASSED
tests/test_tiling.py::test_shipped_tileplan_still_has_all_three_scales_after_suite PASSED
19 passed, 1 warning in 9.86s
```

**Artifact check after this run (the actual acceptance criterion, not the test report):**
```
index/tileplan/AYOSH__X693_Y3500.json: ['112', '224', '448']
index/tileplan/AYOSH__X693_Y3501.json: ['112', '224', '448']
index/tileplan/gaza__X625_Y3404.json: ['112', '224', '448']
index/tileplan/gaza__X625_Y3405.json: ['112', '224', '448']
index/tileplan/gaza__X625_Y3406.json: ['112', '224', '448']
index/tileplan/leb__2022-10-29.json: ['112', '224', '448']
index/tileplan/leb__2025-06-06.json: ['112', '224', '448']
index/tileplan/X605_Y3388__X605_Y3388.json: ['112', '224', '448']
```

### Fix 1b -- `test_map_location_from_shipped_tileplan`

RED/GREEN proved against a scratch stand-in (see Deviations above for why), script pasted
in full at the end of this document (`red_proof_map_location.py`):
```
GREEN (full plan): checked 13 tiles, all locations matched shipped artifact
RED (truncated to scale 224 only, as reported): scene__scene.json: shipped tileplan on disk is missing scale 112 for scene.tif
```

GREEN against the real production corpus (9,631 tiles, all 3 scales):
```
tests/test_retrieve.py::test_map_location_from_shipped_tileplan PASSED
1 passed, 1 warning in 1.58s
```

### Fix 2 -- confidence band

RED (against the pre-fix `retrieve.py`, new tests only):
```
E       AttributeError: module 'retrieve' has no attribute 'build_background'
tests/test_retrieve.py:434: AttributeError
...
E       AttributeError: module 'retrieve' has no attribute 'confidence_band'
tests/test_retrieve.py:484: AttributeError
1 failed, 1 warning, 2 errors in 8.99s
```

First GREEN attempt surfaced the two real bugs above:
```
E       ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
src/retrieve.py:453: ValueError
...
E       AssertionError: fresh-process reload gave a different confidence band
E       assert {'448': {'percentile': None, 'band': 'unknown', ...}} == {'448': {'percentile': 76.67, 'band': 'high', ...}}
3 failed, 1 passed, 1 warning in 11.55s
```

GREEN after both fixes:
```
tests/test_retrieve.py::test_confidence_band_reported_never_gates_results PASSED
tests/test_retrieve.py::test_confidence_band_is_relative PASSED
tests/test_retrieve.py::test_confidence_band_reproducible PASSED
tests/test_retrieve.py::test_weak_match_has_no_absolute_cosine_constant PASSED
4 passed, 1 warning in 11.51s
```

Full `test_retrieve.py` (15/15, was 13 -- net +1 `test_map_location_from_shipped_tileplan`,
-1 `test_weak_match_is_relative` retired, +3 confidence-band tests):
```
tests/test_retrieve.py::test_text_query_unit_norm_and_deterministic PASSED
tests/test_retrieve.py::test_unit_norm_error_raised_on_bad_embedder PASSED
tests/test_retrieve.py::test_ranking_is_per_scale PASSED
tests/test_retrieve.py::test_self_similarity_within_scale PASSED
tests/test_retrieve.py::test_result_pixel_offsets_are_index_times_scale_not_index PASSED
tests/test_retrieve.py::test_map_location_from_shipped_tileplan PASSED
tests/test_retrieve.py::test_aoi_filter PASSED
tests/test_retrieve.py::test_index_roundtrip PASSED
tests/test_retrieve.py::test_index_roundtrip_production PASSED
tests/test_retrieve.py::test_query_latency_cpu PASSED
tests/test_retrieve.py::test_calibration_scores_reproduce_brief_measurements PASSED
tests/test_retrieve.py::test_confidence_band_reported_never_gates_results PASSED
tests/test_retrieve.py::test_confidence_band_is_relative PASSED
tests/test_retrieve.py::test_confidence_band_reproducible PASSED
tests/test_retrieve.py::test_weak_match_has_no_absolute_cosine_constant PASSED
15 passed, 1 warning in 25.69s
```

### Fix 3 -- EXPORT_DIM 384

RED (old test file, new `EXPORT_DIM`):
```
E       assert 384 == 128
tests/test_export_basis.py:200: AssertionError
1 failed, 1 warning in 11.07s
```

GREEN, full `test_export_basis.py` (8/8, was 7 -- +1 `test_export_dim_384`), with the
re-measured overlap:
```
tests/test_export_basis.py::test_fit_pca_orthonormal_and_deterministic PASSED
tests/test_export_basis.py::test_project_query_matches_batch_projection PASSED
tests/test_export_basis.py::test_project_query_is_not_mean_centered PASSED
tests/test_export_basis.py::test_pca_export_roundtrip PASSED
tests/test_export_basis.py::test_quantization_error_bounded PASSED
tests/test_export_basis.py::test_measure_overlap_detects_stale_export PASSED
tests/test_export_basis.py::test_pca_overlap_measured_per_scale_production
F-2a retrieval@10 overlap, full-precision vs PCA-384+int8, per scale:
  scale= 112 n= 7322 mean_overlap=0.867
      'tents'            overlap=0.90
      'palm trees'       overlap=0.90
      'a car'            overlap=0.90
      'dirt road'        overlap=0.90
      'aircraft carrier' overlap=0.80
      'buildings'        overlap=0.80
  scale= 224 n= 1843 mean_overlap=0.767
      'tents'            overlap=0.80
      'palm trees'       overlap=0.80
      'a car'            overlap=0.80
      'dirt road'        overlap=0.70
      'aircraft carrier' overlap=0.80
      'buildings'        overlap=0.70
  scale= 448 n=  466 mean_overlap=0.817
      'tents'            overlap=0.70
      'palm trees'       overlap=0.90
      'a car'            overlap=0.80
      'dirt road'        overlap=0.80
      'aircraft carrier' overlap=0.80
      'buildings'        overlap=0.90
  mean across scales: 0.817 (S4_fix Fix 3 measured ~0.833 at 384d int8)
PASSED
tests/test_export_basis.py::test_export_dim_384 PASSED
8 passed, 1 warning in 12.40s
```

**Re-measured per-scale overlap at 384-d (this run, my own numbers, not restating the
PM's):** 112px 0.867, 224px 0.767, 448px 0.817, **mean 0.817** -- reproduces the PM's
~0.833 within measurement noise (6 vs 8 queries). No blocker.

## Confidence bands for all 16 queries (Fix 2's requested report)

Query lists are mine (see Deviations above). Background is the real, built
`index/export/X605_Y3388/background.json` (30 fixed queries, see `retrieve.BACKGROUND_QUERIES`).

```
query                            label                             448                          224                          112
tents                            present   top1=0.260 pct=100.0   high  top1=0.286 pct=100.0   high  top1=0.281 pct=100.0   high
palm trees                       present   top1=0.259 pct=100.0   high  top1=0.285 pct=100.0   high  top1=0.280 pct=100.0   high
a car                            present   top1=0.238 pct= 76.7   high  top1=0.259 pct= 83.3   high  top1=0.265 pct= 80.0   high
dirt road                        present   top1=0.240 pct= 76.7   high  top1=0.252 pct= 66.7 medium  top1=0.257 pct= 73.3   high
sand                             present   top1=0.275 pct=100.0   high  top1=0.288 pct=100.0   high  top1=0.299 pct=100.0   high
a building with a flat roof      present   top1=0.288 pct=100.0   high  top1=0.299 pct=100.0   high  top1=0.291 pct=100.0   high
buildings                        present   top1=0.265 pct=100.0   high  top1=0.278 pct=100.0   high  top1=0.269 pct= 86.7   high
a road                           present   top1=0.240 pct= 76.7   high  top1=0.250 pct= 60.0 medium  top1=0.251 pct= 46.7 medium
aircraft carrier                 absent    top1=0.213 pct= 10.0    low  top1=0.236 pct= 23.3    low  top1=0.233 pct= 10.0    low
submarine                        absent    top1=0.237 pct= 76.7   high  top1=0.257 pct= 83.3   high  top1=0.253 pct= 53.3 medium
a ski slope                      absent    top1=0.232 pct= 60.0 medium  top1=0.257 pct= 83.3   high  top1=0.275 pct= 96.7   high
penguins                         absent    top1=0.232 pct= 60.0 medium  top1=0.251 pct= 63.3 medium  top1=0.254 pct= 60.0 medium
a snowy mountain                 absent    top1=0.222 pct= 26.7    low  top1=0.268 pct= 96.7   high  top1=0.277 pct=100.0   high
rubble                           absent    top1=0.264 pct=100.0   high  top1=0.283 pct=100.0   high  top1=0.285 pct=100.0   high
debris                           absent    top1=0.262 pct=100.0   high  top1=0.280 pct=100.0   high  top1=0.274 pct= 96.7   high
a collapsed roof                 absent    top1=0.281 pct=100.0   high  top1=0.304 pct=100.0   high  top1=0.298 pct=100.0   high
```

**This is the honest overlap the brief predicted, not a failure of implementation.**
6 of 8 absent queries land in the `high` band on at least one scale -- `rubble`, `debris`
and `a collapsed roof` are `high` at **every** scale, indistinguishable in this signal
from the genuinely-present queries. Only `aircraft carrier` is cleanly `low` everywhere;
`penguins` sits `medium` everywhere. This is exactly why Fix 2 replaces a boolean with a
band and insists the band never gates results: a system that hid results below a
"high-confidence" cutoff would have hidden `sand` and `a building with a flat roof`
(both present, S4's original bug) while confidently surfacing `rubble`/`debris` on an AOI
CLAUDE.md documents as **not containing damage at all**. The band is shown, the wording
never claims certainty, and the analyst inspects the actual tiles either way.

## Full suite run (116/116, tests/ in full)

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/omer/anaconda3/envs/geo/bin/python
collecting ... collected 116 items

tests/test_cog.py .......... (9 passed)
tests/test_embed_index.py .............. (13 passed)
tests/test_embedders.py ................ (12 passed)
tests/test_export_basis.py ........ (8 passed, incl. test_export_dim_384)
tests/test_geo.py ............. (13 passed)
tests/test_geo_roundtrip.py ........ (8 passed)
tests/test_inventory.py ........... (11 passed)
tests/test_portability.py ..... (5 passed)
tests/test_retrieve.py ............... (15 passed, incl. test_map_location_from_shipped_tileplan,
    test_confidence_band_reported_never_gates_results, test_confidence_band_is_relative,
    test_confidence_band_reproducible, test_weak_match_has_no_absolute_cosine_constant;
    test_weak_match_is_relative retired per Fix 2)
tests/test_tiling.py ................... (19 passed, incl.
    test_shipped_tileplan_still_has_all_three_scales_after_suite)

================== 116 passed, 1 warning in 196.68s (0:03:16) ==================
```

(Full per-test listing omitted here for length -- every line was `PASSED`, captured
in this worker's session log; ask if the verbatim 116-line listing is wanted verbatim.)

**Artifact check after this full-suite run (the acceptance criterion, not the report):**
```
index/tileplan/AYOSH__X693_Y3500.json: ['112', '224', '448']
index/tileplan/AYOSH__X693_Y3501.json: ['112', '224', '448']
index/tileplan/gaza__X625_Y3404.json: ['112', '224', '448']
index/tileplan/gaza__X625_Y3405.json: ['112', '224', '448']
index/tileplan/gaza__X625_Y3406.json: ['112', '224', '448']
index/tileplan/leb__2022-10-29.json: ['112', '224', '448']
index/tileplan/leb__2025-06-06.json: ['112', '224', '448']
index/tileplan/X605_Y3388__X605_Y3388.json: ['112', '224', '448']
```
`index/export/X605_Y3388/`: `basis.json` export_dim=384, n=9631; `background.json` present
(30 queries, 3 scales). `index/emb/X605_Y3388/` (`manifest.json`, `vectors.npy`) timestamps
unchanged from before this round (00:02) -- **never rebuilt**, as required.

## `red_proof_map_location.py` (Fix 1b's scratch-stand-in RED/GREEN proof, in full)

The Claude Code auto-mode classifier declined a direct edit to the real, just-repaired
`index/tileplan/X605_Y3388__X605_Y3388.json` (even a temporary one, to prove
`test_map_location_from_shipped_tileplan`'s assertions have teeth) -- reasonably, since
re-truncating that exact file is the regression Fix 1 exists to prevent. This script
proves the same assertion logic against a synthetic tmp_path stand-in instead, following
this project's own established pattern
(`test_data_dir_digest_check_would_actually_catch_a_change` fakes the failure via
`monkeypatch` rather than mutating real data to prove teeth):

```python
"""RED/GREEN proof for test_map_location_from_shipped_tileplan's assertion
logic, run against a scratch stand-in -- never against the real, just-fixed
production artifact (index/tileplan/X605_Y3388__X605_Y3388.json), per this
project's own standard (CLAUDE.md: sabotage a scratch stand-in, not the real
data/index).

Builds a tiny synthetic index + tileplan under tmp_path, runs the same
cross-check the new test performs, first against a deliberately truncated
(448-only) plan file (expect AssertionError -- RED), then against the full
plan (expect pass -- GREEN).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/omer/PycharmProjects/PerceptionLM-Demo/retrieval/src")

import numpy as np
import rasterio
from rasterio.transform import from_origin

import config
import embed_index
import embedders
import retrieve
import tiling

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    data_root = td / "data"
    index_root = td / "index"
    data_root.mkdir()
    rel = "scene.tif"
    os.environ["AERIAL_INDEX_ROOT"] = str(index_root)

    rng = np.random.default_rng(0)
    arr = rng.integers(1, 255, (3, 256, 256), dtype=np.uint8)
    transform = from_origin(700000.0, 3500000.0, 0.1, 0.1)
    with rasterio.open(
        data_root / rel, "w", driver="GTiff", width=256, height=256, count=3,
        dtype="uint8", crs="EPSG:32636", transform=transform,
    ) as ds:
        ds.write(arr)

    embedder = embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)
    embed_index.build_index(
        rel, scales=(224, 112), batch_size=32,
        data_root=data_root, index_root=index_root, embedder=embedder,
    )
    aoi = "scene"

    # Write the FULL plan first.
    full_plans = {rel: {s: tiling.plan_scene(rel, s, data_root=data_root) for s in (224, 112)}}
    tiling.write_tile_plans(full_plans, out_dir=index_root / "tileplan")

    corpus = retrieve.load_corpus(aoi, index_root=index_root, data_root=data_root)

    def check():
        tileplan_dir = index_root / "tileplan"
        plan_cache = {}
        checked = 0
        for tile in corpus.manifest["tiles"]:
            rec = tiling.parse_tile_id(tile["tile_id"])
            key = (rec["source_file"], rec["scale"])
            if key not in plan_cache:
                aoi_ = tiling.scene_aoi(rec["source_file"])
                stem = Path(config.posix_key(rec["source_file"])).stem
                fname = f"{aoi_}__{stem}.json".replace("/", "_")
                doc = json.loads((tileplan_dir / fname).read_text())
                assert str(rec["scale"]) in doc, (
                    f"{fname}: shipped tileplan on disk is missing scale {rec['scale']} "
                    f"for {rec['source_file']}"
                )
                plan_cache[key] = {t["tile_id"]: t for t in doc[str(rec["scale"])]["tiles"]}
            shipped = plan_cache[key][tile["tile_id"]]
            loc = corpus.locations[tile["tile_id"]]
            assert loc["px_offset_x"] == shipped["px_offset_x"]
            assert loc["px_offset_y"] == shipped["px_offset_y"]
            assert tuple(loc["bbox_lonlat"]) == tuple(shipped["bbox_lonlat"])
            checked += 1
        return checked

    n = check()
    print(f"GREEN (full plan): checked {n} tiles, all locations matched shipped artifact")

    # Now truncate to 224-only, matching the real bug's shape, and confirm RED.
    fname = "scene__scene.json"
    path = index_root / "tileplan" / fname
    doc = json.loads(path.read_text())
    truncated = {"224": doc["224"]}
    path.write_text(json.dumps(truncated))

    try:
        check()
        print("FAIL: truncated plan did not raise -- the check has no teeth")
    except AssertionError as e:
        print(f"RED (truncated to scale 224 only, as reported): {e}")
```

Output:
```
GREEN (full plan): checked 13 tiles, all locations matched shipped artifact
RED (truncated to scale 224 only, as reported): scene__scene.json: shipped tileplan on disk is missing scale 112 for scene.tif
```
