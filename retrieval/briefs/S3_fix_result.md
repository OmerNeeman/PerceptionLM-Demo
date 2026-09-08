# S3-fix — handback

**Status: DONE**

All three fixes plus the batch_size addition are complete. Baseline
confirmed before any change: `pytest retrieval/tests/test_embed_index.py` ->
**6 passed**, 44.56s (production index already built; resume near-instant,
as N-2 promises). Final full run: `pytest retrieval/tests/` -> **92 passed**
(85 pre-existing + 7 new), 168.80s, zero failures, zero skips.

Neither of the brief's stop-and-flag conditions fired: the non-finite-aware
sanity check did **not** raise on any of the 9,631 real production vectors
(`test_demo_aoi_tile_count` reloads the real index and passed), and the new
manifest/vectors consistency check did **not** fire on the real index
either (same test, same reload). **No blockers.**

Order executed: Fix 1 (NaN/Inf), Fix 2 (manifest/vectors consistency),
Fix 3 (bound coverage + mutation proof), then the batch_size addition. Test
discipline followed as instructed: every new test's RED was captured against
the pre-fix code before any source change, pasted below, then the fix
applied and the GREEN captured.

---

## Fix 1 — NaN/Inf no longer defeat the sanity check

**DONE.** `src/embed_index.py`, `load_index` (was line 417, now ~line 465
after the docstring additions).

**Change:** before the existing `abs(raw_norms - 1.0) > ATOL` comparison,
non-finite is checked explicitly on both the vectors themselves
(`np.isfinite(vectors_f32).all(axis=1)`) and their norms
(`np.isfinite(raw_norms)`). Either non-finite raises immediately, naming the
first 5 offending row indices and whether each is `NaN` or `Inf` — never
silently dropped, repaired, or renormalised.

**RED** (pre-fix, both tests written first):
```
test_nan_vector_raises:
  Failed: DID NOT RAISE RuntimeError
  (pre-fix: NaN norm makes abs(nan-1.0) > ATOL evaluate False, so `bad.any()`
  is False and load_index returns a NaN-poisoned vector with no exception)

test_inf_vector_raises:
  AssertionError: /.../inf_aoi: 1 stored vectors have norm far from 1.0
  (> 0.01 off -- fp16 quantisation alone cannot explain this); first
  offending indices [2], norms [inf]
  assert 'Inf' in '...' -- FAILED because the pre-fix message reports this
  as a generic "norm far from 1.0" gross-error, not as non-finite. Inf
  happens to trip the ATOL threshold by accident (abs(inf-1)>0.01 is True),
  which is exactly why NaN cannot be trusted to behave the same way -- the
  reviewer's point that the comparison's *domain* is defeated, not just its
  threshold.
```

**GREEN** (post-fix):
```
retrieval/tests/test_embed_index.py::test_nan_vector_raises PASSED
retrieval/tests/test_embed_index.py::test_inf_vector_raises PASSED
```
Messages now read e.g. `"...1 stored vectors contain non-finite values...
first offending row indices [1], kinds ['NaN']"` and `[2], kinds ['Inf']`
respectively — non-finite is reported as its own category, distinct from
the ATOL gross-error path.

---

## Fix 2 — `build_index` validates what it resumed from

**DONE.** `src/embed_index.py`, `_load_state` (~line 126), called from
`build_index` before `todo` is computed.

**Change:** `_load_state` now raises `RuntimeError` naming both counts when
`vectors.npy` holds **fewer** rows than the manifest lists — the corruption
direction the module's own write-ordering argument claims cannot happen. The
existing truncate-to-`n`-when-**more**-rows behaviour (the actually-possible
kill scenario) is unchanged.

**RED** (pre-fix, reproducing the reviewer's exact repro — build a real
16-tile index, truncate `vectors.npy` to 10 rows, call `build_index` again):
```
test_build_index_detects_manifest_vector_mismatch:
  Failed: DID NOT RAISE RuntimeError
  (pre-fix: done_ids covers all 16 manifest tile_ids, todo computes empty,
  the run returns {"embedded_total": 16} -- exactly the reviewer's false
  success report, reproduced here against real tiles from a fresh fixture
  scene, not a synthetic manifest)
```

**GREEN** (post-fix):
```
retrieval/tests/test_embed_index.py::test_build_index_detects_manifest_vector_mismatch PASSED
```
Raised message: `"...manifest lists 16 tiles but vectors.npy has only 10
rows -- inconsistent index, refusing to resume from it..."` — both counts
named, per the acceptance criterion.

---

## Fix 3 — the sanity bound, proven by test and by mutation

**DONE.** No source change here beyond what Fix 1 already contributes
(non-finite is now checked ahead of, and independently of,
`RAW_NORM_SANITY_ATOL`). This fix is entirely new test coverage:
`test_gross_norm_error_raises` (a finite norm-0.5 vector) and
`test_fp16_noise_does_not_raise` (the over-tightening guard — 200 real
unit vectors round-tripped through float16, matching the module's own
measured-noise claim).

**RED / GREEN for the two new tests** — both already passed on first write,
because the pre-fix `ATOL` bound itself was correct (the reviewer's defect
was that nothing *tested* it, not that it didn't work):
```
test_gross_norm_error_raises PASSED
test_fp16_noise_does_not_raise PASSED
```
Since a "born green" result is exactly what the reviewer flagged as
insufficient (`ATOL = 1e9` and all 6 old tests still passed), I proved
coverage by mutation per the brief's instruction, not by the pre-existing
RED/GREEN discipline alone:

**Mutation applied** (`RAW_NORM_SANITY_ATOL = 1e-2` -> `1e9`, restored
immediately after):
```
retrieval/tests/test_embed_index.py::test_nan_vector_raises PASSED
retrieval/tests/test_embed_index.py::test_inf_vector_raises PASSED
retrieval/tests/test_embed_index.py::test_gross_norm_error_raises FAILED
  >       with pytest.raises(RuntimeError) as exc:
  E       Failed: DID NOT RAISE RuntimeError
retrieval/tests/test_embed_index.py::test_fp16_noise_does_not_raise PASSED
1 failed, 3 passed
```
`test_gross_norm_error_raises` fails exactly as expected once the bound is
disabled — proving it is the only thing standing between a norm-0.5 vector
and a silent load. `test_nan_vector_raises` / `test_inf_vector_raises` stay
green under the mutation, which is correct and expected: Fix 1 made
non-finite detection independent of `ATOL` entirely, so disabling the bound
no longer reopens the NaN/Inf hole the reviewer originally found — that
hole is closed by a separate code path now, not by this bound.

**Restored** immediately after (`sed` back to `1e-2`), confirmed via
`grep -n "RAW_NORM_SANITY_ATOL = " src/embed_index.py` ->
`RAW_NORM_SANITY_ATOL = 1e-2`, and the full suite re-run green (below).
**`RAW_NORM_SANITY_ATOL` itself was never tightened**, per the brief's
explicit instruction — only made non-finite-aware ahead of it.

---

## Addition — `batch_size` recorded in the manifest

**DONE.** `src/embed_index.py`, `build_index`:
- Fresh build: `manifest["batch_size"] = batch_size` at creation.
- Resume: `manifest.setdefault("batch_size", batch_size)` for backward
  compatibility with a manifest written before this fix, then a `log.warning`
  naming both the recorded and the requested batch size if they differ (the
  manifest keeps recording the value the index was **originally** built
  with, rather than overwriting it on every differing resume — see
  Deviations).

**RED** (pre-fix):
```
test_manifest_records_batch_size:
  KeyError: 'batch_size'
```

**GREEN** (post-fix):
```
retrieval/tests/test_embed_index.py::test_manifest_records_batch_size PASSED
```

Not in the brief's minimum acceptance list, but added
`test_batch_size_change_on_resume_logs_warning` to cover the "logged loudly"
half of this requirement, which the manifest-content test alone does not
exercise:
```
RED (pre-fix):  AssertionError: [] (no warning record with "batch_size", "8", "4")
GREEN (post-fix): PASSED
```

No embedding-behaviour change: batch size still defaults to
`DEFAULT_BATCH_SIZE = 256` and nothing about embedding itself was made
batch-invariant, per the brief's explicit instruction not to attempt that.

---

## Full final test run

```
retrieval/tests/ -> 92 passed, 1 warning in 168.80s (0:02:48)
```
92 = 85 pre-existing (6 in `test_embed_index.py` + 79 across `test_cog.py`,
`test_embedders.py`, `test_geo.py`, `test_geo_roundtrip.py`,
`test_inventory.py`, `test_portability.py`, `test_tiling.py`) + 7 new in
`test_embed_index.py` (`test_nan_vector_raises`, `test_inf_vector_raises`,
`test_gross_norm_error_raises`, `test_fp16_noise_does_not_raise`,
`test_build_index_detects_manifest_vector_mismatch`,
`test_manifest_records_batch_size`,
`test_batch_size_change_on_resume_logs_warning`). The one warning is the
pre-existing, unrelated `pyproj unable to set PROJ database path` notice.

`test_demo_aoi_tile_count` (runs the real production build against the real
demo AOI and reloads the real on-disk index at `index/emb/X605_Y3388/` in a
fresh process) passed with **both** new checks live — confirming neither
blocker condition fired against the 9,631 real vectors.

---

## Deviations / unspecified decisions

- **`_load_state`'s new raise is a plain `RuntimeError`**, matching
  `load_index`'s existing style for its own manifest/vectors consistency
  check, rather than a new named exception class.
- **On a `batch_size` mismatch at resume, the manifest keeps the
  originally-recorded value** (does not overwrite it to the new run's
  value). Rationale: the manifest should keep saying what the index was
  actually built with; only the log line needs to say what just happened.
  The brief only requires logging, not a specific manifest-update policy,
  so this is a judgment call.
- **Added `test_batch_size_change_on_resume_logs_warning`**, beyond the
  brief's minimum acceptance list (`test_manifest_records_batch_size` only),
  to give the "logged loudly" half of the batch_size requirement its own
  test rather than leaving it unverified.
- **Fix 1's non-finite check also covers non-finite in the computed norm**
  (`np.isfinite(raw_norms)`), not just in the vectors themselves — the brief
  names both explicitly ("in the vectors or in the computed norms"); for
  fp16-upcast-to-fp32 data these are almost always redundant (a non-finite
  vector produces a non-finite norm), but a norm could in principle overflow
  independently on a future dtype/shape, so both are checked directly rather
  than relying on the vector check alone.
- **Test fixtures for Fix 1/Fix 3 hand-build the manifest + vectors.npy
  directly** (`_write_raw_index` helper), bypassing `build_index` and the
  embedder entirely, so they run with no GPU dependency and can plant an
  exact broken value (NaN, Inf, norm 0.5) rather than trying to coax one out
  of a real embedding.
- **Fix 2's fixture uses a fresh, nodata-block-free raster** (not the shared
  `fixture_scene` fixture, which forces one tile to 100%-nodata) so that
  "16 planned == 16 embedded" holds exactly, matching the reviewer's 16-vs-10
  repro precisely.

## Blockers

None. Both of the brief's explicit stop-and-flag conditions were checked and
came back clean:
- The non-finite-aware sanity check did not cause any of the 9,631 real
  production vectors to raise.
- The new manifest/vectors consistency check did not fire on the real
  index.

Neither the production index nor `RAW_NORM_SANITY_ATOL` was touched.
