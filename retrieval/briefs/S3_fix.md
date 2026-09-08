# Brief — S3-fix: close the review's three defects

**Dispatch:** subagent. **Send-back on S3**, not a new stage.
**WIP:** S3's single permitted re-dispatch. A second send-back stops and escalates.

## Why you are being sent back

S3's spec compliance passed on F-1a and the nodata handling, and the adversarial
reviewer could not defeat the mixed-embedder guard, the resume path, tile
identity, or nodata correctness — it tried hard and said so. **The build itself
is good and the production index is correct.** Do not rebuild it.

But the reviewer broke the *safety net* around the fp16 decision in two ways,
and found that a claim asserted in three places is tested in none.

## Fix 1 (severe) — NaN silently bypasses the sanity check and poisons the output

`src/embed_index.py:417`:

```python
bad = np.abs(raw_norms - 1.0) > RAW_NORM_SANITY_ATOL
```

**NumPy comparisons against NaN are always `False`.** A vector containing a NaN
component has a NaN norm, is therefore *not* flagged, and `load_index` then
divides by it — returning NaN in every element as a "successfully loaded" unit
vector, with **no exception**. The reviewer constructed this and reproduced it.

This is worse than a threshold miss: the check's comparison is defeated
entirely, so the value is not merely outside the 1e-2 window, it is outside the
comparison's domain.

**What must be true:** non-finite values (NaN **and** ±inf, in the vectors or in
the computed norms) are detected explicitly and **raise**, naming the offending
row index and what was found. Do not silently drop or repair them — a NaN vector
means something upstream went wrong and the PM must see it.

## Fix 2 — `build_index` reports success without validating what it resumed from

`load_index` checks that `vectors.npy`'s row count matches the manifest.
`build_index` does not. The reviewer hand-built an index whose manifest lists 16
tiles while `vectors.npy` holds 10 rows; `build_index` computed `todo` as empty,
never entered the checkpoint branch, and **returned `{"embedded_total": 16}` — a
false success report** with no exception, warning, or repair.

The module's docstring claims a kill can only leave the index in a recoverable
state. The reviewer could not break that by killing a real build, so the
write-ordering argument does hold today — but **nothing asserts it at runtime**,
so a future change to checkpoint order, or any non-crash corruption, produces a
lying "done".

**What must be true:** `build_index` validates manifest/vectors consistency
**when it loads an existing index**, before deciding what work remains, and
raises on a mismatch naming both counts. An argument that holds is not a
substitute for a check that runs.

## Fix 3 — the sanity bound is asserted in three places and tested in zero

The reviewer set `RAW_NORM_SANITY_ATOL = 1e9`, effectively disabling it, and
**all 6 tests still passed**. No fixture constructs a vector broken enough to
reach it.

**What must be true:** tests that would fail if the bound were removed or
widened. At minimum: a grossly-wrong norm (e.g. 0.5) raises; a NaN raises (Fix
1); and normal fp16 quantisation noise does **not** raise. Prove each by
mutation — disable the bound, watch your new test fail, restore it.

## Also do this — record `batch_size` in the manifest

The reviewer measured that the same tile embedded **alone** differs from the same
tile inside a **batch of 32** by up to **2.574e-4** — 25x N-4's 1e-5 tolerance.
Root cause is GPU fp16 GEMM non-associativity: kernel selection depends on batch
shape. **This is not a bug in your code**, and cross-process determinism is
exact (0.000e+00) when batch composition is held identical.

But it means an index is only reproducible at a **fixed batch size**, and the
manifest does not currently record which one was used.

**What must be true:** the manifest records `batch_size`, and a rebuild that
would use a different one is at minimum logged loudly. Do not attempt to make
embedding batch-invariant — that would mean batch size 1 and a ~50x slowdown for
no retrieval benefit. The spec side of this is being handled by the PM as an
N-4 amendment; your job is only to record the value.

## Explicitly NOT in scope

- **Rebuilding the production index.** It is correct and verified; the reviewer
  confirmed `embedded ∪ skipped == planned` exactly as sets. Leave it.
- **The renormalise-on-load decision.** Upheld by the reviewer, who also showed
  it is load-bearing rather than cosmetic: raw stored norms vary 0.9997–1.0003,
  so an un-renormalised dot product is not a valid cosine *across* tiles, and
  ranking genuinely changes. Keep it.
- **The 1e-2 bound's inability to catch a subtle 0.5%-off vector.** That is its
  designed behaviour — anything between fp16 noise (~3e-4) and 1e-2 is
  genuinely indistinguishable from quantisation. Going to the tech-debt ledger,
  not to you. **Do not tighten it**, or you will start rejecting healthy vectors.

## Acceptance criteria — test-first, paste RED then GREEN

- `test_nan_vector_raises` — a NaN component raises, naming the row. **Prove RED
  against the current code**, where it silently returns NaN.
- `test_inf_vector_raises` — ±inf likewise.
- `test_gross_norm_error_raises` — a norm-0.5 vector raises.
- `test_fp16_noise_does_not_raise` — real quantisation noise passes. This is the
  guard against over-tightening.
- `test_build_index_detects_manifest_vector_mismatch` — 16 listed, 10 stored →
  raises naming both, instead of reporting success.
- `test_manifest_records_batch_size`.
- All **85** existing tests stay green.

## Constraints

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data \
    /home/omer/anaconda3/envs/geo/bin/python
```

**Env vars do not persist between Bash calls — prefix every invocation inline.**

- Data directory READ-ONLY. Fixtures under `retrieval/index/` only. A write-guard
  RED targets a **scratch stand-in**, never the real data root.
- Do not edit `CLAUDE.md`, `spec.md`, `plan.md`, `notes.md`, or the production
  index under `index/emb/X605_Y3388/`.
- dtype via `src/device.py`; paths via `src/config.py`; ids via `posix_key()`.

## Report back

Status, QA checklist with **pasted RED-then-GREEN**, full test run, deviations,
unspecified decisions, blockers.

**Flag as a blocker rather than working around it if:** making the sanity check
non-finite-aware causes any healthy production vector to raise (it must not —
9,631 real vectors currently pass), or the manifest/vectors consistency check
fires on the real index.
