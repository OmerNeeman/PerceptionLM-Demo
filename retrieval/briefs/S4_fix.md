# Brief — S4-fix: three items

**Send-back on S4**, not a new stage. S4's single permitted re-dispatch.

## What was right — leave it alone

Per-scale ranking, the F-3/F-5/F-6 work, latency (9-18 ms against a 200 ms
budget), reload (0.13 s against 5 s), and the **uncentered-PCA diagnosis** were
all correct. That last one was a real catch: this corpus's mean vector has norm
0.906, so textbook mean-centering discards signal rather than noise and gave
0.00-0.10 overlap. Good work; do not revisit it.

## Fix 1 (must) — tests are overwriting a production artifact

`tests/test_tiling.py` writes to the **real** index root:

- line ~205-207: `out_dir = config.get_index_root() / "tileplan"` then `write_tile_plans(...)`
- line ~245-246: `plan_all()` then `write_tile_plans(plans)` (default = real root)
- line ~254-255: **`plan_all(scales=(448,))` written to the real tileplan directory**

That last one **truncates `index/tileplan/*.json` to scale 448 only**. The PM
regenerated the full three-scale plan once already; running the suite silently
destroyed it again. S4 then had to work around the damage by re-planning in
memory — treating a self-inflicted wound as an external constraint.

**What must be true:** no test writes anywhere under the real index root.
Every one of these writes goes to `tmp_path`. After your change, run the full
suite and then confirm `index/tileplan/*.json` still holds **all three scales**
for all 8 scenes — that check is the acceptance criterion, not the test passing.

Then regenerate the plans (`python retrieval/src/tiling.py`, ~1 s) so the
artifact is correct on disk, and **verify S4's map-location join works when
reading the real files**, not only from an in-memory re-plan. F-10 says results
carry their location; that must hold from the shipped artifact.

## Fix 2 (must) — the weak-match boolean is overfit and is being replaced

`WEAK_GAP_MULTIPLE = 0.75` was tuned on the five values the PM supplied and does
not generalise. Measured by the PM on this index:

- **112 px:** misses `submarine`, `a ski slope`, `penguins` (all absent, called
  strong).
- **448 px:** flags `sand` and `a building with a flat roof` — both **present** —
  as weak, and misses three absent queries.

The PM then tested whether **any** relative statistic can do this, over 8
present and 8 absent queries: `top1`, `z_mean`, `gap_top10`, `gap_top50`,
`gap_top100`, `top10_z`, `skew`, `n_within_1pct`. **Every one overlaps.** Raw
scores overlap too — `a snowy mountain` (absent) 0.2769 beats `a car` (present)
0.2650.

The isolation measures are **backwards**: `gap_top10` for absent queries reaches
0.812 against 0.684 for present ones, because an abundant class like `tents` has
many near-equal matches (small gap) while `penguins` retrieves a few odd tiles
that stand alone (large gap). **A reliable boolean does not exist here.** Do not
try to find one.

**Owner decision — implement a calibrated confidence band, no boolean:**

- Build a **background distribution** from a fixed, seeded list of unrelated
  text queries (~30, stored with the export so it is reproducible). Precompute
  their top-1 scores **per scale**.
- For a user query, report where its top-1 falls against that background — a
  percentile or band, per scale.
- **Never gate or hide results on it.** It is an indicator shown alongside them.
- Wording is a requirement, not a detail: low confidence means **"may not be
  present"**, never "is not present". The two are different claims and only one
  is supported.
- Remove `is_weak_match`'s boolean from the public API, or keep it only as an
  internal helper that nothing gates on.

Report the band values for the 8 present and 8 absent queries so the PM can see
what the user will see. **A large overlap between the two groups is the
expected, honest outcome** — do not tune the bands to hide it.

## Fix 3 (must) — EXPORT_DIM 128 -> 384, int8

Owner decision from the PM's measured curve (retrieval@10 overlap vs full
precision, 8 queries, this index):

| dim | dtype | MB | 448 | 224 | 112 | mean |
|---|---|---|---|---|---|---|
| 128 | int8 | 1.23 | 0.738 | 0.750 | 0.650 | **0.713** (current) |
| 256 | int8 | 2.47 | 0.838 | 0.825 | 0.738 | 0.800 |
| **384** | **int8** | **3.70** | 0.850 | 0.838 | 0.812 | **0.833** (chosen) |
| 384 | fp16 | 7.40 | 0.875 | 0.900 | 0.838 | 0.871 |
| 768 | fp16 | 14.79 | 1.000 | 1.000 | 1.000 | 1.000 (lossless) |

Set `EXPORT_DIM = 384`, keep int8, rebuild `index/export/X605_Y3388/`, and
**re-measure per scale to confirm ~0.833**. Report your numbers; do not restate
the PM's.

Note for your own understanding: 768-d fp16 is **exactly 1.000** because the
local index is already fp16, so that path is lossless by construction. The loss
is dimensionality reduction, with int8 adding a few points.

## Acceptance criteria — test-first, paste RED then GREEN

- `test_tests_do_not_write_real_index_root` — or equivalent: after the full
  suite, `index/tileplan/*.json` holds all 3 scales for all 8 scenes. **RED is
  available right now** — the files are truncated to 448 as you read this.
- `test_map_location_from_shipped_tileplan` — F-10 satisfied by reading the real
  plan files, not an in-memory re-plan.
- `test_confidence_band_is_relative` — no absolute cosine constant in the path;
  bands are computed against the stored background set.
- `test_confidence_band_reproducible` — same query, same band, across processes.
- `test_export_dim_384` — export is 384-d int8; per-scale overlap re-measured
  and reported.
- All **111** existing tests stay green, minus any you correctly relocate to
  `tmp_path`.

## Constraints

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data \
    /home/omer/anaconda3/envs/geo/bin/python
```

**Env vars do not persist between Bash calls — prefix every invocation inline.**

- Data directory READ-ONLY. **Do not rebuild `index/emb/X605_Y3388/`** — the
  embedding index is verified; you only rebuild the *export*.
- dtype via `device.py`, paths via `config.py`, ids via `posix_key()`.
- Do not edit `CLAUDE.md`, `spec.md`, `plan.md`, `notes.md` — the PM is amending
  U-3 in parallel.

## Report back

Status, QA checklist with pasted RED-then-GREEN, full test run, the tileplan
scale check **after** the suite runs, the confidence bands for all 16 queries,
the re-measured per-scale overlap at 384-d, deviations, blockers.

**Flag as a blocker rather than working around it if:** relocating the tiling
tests breaks coverage you cannot restore, or 384-d int8 does not reproduce
~0.833.
