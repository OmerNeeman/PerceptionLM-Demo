# Handoff — S11: multi-embedder index + model selector

Paste the block below into a fresh Claude Code session started in
`/home/omer/PycharmProjects/PerceptionLM-Demo`.

---

You are extending a working aerial-imagery retrieval system so the user can
**choose which embedding model to search with** and compare them on the same
imagery. Everything below already exists and works — **do not rebuild it.**

## What the system does today

A user types a description ("tents", "a car", "rubble") and gets back the aerial
tiles that best match, ranked by cosine similarity, with map locations, in three
rows by scale. Nothing is trained; it is off-the-shelf CLIP-style dual-tower
retrieval.

- **8 scenes, 4 AOIs, 108,542 planned tiles, 104,374 embedded** at 448/224/112 px
- Embedder: **RemoteCLIP-ViT-L-14** (768-d), chosen at S0 by measurement
- Local FastAPI app on `127.0.0.1:8420` — free text
- Two standalone offline HTML exports — precomputed queries only
- **166 tests pass**

## Read these first, in this order

1. `retrieval/CLAUDE.md` — problem, environment, **four traps**, standards
2. `retrieval/plan.md` — stages and the progress ledger
3. `retrieval/spec.md` — **F-0, F-1a, F-2, F-3, F-4** especially
4. `retrieval/notes.md` — `#s0-bakeoff` for why RemoteCLIP won; `#research-1`
   and `#arch-confirm` for why patch tokens are rejected
5. `retrieval/briefs/S0_result.md` — the original bake-off evidence

## Your task

**Let the user pick the embedder at query time**, and see the difference on the
same imagery.

**The encoder work is already done.** `retrieval/src/embedders.py` loads all
three candidates behind one interface — `PE-Core-G14-448` (1280-d),
`PE-Core-L14-336` (1024-d) and `RemoteCLIP-ViT-L-14` (768-d) — and
`perception_models` is installed at `/home/omer/perception_models`
(`import core.vision_encoder.pe` works). Do not rewrite the loader.

**The real constraint is F-1a.** Different models occupy **different vector
spaces**, so a query vector cannot be compared against tiles embedded by another
model. A selector therefore requires a **parallel index per model**, kept
strictly separate, with the manifest recording exactly one model id and revision
each — and a build that would mix them must fail loudly. That guard already
exists and is tested; keep it.

### Scope

1. **Index the corpus with PE-Core-L14-336** alongside the existing RemoteCLIP
   index. Measured cost at S0's 84.5 tiles/sec: **~21 minutes** for 104,374
   tiles, ~214 MB at 1024-d fp16.
2. **PE-Core-G14-448 is optional and probably not worth it** — 11.6 tiles/sec is
   **~2.5 hours** and ~267 MB, for the candidate that came last on every axis.
   Ask the owner before spending that; do not start it unprompted.
3. **Add a model selector** to the local app, next to the AOI and date filters.
   Switching model re-runs the current query against that model's index.
   Never merge results across models — the scores are not comparable.
4. **A same-query side-by-side view** is the point of the feature: one query,
   two models, results next to each other, clearly labelled with each model's
   name and dimensionality.

### Be honest about what this is for

S0 already measured PE Core losing: RemoteCLIP got **4/5** genuine vehicle crops
at 112 px against PE-Core-L14's 3/5 and G14's 2/5, and it is the only candidate
whose known-absent control is beaten by all 8 real queries (margin **+0.0135**;
PE-Core-L14 scored **-0.0157** — its "aircraft carrier" outranked its own "car").

So this feature is **not** expected to find a better model. Its value is letting
the owner see the difference on the full corpus rather than trusting a table
built from 336 calibration crops. Report what you measure, including if PE Core
does better than S0 suggested — that would be a genuine finding, since S0's
sample was small.

## Non-negotiables

```bash
export PYTHONNOUSERSITE=1        # else torch silently runs on CPU
export CUDA_VISIBLE_DEVICES=0    # GPU 1 drives the desktop
export AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data
/home/omer/anaconda3/envs/geo/bin/python   # not python3
```

**Your Bash tool does not persist env vars between calls — prefix every python
and pytest invocation inline, every time.** Without `PYTHONNOUSERSITE=1` a
CPU-only torch shadows the CUDA build: the index build drops from ~240 tiles/sec
to ~5 and **still looks like it is working**. A 21-minute job becomes 17 hours.
Throughput is your canary — check it in the first minute.

- **fp16 on this hardware, but do not hardcode it.** `src/device.py`'s
  `select_dtype()` derives it from compute capability: cc < 8.0 → fp16 (these
  are Turing), cc >= 8.0 → bf16, CPU → fp32. Use it.
- **Pooled output only, never patch tokens.** They are 1536-d, never entered the
  contrastive objective, and the published attempt at retrieving on them scored
  **2.5 nDCG@5 against 51.4 for pooling**. Argued twice, settled.
- **No training, no fine-tuning.** Off-the-shelf weights only.
- **`/home/omer/PycharmProjects/Dynamic-Terrain/data` is READ-ONLY.** Another
  project's directory. All output under `retrieval/index/` (gitignored).
- **Do not rebuild the RemoteCLIP index** — 104,374 verified vectors. Add
  alongside.
- **Tests must never write under the real index root.** That defect was found
  and fixed three times here. Use `tmp_path`, and after your suite runs confirm
  `index/tileplan/*.json` still holds 108,542 across three scales.
- **Install `perception_models` with `--no-deps` if you reinstall it.** Its
  requirements pin older numpy/pillow/timm/sklearn and can replace the CUDA
  torch with a CPU build.
- Never commit weights, tiles, embeddings or imagery.

## How this project works

You are the PM. Write a brief per stage into `retrieval/briefs/`, dispatch
subagents against it, then **verify their work yourself** — re-run the tests,
re-check the numbers, and **look at the artifact on disk**, not just the
handback. Keep `plan.md`'s ledger and `notes.md` current.

Tests are test-first: every acceptance item needs the **RED failure and then the
GREEN pass** pasted. Green-only evidence means the test was born green — send it
back. Two hard-won lessons from this project:

- A worker once shipped "adversarial" fixtures that were not adversarial, and
  its tests passed against unfixed code. **Validate that a fixture actually
  reproduces the defect before trusting the test built on it.**
- A stage reported counts its tests computed in memory while **the file on disk
  was wrong**. Verify emitted artifacts by reloading them in a fresh process.

If a spec criterion turns out unsatisfiable, stop and flag it with evidence —
do not quietly weaken it. Unmet is a send-back, not an amendment. Ask the owner
when a decision is genuinely theirs; otherwise proceed and log it.

## Acceptance

- Both indexes exist, each manifest recording exactly one model id + revision;
  a mixed build raises.
- `‖v‖₂ = 1.0 ± 1e-5` and correct dimensionality for both (768 / 1024).
- The selector switches model and re-queries; results are never merged across
  models.
- Side-by-side view works for one query across both models.
- **N-1 still holds: top-10 in < 200 ms on CPU** (currently 65.7 ms at full
  scale — a second index in memory may change RSS, so report it; peak is
  currently 1,419.6 MiB).
- All **166** existing tests stay green.
- Report a same-corpus comparison on at least: `tents`, `a car`, `dirt road`,
  `rubble`, and one known-absent control such as `aircraft carrier`.
