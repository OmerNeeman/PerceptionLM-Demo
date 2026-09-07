# Handoff prompt — read this first

Paste the block below into a fresh Claude Code session started in
`/home/omer/PycharmProjects/PerceptionLM-Demo`.

---

You are taking over an aerial-imagery retrieval project at the start of
implementation. All design work is finished and signed off; nothing has been
built yet. **Do not redesign it, and do not re-run reconnaissance** — the
survey cost two long agent runs and its results are recorded as measured fact.

**The system:** a user types a short description — "tents", "a white car",
"dirt road", "mosque" — and gets back the aerial image tiles that best match,
ranked by cosine similarity, with their map locations. Text and image are
embedded into one shared space by an off-the-shelf CLIP-style dual-tower model.
Nothing is trained.

**Your first job is a running demo**, not a complete system. One AOI, indexed,
queryable, exported as a standalone HTML page the owner can play with. Breadth
comes after.

## Read these, in this order, before doing anything

1. `retrieval/CLAUDE.md` — problem statement, roles, environment, standards.
   **The three traps in it will silently ruin a run if you skip them.**
2. `retrieval/plan.md` — stages, milestones, dependency order, progress ledger.
3. `retrieval/spec.md` — the source of truth. Requirements are checkable
   assertions with concrete expected results. Code exists only to satisfy it.
4. `retrieval/notes.md` — every decision and why, including two rejected
   approaches and the evidence that killed them.
5. `retrieval/docs/DATA.md` — the imagery: exact paths, dimensions, CRS, true
   ground resolution, dates, exclusions, and where the free evaluation ground
   truth lives.

## Then execute

`retrieval/briefs/S0.md`, exactly as written. It is the embedder bake-off —
the only irreversible decision in the project, settled by measurement on one
subregion before anything is built at scale.

After S0 gates green, work down `plan.md`: S1 → S2 → S3 → S4 → S5. **S5 is the
milestone** — after it the owner has something to use. One stage in flight at a
time; do not start N+1 until N is green with pasted evidence.

## How to work

You are the PM. Write a brief per stage into `retrieval/briefs/`, dispatch
subagents to implement against it, then verify their work yourself rather than
trusting the handback — re-run the tests, re-check the numbers. Never write
production code yourself unless it is a few lines. Keep `plan.md`'s ledger and
`notes.md` current as you go.

Tests are test-first. For every acceptance item, the handback must paste **the
RED failure and then the GREEN pass**. Green-only evidence means the test was
born green; send it back.

If a spec criterion turns out to be unsatisfiable, stop and flag it with
evidence — do not quietly weaken it. If it is merely unmet, that is a
send-back, not an amendment.

## Non-negotiables

```bash
export PYTHONNOUSERSITE=1        # or torch silently runs on CPU and looks fine
export CUDA_VISIBLE_DEVICES=0    # GPU 1 drives the desktop
/home/omer/anaconda3/envs/geo/bin/python   # not the default python3
```

- **fp16, never bf16.** Turing cards; bf16 is emulated and 5.6x slower than
  fp16 — slower even than fp32. Model cards recommending bf16 are wrong here.
- **No training, no fine-tuning.** Owner constraint for this phase.
  Off-the-shelf weights only. Recommend training as future work if you like,
  but do not put it in the pipeline.
- **`/home/omer/PycharmProjects/Dynamic-Terrain/data` is READ-ONLY.** It is
  another project's directory. Never write, move or convert in place. All
  derived output goes under `retrieval/index/`, which is gitignored.
- **Never classify imagery by filename.** Only 12 of ~232 rasters in that tree
  are real imagery; the rest are model outputs. Source rasters have 187-256
  distinct values per band, derived ones have <= 6. Indexing a binary mask as
  imagery yields an index that looks fine and means nothing.
- **Ground distances need the Web Mercator correction.** `gdalinfo` prints
  12.5 cm/px for the `leb` scenes; the true ground GSD is **10.47 cm/px**
  (multiply by cos(latitude)). Uncorrected figures are 19.4% too large.
- **Retrieve on pooled embeddings, not patch tokens.** This looks like an
  oversight and is not — it was argued twice and settled on published
  evidence. PE-Core's patch tokens are 1536-d and never entered its
  contrastive objective; the 1280-d text-aligned space exists only after
  attention pooling. MaxSim over a CLIP tower's patch tokens is **ColSigLIP:
  2.5 nDCG@5**, against 51.4 for simply pooling.

  ColPali-style late interaction works only because patch embeddings pass
  **through an LLM decoder** into its text token space — "this enables
  leveraging the ColBERT strategy," in the authors' words. So "bypass the
  decoder **and** do patch MaxSim" is exactly the 2.5 configuration. You must
  pick one. We picked the encoder-only path.

  Small objects are handled by **multi-scale crops** — 448/224/112 px. At
  112 px a car is 5.9% of the tile instead of 0.37%, a 16x gain in signal
  fraction, using the only representation that is actually text-aligned.
  Citations in `notes.md#research-1` and `notes.md#arch-confirm`.

## What the owner cares about

A working, good-enough text-to-tile retrieval demo on real aerial imagery. The
acceptance gate is qualitative — they will judge it by trying queries and
looking at the results. Evaluation numbers (`spec.md` §5) are produced as
information for a product-scoping decision, and never block a stage.

Ask the owner when a decision is genuinely theirs. Otherwise proceed and log it.
