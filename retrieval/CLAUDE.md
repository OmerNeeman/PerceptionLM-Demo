# Aerial Tile Retrieval — project anchor

> Every worker reads this file first, before any brief.

## Problem statement

**Analysts working multi-date aerial imagery cannot ask it open-vocabulary
questions.** Today the only way to find content in a 20179x9784 ortho is to
pan and zoom by eye, or to run a fixed-class segmentation model that knows
exactly ten classes (road, sidewalk, building, rut, rock, wall, fence,
tree/hedge, grass/soil, car). Anything outside that vocabulary — "debris
beside a road", "collapsed roof", "vehicles parked in a courtyard" — has no
tool at all, and the imagery already sits in six AOIs across two or more
dates each.

**Success, from where the analyst sits:** type a description, get back the
ranked tiles that match it with their map locations, in seconds — then ask a
question about any specific tile and get an answer.

> **RATIFICATION PENDING.** The paragraphs above are the PM's *inferred*
> reading, not the owner's stated words. Confirmed by the owner: the data,
> the mechanism (cosine similarity over aligned text/image embeddings), the
> four decisions in `notes.md#intake`. Still **open**: who the primary user
> is, what their current workaround costs, why now, and what accuracy would
> count as good enough. `spec.md` acceptance criteria cannot be finalised
> until these are ratified — see `notes.md` open questions.

## Tier

**Standard** (methodology §1). Solo contributor, but real company data, a
stakeholder-facing deliverable, and an expensive-to-unwind architecture
choice. Full four-doc system, sequentially gated stages, independent review
at PM discretion for high-risk stages. No parallel-worker machinery.

## Platform and roles

| Role | Who |
|---|---|
| Owner | Omer (goal, scope, gates, final decisions) |
| PM | Claude Code, this session. Writes docs + briefs + acceptance criteria. **Never writes production code.** |
| Workers | Subagents, one scoped brief each, isolated context |
| Reviewer | Fresh-context subagent, prompted to refute. Mandatory for the embedding and retrieval-maths stages. |

## Environment — cold start

**Interpreter:** `/home/omer/anaconda3/envs/geo/bin/python` (3.12.13)

```bash
export PYTHONNOUSERSITE=1          # MANDATORY — see trap 1
export CUDA_VISIBLE_DEVICES=0      # GPU 1 drives the desktop
```

Installed and verified: torch 2.6.0+cu118 · transformers 5.12.0 · timm
1.0.27 · rasterio 1.5.0 · GDAL 3.12.0 · opencv 4.13.0 · scikit-learn 1.9.0
· numpy 2.3.5 · Pillow 12.2.0 · huggingface_hub 1.19.0

Absent by design: **faiss** (numpy does exact top-k over 5k x 1280 in
0.37 ms — do not install it), open_clip, sentence-transformers.
`accelerate` is absent and **is** needed for sharded VLM loading.

### Three traps that will silently ruin a run

1. **A CPU-only `torch 2.5.1+cpu` in `~/.local/lib/python3.12/site-packages`
   shadows the CUDA build in every conda env.** Without
   `PYTHONNOUSERSITE=1`, `torch.cuda.is_available()` returns `False` and the
   job runs on CPU while appearing to work. Any worker reporting "no GPU"
   has hit this, not a hardware fault.
2. **Use `float16`, never `bfloat16`.** These are Quadro RTX 6000 (Turing,
   cc 7.5). `is_bf16_supported()` returns True but bf16 is emulated:
   measured 7.0 TFLOP/s vs 38.9 for fp16 — 5.6x slower, and slower than
   fp32. Model cards recommending bf16 are wrong for this machine.
3. **`gdalinfo` reports 12.5 cm/px; the true ground GSD is 10.47 cm/px.**
   The CRS is EPSG:3857, whose metres are inflated by 1/cos(latitude). At
   lat 33.107 the factor is 0.8377. Any ground-distance figure computed from
   the raw geotransform is 19.4% too large. Always apply the correction.

## Hardware

2x Quadro RTX 6000, 24 GiB each. GPU 0 idle (24.0 GiB free); GPU 1 drives
the display. 368 GB free disk on one NVMe partition.

## Models

| Purpose | Model | Status |
|---|---|---|
| Tile + text embedding | `facebook/PE-Core-G14-448` (1280-d) | **ungated, Apache 2.0, 9.01 GiB** — verified downloadable |
| Fallback embedder | `facebook/PE-Core-L14-336` (1024-d, 2.50 GiB) | ungated, Apache 2.0 |
| Tile question answering | swappable behind an interface | **PLM is 403 / `gated: manual`** — ungated VLM is the default until approval |

PE Core loads from its **own repo**, not transformers and not open_clip:
`git clone https://github.com/facebookresearch/perception_models && pip install -e .`
then `pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)`. The timm
route (`vit_pe_core_*`) exposes the **vision tower only** — no text tower,
so it cannot do text retrieval. Do not use it for this project.

PerceptionLM's licence is FAIR **non-commercial research**. If this system
becomes a product, PLM is disqualified on licence regardless of approval;
PE Core (Apache 2.0) is not.

## Standards

- **The data directory is READ-ONLY.**
  `/home/omer/PycharmProjects/Dynamic-Terrain/data` belongs to another
  project. Never write, move, or convert in place. Derived artifacts
  (COGs, tiles, embeddings) go under `retrieval/index/`, which is gitignored.
- Source imagery only. In `leb`, 2 of 50 TIFFs are real RGB; the other 48
  are derived model outputs. Indexing a binary mask as if it were imagery is
  a correctness bug — every worker must verify band count and value
  distribution, not trust filenames.
- Determinism: fixed seeds, pinned model revisions, embeddings written with
  the model id and revision recorded alongside them.
- Structured logging at stage boundaries. No catch-and-silence.
- Never commit weights, tiles, embeddings, or imagery.

## Git

Branch per stage off `main` in the existing `PerceptionLM-Demo` repo.
Protected `main`; no direct commits. Pre-commit: formatter + lint + secret
scan. This subsystem lives in `retrieval/` and can be lifted into its own
repo later with `git subtree split` if the owner wants it separated.

## Handback protocol

Status is one of `DONE` · `DONE_WITH_CONCERNS` · `NEEDS_CONTEXT` ·
`BLOCKED`. For every test-driven acceptance item, paste **the RED failure
and then the GREEN pass**. Green-only evidence is treated as born-green and
sent back. `DONE_WITH_CONCERNS` never auto-advances.

Reversible, local, low-cost decisions: make them and report. Anything
touching scope, an interface, or `spec.md`: stop and flag as a blocker.
