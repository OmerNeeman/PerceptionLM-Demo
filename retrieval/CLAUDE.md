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

> **RATIFIED 2026-09-07** (`notes.md#owner-ratify-2`). **The primary user is
> the owner, scoping a product.** This demo exists to inform a
> build/don't-build decision, so "success" is *enough signal to judge
> feasibility* — not time saved for a third-party analyst. There is no
> external workaround to cost, and the accuracy bar is whatever is enough to
> decide, which is why the gate is **qualitative**.
>
> **The consequence every worker must carry:** a demo whose purpose is a
> go/no-go decision is corrupted by overstating what the imagery supports. So
> **U-5 is the most important UX requirement in the spec**, and D-5's
> resolution limit is an honesty constraint, not a footnote. Never promise
> attribute-level discrimination; measurement in S0 confirms it does not
> exist (`a white car` retrieves the same crops as `car`, and the colour word
> makes results *worse*).
>
> The owner will query all four vocabularies: small objects (tents, cars),
> structures (buildings, flat roofs), terrain (dirt road, sand, palm trees),
> and damage (rubble, debris, collapsed roof). Damage is **not** answerable on
> the `X605_Y3388` demo AOI — it is what `leb`'s 2022->2025 pair and `gaza`
> contain.

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

Added at S0 (pinned; `notes.md#s0-bakeoff`): **`open_clip_torch` 3.3.0** —
now **load-bearing**, it loads the chosen embedder · `pytest` 9.1.1 ·
`accelerate` 1.14.0 (installed, not actually needed — no candidate required
sharded loading) · `perception_models` 1.0.0 @ commit `3e352cca` cloned to
`/home/omer/perception_models`, **outside this repo**.

> **Install `perception_models` with `--no-deps`. Mandatory.** Its
> `requirements.txt` pins `numpy==2.1.2`, `pillow==11.0.0`, `timm==1.0.15`,
> `scikit-learn==1.6.1`, `opencv-python==4.11.0.86` and pulls `torchdata`,
> `torchcodec`, `lm-eval` and `wandb` — it will downgrade five of the verified
> packages above and can replace `torch 2.6.0+cu118` with a CPU build. Its
> real imports are only numpy, torch, torchvision, einops, timm,
> huggingface_hub, ftfy and regex, all already present.

Absent by design: **faiss** (numpy does exact top-k over 5k x 768 in well
under a millisecond — do not install it), sentence-transformers.

### Four traps that will silently ruin a run

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
   **But the correction is conditional on the CRS being Mercator.** The UTM
   scenes (EPSG:32636, k0=0.9996) are within 0.04% of unity — true GSD 10.00
   cm, a 448 px tile is 44.8 m. Applying cos(lat) to a UTM scene is exactly as
   wrong as omitting it from a Mercator one. Derive it from the CRS.
4. **PE's `load_ckpt` prints to stdout.** Any stage that shells out to a PE
   process and returns data on stdout gets a contaminated channel — the bytes
   arrive with checkpoint chatter prepended. Return data via a file instead.
   Found by S0's F-3 test (`notes.md#s0-bakeoff`).

## Hardware

2x Quadro RTX 6000, 24 GiB each. GPU 0 idle (24.0 GiB free); GPU 1 drives
the display. 368 GB free disk on one NVMe partition.

## Models

**The embedder is settled. Do not revisit it without re-embedding the index.**

| Purpose | Model | Status |
|---|---|---|
| **Tile + text embedding** | **`chendelong/RemoteCLIP` ViT-L-14 (768-d)**, revision `bf1d8a3ccf2ddbf7c875705e46373bfe542bce38`, loaded via `open_clip` | **CHOSEN at S0 by measurement.** Pretrained on aerial imagery. **Apache 2.0** — verified at source |
| Fallback embedder | `facebook/PE-Core-G14-448` (1280-d) / `PE-Core-L14-336` (1024-d) | ungated, Apache 2.0. Retained in `src/embedders.py`; reverting is a config change plus a re-embed |
| Tile question answering | swappable behind an interface | **PLM is 403 / `gated: manual`** — ungated VLM is the default until approval |

**Why RemoteCLIP** (`notes.md#s0-bakeoff`): it is the only candidate whose
known-absent control is beaten by all 8 real queries (margin +0.0135 vs
PE-Core-L14's **-0.0157** — L14's `aircraft carrier` outscored its own `car`).
It got **4/5** genuine vehicle crops at 112 px against 3/5 and 2/5, runs at
**265 tiles/sec** (6.8 min for the full 108,542-tile pyramid vs 2h36m for
G14), and its 768-d vectors are the smallest.

**F-2's dimensionality is therefore 768**, not 1280.

**Licence: Apache 2.0** — verified 2026-09-07 at
`github.com/ChenDelong1999/RemoteCLIP`. The HF repo `chendelong/RemoteCLIP`
carries **no model card and no LICENSE file**, so the terms come from the
upstream source repo; record that provenance in the manifest rather than
implying the weights shipped with a licence.

**So the measured winner costs nothing on licence** — the same Apache 2.0 as
PE Core, and unlike PerceptionLM (FAIR non-commercial) it does not disqualify
a future product. *Residual caveat, minor:* the upstream training corpora
(RET-3 / SEG-4 / DET-10, aggregated from third-party remote-sensing datasets)
do not state their own terms, so a commercial launch should diligence the
dataset provenance — not the weights licence, which is clear.

PE Core loads from its **own repo**, not transformers and not open_clip:
`git clone https://github.com/facebookresearch/perception_models && pip install -e . --no-deps`
(**`--no-deps` is mandatory** — see the environment section)
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
