# PLM + SAM 2 Aerial Object Tracking Demo

A small, self-contained demo of the tracking pipeline recommended in the
PerceptionLM review ([arXiv 2504.13180](https://arxiv.org/abs/2504.13180)):
**PerceptionLM grounds a target from a text description → SAM 2 propagates
it as a track through the video → PLM-style analysis summarizes each track.**

Two modes, one output format:

| Mode | Command | Needs |
|------|---------|-------|
| Mock | `python3 run_demo.py --mock` | Nothing — pure stdlib |
| Real | `python3 run_demo.py --video flight.mp4 --query "all vehicles"` | GPU + deps below |

Both write `output/tracks.json` and `output/viewer.html` — open the viewer
in any browser to play the result (play/pause, scrub, trails, per-track
analysis; click a track to highlight it).

## Quick start (any Linux machine, no installs)

```bash
git clone https://github.com/OmerNeeman/PerceptionLM-Demo.git
cd PerceptionLM-Demo
python3 run_demo.py --mock
xdg-open output/viewer.html        # or just open it in a browser
```

Try different queries against the synthetic scene — this exercises the same
acquisition / re-acquisition logic the real pipeline uses:

```bash
python3 run_demo.py --mock --query "white pickup truck"   # single target
python3 run_demo.py --mock --query "dark sedan"
python3 run_demo.py --mock --query "person walking"
python3 run_demo.py --mock --query "all vehicles"         # multi-target;
        # the box truck enters at t+6s and is picked up by re-acquisition
```

The mock scene includes behaviors the analysis stage should catch: the white
pickup stops near the compound between t+8s and t+12.5s, and the box truck
enters the scene mid-sequence.

## Real mode (GPU machine)

1. Install the Python deps: `pip install -r requirements.txt`
2. Install SAM 2 and download a checkpoint:

   ```bash
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2 && pip install -e . && cd checkpoints && ./download_ckpts.sh
   ```

3. Request access to the PLM weights — the `facebook/Perception-LM-*`
   repos are **gated**, so an unauthenticated first run dies with
   `GatedRepoError`. Approve the licence on the model page, then
   `huggingface-cli login` (or set `HF_TOKEN`).
4. Run (the 8B model wants an A100-class GPU — set
   `PLM_CHECKPOINT=facebook/Perception-LM-3B` for a 24 GB card):

   ```bash
   export SAM2_CHECKPOINT=/path/to/sam2/checkpoints/sam2.1_hiera_large.pt
   python3 run_demo.py --video flight.mp4 --query "all vehicles" --out output
   ```

The real pipeline: extracts frames (≤10 fps) → PLM grounds the query on
frame 0 through the tiled *image* path (small aerial targets survive
tiling; they don't survive the 448² video path) → each box seeds a SAM 2
masklet → propagation through the clip, with PLM re-grounding every
`--reacquire` seconds to pick up objects entering the scene (IoU-gated
against live masklets). The viewer embeds a subsample of frames so it
stays a single shareable file.

PLM grounds *referring expressions* — one call returns one region, and the
prompt is Meta's trained wording from `image_grounding.ipynb`. Multiple
targets come from tiling: the same query over overlapping tiles yields a
box per tile that holds an instance, merged back to full-frame coordinates
by [`plm_sam2_tracker/tiled_grounding.py`](plm_sam2_tracker/tiled_grounding.py).
Ask for one target per query ("the white pickup on the dirt road") rather
than a category — a multi-instance prompt is off-distribution for PLM.

> **Real mode has never been executed.** No GPU was available. It has been
> audited line by line against the current transformers PerceptionLM API
> and the sam2 video predictor, and the defects that audit found are fixed
> — box decoding, the bf16/fp32 vision-tower mismatch, the re-acquisition
> loop, the grounding prompt, and VRAM offload. The decode and IoU maths
> are unit-tested; everything that needs a GPU is still unverified. Treat
> the first real run as a bring-up, not a regression test.

## Layout

```
run_demo.py                       CLI entry point
plm_sam2_tracker/
  mock_scene.py                   synthetic world + mock ground/track/analyze
  real_pipeline.py                PLM grounding + SAM 2 propagation
  tiled_grounding.py              overlapping-tile grounding + box merging
  viewer.py, viewer_template.html standalone HTML viewer generator
eval/
  score.py                        grounding / detection / MOTA / IDF1 scorer
  schema.md, README.md            ground-truth format + labeling guide
  example_gt.json                 worked example against the mock scene
tests/                            unit tests (stdlib only, no GPU)
docs/
  index.html                                 docs landing page (Pages root)
  perceptionlm-aerial-tracking-review.html   paper review + pipeline options
  plm-tracking-sandbox.html                  in-browser pipeline sandbox
```

## Docs

Two standalone pages. GitHub serves `.html` files as source, so read them
either on GitHub Pages or by opening the local files after a clone:

| Page | Live | Local |
|------|------|-------|
| Paper review — the PLM/PE papers, four candidate pipelines ranked, and the aerial constraints that pick between them | [read](https://omerneeman.github.io/PerceptionLM-Demo/perceptionlm-aerial-tracking-review.html) | [`docs/perceptionlm-aerial-tracking-review.html`](docs/perceptionlm-aerial-tracking-review.html) |
| Interactive sandbox — run the acquisition / re-acquisition logic against the synthetic scene in the browser | [open](https://omerneeman.github.io/PerceptionLM-Demo/plm-tracking-sandbox.html) | [`docs/plm-tracking-sandbox.html`](docs/plm-tracking-sandbox.html) |

The live links need GitHub Pages switched on: **Settings → Pages → Source:
Deploy from a branch → `main` → `/docs`**.

## Evaluation

Nothing about zero-shot quality on overhead imagery is known until it is
measured on real footage — that is the project's gating question, so the
harness exists before the answer does. [`eval/`](eval/) holds the
ground-truth format, a labeling guide, and a pure-stdlib scorer:

```bash
python3 run_demo.py --mock
python3 eval/score.py --gt eval/example_gt.json --pred output/tracks.json
```

It reports grounding recall @ IoU (split into targets present at clip start
vs. entering later, with acquisition latency), per-frame detection P/R/F1,
and CLEAR MOT — MOTA, MOTP, IDF1, ID switches. Predictions are plain
`tracks.json`, so mock and real runs score through the same path. See
[`eval/README.md`](eval/README.md) for what to label and how much.

## Tests

```bash
python3 -m unittest discover -s tests
```

Stdlib only, no GPU: tile coverage and box merging, the metric maths, and
PLM's box decoding.

## Known limitations

- Mock mode simulates the *pipeline logic* (grounding, propagation,
  re-acquisition, analysis), not model behavior — it will not tell you how
  well PLM grounds real aerial imagery. Build a small labeled eval set for
  that (see the review's future-work section).
- Real mode is unrun (see the note above). The box decoder follows Meta's
  `rescale_2d_bboxes` — zero-padded fixed point, `[040,482,...]` meaning
  `0.040, 0.482, ...` of the frame — but grounding output formats vary
  between checkpoints; inspect the raw generation if boxes look wrong.
- Grounding is one region per call. Multi-target acquisition depends on
  tiling, so a target smaller than a tile is found and one spanning many
  tiles is stitched, but two instances inside a single tile yield one box.
  Genuine multi-instance detection needs an open-vocabulary detector (the
  review's Option 3: PE Spatial + DETA + ByteTrack).
- PLM is released under Meta's research license; check it before any
  non-research use. SAM 2 and PE are Apache 2.0.
