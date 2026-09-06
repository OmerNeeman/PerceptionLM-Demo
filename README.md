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

3. Run (PLM weights download from HuggingFace on first use; the 8B model
   wants an A100-class GPU — set `PLM_CHECKPOINT=facebook/Perception-LM-3B`
   for a 24 GB card):

   ```bash
   export SAM2_CHECKPOINT=/path/to/sam2/checkpoints/sam2.1_hiera_large.pt
   python3 run_demo.py --video flight.mp4 --query "all vehicles" --out output
   ```

The real pipeline: extracts frames (≤10 fps) → PLM grounds the query on
frame 0 using the tiled *image* path (small aerial targets survive tiling;
they don't survive the 448² video path) → each box seeds a SAM 2 masklet →
propagation through the clip, with PLM re-grounding every `--reacquire`
seconds to pick up objects entering the scene (IoU-gated against live
masklets). The viewer embeds a subsample of frames so it stays a single
shareable file.

> Real-mode code is a reference implementation written against the
> transformers PerceptionLM API and the sam2 video predictor API as of
> mid-2026 — verify against the current repos before relying on it.

## Layout

```
run_demo.py                       CLI entry point
plm_sam2_tracker/
  mock_scene.py                   synthetic world + mock ground/track/analyze
  real_pipeline.py                PLM grounding + SAM 2 propagation
  viewer.py, viewer_template.html standalone HTML viewer generator
docs/
  perceptionlm-aerial-tracking-review.html   paper review + pipeline options
  plm-tracking-sandbox.html                  in-browser pipeline sandbox
```

## Docs

Two standalone pages (no server needed — open them in a browser):

- [`docs/perceptionlm-aerial-tracking-review.html`](docs/perceptionlm-aerial-tracking-review.html)
  — review of the PLM/PE papers and the four candidate pipelines, with the
  reasoning behind the PLM-grounding + SAM 2-propagation choice this demo
  implements.
- [`docs/plm-tracking-sandbox.html`](docs/plm-tracking-sandbox.html)
  — interactive sandbox: run the same acquisition / re-acquisition logic
  against the synthetic scene and watch it step through, in the browser.

## Known limitations

- Mock mode simulates the *pipeline logic* (grounding, propagation,
  re-acquisition, analysis), not model behavior — it will not tell you how
  well PLM grounds real aerial imagery. Build a small labeled eval set for
  that (see the review's future-work section).
- Real-mode PLM box parsing handles pixel and 0–1000-normalized outputs but
  VLM grounding output formats vary between checkpoints; inspect the raw
  generation if boxes look wrong.
- PLM is released under Meta's research license; check it before any
  non-research use. SAM 2 and PE are Apache 2.0.
