#!/usr/bin/env python3
"""PLM + SAM 2 aerial tracking demo.

Mock mode (no dependencies, runs anywhere):
    python3 run_demo.py --mock
    python3 run_demo.py --mock --query "white pickup truck"

Real mode (GPU machine with torch/transformers/sam2/opencv installed):
    python3 run_demo.py --video flight.mp4 --query "all vehicles"

Both write to --out (default ./output):
    tracks.json   the tracking result
    viewer.html   interactive viewer - open it in any browser
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plm_sam2_tracker.viewer import write_viewer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mock", action="store_true",
                     help="synthetic aerial scene, zero dependencies")
    src.add_argument("--video", help="input aerial video (real mode, needs GPU deps)")
    p.add_argument("--query", default="all vehicles",
                   help="natural-language target description (default: 'all vehicles')")
    p.add_argument("--out", default="output", help="output directory")
    p.add_argument("--duration", type=float, default=20.0,
                   help="mock mode: simulated seconds (default 20)")
    p.add_argument("--fps", type=float, default=10.0,
                   help="mock mode: simulation fps (default 10)")
    p.add_argument("--reacquire", type=float, default=3.0,
                   help="seconds between PLM re-grounding passes (default 3)")
    args = p.parse_args()

    if args.mock:
        from plm_sam2_tracker.mock_scene import run_mock_pipeline
        payload = run_mock_pipeline(args.query, duration_s=args.duration,
                                    fps=args.fps,
                                    reacquire_every_s=args.reacquire)
    else:
        from plm_sam2_tracker.real_pipeline import run_real_pipeline
        payload = run_real_pipeline(args.video, args.query, args.out,
                                    reacquire_every_s=args.reacquire)

    os.makedirs(args.out, exist_ok=True)
    tracks_path = os.path.join(args.out, "tracks.json")
    with open(tracks_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    viewer_path = write_viewer(payload, os.path.join(args.out, "viewer.html"))

    n = len(payload["tracks"])
    print(f"[done] {n} track(s) for query '{args.query}'")
    for tr in payload["tracks"]:
        print(f"  #{tr['id']} {tr['label']}: {len(tr['boxes'])} frames"
              + (f" | {tr['analysis']}" if tr.get("analysis") else ""))
    print(f"[out]  {tracks_path}")
    print(f"[out]  {viewer_path}  <- open this in a browser")


if __name__ == "__main__":
    main()
