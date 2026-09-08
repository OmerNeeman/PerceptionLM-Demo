"""S0 embedder bake-off on one calibration subregion.

Cuts one subregion from the tent-camp scene, tiles it at 448 / 224 / 112 px
with 0% overlap, embeds every crop with each candidate, ranks a fixed query
set by cosine, and writes the top-5 crops as PNGs for visual judgement.

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 <python> retrieval/src/calibrate.py

(see INSTRUCTIONS.md for the interpreter path and AERIAL_DATA_ROOT on this
dev machine)

Reads the source raster read-only. Writes only under retrieval/index/calib/.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from embedders import CANDIDATES, load_embedder  # noqa: E402

log = logging.getLogger("calibrate")

# --- the calibration subregion -------------------------------------------
# READ-ONLY source (docs/DATA.md): coastal Khan Yunis / al-Mawasi tent camp,
# EPSG:32636, true GSD 10.00 cm/px, 3-band uint8 RGB. Filename only -- the
# folder it lives in comes from config.py (AERIAL_DATA_ROOT), never a
# hardcoded machine path (N-8).
SRC = config.get_data_root() / "X605_Y3388.tif"

# Chosen by eye from a thumbnail (see _scene_thumb.png / _cand_D.png).
# Contains: tents and tarpaulins (white, blue, yellow, orange), date palms,
# open sand and sand tracks, a large flat-roofed hall, and vehicles verified
# at native resolution.
SUB_X, SUB_Y = 4608, 5120

# 1792 = 4*448 = 8*224 = 16*112 -- all three scales tile the *identical*
# extent with zero remainder, so the per-scale comparison is exact.
SUB_SIZE = 1792
SCALES = (448, 224, 112)

GSD_M = 0.10  # true ground sample distance, metres (UTM scene, no correction)

QUERIES = [
    "tent",
    "tents",
    "a white car",
    "car",
    "palm trees",
    "a building with a flat roof",
    "dirt road",
    "sand",
    "aircraft carrier",  # known-absent control
]

OUT_ROOT = Path(__file__).resolve().parents[1] / "index" / "calib"

TOP_K = 5


def slug(text: str) -> str:
    return text.replace(" ", "_")


# --- tiling ---------------------------------------------------------------


def cut_subregion() -> np.ndarray:
    with rasterio.open(SRC, "r") as ds:
        arr = ds.read(
            [1, 2, 3], window=Window(SUB_X, SUB_Y, SUB_SIZE, SUB_SIZE)
        )
    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    zero = float((arr.sum(axis=2) == 0).mean())
    log.info(
        "subregion (%d,%d) %dx%d  zero-fill=%.4f  mean=%.1f",
        SUB_X, SUB_Y, SUB_SIZE, SUB_SIZE, zero, arr.mean(),
    )
    if zero > 0.01:
        raise RuntimeError(f"subregion is {zero:.1%} nodata; pick another")
    return arr


def build_crops(arr: np.ndarray):
    """-> (list[PIL.Image], list[dict]) with pixel offsets kept."""
    crops, meta = [], []
    for scale in SCALES:
        n = SUB_SIZE // scale
        assert n * scale == SUB_SIZE, "scale must divide the subregion"
        for gy in range(n):
            for gx in range(n):
                y0, x0 = gy * scale, gx * scale
                crops.append(
                    Image.fromarray(arr[y0 : y0 + scale, x0 : x0 + scale])
                )
                meta.append(
                    {
                        "scale": scale,
                        "local_x": x0,
                        "local_y": y0,
                        "scene_x": SUB_X + x0,
                        "scene_y": SUB_Y + y0,
                        "ground_m": round(scale * GSD_M, 2),
                    }
                )
    log.info(
        "%d crops: %s",
        len(crops),
        {s: sum(1 for m in meta if m["scale"] == s) for s in SCALES},
    )
    return crops, meta


# --- reporting ------------------------------------------------------------


def contact_sheet(crops, meta, sims, query, path: Path, cell=190):
    """One sheet per candidate x query: a row per scale, top-5 within it."""
    pad, bar = 6, 20
    w = 5 * (cell + pad) + pad + 70
    h = len(SCALES) * (cell + bar + pad) + pad
    sheet = Image.new("RGB", (w, h), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for r, scale in enumerate(SCALES):
        idx = [i for i, m in enumerate(meta) if m["scale"] == scale]
        order = sorted(idx, key=lambda i: -sims[i])[:5]
        y = pad + r * (cell + bar + pad)
        d.text((4, y + cell // 2), f"{scale}\npx", fill=(255, 210, 80))
        for c, i in enumerate(order):
            x = 70 + pad + c * (cell + pad)
            sheet.paste(crops[i].resize((cell, cell), Image.LANCZOS), (x, y))
            d.text(
                (x + 2, y + cell + 4),
                f"{sims[i]:+.4f} @({meta[i]['local_x']},{meta[i]['local_y']})",
                fill=(230, 230, 230),
            )
    d.text((4, 2), query, fill=(120, 220, 255))
    sheet.save(path)


def run_candidate(cid, crops, meta):
    emb = load_embedder(cid)

    # warm-up, excluded from timing
    emb.embed_images(crops[:8])

    t0 = time.perf_counter()
    V = emb.embed_images(crops, batch_size=32)
    dt = time.perf_counter() - t0
    tps = len(crops) / dt
    log.info(
        "%s: %d crops in %.2fs = %.1f tiles/sec (dim=%d, %s)",
        cid, len(crops), dt, tps, V.shape[1], emb.dtype,
    )

    Q = emb.embed_texts(QUERIES)
    sims_all = V @ Q.T  # cosine: both sides unit-norm

    root = OUT_ROOT / cid
    result = {
        "model_id": emb.model_id,
        "revision": emb.revision,
        "dim": int(V.shape[1]),
        "image_size": emb.image_size,
        "dtype": str(emb.dtype),
        "n_crops": len(crops),
        "embed_seconds": round(dt, 3),
        "tiles_per_sec": round(tps, 1),
        "queries": {},
    }

    for qi, query in enumerate(QUERIES):
        sims = sims_all[:, qi]
        qdir = root / slug(query)
        qdir.mkdir(parents=True, exist_ok=True)

        top = np.argsort(-sims)[:TOP_K]
        for rank, i in enumerate(top, 1):
            m = meta[i]
            name = (
                f"{rank:02d}_s{sims[i]:+.4f}_sc{m['scale']}"
                f"_x{m['local_x']}_y{m['local_y']}.png"
            )
            crops[i].resize((448, 448), Image.LANCZOS).save(qdir / name)

        contact_sheet(crops, meta, sims, query, qdir / "_sheet.png")

        per_scale = {}
        for scale in SCALES:
            idx = [i for i, m in enumerate(meta) if m["scale"] == scale]
            order = sorted(idx, key=lambda i: -sims[i])[:TOP_K]
            per_scale[str(scale)] = [
                {
                    "score": round(float(sims[i]), 4),
                    "local_x": meta[i]["local_x"],
                    "local_y": meta[i]["local_y"],
                }
                for i in order
            ]
        result["queries"][query] = {
            "top5_overall": [
                {
                    "score": round(float(sims[i]), 4),
                    "scale": meta[i]["scale"],
                    "local_x": meta[i]["local_x"],
                    "local_y": meta[i]["local_y"],
                }
                for i in top
            ],
            "score_min": round(float(sims.min()), 4),
            "score_max": round(float(sims.max()), 4),
            "score_mean": round(float(sims.mean()), 4),
            "top5_per_scale": per_scale,
        }

    del emb
    import torch

    torch.cuda.empty_cache()
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    arr = cut_subregion()
    Image.fromarray(arr).save(OUT_ROOT / "_subregion.png")
    crops, meta = build_crops(arr)

    wanted = sys.argv[1:] or list(CANDIDATES)
    out = {
        "source": str(SRC),
        "subregion": {
            "x": SUB_X, "y": SUB_Y, "size": SUB_SIZE,
            "gsd_m": GSD_M, "ground_m": SUB_SIZE * GSD_M,
        },
        "scales": list(SCALES),
        "queries": QUERIES,
        "candidates": {},
    }
    for cid in wanted:
        out["candidates"][cid] = run_candidate(cid, crops, meta)

    path = OUT_ROOT / "results.json"
    path.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
