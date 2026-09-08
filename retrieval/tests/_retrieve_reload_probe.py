"""Fresh-process reload probe for S4 (brief's warning: verify what a fresh
process actually gets back, not what a build call returned in memory) --
spec F-5. Takes a pre-computed query vector from disk (not the text
embedder) so this probes exactly F-5's claim (index reload + ranking maths
are deterministic across processes) without also depending on F-3's
separate, looser (1e-6) cross-process text-embedding guarantee.

Usage: <python> _retrieve_reload_probe.py <aoi> <index_root> <qvec_npy_path> [top_k]

Prints one JSON line to stdout: {"reload_s": float, "rankings": {scale:
[{tile_id, score}, ...]}}.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import retrieve  # noqa: E402

aoi = sys.argv[1]
index_root = Path(sys.argv[2])
qvec = np.load(sys.argv[3])
top_k = int(sys.argv[4]) if len(sys.argv) > 4 else retrieve.TOP_K

t0 = time.perf_counter()
corpus = retrieve.load_corpus(aoi, index_root=index_root)
reload_s = time.perf_counter() - t0

out = retrieve.rank_per_scale(corpus, qvec, top_k=top_k)
rankings = {
    str(scale): [{"tile_id": r["tile_id"], "score": r["score"]} for r in ranking["results"]]
    for scale, ranking in out["rankings"].items()
}
print(json.dumps({"reload_s": reload_s, "rankings": rankings}))
