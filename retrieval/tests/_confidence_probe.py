"""Fresh-process reproducibility probe for the confidence-band feature
(S4_fix Fix 2) -- spec U-3. Confirms the same query gets the same
confidence band when the corpus + background reference set are reloaded in
a separate process, mirroring `_retrieve_reload_probe.py`'s F-5 pattern.

Usage: <python> _confidence_probe.py <aoi> <index_root> <qvec_npy_path>

Prints one JSON line to stdout: {scale: {"percentile": float|None, "band":
str, "message": str}, ...}.
"""

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import retrieve  # noqa: E402

aoi = sys.argv[1]
index_root = Path(sys.argv[2])
qvec = np.load(sys.argv[3])

corpus = retrieve.load_corpus(aoi, index_root=index_root)
background = retrieve.load_background(aoi, index_root=index_root)
out = retrieve.rank_per_scale(corpus, qvec, top_k=1, background=background["scores_by_scale"])
result = {str(scale): r["confidence"] for scale, r in out["rankings"].items()}
print(json.dumps(result))
