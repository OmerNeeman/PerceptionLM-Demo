"""Fresh-process reload probe for S3 (brief's warning: verify what is
actually on disk, in a fresh process -- not what a build call returned in
memory). Prints one JSON line to stdout.

Usage: <python> _reload_probe.py <aoi> <index_root>
"""

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

import embed_index  # noqa: E402

aoi = sys.argv[1]
index_root = Path(sys.argv[2])

loaded = embed_index.load_index(aoi, index_root=index_root)
manifest = loaded["manifest"]
vectors = loaded["vectors"]
raw_norms = loaded["raw_norms"]

norms = np.linalg.norm(vectors, axis=1)
print(
    json.dumps(
        {
            "count": int(vectors.shape[0]),
            "dim": int(vectors.shape[1]) if vectors.shape[0] else manifest.get("dim"),
            "model_id": manifest["model_id"],
            "revision": manifest["revision"],
            "norm_max_dev": float(np.abs(norms - 1.0).max()) if len(norms) else 0.0,
            "raw_norm_max_dev": float(np.abs(raw_norms - 1.0).max()) if len(raw_norms) else 0.0,
            "raw_norm_mean_dev": float(np.abs(raw_norms - 1.0).mean()) if len(raw_norms) else 0.0,
            "planned_count": manifest["planned_count"],
            "skipped_all_nodata": len(manifest["skipped_tile_ids"]),
            "vectors_dtype_on_disk": "float16",
        }
    )
)
