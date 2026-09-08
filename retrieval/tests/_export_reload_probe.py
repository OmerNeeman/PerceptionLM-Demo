"""Fresh-process reload probe for S4 (brief's warning: verify what is
actually on disk, in a fresh process -- not what a build call returned in
memory) -- spec F-2a. Prints one JSON line to stdout.

Usage: <python> _export_reload_probe.py <aoi> <index_root>
"""

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import export_basis  # noqa: E402

aoi = sys.argv[1]
index_root = Path(sys.argv[2])

loaded = export_basis.load_export(aoi, index_root=index_root)
basis = loaded["basis"]
components = loaded["components"]
vectors_i8 = loaded["vectors_i8"]

print(
    json.dumps(
        {
            "n": int(vectors_i8.shape[0]),
            "export_dim": basis["export_dim"],
            "dim": basis["dim"],
            "components_shape": list(components.shape),
            "vectors_i8_dtype": str(vectors_i8.dtype),
            "int8_scale": basis["int8_scale"],
            "first_row_i8": vectors_i8[0].tolist() if vectors_i8.shape[0] else [],
            "model_id": basis["model_id"],
            "revision": basis["revision"],
        }
    )
)
