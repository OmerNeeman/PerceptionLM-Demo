"""Child process for the cross-process determinism check (spec F-3).

Writes the raw float32 bytes of the embedding of argv[2] under candidate
argv[1] to the file at argv[3]. A file, not stdout: PE's `load_ckpt` prints
to stdout, so stdout is not a clean channel here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedders import load_embedder  # noqa: E402

if __name__ == "__main__":
    candidate, text, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    v = load_embedder(candidate).embed_texts([text])[0]
    Path(out_path).write_bytes(v.tobytes())
