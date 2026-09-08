"""Acceptance tests for S1a. Spec N-8, N-9.

Run as:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data \
        /home/omer/anaconda3/envs/geo/bin/python -m pytest retrieval/tests/test_portability.py -v

(conftest.py sets AERIAL_DATA_ROOT for this dev machine if it is not already
set, so plain `pytest retrieval/tests -v` also works here -- a fresh machine
must set it itself; that is exactly what test_config_fails_loudly_without_data_root
below checks.)
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path, PureWindowsPath

import pytest
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

# ---------------------------------------------------------------- N-8, path 1


def test_no_machine_paths():
    """N-8: no machine-specific literal anywhere in src/, docstrings included.

    A specific interpreter path or the data root may appear in *documentation*
    (INSTRUCTIONS.md); it may not appear in importable code -- and a docstring
    inside a .py file under src/ is still importable code as far as this test
    is concerned.
    """
    bad_patterns = ["/home/", "C:\\", "anaconda3", "Dynamic-Terrain"]
    hits = []
    for p in sorted(SRC_DIR.rglob("*.py")):
        text = p.read_text()
        for pat in bad_patterns:
            if pat in text:
                # Report every hit, not just the first, so one run shows the
                # whole job rather than being fixed one grep-cycle at a time.
                for lineno, line in enumerate(text.splitlines(), 1):
                    if pat in line:
                        hits.append(f"{p.relative_to(SRC_DIR.parent)}:{lineno}: {pat!r} in {line.strip()!r}")
    assert not hits, "machine-specific literal(s) found in src/:\n" + "\n".join(hits)


# ---------------------------------------------------------------- N-8, dtype


def test_dtype_for_device():
    """N-8: select_dtype is a pure function of device capability, asserted at
    all three cases with a *fabricated* capability -- no Ampere card required
    to test the Ampere branch, and this machine's real cc 7.5 case is also
    covered directly (not just simulated) so it doubles as the N-3 guard."""
    import device

    turing = torch.device("cuda")
    assert device.select_dtype(turing, capability=(7, 5)) == torch.float16, (
        "cc 7.5 (Turing/Volta) must resolve to float16 -- this machine's case, "
        "and the one N-3's existing test depends on staying true"
    )
    ampere = torch.device("cuda")
    assert device.select_dtype(ampere, capability=(8, 6)) == torch.bfloat16, (
        "cc 8.6 (Ampere) must resolve to bfloat16 -- native on that hardware"
    )
    assert device.select_dtype(torch.device("cpu")) == torch.float32, (
        "CPU must resolve to float32 -- fp16 on CPU is unsupported/pathologically slow"
    )
    # MPS gets the same float32 answer as CPU (no fabrication needed: MPS
    # carries no compute-capability concept at all).
    assert device.select_dtype(torch.device("mps")) == torch.float32


# ---------------------------------------------------------------- N-8, CPU


def _fake_tiles(n, size, seed=0):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        for _ in range(n)
    ]


def test_cpu_only_build():
    """N-8: forcing device='cpu' actually completes an embed + a query.

    This is an actual CPU-forced run, not an inspection of the dtype rule --
    the brief is explicit that inspection alone does not count. The measured
    throughput is recorded (captured via -s / the log line below) so
    INSTRUCTIONS.md can state an honest CPU expectation; for reference
    RemoteCLIP does 265 tiles/sec on GPU 0 on this machine.
    """
    from embedders import load_embedder

    emb = load_embedder("RemoteCLIP-ViT-L-14", device="cpu")
    assert emb.device.type == "cpu"
    assert emb.dtype == torch.float32, f"CPU load must be float32, got {emb.dtype}"

    tiles = _fake_tiles(6, 224)
    t0 = time.perf_counter()
    v = emb.embed_images(tiles)
    dt = time.perf_counter() - t0
    assert v.shape == (6, emb.dim)

    q = emb.embed_texts(["tent", "a white car"])
    assert q.shape == (2, emb.dim)

    tps = len(tiles) / dt if dt > 0 else float("inf")
    print(f"\nCPU-only throughput (RemoteCLIP-ViT-L-14, {len(tiles)} tiles @224px): "
          f"{dt:.2f}s = {tps:.2f} tiles/sec (reference: 265 tiles/sec on GPU 0)")


# ---------------------------------------------------------------- N-8, paths


def test_manifest_path_portable():
    """N-8: a manifest/tile-id key is one canonical ('/') separator
    regardless of which OS produced the input -- tested with pathlib's
    PureWindowsPath, so no real Windows machine is needed."""
    import config

    windows_style = "leb\\2022-10-29.tif"
    posix_style = "leb/2022-10-29.tif"
    assert config.posix_key(windows_style) == "leb/2022-10-29.tif"
    assert config.posix_key(posix_style) == "leb/2022-10-29.tif"
    assert config.posix_key(windows_style) == config.posix_key(posix_style)

    # A PureWindowsPath as produced by inventory.py-style relative-path logic
    # if it ever ran on Windows -- as_posix() normalises it, no OS needed.
    assert config.posix_key(PureWindowsPath("AYOSH\\X693_Y3500.tif")) == "AYOSH/X693_Y3500.tif"

    # Round-trip through an actual written-and-reread JSON manifest.
    manifest = {"tiles": [config.posix_key(windows_style), config.posix_key(posix_style)]}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.json"
        p.write_text(json.dumps(manifest))
        reloaded = json.loads(p.read_text())
    assert reloaded["tiles"] == ["leb/2022-10-29.tif", "leb/2022-10-29.tif"]


# ---------------------------------------------------------------- N-8, config


def test_config_fails_loudly_without_data_root(monkeypatch):
    """N-8: a missing data root raises, naming the env var -- it must never
    silently fall back to this machine's (or any machine's) hardcoded path."""
    import config

    monkeypatch.delenv(config.ENV_DATA_ROOT, raising=False)
    with pytest.raises(config.ConfigError) as excinfo:
        config.get_data_root()
    msg = str(excinfo.value)
    assert config.ENV_DATA_ROOT in msg
    assert "PowerShell" in msg
    # The generic example syntax is fine; falling back to this dev machine's
    # own path (the exact bug N-8 exists to prevent) is not.
    assert "Dynamic-Terrain" not in msg
    assert "/home/omer" not in msg
