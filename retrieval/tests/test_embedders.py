"""Acceptance tests for S0. Spec F-2, F-3, F-4(lite), N-3.

Run as:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        /home/omer/anaconda3/envs/geo/bin/python -m pytest retrieval/tests -v
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from embedders import CANDIDATES, load_embedder

CANDIDATE_IDS = list(CANDIDATES)

# One model at a time: 9 GiB for G14 alone.
_CACHE: dict = {}


@pytest.fixture(scope="session")
def embedder(request):
    cid = request.param
    if cid not in _CACHE:
        for stale in list(_CACHE):
            del _CACHE[stale]
        torch.cuda.empty_cache()
        _CACHE[cid] = load_embedder(cid)
    return _CACHE[cid]


def pytest_generate_tests(metafunc):  # pragma: no cover - pytest hook
    if "embedder" in metafunc.fixturenames:
        metafunc.parametrize("embedder", CANDIDATE_IDS, indirect=True, ids=CANDIDATE_IDS)


def _fake_tiles(n, size, seed=0):
    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        for _ in range(n)
    ]


# ---------------------------------------------------------------- N-3


def test_dtype_and_cuda(embedder):
    """N-3: CUDA really available, model really fp16, never bf16.

    A failure on the CUDA assertion means PYTHONNOUSERSITE was not set
    (CLAUDE.md trap 1), not that the hardware is broken.
    """
    assert torch.cuda.is_available() is True, (
        "torch.cuda.is_available() is False -- shadowed CPU-only torch. "
        f"torch={torch.__version__}"
    )
    assert embedder.dtype == torch.float16, (
        f"{embedder.model_id} loaded as {embedder.dtype}, expected float16"
    )
    assert embedder.dtype != torch.bfloat16
    assert embedder.device.type == "cuda"


# ---------------------------------------------------------------- F-2


def test_embed_unit_norm(embedder):
    """F-2: every image vector is unit-norm; dim matches the stated dim."""
    tiles = _fake_tiles(6, 112) + _fake_tiles(2, 448, seed=1)
    v = embedder.embed_images(tiles)

    assert v.shape == (8, embedder.dim), (
        f"{embedder.model_id}: got {v.shape}, expected (8, {embedder.dim})"
    )
    assert v.dtype == np.float32
    norms = np.linalg.norm(v, axis=1)
    assert np.all(np.isfinite(v)), f"{embedder.model_id}: non-finite values"
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"{embedder.model_id}: norms off unit by up to "
        f"{np.abs(norms - 1.0).max():.3e}; norms={norms}"
    )
    # F-2 also fixes the recorded identity of the vectors.
    assert embedder.model_id in CANDIDATES
    assert embedder.revision


# ---------------------------------------------------------------- F-3


def test_text_embed_deterministic(embedder):
    """F-3: same query text -> bitwise identical in-process, 1e-6 across."""
    query = "a white car"
    a = embedder.embed_texts([query])
    b = embedder.embed_texts([query])

    assert a.shape == (1, embedder.dim)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5), (
        f"{embedder.model_id}: text norm not unit: {np.linalg.norm(a, axis=1)}"
    )
    assert a.tobytes() == b.tobytes(), (
        f"{embedder.model_id}: same text embedded differently within one "
        f"process; max abs diff {np.abs(a - b).max():.3e}"
    )

    probe = os.path.join(os.path.dirname(__file__), "_text_embed_probe.py")
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="0")
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "vec.f32")
        subprocess.run(
            [sys.executable, probe, embedder.model_id, query, dst],
            capture_output=True, text=True, env=env, check=True,
        )
        other = np.fromfile(dst, dtype=np.float32)
    assert other.shape == (embedder.dim,)
    delta = np.abs(a[0] - other).max()
    assert delta <= 1e-6, (
        f"{embedder.model_id}: cross-process text embedding differs by "
        f"{delta:.3e} (> 1e-6)"
    )


# ---------------------------------------------------------------- F-4 lite


def test_self_similarity(embedder):
    """F-4-lite: a vector's cosine with itself is 1.0 +/- 1e-5."""
    v = embedder.embed_images(_fake_tiles(4, 224, seed=2))
    q = embedder.embed_texts(["tent", "dirt road"])

    for name, m in (("image", v), ("text", q)):
        self_cos = np.einsum("ij,ij->i", m, m)
        assert np.allclose(self_cos, 1.0, atol=1e-5), (
            f"{embedder.model_id} {name}: self-cosine off by up to "
            f"{np.abs(self_cos - 1.0).max():.3e}: {self_cos}"
        )
    # And cross-cosines must stay inside [-1, 1].
    sims = v @ q.T
    assert sims.min() >= -1.0 - 1e-5 and sims.max() <= 1.0 + 1e-5, (
        f"{embedder.model_id}: cosines out of range [{sims.min()}, {sims.max()}]"
    )
