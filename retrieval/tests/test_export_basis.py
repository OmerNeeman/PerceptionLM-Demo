"""Acceptance tests for S4. Spec F-2a.

Run:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        AERIAL_DATA_ROOT=<data root> \
        <python> -m pytest retrieval/tests/test_export_basis.py -v -s
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import config
import embed_index
import embedders
import export_basis

DEMO_AOI = "X605_Y3388"
PROBE = Path(__file__).resolve().parent / "_export_reload_probe.py"


def _write_raw_index(
    aoi_dir: Path, vectors: np.ndarray, model_id="RemoteCLIP-ViT-L-14", revision="deadbeef", scale=112
) -> None:
    """Hand-build a manifest + vectors.npy pair -- the on-disk shape
    `embed_index.load_index` reads -- with no GPU/embedder involved (same
    pattern `test_embed_index.py` already uses for its own non-finite-vector
    tests)."""
    aoi_dir.mkdir(parents=True, exist_ok=True)
    n, dim = vectors.shape
    manifest = {
        "aoi": aoi_dir.name,
        "model_id": model_id,
        "revision": revision,
        "dim": dim,
        "dtype": "float16",
        "scales": [scale],
        "planned_count": n,
        "tiles": [{"tile_id": f"t{i}", "scale": scale, "nodata_fraction": 0.0} for i in range(n)],
        "skipped_tile_ids": [],
    }
    (aoi_dir / embed_index.MANIFEST_NAME).write_text(json.dumps(manifest))
    np.save(aoi_dir / embed_index.VECTORS_NAME, vectors.astype(np.float16))


@pytest.fixture(scope="module")
def real_embedder():
    return embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)


# --------------------------------------------------------------------------
# PCA maths in isolation -- no GPU, no on-disk index needed.
# --------------------------------------------------------------------------


def test_fit_pca_orthonormal_and_deterministic():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(200, 64)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    components_a = export_basis.fit_pca(raw, export_dim=8)
    components_b = export_basis.fit_pca(raw, export_dim=8)
    assert components_a.shape == (8, 64)
    assert np.array_equal(components_a, components_b), "PCA fit is not deterministic"

    gram = components_a @ components_a.T
    assert np.allclose(gram, np.eye(8), atol=1e-4), "components are not orthonormal"


def test_project_query_matches_batch_projection():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(300, 32)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    components = export_basis.fit_pca(raw, export_dim=6)

    projected_all = export_basis.project_query(raw, components)
    one = export_basis.project_query(raw[5], components)
    assert np.allclose(one, projected_all[5])


def test_project_query_is_not_mean_centered():
    """Regression guard for the exact bug this module's docstring documents:
    projecting the zero vector through a basis fit on non-zero-mean data
    must give the zero vector -- centered PCA would instead give `-mean @
    components.T`, nonzero whenever the corpus mean is nonzero (measured
    0.906 on the real production corpus)."""
    rng = np.random.default_rng(2)
    raw = rng.normal(loc=0.3, size=(200, 40)).astype(np.float32)  # a clearly nonzero mean
    assert np.linalg.norm(raw.mean(axis=0)) > 0.1
    components = export_basis.fit_pca(raw, export_dim=5)
    zero = np.zeros(40, dtype=np.float32)
    projected_zero = export_basis.project_query(zero, components)
    assert np.allclose(projected_zero, 0.0, atol=1e-6)


# --------------------------------------------------------------------------
# F-2a -- the basis is stored so a query vector projects identically, and
# what actually landed on disk is verified by reloading it in a FRESH
# PROCESS, not by trusting what build_export returned in memory.
# --------------------------------------------------------------------------


def test_pca_export_roundtrip(tmp_path):
    index_root = tmp_path / "index"
    aoi_dir = embed_index.emb_dir("tiny_aoi", index_root)
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(50, 16)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    _write_raw_index(aoi_dir, raw)

    report = export_basis.build_export("tiny_aoi", index_root=index_root, export_dim=4)
    assert report["n"] == 50
    assert report["export_dim"] == 4

    proc = subprocess.run(
        [sys.executable, str(PROBE), "tiny_aoi", str(index_root)],
        capture_output=True, text=True, check=True,
    )
    result = json.loads(proc.stdout)
    assert result["n"] == 50
    assert result["export_dim"] == 4
    assert result["dim"] == 16
    assert result["components_shape"] == [4, 16]
    assert result["vectors_i8_dtype"] == "int8"
    assert all(-127 <= v <= 127 for v in result["first_row_i8"])
    assert result["model_id"] == "RemoteCLIP-ViT-L-14"
    assert result["revision"] == "deadbeef"


def test_quantization_error_bounded(tmp_path):
    index_root = tmp_path / "index"
    aoi_dir = embed_index.emb_dir("quant_aoi", index_root)
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(400, 24)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    _write_raw_index(aoi_dir, raw)
    export_basis.build_export("quant_aoi", index_root=index_root, export_dim=10)

    exp = export_basis.load_export("quant_aoi", index_root=index_root)
    dequant = export_basis.exported_vectors_f32(exp)
    projected = export_basis.project_query(raw.astype(np.float32), exp["components"])
    max_err = float(np.abs(dequant - projected).max())
    half_step = exp["basis"]["int8_scale"] / 2.0
    # A small allowance above the exact half-step bound for fp32 rounding in
    # the matmul reduction itself (not the quantiser) -- not a razor-thin
    # fudge: half_step here is ~2.8e-3, and this allows ~7% slack on top.
    assert max_err <= half_step * 1.1 + 1e-4, (
        f"dequantised error {max_err:.3e} exceeds half a quantisation step ({half_step:.3e}) "
        "by more than fp32 rounding can explain"
    )


def test_measure_overlap_detects_stale_export(tmp_path):
    """A basis whose recorded tile order no longer matches the current
    index (e.g. the index was rebuilt/appended after the export) must raise
    rather than silently mismeasure overlap against mismatched rows."""
    index_root = tmp_path / "index"
    aoi_dir = embed_index.emb_dir("stale_aoi", index_root)
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(30, 8)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    _write_raw_index(aoi_dir, raw)
    export_basis.build_export("stale_aoi", index_root=index_root, export_dim=4)

    basis_path = export_basis.export_dir("stale_aoi", index_root) / export_basis.BASIS_NAME
    basis = json.loads(basis_path.read_text())
    basis["tile_ids"] = list(reversed(basis["tile_ids"]))
    basis_path.write_text(json.dumps(basis))

    with pytest.raises(RuntimeError):
        export_basis.measure_overlap("stale_aoi", ["x"], index_root=index_root, embedder=object())


# --------------------------------------------------------------------------
# F-2a -- the actual deliverable: build the real export artifact and
# measure the retrieval@10 overlap cost, per scale, against the real
# production index.
# --------------------------------------------------------------------------


def test_pca_overlap_measured_per_scale_production(real_embedder):
    build_report = export_basis.build_export(DEMO_AOI)
    assert build_report["n"] == 9631
    assert build_report["export_dim"] == export_basis.EXPORT_DIM

    # Fresh-process verification of what actually landed on disk.
    proc = subprocess.run(
        [sys.executable, str(PROBE), DEMO_AOI, str(config.get_index_root())],
        capture_output=True, text=True, check=True,
    )
    reloaded = json.loads(proc.stdout)
    assert reloaded["n"] == 9631
    assert reloaded["dim"] == 768
    assert reloaded["export_dim"] == 128
    assert reloaded["model_id"] == "RemoteCLIP-ViT-L-14"

    # "at least the six [calibration queries]" (F-2a): the five real
    # measurements the brief gives, plus one more.
    queries = ["tents", "palm trees", "a car", "dirt road", "aircraft carrier", "buildings"]
    overlap = export_basis.measure_overlap(DEMO_AOI, queries, embedder=real_embedder)

    print("\nF-2a retrieval@10 overlap, full-precision vs PCA-128+int8, per scale:")
    for scale in sorted(overlap):
        r = overlap[scale]
        print(f"  scale={scale:>4d} n={r['n']:>5d} mean_overlap={r['mean_overlap']:.3f}")
        for pq in r["per_query"]:
            print(f"      {pq['query']!r:18s} overlap={pq['overlap']:.2f}")

    assert set(overlap.keys()) == {448, 224, 112}
    for scale, r in overlap.items():
        assert 0.0 <= r["mean_overlap"] <= 1.0
        # Regression guard (not a quality bar): the mean-centering bug this
        # module's docstring documents measured 0.00-0.10 at every scale;
        # today's real number is ~0.68-0.73. This threshold catches a
        # reintroduction of that bug without hard-coding today's exact figure.
        assert r["mean_overlap"] > 0.3, (
            f"scale {scale}: mean_overlap={r['mean_overlap']:.3f} looks like the "
            "mean-centering regression this module's docstring documents"
        )
