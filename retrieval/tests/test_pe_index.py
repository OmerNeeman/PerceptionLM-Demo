"""Acceptance tests for S11a: index the corpus with PE-Core-L14-336
alongside the existing RemoteCLIP index. Spec F-1a (at two models), F-2,
N-2, N-4. Brief: briefs/S11a.md.

Layout under test: `<index_root>/emb/<model_id>/<aoi>/{manifest.json,
vectors.npy}` -- model-scoped storage, so two embedders can coexist for the
same AOI without ever sharing a directory (see briefs/S11a_result.md for the
one-paragraph rationale). `emb_dir` / `list_indexed_aois` / `load_index` all
grew an optional `model_id` parameter defaulting to
`embed_index.DEFAULT_MODEL_ID` -- every pre-S11a call site (retrieve.py,
export_basis.py, export_html.py, app.py, and every pre-S11a test) passes at
most two positional/keyword args to these functions and keeps working
unchanged against the RemoteCLIP index, now at
`emb/RemoteCLIP-ViT-L-14/<aoi>/` instead of `emb/<aoi>/` (moved, not
re-embedded, at S11a).

Run:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        AERIAL_DATA_ROOT=<data root> \
        <python> -m pytest retrieval/tests/test_pe_index.py -v -s
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import embed_index
import embedders

PE_MODEL_ID = "PE-Core-L14-336"
RC_MODEL_ID = embed_index.DEFAULT_MODEL_ID  # "RemoteCLIP-ViT-L-14"
PE_DIM = 1024

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
PROBE = Path(__file__).resolve().parent / "_reload_probe.py"


@pytest.fixture(scope="module")
def pe_embedder():
    return embedders.load_embedder(PE_MODEL_ID)


@pytest.fixture(scope="module")
def rc_embedder():
    return embedders.load_embedder(RC_MODEL_ID)


def _make_fixture_raster(path: Path, size: int = 448, seed: int = 0, nodata_block=None) -> Path:
    """Same recipe as test_embed_index.py's own fixture (not imported from
    there to keep this file runnable standalone) -- real CRS/transform
    (EPSG:32636 UTM, 0.1 m/px, the demo AOI's own), non-degenerate random
    content. `nodata_block`, if given, is `(x0, y0, s)`: that block is
    forced to all-zero so the 100%-nodata skip path has something real to
    exercise."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(1, 255, (3, size, size), dtype=np.uint8)
    if nodata_block is not None:
        x0, y0, s = nodata_block
        arr[:, y0 : y0 + s, x0 : x0 + s] = 0
    transform = from_origin(700000.0, 3500000.0, 0.1, 0.1)
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3,
        dtype="uint8", crs="EPSG:32636", transform=transform,
    ) as ds:
        ds.write(arr)
    return path


@pytest.fixture
def fixture_scene(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    rel = "tiny_scene.tif"
    _make_fixture_raster(data_root / rel, size=448, nodata_block=(0, 0, 112))
    return data_root, rel


def _reload(aoi: str, index_root: Path, model_id: str) -> dict:
    """The brief's mandated check: reload in a FRESH process, not trust
    what a build call returned in this process's memory."""
    out = subprocess.run(
        [sys.executable, str(PROBE), aoi, str(index_root), model_id],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


# --------------------------------------------------------------------------
# F-2 -- PE-Core vectors are unit-normalised, 1024-d
# --------------------------------------------------------------------------


def test_pe_index_unit_norm(fixture_scene, pe_embedder, tmp_path):
    data_root, rel = fixture_scene
    index_root = tmp_path / "index"
    report = embed_index.build_index(
        rel, model_id=PE_MODEL_ID, scales=(224, 112), batch_size=8,
        data_root=data_root, index_root=index_root, embedder=pe_embedder,
    )
    # 224-scale: 2x2=4 tiles, none skipped. 112-scale: 4x4=16, tile (0,0)
    # is 100% nodata -- same fixture geometry as test_embed_index.py's own
    # test_all_vectors_unit_norm.
    assert report["planned_count"] == 20
    assert report["skipped_all_nodata_total"] == 1
    assert report["embedded_total"] == 19

    result = _reload("tiny_scene", index_root, PE_MODEL_ID)
    assert result["count"] == 19
    assert result["dim"] == PE_DIM
    assert result["model_id"] == PE_MODEL_ID
    assert result["norm_max_dev"] <= 1e-5, (
        f"reloaded (renormalised) PE vectors: max |norm - 1| = {result['norm_max_dev']:.3e} > 1e-5"
    )


# --------------------------------------------------------------------------
# The layout's core promise -- two models coexist for the same AOI, each
# manifest recording exactly one model id + revision (F-1a).
# --------------------------------------------------------------------------


def test_manifest_single_embedder_per_index(fixture_scene, pe_embedder, rc_embedder, tmp_path):
    data_root, rel = fixture_scene
    index_root = tmp_path / "index"
    embed_index.build_index(
        rel, model_id=RC_MODEL_ID, scales=(112,), batch_size=8,
        data_root=data_root, index_root=index_root, embedder=rc_embedder,
    )
    embed_index.build_index(
        rel, model_id=PE_MODEL_ID, scales=(112,), batch_size=8,
        data_root=data_root, index_root=index_root, embedder=pe_embedder,
    )

    rc = _reload("tiny_scene", index_root, RC_MODEL_ID)
    pe = _reload("tiny_scene", index_root, PE_MODEL_ID)

    # Each index records exactly one model id + one revision -- not a
    # union, not the other model's identity leaking in.
    assert rc["model_id"] == RC_MODEL_ID
    assert rc["revision"] == rc_embedder.revision
    assert pe["model_id"] == PE_MODEL_ID
    assert pe["revision"] == pe_embedder.revision
    assert rc["model_id"] != pe["model_id"]

    # Coexistence: both indexes are independently loadable from the SAME
    # aoi + index_root, over the SAME source tiles -- 16 tiles at scale 112,
    # tile (0,0) is 100% nodata (fixture_scene's nodata_block) and skipped
    # by both, so 15 embedded each.
    assert rc["count"] == pe["count"] == 15
    assert rc["dim"] == 768
    assert pe["dim"] == PE_DIM

    # Directory layout itself: model-scoped, not aoi-scoped.
    rc_dir = embed_index.emb_dir("tiny_scene", index_root, model_id=RC_MODEL_ID)
    pe_dir = embed_index.emb_dir("tiny_scene", index_root, model_id=PE_MODEL_ID)
    assert rc_dir != pe_dir
    assert rc_dir.parent.name == RC_MODEL_ID
    assert pe_dir.parent.name == PE_MODEL_ID


# --------------------------------------------------------------------------
# F-1a -- the mixed-embedder guard, adapted to the model-scoped layout.
#
# Two DIFFERENT models building at the same aoi+index_root no longer collide
# by directory routing alone -- that is the entire point of this layout
# (S11b needs both to coexist; see test_manifest_single_embedder_per_index
# above). What must still raise is on-disk corruption: a manifest at a
# model-scoped directory recording a model_id/revision that does not match
# what a build for that directory is about to write. This generalises the
# pre-S11a "same model id, tampered revision" case (test_embed_index.py's
# Case B, unaffected by this stage and left as-is) to the model id field
# itself.
# --------------------------------------------------------------------------


def test_mixed_embedder_still_raises(fixture_scene, rc_embedder, tmp_path):
    data_root, rel = fixture_scene
    index_root = tmp_path / "index"
    embed_index.build_index(
        rel, model_id=RC_MODEL_ID, scales=(112,), batch_size=8,
        data_root=data_root, index_root=index_root, embedder=rc_embedder,
    )
    aoi_dir = embed_index.emb_dir("tiny_scene", index_root, model_id=RC_MODEL_ID)
    manifest_path = aoi_dir / embed_index.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    real_model_id = manifest["model_id"]
    manifest["model_id"] = PE_MODEL_ID  # simulate corruption: dir says RC, manifest says PE
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(embed_index.MixedEmbedderError) as exc:
        embed_index.build_index(
            rel, model_id=RC_MODEL_ID, scales=(112,), batch_size=8,
            data_root=data_root, index_root=index_root, embedder=rc_embedder,
        )
    msg = str(exc.value)
    assert PE_MODEL_ID in msg, msg
    assert real_model_id in msg, msg


# --------------------------------------------------------------------------
# N-4 -- determinism, for PE-Core same as for RemoteCLIP.
# --------------------------------------------------------------------------


def test_pe_embedding_determinism(fixture_scene, pe_embedder, tmp_path):
    data_root, rel = fixture_scene
    index_root_1 = tmp_path / "index_1"
    index_root_2 = tmp_path / "index_2"
    embed_index.build_index(
        rel, model_id=PE_MODEL_ID, scales=(112,), batch_size=8,
        data_root=data_root, index_root=index_root_1, embedder=pe_embedder,
    )
    embed_index.build_index(
        rel, model_id=PE_MODEL_ID, scales=(112,), batch_size=8,
        data_root=data_root, index_root=index_root_2, embedder=pe_embedder,
    )
    d1 = embed_index.emb_dir("tiny_scene", index_root_1, model_id=PE_MODEL_ID)
    d2 = embed_index.emb_dir("tiny_scene", index_root_2, model_id=PE_MODEL_ID)
    m1 = json.loads((d1 / embed_index.MANIFEST_NAME).read_text())
    m2 = json.loads((d2 / embed_index.MANIFEST_NAME).read_text())
    assert [t["tile_id"] for t in m1["tiles"]] == [t["tile_id"] for t in m2["tiles"]]

    v1 = np.load(d1 / embed_index.VECTORS_NAME)
    v2 = np.load(d2 / embed_index.VECTORS_NAME)
    assert v1.shape == v2.shape
    max_diff = float(np.abs(v1.astype(np.float32) - v2.astype(np.float32)).max())
    assert max_diff <= 1e-5, (
        f"same tile bytes embedded differently across runs: max abs diff {max_diff:.3e} > 1e-5"
    )


# --------------------------------------------------------------------------
# The candidate-pool identity requirement -- run against the REAL, already-
# built production indexes (RemoteCLIP at S3/S6, PE-Core at S11a). No
# meaningful RED for this one beyond "the PE index does not exist yet" --
# same shape as S3/S6's own production-scale tests.
# --------------------------------------------------------------------------


PRODUCTION_AOIS = ("AYOSH", "X605_Y3388", "gaza", "leb")


def test_two_indexes_same_tile_set():
    for aoi in PRODUCTION_AOIS:
        rc = embed_index.load_index(aoi, model_id=RC_MODEL_ID)
        pe = embed_index.load_index(aoi, model_id=PE_MODEL_ID)
        rc_ids = {t["tile_id"] for t in rc["manifest"]["tiles"]}
        pe_ids = {t["tile_id"] for t in pe["manifest"]["tiles"]}
        rc_skipped = set(rc["manifest"]["skipped_tile_ids"])
        pe_skipped = set(pe["manifest"]["skipped_tile_ids"])
        assert rc_ids == pe_ids, (
            f"{aoi}: embedded tile id sets differ between RemoteCLIP and PE-Core "
            f"-- candidate pools are not identical "
            f"(RC only: {len(rc_ids - pe_ids)}, PE only: {len(pe_ids - rc_ids)})"
        )
        assert rc_skipped == pe_skipped, (
            f"{aoi}: 100%-nodata skip sets differ between RemoteCLIP and PE-Core"
        )
    print(f"\ntwo-index tile-set identity confirmed for {PRODUCTION_AOIS}")
