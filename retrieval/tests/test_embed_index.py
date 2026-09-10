"""Acceptance tests for S3. Spec F-1a, F-2, N-2, N-4; brief's demo-AOI count.

Run:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        AERIAL_DATA_ROOT=<data root> \
        <python> -m pytest retrieval/tests/test_embed_index.py -v -s
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import config
import embed_index
import embedders
import tiling

DEMO_REL_PATH = "X605_Y3388.tif"
DEMO_AOI = "X605_Y3388"
DEMO_TOTAL = 11109  # briefs/S2.md + tests/test_tiling.py DEMO_AOI_TOTAL

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
PROBE = Path(__file__).resolve().parent / "_reload_probe.py"


@pytest.fixture(scope="module")
def real_embedder():
    return embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)


def _make_fixture_raster(path: Path, size: int = 448, seed: int = 0, nodata_block=None) -> Path:
    """A small synthetic 3-band uint8 GeoTIFF: real CRS/transform (EPSG:32636
    UTM, 0.1 m/px -- the demo AOI's own CRS and GSD, so `tiling.scene_meta`'s
    ground-resolution guard passes with no special-casing), non-degenerate
    random content. `nodata_block`, if given, is `(x0, y0, s)`: that s x s
    block is forced to all-zero so the 100%-nodata skip path has something
    real to exercise.
    """
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
    """One small synthetic scene, its own data root, with a forced
    100%-nodata corner block sized to be a whole tile at scale 112 but only
    a quarter of a tile at scale 224 (so both the skip path and the
    "partially nodata is still embedded" rule get exercised)."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    rel = "tiny_scene.tif"
    _make_fixture_raster(data_root / rel, size=448, nodata_block=(0, 0, 112))
    return data_root, rel


def _reload(aoi: str, index_root: Path) -> dict:
    """The brief's mandated check: reload in a FRESH process, not trust
    what a build call returned in this process's memory."""
    out = subprocess.run(
        [sys.executable, str(PROBE), aoi, str(index_root)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _child_env(data_root: Path, index_root: Path) -> dict:
    env = dict(os.environ)
    env["AERIAL_DATA_ROOT"] = str(data_root)
    env["AERIAL_INDEX_ROOT"] = str(index_root)
    return env


def _wait_for_progress(manifest_path: Path, at_least: int, timeout: float = 60.0) -> int:
    deadline = time.time() + timeout
    last = 0
    while time.time() < deadline:
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                last = len(manifest["tiles"])
                if last >= at_least:
                    return last
            except (json.JSONDecodeError, KeyError):
                pass  # mid-write race shouldn't happen (os.replace is atomic); be defensive anyway
        time.sleep(0.05)
    raise TimeoutError(f"manifest never reached {at_least} tiles within {timeout}s (last seen {last})")


# --------------------------------------------------------------------------
# F-2 -- unit-normalised vectors, with provenance
# --------------------------------------------------------------------------


def test_all_vectors_unit_norm(fixture_scene, real_embedder, tmp_path):
    data_root, rel = fixture_scene
    index_root = tmp_path / "index"
    report = embed_index.build_index(
        rel, scales=(224, 112), batch_size=8, data_root=data_root,
        index_root=index_root, embedder=real_embedder,
    )
    # 224-scale: 2x2=4 tiles, none skipped (worst nodata_fraction is 0.25).
    # 112-scale: 4x4=16 tiles, tile (0,0) is 100% nodata and must be skipped.
    assert report["planned_count"] == 20
    assert report["skipped_all_nodata_total"] == 1
    assert report["embedded_total"] == 19

    result = _reload("tiny_scene", index_root)
    assert result["count"] == 19
    assert result["dim"] == 768

    # The raw fp16-on-disk vectors do NOT generally satisfy 1e-5 -- this is
    # measured fp16 quantisation noise (see embed_index.py's module
    # docstring: up to 2.26e-4 measured on 300 real crops), not a bug.
    # Documented here, not hidden -- if this assertion ever starts failing
    # (deviation <= 1e-5 on raw storage) the module's design rationale for
    # renormalising on load needs revisiting, not deleting.
    assert result["raw_norm_max_dev"] > 1e-5, (
        f"raw fp16 storage showed only {result['raw_norm_max_dev']:.3e} deviation; "
        "expected measurable fp16 quantisation noise here"
    )
    # What later stages actually receive (load_index's default,
    # renormalize=True) must satisfy F-2 exactly, for every vector.
    assert result["norm_max_dev"] <= 1e-5, (
        f"reloaded (renormalised) vectors: max |norm - 1| = {result['norm_max_dev']:.3e} > 1e-5"
    )


# --------------------------------------------------------------------------
# F-1a -- exactly one embedder for the whole index
# --------------------------------------------------------------------------


def test_manifest_single_embedder(fixture_scene, real_embedder, tmp_path):
    data_root, rel = fixture_scene
    index_root = tmp_path / "index"
    embed_index.build_index(
        rel, scales=(112,), batch_size=8, data_root=data_root,
        index_root=index_root, embedder=real_embedder,
    )
    result = _reload("tiny_scene", index_root)
    assert result["model_id"] == "RemoteCLIP-ViT-L-14"
    assert result["revision"] == "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    assert isinstance(result["model_id"], str)
    assert isinstance(result["revision"], str)


def test_mixed_embedder_build_fails_loudly(fixture_scene, real_embedder, tmp_path):
    """Case A below was rewritten at S11a (briefs/S11a.md): the index layout
    became model-scoped (`emb/<model_id>/<aoi>/`, see embed_index.emb_dir),
    specifically so a second model CAN be indexed for the same aoi+index_root
    without colliding with the first -- that is the feature S11a adds, not a
    bug. So "request a different model_id at an aoi+index_root that already
    has an index" can no longer mean "silently corrupt/append" and must not
    raise either; it now means "build a second, independent, coexisting
    index," asserted here. The guard this case used to cover (a genuinely
    different model ending up appended into ONE directory) is retargeted at
    the layout's remaining collision surface -- see
    test_pe_index.py::test_mixed_embedder_still_raises, which reproduces it
    via a tampered on-disk manifest (the same technique Case B below already
    used for a mismatched revision) and still raises, naming both values.
    Case B itself is untouched: same model id -> same directory, still
    collides exactly as before.
    """
    data_root, rel = fixture_scene

    # Case A (rewritten at S11a): a different model id entirely now builds a
    # SEPARATE, coexisting index under its own model-scoped subdirectory --
    # it must NOT raise, and the original RemoteCLIP index must be untouched.
    index_root_a = tmp_path / "index_a"
    embed_index.build_index(
        rel, scales=(112,), batch_size=8, data_root=data_root,
        index_root=index_root_a, embedder=real_embedder,
    )
    pe_report = embed_index.build_index(
        rel, scales=(112,), batch_size=8, data_root=data_root,
        index_root=index_root_a, model_id="PE-Core-L14-336",
    )
    assert pe_report["model_id"] == "PE-Core-L14-336"
    rc_dir = embed_index.emb_dir("tiny_scene", index_root_a, model_id="RemoteCLIP-ViT-L-14")
    pe_dir = embed_index.emb_dir("tiny_scene", index_root_a, model_id="PE-Core-L14-336")
    assert rc_dir != pe_dir
    assert json.loads((rc_dir / embed_index.MANIFEST_NAME).read_text())["model_id"] == (
        "RemoteCLIP-ViT-L-14"
    )
    assert json.loads((pe_dir / embed_index.MANIFEST_NAME).read_text())["model_id"] == (
        "PE-Core-L14-336"
    )

    # Case B: same model id, a different (fabricated) revision.
    index_root_b = tmp_path / "index_b"
    embed_index.build_index(
        rel, scales=(112,), batch_size=8, data_root=data_root,
        index_root=index_root_b, embedder=real_embedder,
    )
    manifest_path = embed_index.emb_dir("tiny_scene", index_root_b) / embed_index.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    real_revision = manifest["revision"]
    manifest["revision"] = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(embed_index.MixedEmbedderError) as exc_b:
        embed_index.build_index(
            rel, scales=(112,), batch_size=8, data_root=data_root,
            index_root=index_root_b, embedder=real_embedder,
        )
    msg_b = str(exc_b.value)
    assert "deadbeef" in msg_b
    assert real_revision in msg_b

    # Neither rejected attempt corrupted the pre-existing index on disk.
    result_a = _reload("tiny_scene", index_root_a)
    assert result_a["model_id"] == "RemoteCLIP-ViT-L-14"


# --------------------------------------------------------------------------
# N-4 -- determinism
# --------------------------------------------------------------------------


def test_embedding_determinism(fixture_scene, real_embedder, tmp_path):
    data_root, rel = fixture_scene
    index_root_1 = tmp_path / "index_1"
    index_root_2 = tmp_path / "index_2"
    embed_index.build_index(
        rel, scales=(112,), batch_size=8, data_root=data_root,
        index_root=index_root_1, embedder=real_embedder,
    )
    embed_index.build_index(
        rel, scales=(112,), batch_size=8, data_root=data_root,
        index_root=index_root_2, embedder=real_embedder,
    )
    m1 = json.loads((embed_index.emb_dir("tiny_scene", index_root_1) / embed_index.MANIFEST_NAME).read_text())
    m2 = json.loads((embed_index.emb_dir("tiny_scene", index_root_2) / embed_index.MANIFEST_NAME).read_text())
    assert [t["tile_id"] for t in m1["tiles"]] == [t["tile_id"] for t in m2["tiles"]]

    v1 = np.load(embed_index.emb_dir("tiny_scene", index_root_1) / embed_index.VECTORS_NAME)
    v2 = np.load(embed_index.emb_dir("tiny_scene", index_root_2) / embed_index.VECTORS_NAME)
    assert v1.shape == v2.shape
    max_diff = float(np.abs(v1.astype(np.float32) - v2.astype(np.float32)).max())
    assert max_diff <= 1e-5, (
        f"same tile bytes embedded differently across runs: max abs diff {max_diff:.3e} > 1e-5"
    )


# --------------------------------------------------------------------------
# N-2 -- unattended, resumable build
# --------------------------------------------------------------------------


def test_resume_does_not_reembed(tmp_path):
    """Kill a build partway (a real SIGKILL) and restart it. Checks BOTH
    halves of the acceptance criterion: the finished index is identical to
    an uninterrupted control build, AND the resumed run embedded strictly
    fewer tiles than a full build (16) would -- a resume that silently
    re-embeds everything passes the first check alone and fails this one.
    """
    data_root = tmp_path / "data"
    data_root.mkdir()
    rel = "resume_scene.tif"
    # No nodata block: isolates this test from nodata-skip bookkeeping so
    # "16 tiles planned -> 16 embedded" holds exactly.
    _make_fixture_raster(data_root / rel, size=448, seed=7)
    aoi = "resume_scene"

    # ---- control: one full, uninterrupted build ----
    control_root = tmp_path / "control_index"
    control_env = _child_env(data_root, control_root)
    subprocess.run(
        [sys.executable, str(SRC_DIR / "embed_index.py"),
         "--source", rel, "--scales", "112", "--batch-size", "4"],
        capture_output=True, text=True, check=True, env=control_env,
    )
    control_manifest = json.loads(
        (embed_index.emb_dir(aoi, control_root) / embed_index.MANIFEST_NAME).read_text()
    )
    total = len(control_manifest["tiles"])
    assert total == 16  # 448 / 112 = 4x4 grid, nothing skipped in this fixture
    control_vectors = np.load(embed_index.emb_dir(aoi, control_root) / embed_index.VECTORS_NAME)

    # ---- interrupted build: throttled so a real kill can land mid-build ----
    live_root = tmp_path / "live_index"
    live_env = _child_env(data_root, live_root)
    manifest_path = embed_index.emb_dir(aoi, live_root) / embed_index.MANIFEST_NAME
    proc = subprocess.Popen(
        [sys.executable, str(SRC_DIR / "embed_index.py"),
         "--source", rel, "--scales", "112", "--batch-size", "4",
         "--batch-sleep-s", "0.5"],
        env=live_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        done_at_kill = _wait_for_progress(manifest_path, at_least=4, timeout=60)
    finally:
        proc.kill()  # SIGKILL -- a real, uncontrolled kill
        proc.wait(timeout=10)
    assert 0 < done_at_kill < total, (
        f"expected a partial manifest at kill time, got {done_at_kill}/{total}"
    )

    # ---- resume: same command, same (now partially-populated) index dir ----
    resume_proc = subprocess.run(
        [sys.executable, str(SRC_DIR / "embed_index.py"),
         "--source", rel, "--scales", "112", "--batch-size", "4"],
        capture_output=True, text=True, check=True, env=live_env,
    )
    resume_report = json.loads(resume_proc.stdout)
    assert resume_report["embedded_total"] == total
    assert resume_report["embedded_this_run"] == total - done_at_kill
    assert resume_report["embedded_this_run"] < total, (
        "resume embedded as many tiles as a full build would have -- "
        "it silently re-embedded everything instead of skipping completed tiles"
    )

    live_manifest = json.loads(manifest_path.read_text())
    assert [t["tile_id"] for t in live_manifest["tiles"]] == [
        t["tile_id"] for t in control_manifest["tiles"]
    ]
    live_vectors = np.load(embed_index.emb_dir(aoi, live_root) / embed_index.VECTORS_NAME)
    assert live_vectors.shape == control_vectors.shape
    max_diff = float(np.abs(live_vectors.astype(np.float32) - control_vectors.astype(np.float32)).max())
    assert max_diff <= 1e-5, (
        f"resumed index differs from an uninterrupted control build: max abs diff {max_diff:.3e}"
    )


# --------------------------------------------------------------------------
# counts -- the demo AOI's tile plan and the real build
# --------------------------------------------------------------------------


def test_demo_aoi_tile_count(real_embedder):
    """The plan holds 11,109 tiles (brief); embedded = 11,109 minus the
    100%-nodata ones. This runs the REAL production build against the real
    demo AOI, at the real (default) index root -- the actual deliverable,
    not a stand-in. A second run in the same session (or a later pytest
    invocation) is expected to be near-instant: N-2's resumability means
    every tile already on disk is skipped, not re-embedded.
    """
    planned = sum(
        tiling.plan_scene(DEMO_REL_PATH, s)["planned_count"] for s in tiling.SCALES
    )
    assert planned == DEMO_TOTAL

    report = embed_index.build_index(
        DEMO_REL_PATH, scales=tiling.SCALES, embedder=real_embedder,
    )
    assert report["planned_count"] == DEMO_TOTAL
    assert report["embedded_total"] == DEMO_TOTAL - report["skipped_all_nodata_total"]

    # Regression guard for a real bug found while building this index: a
    # second call against an already-complete index must be a true no-op --
    # nothing embedded, and (this is the part that actually broke) the
    # skipped-nodata count must NOT grow. The first version of this module
    # tracked skips as a bare counter incremented every time a 100%-nodata
    # tile was seen; since skipped tiles were never added to `done_ids`,
    # every resumed run re-examined and re-counted the same ~1,478 tiles,
    # so a second real run on this AOI inflated skipped_all_nodata_total
    # from 1,478 to 2,956 while embedded_total stayed correct at 9,631 --
    # the bug was invisible unless you ran the build more than once.
    report2 = embed_index.build_index(
        DEMO_REL_PATH, scales=tiling.SCALES, embedder=real_embedder,
    )
    assert report2["embedded_this_run"] == 0
    assert report2["embedded_total"] == report["embedded_total"]
    assert report2["skipped_all_nodata_total"] == report["skipped_all_nodata_total"]

    print(
        f"\ndemo AOI: planned={report['planned_count']} embedded_total={report['embedded_total']} "
        f"skipped_all_nodata={report['skipped_all_nodata_total']} "
        f"embedded_this_run={report['embedded_this_run']} "
        f"wall_s={report['wall_s']:.2f} tiles_per_sec={report['tiles_per_sec']:.2f}"
    )

    # F-2/F-5, once more, against the actual on-disk production artifact,
    # reloaded in a fresh process -- not the in-memory report above.
    result = _reload(DEMO_AOI, config.get_index_root())
    assert result["count"] == report["embedded_total"]
    assert result["dim"] == 768
    assert result["norm_max_dev"] <= 1e-5
    assert result["model_id"] == "RemoteCLIP-ViT-L-14"
    assert result["revision"] == "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"


# --------------------------------------------------------------------------
# S3-fix, Fix 1 -- non-finite must raise, never be silently renormalised.
#
# `np.abs(nan - 1.0) > ATOL` is always False, so the pre-fix sanity check is
# defeated entirely by a NaN component, not merely under-threshold: a NaN
# vector loads "successfully" and poisons every later dot product. These
# tests hand-build a manifest + vectors.npy directly (no GPU/model needed)
# so the fixture can plant exactly the broken value under test.
# --------------------------------------------------------------------------


def _write_raw_index(
    aoi_dir: Path, vectors: np.ndarray, model_id="RemoteCLIP-ViT-L-14", revision="deadbeef"
) -> None:
    """Hand-build a manifest + vectors.npy pair -- the on-disk shape
    `load_index` reads -- without going through `build_index` at all, so
    these tests can plant an exact broken value with no embedder/GPU
    involved."""
    aoi_dir.mkdir(parents=True, exist_ok=True)
    n, dim = vectors.shape
    manifest = {
        "aoi": aoi_dir.name,
        "model_id": model_id,
        "revision": revision,
        "dim": dim,
        "dtype": "float16",
        "scales": [112],
        "planned_count": n,
        "tiles": [{"tile_id": f"t{i}", "scale": 112, "nodata_fraction": 0.0} for i in range(n)],
        "skipped_tile_ids": [],
    }
    (aoi_dir / embed_index.MANIFEST_NAME).write_text(json.dumps(manifest))
    np.save(aoi_dir / embed_index.VECTORS_NAME, vectors.astype(np.float16))


def test_nan_vector_raises(tmp_path):
    """RED against the pre-fix code: a NaN component's norm is NaN,
    `np.abs(nan - 1.0) > ATOL` is False, so `bad.any()` is False and
    `load_index` returns a NaN-poisoned vector with no exception at all."""
    index_root = tmp_path / "index"
    aoi_dir = embed_index.emb_dir("nan_aoi", index_root)
    vecs = np.full((3, 4), 0.5, dtype=np.float32)  # norm 1.0, healthy
    vecs[1, 2] = np.nan
    _write_raw_index(aoi_dir, vecs)
    with pytest.raises(RuntimeError) as exc:
        embed_index.load_index("nan_aoi", index_root=index_root)
    msg = str(exc.value)
    assert "1" in msg, msg  # offending row index
    assert "NaN" in msg, msg


def test_inf_vector_raises(tmp_path):
    """+/-Inf must raise too, not just NaN -- same defeated-comparison
    failure mode (abs(inf - 1.0) > ATOL is True here, so this one happens to
    slip through the pre-fix threshold check by accident, but must still be
    reported as non-finite, not as a merely-gross norm)."""
    index_root = tmp_path / "index"
    aoi_dir = embed_index.emb_dir("inf_aoi", index_root)
    vecs = np.full((3, 4), 0.5, dtype=np.float32)
    vecs[2, 0] = np.inf
    _write_raw_index(aoi_dir, vecs)
    with pytest.raises(RuntimeError) as exc:
        embed_index.load_index("inf_aoi", index_root=index_root)
    msg = str(exc.value)
    assert "2" in msg, msg
    assert "Inf" in msg, msg


# --------------------------------------------------------------------------
# S3-fix, Fix 3 -- the sanity bound itself, proven by a test that would fail
# if RAW_NORM_SANITY_ATOL were removed or widened (the reviewer set it to
# 1e9 and all 6 pre-fix tests still passed).
# --------------------------------------------------------------------------


def test_gross_norm_error_raises(tmp_path):
    """A finite but grossly-wrong norm (0.5, nowhere near fp16 quantisation
    noise) must still raise -- the bound this module has asserted in prose
    in three places but a test exercised in none."""
    index_root = tmp_path / "index"
    aoi_dir = embed_index.emb_dir("gross_aoi", index_root)
    vecs = np.zeros((2, 4), dtype=np.float32)
    vecs[0] = [1.0, 0.0, 0.0, 0.0]  # norm 1.0, healthy
    vecs[1] = [0.5, 0.0, 0.0, 0.0]  # norm 0.5, grossly wrong -- not quantisation
    _write_raw_index(aoi_dir, vecs)
    with pytest.raises(RuntimeError) as exc:
        embed_index.load_index("gross_aoi", index_root=index_root)
    assert "1" in str(exc.value)


def test_fp16_noise_does_not_raise(tmp_path):
    """The guard against over-tightening (brief: do not tighten
    RAW_NORM_SANITY_ATOL): real fp16 quantisation noise on an otherwise
    healthy unit vector must NOT raise, and must renormalise to satisfy F-2's
    1e-5 bound on reload."""
    index_root = tmp_path / "index"
    aoi_dir = embed_index.emb_dir("fp16_aoi", index_root)
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(200, 768)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    _write_raw_index(aoi_dir, raw)  # _write_raw_index casts to float16, as production storage does
    result = embed_index.load_index("fp16_aoi", index_root=index_root)
    assert result["vectors"].shape == (200, 768)
    assert np.abs(np.linalg.norm(result["vectors"], axis=1) - 1.0).max() <= 1e-5


# --------------------------------------------------------------------------
# S3-fix, Fix 2 -- build_index must validate what it resumed from, not just
# what load_index validates on a later, separate reload.
# --------------------------------------------------------------------------


def test_build_index_detects_manifest_vector_mismatch(real_embedder, tmp_path):
    """RED against the pre-fix code: the reviewer's exact reproduction --
    truncate vectors.npy to fewer rows than the manifest lists, then call
    build_index again. Pre-fix, `todo` computes as empty (every tile_id in
    the manifest already counts as 'done') and the call returns
    `{"embedded_total": 16}` -- a false success report, no exception."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    rel = "mismatch_scene.tif"
    _make_fixture_raster(data_root / rel, size=448, seed=11)  # no nodata block -> exactly 16/16
    index_root = tmp_path / "index"
    embed_index.build_index(
        rel, scales=(112,), batch_size=8, data_root=data_root,
        index_root=index_root, embedder=real_embedder,
    )
    aoi_dir = embed_index.emb_dir("mismatch_scene", index_root)
    manifest_path = aoi_dir / embed_index.MANIFEST_NAME
    vectors_path = aoi_dir / embed_index.VECTORS_NAME
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["tiles"]) == 16
    vectors = np.load(vectors_path)
    assert vectors.shape[0] == 16
    # The corruption the docstring's write-ordering argument claims cannot
    # happen -- nothing previously asserted it at runtime, so a future
    # change to checkpoint order (or any non-crash corruption) would lie.
    np.save(vectors_path, vectors[:10])

    with pytest.raises(RuntimeError) as exc:
        embed_index.build_index(
            rel, scales=(112,), batch_size=8, data_root=data_root,
            index_root=index_root, embedder=real_embedder,
        )
    msg = str(exc.value)
    assert "16" in msg, msg
    assert "10" in msg, msg


# --------------------------------------------------------------------------
# S3-fix -- record batch_size in the manifest (N-4 amendment).
# --------------------------------------------------------------------------


def test_manifest_records_batch_size(fixture_scene, real_embedder, tmp_path):
    data_root, rel = fixture_scene
    index_root = tmp_path / "index"
    embed_index.build_index(
        rel, scales=(112,), batch_size=8, data_root=data_root,
        index_root=index_root, embedder=real_embedder,
    )
    manifest = json.loads(
        (embed_index.emb_dir("tiny_scene", index_root) / embed_index.MANIFEST_NAME).read_text()
    )
    assert manifest["batch_size"] == 8


def test_batch_size_change_on_resume_logs_warning(fixture_scene, real_embedder, tmp_path, caplog):
    """Not in the brief's minimum acceptance list, but directly implements
    its "a rebuild that would use a different [batch size] is at minimum
    logged loudly" requirement -- added as coverage for that half of Fix 4,
    which test_manifest_records_batch_size alone does not exercise."""
    data_root, rel = fixture_scene
    index_root = tmp_path / "index"
    embed_index.build_index(
        rel, scales=(112,), batch_size=8, data_root=data_root,
        index_root=index_root, embedder=real_embedder,
    )
    with caplog.at_level("WARNING"):
        embed_index.build_index(
            rel, scales=(112,), batch_size=4, data_root=data_root,
            index_root=index_root, embedder=real_embedder,
        )
    assert any(
        "batch_size" in rec.message and "8" in rec.message and "4" in rec.message
        for rec in caplog.records
    ), [rec.message for rec in caplog.records]
