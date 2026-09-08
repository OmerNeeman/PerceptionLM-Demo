"""Acceptance tests for S6: index all 8 scenes, and let the app reach them.

Spec F-1 (at scale), F-1a, F-6, F-7, N-1 (at scale), N-2. Brief:
briefs/S6.md.

Two kinds of test live here, deliberately not mixed into test_embed_index.py
/ test_retrieve.py / test_app.py:

* Data-reconciliation tests (`test_full_index_tile_counts`,
  `test_single_embedder_across_all_aois`) exercise the real, already-built
  production index -- all 4 AOIs (leb, AYOSH, gaza, X605_Y3388), 108,542
  planned tiles. There is no meaningful "RED" for these other than the
  literal pre-build state (only X605_Y3388 indexed, 11,109 tiles across 1
  AOI) -- see briefs/S6_result.md for that evidence, captured before this
  stage's build ran. Building an index is not something you can TDD in the
  usual sense; S3 (the first indexing stage) had the same shape.
* Logic tests (multi-AOI corpus loading, the app's AOI selector and F-7 date
  filter, N-1's latency) are ordinary code -- these were run RED (against
  the pre-S6 app.py, which had no `aoi` parameter at all) before the
  implementation landed; see briefs/S6_result.md for the pasted failure.

Every test here only *reads* the on-disk production index
(`retrieve.load_corpus` / `load_corpus_multi`, `app.Engine`) -- nothing in
this file writes under the real index root. Fixture-scale tests that do
build an index (the mixed-embedder guard, `list_indexed_aois`) use
`tmp_path_factory`, mirroring test_retrieve.py's / test_app.py's own
fixture pattern.

Run:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \\
        AERIAL_DATA_ROOT=<data root> \\
        <python> -m pytest retrieval/tests/test_s6.py -v -s
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import app
import config
import embed_index
import embedders
import retrieve
import tiling

#: Sorted, matching `embed_index.list_indexed_aois`'s own return order.
ALL_PRODUCTION_AOIS = ("AYOSH", "X605_Y3388", "gaza", "leb")

#: F-1's padded-planned total, over all 8 source scenes at [448, 224, 112]
#: (docs/DATA.md, spec.md F-1) -- embedded + 100%-nodata-skipped must
#: reconcile to exactly this.
EXPECTED_PLANNED_TOTAL = 108_542
EXPECTED_PLANNED_BY_SCALE = {448: 5198, 224: 20704, 112: 82640}

#: The *embedded* total (vectors actually present -- planned minus the
#: 100%-nodata skips) -- this is what N-1's "full index" search actually
#: ranks over, since a skipped tile has no vector to compare against a
#: query. Measured at build time (briefs/S6_result.md's per-scene table):
#: 108,542 planned - 4,168 skipped (100% nodata: 2,690 in AYOSH's
#: X693_Y3501, 1,478 in X605_Y3388) = 104,374.
EXPECTED_EMBEDDED_TOTAL = 104_374


@pytest.fixture(scope="module")
def real_embedder():
    return embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)


def _make_fixture_raster(path: Path, size: int, seed: int = 0) -> Path:
    """Same synthetic-raster recipe as test_retrieve.py's / test_app.py's own
    fixture: real CRS/transform (EPSG:32636 UTM, 0.1 m/px), non-degenerate
    random content -- used only by this file's mixed-embedder-guard and
    `list_indexed_aois` tests, never by the production-index tests below."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(1, 255, (3, size, size), dtype=np.uint8)
    transform = from_origin(700000.0, 3500000.0, 0.1, 0.1)
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3,
        dtype="uint8", crs="EPSG:32636", transform=transform,
    ) as ds:
        ds.write(arr)
    return path


@pytest.fixture(scope="module")
def production_engine():
    """The real app.Engine over the real, already-built 4-AOI production
    index -- read-only, never rebuilt here (this is what the local app
    actually serves)."""
    return app.Engine()


@pytest.fixture(scope="module")
def full_corpus():
    """All 4 production AOIs concatenated into one Corpus -- exactly what
    the app's "all AOIs" option searches, and what N-1's latency budget is
    measured against at real scale."""
    return retrieve.load_corpus_multi(sorted(embed_index.list_indexed_aois()))


# --------------------------------------------------------------------------
# F-1 at scale -- the full 8-scene, 3-scale plan reconciles to 108,542.
# --------------------------------------------------------------------------


def test_full_index_tile_counts():
    aois = embed_index.list_indexed_aois()
    assert set(aois) == set(ALL_PRODUCTION_AOIS), f"expected the 4 production AOIs, found {aois}"

    total = 0
    by_scale = {s: 0 for s in tiling.SCALES}
    for aoi in aois:
        loaded = embed_index.load_index(aoi)  # F-5's own reload path -- raises on any inconsistency
        manifest = loaded["manifest"]
        embedded = manifest["tiles"]
        skipped_ids = manifest["skipped_tile_ids"]
        assert loaded["vectors"].shape[0] == len(embedded), (
            f"{aoi}: vectors.npy row count disagrees with manifest tile count"
        )
        total += len(embedded) + len(skipped_ids)
        for t in embedded:
            by_scale[t["scale"]] += 1
        for tid in skipped_ids:
            by_scale[tiling.parse_tile_id(tid)["scale"]] += 1

    print(f"\nF-1 full-index reconciliation: total={total}, by_scale={by_scale}")
    assert total == EXPECTED_PLANNED_TOTAL, f"reconciled total {total} != {EXPECTED_PLANNED_TOTAL}"
    assert by_scale == EXPECTED_PLANNED_BY_SCALE, f"per-scale totals {by_scale} != {EXPECTED_PLANNED_BY_SCALE}"


# --------------------------------------------------------------------------
# F-1a -- one embedder serves every scale, and now every AOI too.
# --------------------------------------------------------------------------


def test_single_embedder_across_all_aois():
    aois = embed_index.list_indexed_aois()
    model_ids = {embed_index.load_index(a)["manifest"]["model_id"] for a in aois}
    revisions = {embed_index.load_index(a)["manifest"]["revision"] for a in aois}
    batch_sizes = {embed_index.load_index(a)["manifest"]["batch_size"] for a in aois}
    print(f"\nF-1a: model_ids={model_ids} revisions={revisions} batch_sizes={batch_sizes}")
    assert len(model_ids) == 1, f"more than one model_id across indexed AOIs: {model_ids}"
    assert len(revisions) == 1, f"more than one revision across indexed AOIs: {revisions}"
    assert model_ids == {embed_index.DEFAULT_MODEL_ID}
    # `load_corpus_multi` re-derives and enforces this same guarantee at load
    # time (not just at build time) -- reaching this line is proof it did not
    # raise for the real production AOIs.
    combined = retrieve.load_corpus_multi(sorted(aois))
    assert combined.manifest["model_id"] == embed_index.DEFAULT_MODEL_ID


def test_load_corpus_multi_raises_on_mixed_embedders(real_embedder, tmp_path_factory):
    """F-1a extended to the query side: combining two AOIs indexed with
    different embedders/revisions must raise loudly, never silently compare
    incompatible vector spaces (the same `MixedEmbedderError` `build_index`
    itself raises -- reused, not reimplemented)."""
    data_root = tmp_path_factory.mktemp("mixed_data")
    index_root = tmp_path_factory.mktemp("mixed_index")
    rel_a, rel_b = "mixed_a.tif", "mixed_b.tif"
    _make_fixture_raster(data_root / rel_a, size=224, seed=11)
    _make_fixture_raster(data_root / rel_b, size=224, seed=12)
    embed_index.build_index(
        rel_a, scales=(112,), batch_size=8,
        data_root=data_root, index_root=index_root, embedder=real_embedder,
    )

    class _FakeEmbedder:
        model_id = "fake-model"
        revision = "fake-rev"
        dim = real_embedder.dim

        def embed_images(self, images):
            v = np.random.default_rng(0).normal(size=(len(images), self.dim)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            return v

    embed_index.build_index(
        rel_b, scales=(112,), batch_size=8,
        data_root=data_root, index_root=index_root, embedder=_FakeEmbedder(),
    )

    aoi_a, aoi_b = tiling.scene_aoi(rel_a), tiling.scene_aoi(rel_b)
    with pytest.raises(embed_index.MixedEmbedderError):
        retrieve.load_corpus_multi([aoi_a, aoi_b], index_root=index_root, data_root=data_root)


def test_list_indexed_aois(tmp_path_factory, real_embedder):
    index_root = tmp_path_factory.mktemp("empty_index")
    assert embed_index.list_indexed_aois(index_root) == []

    data_root = tmp_path_factory.mktemp("solo_data")
    rel = "solo.tif"
    _make_fixture_raster(data_root / rel, size=224, seed=21)
    embed_index.build_index(
        rel, scales=(112,), batch_size=8,
        data_root=data_root, index_root=index_root, embedder=real_embedder,
    )
    assert embed_index.list_indexed_aois(index_root) == [tiling.scene_aoi(rel)]
    # Reflects the real production index too -- all 4 AOIs, sorted.
    assert embed_index.list_indexed_aois() == sorted(ALL_PRODUCTION_AOIS)


# --------------------------------------------------------------------------
# F-7 -- the date filter, live for `leb` at real scale, exercised through
# the app layer (the same path the owner actually uses).
# --------------------------------------------------------------------------


def test_date_filter_leb(production_engine):
    full = app.answer_query(production_engine, "a building", aoi="leb")
    filtered = app.answer_query(production_engine, "a building", aoi="leb", date="2025-06-06")
    checked_any = False
    for scale in tiling.SCALES:
        full_n = full["rankings"][scale]["corpus_size"]
        filt_n = filtered["rankings"][scale]["corpus_size"]
        assert filt_n == full_n // 2, (
            f"scale {scale}: date filter did not halve leb's pool "
            f"({filt_n} of {full_n}, expected exactly {full_n // 2})"
        )
        for r in filtered["rankings"][scale]["results"]:
            assert r["date"] == "2025-06-06", r
            checked_any = True
    assert checked_any, "expected at least one dated leb result"


def test_date_filter_undated_aoi_returns_nothing(production_engine):
    """gaza carries no acquisition date (docs/DATA.md) -- an active date
    filter must exclude every one of its tiles, and must not raise."""
    out = app.answer_query(production_engine, "a car", aoi="gaza", date="2025-06-06")
    for scale in tiling.SCALES:
        assert out["rankings"][scale]["results"] == []
        assert out["rankings"][scale]["corpus_size"] == 0
    # Reaching this line at all is the "does not raise" half of the assertion.


def test_date_filter_all_aois_selects_only_leb(production_engine):
    """Under the "all AOIs" scope, a date filter must still only ever
    surface the one AOI that carries that date -- every other AOI's tiles
    are `unknown`-dated and therefore excluded, per F-7's amendment."""
    out = app.answer_query(production_engine, "a car", aoi=app.ALL_AOIS, date="2025-06-06")
    any_result = False
    for scale in tiling.SCALES:
        for r in out["rankings"][scale]["results"]:
            any_result = True
            assert r["tile_id"].startswith("leb::"), r["tile_id"]
            assert r["date"] == "2025-06-06"
    assert any_result


# --------------------------------------------------------------------------
# Part 3 -- the AOI selector: single-AOI restriction, and "all" spanning
# every indexed AOI.
# --------------------------------------------------------------------------


def test_aoi_selector(production_engine):
    out = app.answer_query(production_engine, "rubble", aoi="leb")
    any_result = False
    for scale, row in out["rankings"].items():
        for r in row["results"]:
            any_result = True
            assert r["tile_id"].startswith("leb::"), r["tile_id"]
    assert any_result, "expected at least one leb result for 'rubble'"

    out_all = app.answer_query(production_engine, "rubble", aoi=app.ALL_AOIS)
    for scale in tiling.SCALES:
        expected = sum(
            production_engine.get_corpus(a).scale_indices[scale].size
            for a in production_engine.available_aois
        )
        assert out_all["rankings"][scale]["corpus_size"] == expected, (
            f"scale {scale}: 'all' corpus_size {out_all['rankings'][scale]['corpus_size']} "
            f"!= sum of per-AOI sizes {expected}"
        )
        for r in out_all["rankings"][scale]["results"]:
            assert r["tile_id"].split("::", 1)[0] in production_engine.available_aois


def test_aoi_selector_rejects_unknown_aoi(production_engine):
    with pytest.raises(ValueError):
        app.answer_query(production_engine, "a car", aoi="not_a_real_aoi")


def test_available_aois_default_is_one_aoi(production_engine):
    """U-6 / Part 3: the app defaults to one AOI, not everything, so results
    stay interpretable at rest."""
    assert production_engine.aoi != app.ALL_AOIS
    assert production_engine.aoi in production_engine.available_aois


def test_aoi_selector_rendered_in_page(production_engine):
    html = app.render_index_html(production_engine)
    assert 'id="aoi-select"' in html
    for aoi in production_engine.available_aois:
        assert f'value="{aoi}"' in html
    assert f'value="{app.ALL_AOIS}"' in html


# --------------------------------------------------------------------------
# N-1 at scale -- top-10 over the full (all-AOI) index in < 200 ms on CPU.
# --------------------------------------------------------------------------


def test_query_latency_full_index_cpu(full_corpus, real_embedder):
    n_total = int(full_corpus.vectors.shape[0])
    assert n_total == EXPECTED_EMBEDDED_TOTAL, (
        f"full corpus holds {n_total} embedded tiles, expected {EXPECTED_EMBEDDED_TOTAL}"
    )

    qvec = retrieve.embed_query("a car", embedder=real_embedder)
    retrieve.rank_per_scale(full_corpus, qvec, top_k=retrieve.TOP_K)  # warm-up
    n_reps = 20
    latencies_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        retrieve.rank_per_scale(full_corpus, qvec, top_k=retrieve.TOP_K)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    max_ms = max(latencies_ms)
    print(
        f"\nN-1 FULL INDEX ({n_total} tiles) CPU query latency over {n_reps} reps: "
        f"mean={mean_ms:.3f} ms max={max_ms:.3f} ms"
    )
    assert max_ms < 200.0, f"top-{retrieve.TOP_K} search over the full index took up to {max_ms:.3f} ms (>= 200 ms)"
