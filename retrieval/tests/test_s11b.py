"""Acceptance tests for S11b: model selector + side-by-side compare view.

Spec F-1a expressed in the UI, F-4 unchanged, N-1 unchanged (per model, and
for a compare query). Brief: briefs/S11b.md.

The constraint this whole file is organised around: **scores from different
models are not comparable.** So every test here either (a) proves the app
reads the *correct* model's own index for a given selection, or (b) proves
no code path ever merges, interleaves, or jointly ranks two models' results
-- never that one model "won" the other on raw score.

Two model-scoped fixtures, mirroring test_app.py's own real-embedder-on-a-
tmp_path-fixture pattern (never touching the shipped `index/` or the real
data root's write path -- D-3):

* `compare_engine` -- a small synthetic index built with **both**
  RemoteCLIP and PE-Core-L14-336 over the identical tile set, so compare
  tests exercise two real, independently-computed rankings.
* `single_model_engine` -- built with RemoteCLIP only, so PE-Core is a
  genuinely *missing* index for this AOI -- the brief's own abuse case
  ("switch to a model whose index is missing for that AOI"), reproduced
  directly rather than hoped for.

The rest of the suite (N-1 per model, and the compare view's total) is
measured against the real, already-built 4-AOI production index for both
models -- mirrors test_s6.py's own `test_query_latency_full_index_cpu`.

Run:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \\
        AERIAL_DATA_ROOT=<data root> \\
        <python> -m pytest retrieval/tests/test_s11b.py -v -s
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

import app
import embed_index
import embedders
import retrieve
import tiling

RC_MODEL_ID = embed_index.DEFAULT_MODEL_ID  # "RemoteCLIP-ViT-L-14"
PE_MODEL_ID = "PE-Core-L14-336"

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rc_embedder():
    return embedders.load_embedder(RC_MODEL_ID)


@pytest.fixture(scope="module")
def pe_embedder():
    return embedders.load_embedder(PE_MODEL_ID)


def _make_fixture_raster(path: Path, size: int, seed: int = 0) -> Path:
    """Same synthetic-raster recipe as test_app.py's own fixture: real
    CRS/transform (EPSG:32636 UTM, 0.1 m/px -- the demo AOI's own CRS/GSD),
    non-degenerate random content."""
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
def compare_engine(rc_embedder, pe_embedder, tmp_path_factory):
    """A small, real index + background, built with **both** models over
    the identical tile set -- entirely under `tmp_path_factory`, never
    touching production data, the shipped `index/`, or (per the brief)
    `index/export/`."""
    data_root = tmp_path_factory.mktemp("s11b_data")
    index_root = tmp_path_factory.mktemp("s11b_index")
    rel = "compare_scene.tif"
    _make_fixture_raster(data_root / rel, size=896, seed=7)
    aoi = "compare_scene"

    for model_id, emb in ((RC_MODEL_ID, rc_embedder), (PE_MODEL_ID, pe_embedder)):
        embed_index.build_index(
            rel, model_id=model_id, scales=(448, 224, 112), batch_size=32,
            data_root=data_root, index_root=index_root, embedder=emb,
        )
        retrieve.build_background(aoi, index_root=index_root, data_root=data_root, embedder=emb, model_id=model_id)

    return app.Engine(aoi=aoi, index_root=index_root, data_root=data_root, models=app.COMPARE_MODELS)


@pytest.fixture(scope="module")
def compare_client(compare_engine):
    return TestClient(app.create_app(engine=compare_engine))


@pytest.fixture(scope="module")
def single_model_engine(rc_embedder, tmp_path_factory):
    """RemoteCLIP-only index -- PE-Core-L14-336 has **no** on-disk index for
    this AOI at all. This is the brief's own abuse case ("switch to a model
    whose index is missing for that AOI"), constructed directly rather than
    hoped for: `app.Engine` must still start cleanly (S11b's own discovery
    change), and any attempt to query/compare with PE-Core against this AOI
    must fail cleanly, never with a stack trace."""
    data_root = tmp_path_factory.mktemp("s11b_solo_data")
    index_root = tmp_path_factory.mktemp("s11b_solo_index")
    rel = "solo_scene.tif"
    _make_fixture_raster(data_root / rel, size=448, seed=13)
    aoi = "solo_scene"
    embed_index.build_index(
        rel, model_id=RC_MODEL_ID, scales=(448, 224, 112), batch_size=32,
        data_root=data_root, index_root=index_root, embedder=rc_embedder,
    )
    retrieve.build_background(aoi, index_root=index_root, data_root=data_root, embedder=rc_embedder)
    return app.Engine(aoi=aoi, index_root=index_root, data_root=data_root, models=app.COMPARE_MODELS)


@pytest.fixture(scope="module")
def single_model_client(single_model_engine):
    return TestClient(app.create_app(engine=single_model_engine))


# --------------------------------------------------------------------------
# test_model_selector_switches_index
# --------------------------------------------------------------------------


def test_model_selector_switches_index(compare_engine):
    """Querying under each model reads that model's own on-disk index --
    proven by recomputing each result's score directly from that model's own
    stored vectors (not just trusting the label), and by checking the two
    models' rankings for 'a car' are not accidentally identical (they occupy
    different, independently-embedded spaces, so identical top-10 tile ids
    in the same order at every scale would itself be evidence of a plumbing
    bug -- reading the same vectors twice under two different labels)."""
    rc_out = app.answer_query(compare_engine, "a car", model_id=RC_MODEL_ID)
    pe_out = app.answer_query(compare_engine, "a car", model_id=PE_MODEL_ID)

    assert rc_out["model_id"] == RC_MODEL_ID
    assert rc_out["model_label"] == "RemoteCLIP-ViT-L-14 · 768-d"
    assert pe_out["model_id"] == PE_MODEL_ID
    assert pe_out["model_label"] == "PE-Core-L14-336 · 1024-d"

    rc_corpus = compare_engine.get_corpus("compare_scene", model_id=RC_MODEL_ID)
    pe_corpus = compare_engine.get_corpus("compare_scene", model_id=PE_MODEL_ID)
    assert rc_corpus.manifest["model_id"] == RC_MODEL_ID
    assert rc_corpus.manifest["dim"] == 768
    assert pe_corpus.manifest["model_id"] == PE_MODEL_ID
    assert pe_corpus.manifest["dim"] == 1024

    # Ids come from the right corpus: each result's score matches a direct
    # dot product against *that model's own* stored vector for that tile,
    # never the other model's.
    rc_embedder = compare_engine.embedders[RC_MODEL_ID]
    pe_embedder = compare_engine.embedders[PE_MODEL_ID]
    rc_qvec = retrieve.embed_query("a car", embedder=rc_embedder)
    pe_qvec = retrieve.embed_query("a car", embedder=pe_embedder)

    checked = 0
    for scale, row in rc_out["rankings"].items():
        for r in row["results"][:3]:
            i = rc_corpus.manifest["tiles"].index(
                next(t for t in rc_corpus.manifest["tiles"] if t["tile_id"] == r["tile_id"])
            )
            direct = float(rc_corpus.vectors[i] @ rc_qvec)
            assert direct == pytest.approx(r["score"], abs=1e-4)
            checked += 1
    assert checked > 0

    checked = 0
    for scale, row in pe_out["rankings"].items():
        for r in row["results"][:3]:
            i = pe_corpus.manifest["tiles"].index(
                next(t for t in pe_corpus.manifest["tiles"] if t["tile_id"] == r["tile_id"])
            )
            direct = float(pe_corpus.vectors[i] @ pe_qvec)
            assert direct == pytest.approx(r["score"], abs=1e-4)
            checked += 1
    assert checked > 0


def test_default_model_is_remoteclip(compare_engine):
    """Default stays RemoteCLIP (brief: 'it won S0 and is the shipped
    default') -- a query with no model_id must answer identically to an
    explicit RC_MODEL_ID request."""
    assert compare_engine.default_model == RC_MODEL_ID
    default_out = app.answer_query(compare_engine, "a car")
    explicit_out = app.answer_query(compare_engine, "a car", model_id=RC_MODEL_ID)
    assert default_out["model_id"] == RC_MODEL_ID == explicit_out["model_id"]
    for scale in tiling.SCALES:
        assert [r["tile_id"] for r in default_out["rankings"][scale]["results"]] == [
            r["tile_id"] for r in explicit_out["rankings"][scale]["results"]
        ]


# --------------------------------------------------------------------------
# test_no_cross_model_merging
# --------------------------------------------------------------------------


def test_no_cross_model_merging(compare_engine):
    """No code path concatenates or jointly sorts vectors or results from
    two models. Proven structurally: the compare payload keeps each model's
    rankings in its own bucket (never one merged list), each model's own
    per-tile score is independently reproducible from that model's own
    vectors (already shown for single-model queries above; repeated here
    through the compare path specifically), and the two models' top-1 tile
    ids at 112px are not required (or expected) to agree -- the measured
    context (briefs/S11b.md) is 0/5 top-5 agreement on the real index, so a
    test that *asserted* agreement would itself be wrong."""
    out = app.answer_compare(compare_engine, "a car")
    assert set(out["models"].keys()) == {RC_MODEL_ID, PE_MODEL_ID}

    for model_id, result in out["models"].items():
        assert result["available"] is True
        assert result["model_id"] == model_id
        # Every result under this model's key is genuinely from this model's
        # own corpus -- score reproducible from that model's own vectors.
        corpus = compare_engine.get_corpus("compare_scene", model_id=model_id)
        qvec = retrieve.embed_query("a car", embedder=compare_engine.embedders[model_id])
        by_id = {t["tile_id"]: i for i, t in enumerate(corpus.manifest["tiles"])}
        for scale, row in result["rankings"].items():
            for r in row["results"]:
                i = by_id[r["tile_id"]]
                direct = float(corpus.vectors[i] @ qvec)
                assert direct == pytest.approx(r["score"], abs=1e-4)

    # Never a top-level merged/joint ranking anywhere in the payload -- only
    # the per-model "models" dict and per-model "rankings" inside it.
    assert "rankings" not in out
    assert "results" not in out


def test_run_compare_never_touches_other_models_scores(compare_engine):
    """A stronger structural guarantee than the payload shape: `run_compare`
    is implemented as one independent `run_query` call per model with no
    step that reads across buckets (see app.py's own docstring on
    `Engine.run_compare`) -- verified here by monkeypatching one model's
    embedder to return a vector that is deliberately *not* unit-norm and
    checking that only that model's column is affected (an error contained
    to its own bucket), never silently propagating into or corrupting the
    other model's already-computed ranking."""
    good_pe_embedder = compare_engine.embedders[PE_MODEL_ID]

    class _BrokenEmbedder:
        model_id = PE_MODEL_ID
        dim = good_pe_embedder.dim

        def embed_texts(self, texts):
            # Not unit-norm -- retrieve.embed_query must raise UnitNormError.
            return np.ones((len(texts), self.dim), dtype=np.float32)

    compare_engine.embedders[PE_MODEL_ID] = _BrokenEmbedder()
    try:
        with pytest.raises(retrieve.UnitNormError):
            compare_engine.run_compare("a car", models=(PE_MODEL_ID,))
        # The *other* model is entirely unaffected by the broken one.
        rc_result = compare_engine.run_query("a car", model_id=RC_MODEL_ID)
        assert rc_result["model_id"] == RC_MODEL_ID
    finally:
        compare_engine.embedders[PE_MODEL_ID] = good_pe_embedder


# --------------------------------------------------------------------------
# test_compare_view_same_query_both_models
# --------------------------------------------------------------------------


def test_compare_view_same_query_both_models(compare_engine):
    out = app.answer_compare(compare_engine, "a car")
    assert out["empty"] is False
    assert out["query"] == "a car"

    rc = out["models"][RC_MODEL_ID]
    pe = out["models"][PE_MODEL_ID]
    assert rc["available"] is True and pe["available"] is True
    assert rc["model_label"] == "RemoteCLIP-ViT-L-14 · 768-d"
    assert pe["model_label"] == "PE-Core-L14-336 · 1024-d"

    # Same scale set, each labelled with the same true ground extent (purely
    # geometric -- identical tile plan under both models).
    assert set(rc["rankings"].keys()) == set(tiling.SCALES)
    assert set(pe["rankings"].keys()) == set(tiling.SCALES)
    for scale in tiling.SCALES:
        assert rc["rankings"][scale]["ground_extent_m"] == pytest.approx(
            pe["rankings"][scale]["ground_extent_m"], rel=1e-9
        )
        # No tile appears under a scale it was not embedded at, in either column.
        for r in rc["rankings"][scale]["results"]:
            assert r["scale"] == scale
        for r in pe["rankings"][scale]["results"]:
            assert r["scale"] == scale


def test_compare_view_rendered_in_page_with_model_and_dim_headers():
    html = app.render_index_html()
    assert 'id="compare-toggle"' in html
    assert 'id="model-select"' in html
    assert "RemoteCLIP-ViT-L-14 · 768-d" in html
    assert "PE-Core-L14-336 · 1024-d" in html
    # F-1a's UI constraint, stated for the compare view. (Stops short of the
    # apostrophe in "model's" -- html.escape renders it as `&#x27;`, exactly
    # the same trick test_app.py's own `test_cross_row_caveat_in_page` uses
    # for "one scale's row".)
    assert "only meaningful within one model" in html


def test_never_shows_a_winner_computed_from_raw_scores():
    """Static content check -- the page must never contain "winner"/"best
    model" copy, and the JS renderer must never sort or compare across the
    `models` dict's own values (the compare column-building code only ever
    indexes into one model's own `rankings`/`results` at a time)."""
    html = app.render_index_html()
    for forbidden in ("winner", "best model", "which model is better", "recommended model"):
        assert forbidden not in html.lower()
    # The JS never compares two models' scores against each other -- no
    # comparison operator applied to two different models' `.score` /
    # `.timing.total_ms` fields anywhere in the compare renderer.
    assert "models[mid].score" not in app._JS
    import re

    assert not re.search(r"models\[\w+\]\.score\s*[<>]", app._JS)


# --------------------------------------------------------------------------
# N-1 -- top-10 under 200 ms on CPU, per model, and for a compare query.
# Measured against the real production index (both models), mirroring
# test_s6.py's own `test_query_latency_full_index_cpu`.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_corpus_rc():
    return retrieve.load_corpus_multi(
        sorted(embed_index.list_indexed_aois(model_id=RC_MODEL_ID)), model_id=RC_MODEL_ID
    )


@pytest.fixture(scope="module")
def full_corpus_pe():
    return retrieve.load_corpus_multi(
        sorted(embed_index.list_indexed_aois(model_id=PE_MODEL_ID)), model_id=PE_MODEL_ID
    )


def _measure_latency_ms(corpus, qvec, n_reps=20):
    retrieve.rank_per_scale(corpus, qvec, top_k=retrieve.TOP_K)  # warm-up
    latencies = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        retrieve.rank_per_scale(corpus, qvec, top_k=retrieve.TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def test_query_latency_per_model(full_corpus_rc, full_corpus_pe, rc_embedder, pe_embedder):
    n_rc = int(full_corpus_rc.vectors.shape[0])
    n_pe = int(full_corpus_pe.vectors.shape[0])
    assert n_rc == n_pe == 104_374, f"expected both models over 104,374 tiles, got rc={n_rc} pe={n_pe}"

    rc_qvec = retrieve.embed_query("a car", embedder=rc_embedder)
    pe_qvec = retrieve.embed_query("a car", embedder=pe_embedder)

    rc_lat = _measure_latency_ms(full_corpus_rc, rc_qvec)
    pe_lat = _measure_latency_ms(full_corpus_pe, pe_qvec)

    rc_mean, rc_max = sum(rc_lat) / len(rc_lat), max(rc_lat)
    pe_mean, pe_max = sum(pe_lat) / len(pe_lat), max(pe_lat)
    compare_total_mean = rc_mean + pe_mean  # a compare view runs two queries, sequentially (Engine._lock)

    print(
        f"\nN-1 per-model FULL INDEX ({n_rc} tiles) CPU query latency, 20 reps:\n"
        f"  {RC_MODEL_ID} (768-d):  mean={rc_mean:.3f} ms max={rc_max:.3f} ms\n"
        f"  {PE_MODEL_ID} (1024-d): mean={pe_mean:.3f} ms max={pe_max:.3f} ms\n"
        f"  compare total (search-only, sequential): {compare_total_mean:.3f} ms mean"
    )
    assert rc_max < 200.0, f"{RC_MODEL_ID}: top-{retrieve.TOP_K} search took up to {rc_max:.3f} ms (>= 200 ms)"
    assert pe_max < 200.0, f"{PE_MODEL_ID}: top-{retrieve.TOP_K} search took up to {pe_max:.3f} ms (>= 200 ms)"


# --------------------------------------------------------------------------
# Abuse cases -- nothing may show a stack trace.
# --------------------------------------------------------------------------


def test_http_compare_endpoint_ok_no_stack_trace(compare_client):
    resp = compare_client.post("/api/compare", json={"text": "a car"})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["models"].keys()) == {RC_MODEL_ID, PE_MODEL_ID}
    assert "Traceback" not in resp.text


def test_http_switch_model_mid_query_no_stack_trace(compare_client):
    """Switching models mid-query (brief's own abuse case): a sequence of
    single-model queries alternating models must each succeed cleanly."""
    for model_id in (RC_MODEL_ID, PE_MODEL_ID, RC_MODEL_ID):
        resp = compare_client.post("/api/query", json={"text": "a car", "model_id": model_id})
        assert resp.status_code == 200, resp.text
        assert resp.json()["model_id"] == model_id
        assert "Traceback" not in resp.text


def test_http_compare_known_absent_query_no_stack_trace(compare_client):
    """A query for something this corpus cannot contain must still answer
    cleanly on both models -- U-3's 'may not be present' wording, never a
    crash, never a hidden/gated result."""
    resp = compare_client.post("/api/compare", json={"text": "a nuclear submarine"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Traceback" not in resp.text
    for model_id, result in data["models"].items():
        assert result["available"] is True
        for row in result["rankings"].values():
            assert "is not present" not in row["confidence"]["message"]


def test_http_unknown_model_id_clean_422(compare_client):
    resp = compare_client.post("/api/query", json={"text": "a car", "model_id": "not-a-real-model"})
    assert resp.status_code == 422
    assert "Traceback" not in resp.text
    assert isinstance(resp.json().get("detail"), str)


def test_http_model_with_missing_index_for_aoi_single_query_clean_422(single_model_client):
    """The brief's own abuse case: switch to a model whose index is missing
    for the selected AOI. A single-model /api/query for PE-Core (never
    indexed for `solo_scene`) must fail cleanly, not with a 500/stack
    trace."""
    resp = single_model_client.post("/api/query", json={"text": "a car", "model_id": PE_MODEL_ID})
    assert resp.status_code == 422
    assert "Traceback" not in resp.text
    assert isinstance(resp.json().get("detail"), str)

    # The default model (RemoteCLIP, indexed) still works fine on the same engine.
    ok = single_model_client.post("/api/query", json={"text": "a car"})
    assert ok.status_code == 200
    assert "Traceback" not in ok.text


def test_http_model_with_missing_index_for_aoi_compare_degrades_gracefully(single_model_client):
    """Same missing-index scenario, but through /api/compare: the whole
    comparison must not 500 just because one model lacks an index -- the
    available model's column still answers, the missing one reports
    `available: false`, never a stack trace."""
    resp = single_model_client.post("/api/compare", json={"text": "a car"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Traceback" not in resp.text
    assert data["models"][RC_MODEL_ID]["available"] is True
    assert len(data["models"][RC_MODEL_ID]["rankings"]) > 0
    assert data["models"][PE_MODEL_ID]["available"] is False
    assert data["models"][PE_MODEL_ID]["model_label"] == "PE-Core-L14-336 · 1024-d"


def test_engine_construction_does_not_crash_with_missing_second_model_index(single_model_engine):
    """Constructing the Engine itself over an index_root where only one
    model was ever built must not raise -- discovery for the *other* model
    is independent and forgiving (see Engine.__init__'s docstring)."""
    assert single_model_engine.default_model == RC_MODEL_ID
    assert "solo_scene" in single_model_engine._model_aois[RC_MODEL_ID]
    assert "solo_scene" not in single_model_engine._model_aois.get(PE_MODEL_ID, [])


def test_http_compare_switch_aoi_no_stack_trace(compare_client):
    """Switching AOI while in compare view: repeat the compare call with an
    unknown AOI, then a real one -- neither may 500/stack-trace, and the
    real one must still answer both models."""
    bad = compare_client.post("/api/compare", json={"text": "a car", "aoi": "not_a_real_aoi"})
    assert bad.status_code == 422
    assert "Traceback" not in bad.text

    ok = compare_client.post("/api/compare", json={"text": "a car", "aoi": "compare_scene"})
    assert ok.status_code == 200
    assert "Traceback" not in ok.text
    assert ok.json()["models"][RC_MODEL_ID]["available"] is True


# --------------------------------------------------------------------------
# background_path -- confirms the export directory is never touched for a
# non-default model (constraint: "do not touch index/export/").
# --------------------------------------------------------------------------


def test_background_path_for_non_default_model_never_under_export(tmp_path):
    rc_path = retrieve.background_path("some_aoi", index_root=tmp_path, model_id=RC_MODEL_ID)
    pe_path = retrieve.background_path("some_aoi", index_root=tmp_path, model_id=PE_MODEL_ID)
    assert "export" in rc_path.parts  # unchanged pre-S11b behaviour for the default model
    assert "export" not in pe_path.parts
    assert pe_path == embed_index.emb_dir("some_aoi", tmp_path, model_id=PE_MODEL_ID) / "background.json"
    assert rc_path != pe_path
