"""Acceptance tests for S5 -- the local query app. Spec U-1..U-8, D-5, and
the free-text half of F-8's amendment.

Tests the app's *logic* (query -> per-scale results -> confidence band ->
geo location, plus the page's static content), not pixel layout -- per
brief S5.md: "Test the app's logic ... not pixel layout." Pixel/viewport
behaviour (U-7 responsiveness) is verified manually with a real browser
(see briefs/S5_result.md).

Mirrors test_retrieve.py's own fixture pattern: a real, small (fixture-
scale) index built with the *real* embedder under `tmp_path`, never the
shipped production index/tileplan -- this file never writes under the real
index root.

Run:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        AERIAL_DATA_ROOT=<data root> \
        <python> -m pytest retrieval/tests/test_app.py -v -s
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

import app
import embed_index
import embedders
import geo
import retrieve
import tiling

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_embedder():
    return embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)


def _make_fixture_raster(path: Path, size: int, seed: int = 0) -> Path:
    """Same synthetic-raster recipe as test_retrieve.py's own fixture: real
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
def app_engine(real_embedder, tmp_path_factory):
    """A small, real index + background reference set, wrapped in an
    `app.Engine` the same way `create_app` builds the real one -- entirely
    under `tmp_path_factory`, never touching production data or the shipped
    `index/` (S5.md: tests must not write under the real index root)."""
    data_root = tmp_path_factory.mktemp("data")
    index_root = tmp_path_factory.mktemp("index")
    rel = "app_scene.tif"
    _make_fixture_raster(data_root / rel, size=896, seed=5)
    aoi = "app_scene"
    embed_index.build_index(
        rel, scales=(448, 224, 112), batch_size=32,
        data_root=data_root, index_root=index_root, embedder=real_embedder,
    )
    retrieve.build_background(aoi, index_root=index_root, data_root=data_root, embedder=real_embedder)
    return app.Engine(aoi=aoi, index_root=index_root, data_root=data_root)


@pytest.fixture(scope="module")
def client(app_engine):
    fastapi_app = app.create_app(engine=app_engine)
    return TestClient(fastapi_app)


# --------------------------------------------------------------------------
# U-2 / F-4 -- one ranking per scale, each tagged with its true ground
# extent (from geo.py, never hardcoded).
# --------------------------------------------------------------------------


def test_query_returns_one_ranking_per_scale(app_engine):
    out = app.answer_query(app_engine, "a car")
    rankings = out["rankings"]

    assert set(rankings.keys()) == set(tiling.SCALES)
    assert len(rankings) == len(tiling.SCALES)
    for scale, row in rankings.items():
        assert row["scale"] == scale
        expected_extent = geo.tile_ground_extent_m(0.1, scale)  # fixture's 0.1 m/px UTM GSD
        assert row["ground_extent_m"] == pytest.approx(expected_extent, rel=1e-6)
        # F-4's clarification, surfaced at the app layer: no tile appears
        # under a scale it was not embedded at.
        for r in row["results"]:
            assert r["scale"] == scale


def test_no_cross_scale_pooling(app_engine):
    """The app must not invent a pooled ranking on top of retrieve.py's
    per-scale ones -- same tile id must never appear under two scales."""
    out = app.answer_query(app_engine, "a car")
    seen: dict[str, int] = {}
    for scale, row in out["rankings"].items():
        for r in row["results"]:
            assert r["tile_id"] not in seen, (
                f"tile {r['tile_id']} appears under both scale {seen.get(r['tile_id'])} and {scale}"
            )
            seen[r["tile_id"]] = scale


# --------------------------------------------------------------------------
# U-2 / F-10 -- every hit carries its map location.
# --------------------------------------------------------------------------


def test_results_carry_location(app_engine):
    out = app.answer_query(app_engine, "a car")
    checked = 0
    for scale, row in out["rankings"].items():
        for r in row["results"]:
            assert -90.0 <= r["lat"] <= 90.0
            assert -180.0 <= r["lon"] <= 180.0
            assert r["date"] == "unknown"  # fixture raster's filename carries no date
            assert r["source_file"]
            assert isinstance(r["px_offset_x"], int) and isinstance(r["px_offset_y"], int)
            # px offset is index*scale, never the bare grid index (CLAUDE.md's trap).
            rec = tiling.parse_tile_id(r["tile_id"])
            assert r["px_offset_x"] == rec["col"] * rec["scale"]
            assert r["px_offset_y"] == rec["row"] * rec["scale"]
            # U-2: fixed-width score formatting.
            assert re.fullmatch(r"[+-]\d\.\d{4}", r["score_display"]), r["score_display"]
            checked += 1
    assert checked > 0, "expected at least one result across all scales"


# --------------------------------------------------------------------------
# U-3 -- the confidence band is shown, never gates or reorders results.
# --------------------------------------------------------------------------


def test_confidence_band_present_not_gating(app_engine):
    embedder, _, _ = app_engine.get_embedder()
    qvec = retrieve.embed_query("a car", embedder=embedder)
    with_bg = retrieve.rank_per_scale(app_engine.corpus, qvec, background=app_engine.background)
    without_bg = retrieve.rank_per_scale(app_engine.corpus, qvec, background=None)

    for scale in tiling.SCALES:
        band = with_bg["rankings"][scale]["confidence"]
        assert band["band"] in {"low", "medium", "high", "unknown"}
        assert "is not present" not in band["message"]
        # Never gates or reorders: identical result set/order regardless of
        # whether a background reference was supplied.
        assert with_bg["rankings"][scale]["results"] == without_bg["rankings"][scale]["results"]

    # And the app's own enrichment path surfaces the band unchanged.
    out = app.answer_query(app_engine, "a car")
    for scale, row in out["rankings"].items():
        assert "confidence" in row
        assert "is not present" not in row["confidence"]["message"]
        if row["confidence"]["band"] == "low":
            assert "may not be present" in row["confidence"]["message"]


# --------------------------------------------------------------------------
# U-1 -- opens in a working state: examples visible/clickable, never a
# blank box.
# --------------------------------------------------------------------------


def test_examples_present_at_rest():
    html = app.render_index_html()
    for q in app.EXAMPLE_QUERIES:
        assert q in html, f"example query {q!r} not rendered"
    assert 'class="example-chip"' in html
    assert html.count('data-query="') == len(app.EXAMPLE_QUERIES)
    # The input itself is pre-filled -- not a literal blank box the user
    # must guess vocabulary for.
    assert re.search(r'id="query-input"[^>]*value="[^"]+"', html)
    # And the page auto-runs a query at load, so it opens already populated.
    assert "runQuery(state.lastQuery)" in html


# --------------------------------------------------------------------------
# U-5 / D-5 -- the resolution caveat is stated in the interface, and no
# attribute-level claim is made anywhere in the page.
# --------------------------------------------------------------------------


def test_resolution_caveat_in_page():
    html = app.render_index_html()
    assert 'id="caveat"' in html
    assert "35" in html and "45" in html and "cm" in html
    assert "presence and coarse class" in html
    assert "white car" in html  # the measured evidence the caveat cites
    assert "does not do attribute-level search" in html
    # D-5: no claim of attribute-level discrimination anywhere in the copy.
    assert "vehicle colour" not in html.lower()
    assert "vehicle model" not in html.lower()


def test_no_attribute_search_offered():
    """U-5: 'Do not offer attribute search.' No colour/attribute filter
    control exists anywhere in the page."""
    html = app.render_index_html()
    for forbidden in ("colour filter", "color filter", "attribute filter", "filter by colour", "filter by color"):
        assert forbidden not in html.lower()


# --------------------------------------------------------------------------
# U-2 -- scores are only ever compared within a scale row; the UI must not
# invite cross-row comparison.
# --------------------------------------------------------------------------


def test_cross_row_caveat_in_page():
    html = app.render_index_html()
    assert "only meaningful within one scale" in html or "only comparable within this row" in html


# --------------------------------------------------------------------------
# No external CDN, no network calls at runtime.
# --------------------------------------------------------------------------


def test_no_external_refs():
    html = app.render_index_html()
    for m in re.finditer(r'(?:src|href)\s*=\s*"([^"]*)"', html):
        val = m.group(1)
        assert not re.match(r"^(https?:)?//", val), f"external ref found: {val!r}"
    assert "http://" not in html
    assert "https://" not in html
    assert "cdn." not in html.lower()
    assert "googleapis" not in html.lower()


# --------------------------------------------------------------------------
# U-4 -- feedback on anything over 300 ms, specifically first-query model
# warm-up (~9 s here on the real GPU model).
# --------------------------------------------------------------------------


def test_first_query_latency_reported(app_engine):
    """A *fresh* Engine over the same on-disk fixture index/background (not
    `app_engine` itself, which other tests in this module may already have
    warmed) -- so the first-vs-second call transition is genuine, not
    already-warm."""
    fresh = app.Engine(aoi=app_engine.aoi, index_root=app_engine.index_root, data_root=app_engine.data_root)
    assert fresh.embedder is None

    first = app.answer_query(fresh, "a car")
    assert first["timing"]["warm_up"] is True
    assert first["timing"]["model_load_ms"] > 0.0
    assert first["timing"]["total_ms"] >= first["timing"]["model_load_ms"]

    second = app.answer_query(fresh, "a car")
    assert second["timing"]["warm_up"] is False
    assert second["timing"]["model_load_ms"] == 0.0
    assert first["timing"]["total_ms"] > second["timing"]["total_ms"]

    # The page itself surfaces this to the user (U-4's actual requirement:
    # feedback must be visible, not just measured server-side).
    html = app.render_index_html()
    assert "first search can take" in html or "model warm-up" in html


# --------------------------------------------------------------------------
# Empty / whitespace query -- must guide, never raise (mirrors U-3's own
# "empty state guides" spirit for the query box itself).
# --------------------------------------------------------------------------


def test_empty_and_whitespace_query_does_not_call_embedder(app_engine):
    for text in ("", "   ", "\t\n  "):
        out = app.answer_query(app_engine, text)
        assert out["empty"] is True
        assert out["rankings"] == {}
        assert out["message"]
        assert out["timing"]["total_ms"] == 0.0


# --------------------------------------------------------------------------
# F-7's date filter, exercised at the app layer (retrieve.py itself has no
# date-filtering function -- this module adds the thin candidate-narrowing
# wrapper `filter_corpus_by_date`, never reimplementing rank_per_scale's own
# cosine/sort maths).
# --------------------------------------------------------------------------


def test_date_filter_excludes_unknown_dated_tiles(app_engine):
    """The fixture raster's filename carries no date, so every tile is
    'unknown'. Filtering to any real date value must exclude all of them
    (F-7's amendment: undated tiles are excluded under an active date
    filter, never silently included) -- and must not raise."""
    filtered = app.filter_corpus_by_date(app_engine.corpus, "2025-06-06")
    for scale in tiling.SCALES:
        assert filtered.scale_indices[scale].size == 0

    # No filter (None / "" / "any") is the identity.
    for none_ish in (None, "", "any"):
        same = app.filter_corpus_by_date(app_engine.corpus, none_ish)
        for scale in tiling.SCALES:
            assert same.scale_indices[scale].size == app_engine.corpus.scale_indices[scale].size


# --------------------------------------------------------------------------
# HTTP layer -- thin smoke tests over the same logic, proving the FastAPI
# wiring does not introduce its own bugs (still logic, not pixel layout).
# --------------------------------------------------------------------------


def test_http_index_page_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Aerial Tile Retrieval" in resp.text


def test_http_query_endpoint_ok(client):
    resp = client.post("/api/query", json={"text": "a car"})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["rankings"].keys()) == {str(s) for s in tiling.SCALES} or set(
        int(k) for k in data["rankings"].keys()
    ) == set(tiling.SCALES)


def test_http_empty_query_no_stack_trace(client):
    resp = client.post("/api/query", json={"text": "   "})
    assert resp.status_code == 200
    assert resp.json()["empty"] is True
    assert "Traceback" not in resp.text


def test_http_bbox_matching_nothing_no_stack_trace(client):
    resp = client.post("/api/query", json={"text": "a car", "bbox": [0.0, 0.0, 0.001, 0.001]})
    assert resp.status_code == 200
    data = resp.json()
    for row in data["rankings"].values():
        assert row["results"] == []
    assert "Traceback" not in resp.text


def test_http_malformed_bbox_returns_clean_422(client):
    resp = client.post("/api/query", json={"text": "a car", "bbox": [1.0, 2.0]})
    assert resp.status_code == 422
    assert "Traceback" not in resp.text
    assert isinstance(resp.json().get("detail"), str)


def test_http_thumb_ok_and_unknown_tile_id_is_clean_404(app_engine, client):
    out = app.answer_query(app_engine, "a car")
    any_row = next(r for r in out["rankings"].values() if r["results"])
    tile_id = any_row["results"][0]["tile_id"]

    resp = client.get("/api/thumb", params={"tile_id": tile_id, "size": 64})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    bad = client.get("/api/thumb", params={"tile_id": "not-a-real-tile-id", "size": 64})
    assert bad.status_code == 404
    assert "Traceback" not in bad.text


def test_http_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["n_tiles"] > 0
