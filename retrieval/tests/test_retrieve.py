"""Acceptance tests for S4. Spec F-3, F-4, F-5, F-6, U-3, N-1.

Run:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        AERIAL_DATA_ROOT=<data root> \
        <python> -m pytest retrieval/tests/test_retrieve.py -v -s
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
import geo
import retrieve
import tiling

PROBE = Path(__file__).resolve().parent / "_retrieve_reload_probe.py"
CONFIDENCE_PROBE = Path(__file__).resolve().parent / "_confidence_probe.py"

DEMO_AOI = "X605_Y3388"
DEMO_REL_PATH = "X605_Y3388.tif"

CALIBRATION_QUERIES_112PX = {
    # brief S4.md: real measurements taken on the actual production index at
    # 112 px, top-1 cosine score -- used below both to sanity-check that this
    # module reproduces them and to calibrate/validate the U-3 weak-match
    # rule. The known-absent control is last.
    "tents": 0.2805,
    "palm trees": 0.2801,
    "a car": 0.2650,
    "dirt road": 0.2571,
    "aircraft carrier": 0.2332,
}
KNOWN_ABSENT_QUERY = "aircraft carrier"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_embedder():
    return embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)


def _make_fixture_raster(path: Path, size: int, seed: int = 0) -> Path:
    """A synthetic 3-band uint8 GeoTIFF: real CRS/transform (EPSG:32636 UTM,
    0.1 m/px -- the demo AOI's own CRS/GSD, so ground-resolution/guard logic
    needs no special-casing), non-degenerate random content."""
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
def synthetic_index(real_embedder, tmp_path_factory):
    """A small multi-scale index (896x896 -> 2x2 @448, 4x4 @224, 8x8 @112 --
    enough tiles per scale to test ranking/filtering without depending on
    the real production data)."""
    data_root = tmp_path_factory.mktemp("data")
    index_root = tmp_path_factory.mktemp("index")
    rel = "retrieve_scene.tif"
    _make_fixture_raster(data_root / rel, size=896, seed=3)
    aoi = "retrieve_scene"
    embed_index.build_index(
        rel, scales=(448, 224, 112), batch_size=32,
        data_root=data_root, index_root=index_root, embedder=real_embedder,
    )
    return data_root, index_root, aoi


@pytest.fixture(scope="module")
def corpus(synthetic_index):
    data_root, index_root, aoi = synthetic_index
    return retrieve.load_corpus(aoi, index_root=index_root, data_root=data_root)


@pytest.fixture(scope="module")
def production_corpus():
    """The real, already-built production index (S3) -- read-only, never
    rebuilt here. Used for the tests that must exercise real scale (N-1's
    latency budget, F-5's reload budget) and the real calibration numbers
    (U-3)."""
    return retrieve.load_corpus(DEMO_AOI)


# --------------------------------------------------------------------------
# F-3 -- text queries embed through the same model's text tower.
# --------------------------------------------------------------------------


def test_text_query_unit_norm_and_deterministic(real_embedder, tmp_path):
    query = "a white car"
    a = retrieve.embed_query(query, embedder=real_embedder)
    b = retrieve.embed_query(query, embedder=real_embedder)

    assert a.dtype == np.float32
    assert abs(float(np.linalg.norm(a)) - 1.0) <= 1e-5
    assert a.tobytes() == b.tobytes(), "same text embedded differently within one process"

    # Cross-process: retrieve.embed_query is a thin wrapper over
    # embedders.load_embedder(...).embed_texts(...) -- reuse the existing
    # cross-process probe (tests/_text_embed_probe.py) rather than spawning
    # a second, redundant GPU load just to re-derive the same number.
    probe = Path(__file__).resolve().parent / "_text_embed_probe.py"
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="0")
    dst = tmp_path / "vec.f32"
    subprocess.run(
        [sys.executable, str(probe), embed_index.DEFAULT_MODEL_ID, query, str(dst)],
        capture_output=True, text=True, env=env, check=True,
    )
    other = np.fromfile(dst, dtype=np.float32)
    delta = float(np.abs(a - other).max())
    assert delta <= 1e-6, f"cross-process query embedding differs by {delta:.3e} (> 1e-6)"


def test_unit_norm_error_raised_on_bad_embedder():
    """F-3's own norm guard: a stub embedder that hands back a non-unit
    vector must be rejected here, not silently fed into F-4's cosine
    identity (which assumes both sides are unit-norm)."""

    class _BadEmbedder:
        model_id = "stub"

        def embed_texts(self, texts):
            return np.full((len(texts), 8), 2.0, dtype=np.float32)  # norm 5.66, not 1.0

    with pytest.raises(retrieve.UnitNormError):
        retrieve.embed_query("anything", embedder=_BadEmbedder())


# --------------------------------------------------------------------------
# F-4 -- ranking is cosine similarity, one ranking per scale.
# --------------------------------------------------------------------------


def test_ranking_is_per_scale(corpus, real_embedder):
    qvec = retrieve.embed_query("a car", embedder=real_embedder)
    out = retrieve.rank_per_scale(corpus, qvec, top_k=3)
    rankings = out["rankings"]

    expected_scales = set(corpus.manifest["scales"])
    assert set(rankings.keys()) == expected_scales
    assert len(rankings) == len(corpus.manifest["scales"]) == len(expected_scales)

    ids_by_scale = {}
    for scale, r in rankings.items():
        assert r["scale"] == scale
        expected_extent = geo.tile_ground_extent_m(0.1, scale)  # fixture's 0.1 m/px UTM GSD
        assert r["ground_extent_m"] == pytest.approx(expected_extent, rel=1e-6)

        scores = [res["score"] for res in r["results"]]
        assert scores == sorted(scores, reverse=True), "results not sorted descending"
        assert all(-1.0 <= s <= 1.0 for s in scores), f"score outside [-1,1]: {scores}"
        for res in r["results"]:
            assert res["scale"] == scale, "a result claims a scale other than its own ranking's"
        ids_by_scale[scale] = {res["tile_id"] for res in r["results"]}

    # No cross-scale pooling: no tile id appears under more than one scale.
    scales_list = list(ids_by_scale)
    for i in range(len(scales_list)):
        for j in range(i + 1, len(scales_list)):
            overlap = ids_by_scale[scales_list[i]] & ids_by_scale[scales_list[j]]
            assert not overlap, f"tile(s) {overlap} appear in both scale {scales_list[i]} and {scales_list[j]}"


def test_self_similarity_within_scale(corpus):
    """F-4: a query embedding compared against itself scores 1.0 +/- 1e-5.
    Uses one of the corpus's own stored vectors as the query -- exercises
    the ranking maths directly, with no embedder call needed."""
    scale = corpus.manifest["scales"][0]
    idx = corpus.scale_indices[scale]
    self_qvec = corpus.vectors[idx[0]].copy()
    tile_id = corpus.manifest["tiles"][idx[0]]["tile_id"]

    out = retrieve.rank_per_scale(corpus, self_qvec, top_k=1)
    top = out["rankings"][scale]["results"][0]
    assert top["tile_id"] == tile_id
    assert abs(top["score"] - 1.0) <= 1e-5


def test_result_pixel_offsets_are_index_times_scale_not_index(corpus):
    """F-10's trap, guarded directly: a tile id encodes GRID INDICES
    (`...::448::00012::00000` is column 12, row 0), and CLAUDE.md records a
    real incident where the PM read pixels at `(12, 0)` -- the index itself
    -- and silently got solid black. Every result's px_offset_x/y must be
    `col * scale` / `row * scale`, taken from `tiling.py`'s own plan (never
    recomputed from `col`/`row` a second, independent way here)."""
    for tile in corpus.manifest["tiles"]:
        rec = tiling.parse_tile_id(tile["tile_id"])
        loc = corpus.locations[tile["tile_id"]]
        assert loc["px_offset_x"] == rec["col"] * rec["scale"]
        assert loc["px_offset_y"] == rec["row"] * rec["scale"]
        # The trap, made concrete: for any tile whose grid index is >= 1,
        # the pixel offset must differ from the bare index.
        if rec["col"] >= 1:
            assert loc["px_offset_x"] != rec["col"], "px_offset_x looks like the raw column index, not index*scale"


def test_map_location_from_shipped_tileplan(production_corpus):
    """S4_fix Fix 1's second half: F-10 must hold against the real, shipped
    `index/tileplan/*.json` files, not only an in-memory re-plan. Before Fix
    1 those files were truncated to scale 448 only (three tests in
    test_tiling.py were overwriting them), which is why retrieve.py's own
    join calls `tiling.plan_scene` directly rather than trusting the file
    (see its module docstring). Now that Fix 1 regenerates the full
    three-scale artifact, this reads THOSE files -- not `tiling.plan_scene`
    a second time -- and cross-checks every production tile's location
    against what is actually on disk."""
    tileplan_dir = config.get_index_root() / "tileplan"
    plan_cache: dict[tuple[str, int], dict] = {}
    checked = 0
    for tile in production_corpus.manifest["tiles"]:
        rec = tiling.parse_tile_id(tile["tile_id"])
        key = (rec["source_file"], rec["scale"])
        if key not in plan_cache:
            aoi = tiling.scene_aoi(rec["source_file"])
            stem = Path(config.posix_key(rec["source_file"])).stem
            fname = f"{aoi}__{stem}.json".replace("/", "_")
            doc = json.loads((tileplan_dir / fname).read_text())
            assert str(rec["scale"]) in doc, (
                f"{fname}: shipped tileplan on disk is missing scale {rec['scale']} "
                f"for {rec['source_file']}"
            )
            plan_cache[key] = {t["tile_id"]: t for t in doc[str(rec["scale"])]["tiles"]}
        shipped = plan_cache[key][tile["tile_id"]]
        loc = production_corpus.locations[tile["tile_id"]]
        assert loc["px_offset_x"] == shipped["px_offset_x"]
        assert loc["px_offset_y"] == shipped["px_offset_y"]
        assert tuple(loc["bbox_lonlat"]) == tuple(shipped["bbox_lonlat"])
        assert loc["source_file"] == shipped["source_file"]
        assert loc["date"] == shipped["date"]
        checked += 1
    assert checked == len(production_corpus.manifest["tiles"]) == 9631


# --------------------------------------------------------------------------
# F-6 -- AOI filter restricts candidates by geographic bbox.
# --------------------------------------------------------------------------


def test_aoi_filter(corpus, real_embedder):
    scale = 224  # 4x4 grid in the fixture -- enough neighbours to prove isolation
    tile_id = next(t["tile_id"] for t in corpus.manifest["tiles"] if t["scale"] == scale)
    min_lon, min_lat, max_lon, max_lat = corpus.locations[tile_id]["bbox_lonlat"]

    # Shrink 25% inward on every side: strictly inside this tile's own
    # footprint, so it can never touch a neighbour's edge (OVERLAP=0 tiles
    # are contiguous, not separated -- shrinking, not just picking the exact
    # bbox, is what keeps this unambiguous).
    dlon, dlat = (max_lon - min_lon) * 0.25, (max_lat - min_lat) * 0.25
    inner_bbox = (min_lon + dlon, min_lat + dlat, max_lon - dlon, max_lat - dlat)

    qvec = retrieve.embed_query("a car", embedder=real_embedder)
    n_at_scale = len(corpus.scale_indices[scale])
    out = retrieve.rank_per_scale(corpus, qvec, top_k=n_at_scale, bbox=inner_bbox)
    ids = {r["tile_id"] for r in out["rankings"][scale]["results"]}
    assert ids == {tile_id}, f"expected exactly {{{tile_id}}}, got {ids}"

    # A bbox nowhere near the AOI: no results at any scale, no exception.
    far_away = (min_lon - 100.0, min_lat - 100.0, min_lon - 99.0, min_lat - 99.0)
    out_far = retrieve.rank_per_scale(corpus, qvec, bbox=far_away)
    for s, r in out_far["rankings"].items():
        assert r["results"] == [], f"scale {s}: expected no results for a bbox far from the AOI"

    # A degenerate (min > max) bbox: also no results, also no exception --
    # F-6's "empty bbox" requirement, covered from the other direction.
    degenerate = (min_lon, min_lat, min_lon - 1.0, min_lat - 1.0)
    out_degenerate = retrieve.rank_per_scale(corpus, qvec, bbox=degenerate)
    for s, r in out_degenerate["rankings"].items():
        assert r["results"] == [], f"scale {s}: degenerate bbox must not raise and must match nothing"


# --------------------------------------------------------------------------
# F-5 -- the index persists and reloads without recomputation.
# --------------------------------------------------------------------------


def test_index_roundtrip(synthetic_index, real_embedder, corpus, tmp_path):
    """Fixture-scale mechanism proof: byte-identical top-1 ids/scores across
    a fresh process, using a FIXED, pre-computed query vector (not a fresh
    text-embedder call in the child) -- isolates F-5's own claim (index
    reload + ranking maths are deterministic) from F-3's separate, looser
    (1e-6) cross-process text-embedding guarantee.

    top_k=1 deliberately, not TOP_K: this fixture's content is unstructured
    random noise (no genuine best match), so several of its candidates score
    within noise of each other -- multi-threaded OpenBLAS does not guarantee
    a fixed reduction order between two independent process launches, so a
    near-tied *tail* ranking (e.g. rank 4 vs 5) can legitimately swap by
    ~1e-4 without any bug in `load_index`/`rank_per_scale`. That is a
    property of floating-point non-associativity under threaded BLAS, not of
    this module -- and it does not appear on real, semantically-separated
    data: `test_index_roundtrip_production` below proves byte-identical
    top-TOP_K (not just top-1) against the actual 9,631-vector production
    index, which is the claim F-5 is actually about.
    """
    data_root, index_root, aoi = synthetic_index
    qvec = retrieve.embed_query("a car", embedder=real_embedder)
    qvec_path = tmp_path / "qvec.npy"
    np.save(qvec_path, qvec)

    out = retrieve.rank_per_scale(corpus, qvec, top_k=1)
    expected = {
        str(scale): [{"tile_id": r["tile_id"], "score": r["score"]} for r in ranking["results"]]
        for scale, ranking in out["rankings"].items()
    }

    env = dict(os.environ, PYTHONNOUSERSITE="1", AERIAL_DATA_ROOT=str(data_root))
    proc = subprocess.run(
        [sys.executable, str(PROBE), aoi, str(index_root), str(qvec_path), "1"],
        capture_output=True, text=True, env=env, check=True,
    )
    result = json.loads(proc.stdout)
    assert result["rankings"] == expected, "fresh-process reload gave a different top-1 id/score"


def test_index_roundtrip_production(real_embedder, tmp_path):
    """Real-scale proof against the actual production index (9,631 vectors,
    read-only) -- reload completes in < 5 s (F-5) and gives byte-identical
    results in a fresh process, using the five real calibration queries."""
    qvec = retrieve.embed_query(KNOWN_ABSENT_QUERY, embedder=real_embedder)
    qvec_path = tmp_path / "qvec.npy"
    np.save(qvec_path, qvec)

    t0 = time.perf_counter()
    corpus = retrieve.load_corpus(DEMO_AOI)
    in_process_reload_s = time.perf_counter() - t0

    out = retrieve.rank_per_scale(corpus, qvec, top_k=10)
    expected = {
        str(scale): [{"tile_id": r["tile_id"], "score": r["score"]} for r in ranking["results"]]
        for scale, ranking in out["rankings"].items()
    }

    proc = subprocess.run(
        [sys.executable, str(PROBE), DEMO_AOI, str(config.get_index_root()), str(qvec_path)],
        capture_output=True, text=True,
        env=dict(os.environ, PYTHONNOUSERSITE="1", AERIAL_DATA_ROOT=str(config.get_data_root())),
        check=True,
    )
    result = json.loads(proc.stdout)
    print(
        f"\nF-5 production reload: in_process={in_process_reload_s:.3f}s "
        f"fresh_process={result['reload_s']:.3f}s"
    )
    assert result["reload_s"] < 5.0, f"fresh-process reload took {result['reload_s']:.3f}s (>= 5s)"
    assert in_process_reload_s < 5.0, f"in-process reload took {in_process_reload_s:.3f}s (>= 5s)"
    assert result["rankings"] == expected, "fresh-process reload gave different top-k ids/scores"


# --------------------------------------------------------------------------
# N-1 -- query latency: top-10 over the full index in < 200 ms on CPU.
# --------------------------------------------------------------------------


def test_query_latency_cpu(production_corpus, real_embedder):
    """Times only `rank_per_scale` (the numpy brute-force search) against
    the real production index -- not text embedding, which is a separate
    GPU model call N-1 is not about (see the baseline note in spec.md: numpy
    doing 5000x1280 in 0.37 ms is the thing this guards)."""
    qvec = retrieve.embed_query("a car", embedder=real_embedder)

    # One warm-up call (first-touch page faults / allocator warm-up), then
    # measure repeatedly.
    retrieve.rank_per_scale(production_corpus, qvec, top_k=retrieve.TOP_K)
    n_reps = 20
    latencies_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        retrieve.rank_per_scale(production_corpus, qvec, top_k=retrieve.TOP_K)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    max_ms = max(latencies_ms)
    print(f"\nN-1 CPU query latency over {n_reps} reps: mean={mean_ms:.3f} ms max={max_ms:.3f} ms")
    assert max_ms < 200.0, f"top-{retrieve.TOP_K} search took up to {max_ms:.3f} ms (>= 200 ms)"


# --------------------------------------------------------------------------
# U-3 -- S4_fix Fix 2: the match indicator is a calibrated CONFIDENCE BAND
# against a fixed background set, never a present/absent boolean. The old
# `weak_match` boolean (`WEAK_GAP_MULTIPLE`, tuned on five values) was
# measured by the PM to fail on unseen queries at both 112px and 448px, and
# every one of eight relative statistics tried over 8 present/8 absent
# queries overlapped between the groups -- see retrieve.py's U-3 section
# banner. `test_weak_match_has_no_absolute_cosine_constant` below still
# exercises `is_weak_match` directly (it is retained as an internal,
# unit-tested helper nothing gates on); `rank_per_scale`'s output no longer
# has a `weak_match` key at all.
# --------------------------------------------------------------------------


def test_calibration_scores_reproduce_brief_measurements(production_corpus, real_embedder):
    """Sanity check before trusting the confidence-band measurements below:
    this exact index + embedder must reproduce the brief's stated 112 px
    top-1 scores (to loose tolerance -- they were read off a printed
    table)."""
    for text, expected_top1 in CALIBRATION_QUERIES_112PX.items():
        qvec = retrieve.embed_query(text, embedder=real_embedder)
        out = retrieve.rank_per_scale(production_corpus, qvec, top_k=1)
        top1 = out["rankings"][112]["results"][0]["score"]
        assert top1 == pytest.approx(expected_top1, abs=2e-3), (
            f"{text!r} at 112px: got top1={top1:.4f}, brief measured {expected_top1:.4f}"
        )


@pytest.fixture(scope="module")
def production_background(real_embedder):
    """Builds the real AOI's background reference set once per module -- the
    artifact `confidence_band` is calibrated against (S4_fix Fix 2), stored
    alongside the F-2a export (`retrieve.background_path`) so it is a
    reproducible build artifact, not something recomputed per query."""
    doc = retrieve.build_background(DEMO_AOI, embedder=real_embedder)
    return doc["scores_by_scale"]


def test_confidence_band_reported_never_gates_results(production_corpus, production_background, real_embedder):
    """S4_fix Fix 2's actual requirement: every scale's ranking carries a
    `confidence` band (never the retired `weak_match` boolean), the wording
    never claims certainty of absence, and results are identical regardless
    of the band -- this is an indicator shown alongside results, not a
    filter. Deliberately does NOT assert that present and absent queries
    land in different bands: the PM measured a large, honest overlap between
    the two groups on this index, and tuning this test to hide that would be
    exactly the overfitting Fix 2 exists to undo."""
    for text in CALIBRATION_QUERIES_112PX:
        qvec = retrieve.embed_query(text, embedder=real_embedder)
        out_with_bg = retrieve.rank_per_scale(
            production_corpus, qvec, top_k=10, background=production_background
        )
        out_without_bg = retrieve.rank_per_scale(production_corpus, qvec, top_k=10)
        for scale, ranking in out_with_bg["rankings"].items():
            assert "weak_match" not in ranking, "the retired boolean must not reappear in the output"
            confidence = ranking["confidence"]
            assert confidence["band"] in {"low", "medium", "high"}
            assert 0.0 <= confidence["percentile"] <= 100.0
            assert "is not present" not in confidence["message"], (
                "U-3 wording requirement: confidence must never claim certainty of absence"
            )
            if confidence["band"] == "low":
                assert "may not be present" in confidence["message"]
            # Never gates: identical results with/without a background set.
            assert ranking["results"] == out_without_bg["rankings"][scale]["results"]
        # No background reference available -> reported as "unknown", not raised.
        for scale, ranking in out_without_bg["rankings"].items():
            assert ranking["confidence"]["band"] == "unknown"
            assert ranking["confidence"]["percentile"] is None


def test_confidence_band_is_relative():
    """Proof, not assertion-by-inspection -- mirrors
    `test_weak_match_has_no_absolute_cosine_constant`'s style:
    `confidence_band`'s percentile/band is invariant under an arbitrary
    positive affine rescale (`scores -> a*scores + b`, a > 0) applied
    identically to `top_score` and to every background score. An
    implementation hiding a fixed cosine cut-off would not be invariant;
    an empirical percentile against the same background, rescaled the same
    way, is."""
    rng = np.random.default_rng(11)
    background = rng.normal(loc=0.20, scale=0.02, size=30)
    top_score = float(np.percentile(background, 80))

    base = retrieve.confidence_band(top_score, background)
    assert base["band"] in {"low", "medium", "high"}

    for a, b in [(2.0, 0.0), (0.001, 5.0), (1000.0, -37.0), (50.0, 123.4)]:
        out = retrieve.confidence_band(a * top_score + b, a * background + b)
        assert out["band"] == base["band"], f"band changed under affine rescale a={a}, b={b}"
        assert out["percentile"] == pytest.approx(base["percentile"]), (
            f"percentile changed under affine rescale a={a}, b={b} -- "
            "an absolute cosine constant must be hiding in the path"
        )


def test_confidence_band_reproducible(real_embedder, production_background, tmp_path):
    """Same query, same band, across processes -- mirrors F-5's
    `test_index_roundtrip_production` reload-probe pattern. Reloads both the
    corpus and the background reference set from disk in a fresh process
    (not the in-memory values this test's own process just computed)."""
    qvec = retrieve.embed_query("a car", embedder=real_embedder)
    corpus = retrieve.load_corpus(DEMO_AOI)
    out = retrieve.rank_per_scale(corpus, qvec, top_k=1, background=production_background)
    in_process = {str(scale): r["confidence"] for scale, r in out["rankings"].items()}

    qvec_path = tmp_path / "qvec.npy"
    np.save(qvec_path, qvec)
    proc = subprocess.run(
        [sys.executable, str(CONFIDENCE_PROBE), DEMO_AOI, str(config.get_index_root()), str(qvec_path)],
        capture_output=True, text=True,
        env=dict(os.environ, PYTHONNOUSERSITE="1", AERIAL_DATA_ROOT=str(config.get_data_root())),
        check=True,
    )
    result = json.loads(proc.stdout)
    assert result == in_process, "fresh-process reload gave a different confidence band"


def test_weak_match_has_no_absolute_cosine_constant():
    """Proof, not assertion-by-inspection: `is_weak_match`'s classification
    is invariant under an arbitrary positive affine rescaling of the score
    axis (`scores -> a*scores + b`, a > 0). An implementation hiding a fixed
    cosine cut-off (e.g. "weak if top1 < 0.25") would NOT be invariant under
    such a rescaling; a true relative-gap rule (a ratio of two quantities
    that both scale by `a` and are both shifted by `b` identically) is."""
    rng = np.random.default_rng(0)
    background = rng.normal(loc=0.20, scale=0.02, size=500)
    spike = float(background.max() + 0.05)  # an isolated, unsupported peak
    scores = np.append(background, spike)
    top_score = float(scores.max())
    assert top_score == spike

    base = retrieve.is_weak_match(scores, top_score)
    assert base is True, "fixture spike should itself register as weak (isolated peak)"

    for a, b in [(2.0, 0.0), (0.001, 5.0), (1000.0, -37.0), (50.0, 123.4)]:
        transformed_scores = a * scores + b
        transformed_top = a * top_score + b
        assert retrieve.is_weak_match(transformed_scores, transformed_top) == base, (
            f"classification changed under affine rescale a={a}, b={b} -- "
            "an absolute cosine constant must be hiding in the threshold path"
        )

    # And the mirror case: a real cluster (the top 10 -- TOP_K -- scores
    # sitting tightly together, i.e. genuine peers, not one lone spike) must
    # register as NOT weak, and stay not-weak under the same rescalings.
    # (A naive "top1 vs background.max()" construction is NOT a valid
    # negative fixture here: 500 iid normal draws already have a sizeable
    # natural gap between their own 1st and 10th order statistics, which
    # would register as weak regardless of any added spike -- the point of
    # this fixture is a tight top-TOP_K cluster, not merely "close to the
    # sample max".)
    background2 = rng.normal(loc=0.20, scale=0.02, size=490)
    cluster = 0.20 + 0.06 + rng.normal(scale=0.0005, size=10)  # 10 genuine, tightly-scored peers
    cluster_scores = np.concatenate([background2, cluster])
    base_cluster = retrieve.is_weak_match(cluster_scores, float(cluster_scores.max()))
    assert base_cluster is False
    for a, b in [(2.0, 0.0), (0.001, 5.0), (1000.0, -37.0)]:
        assert retrieve.is_weak_match(a * cluster_scores + b, a * float(cluster_scores.max()) + b) == base_cluster
