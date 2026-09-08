"""Acceptance tests for S5b: the standalone offline HTML export.

Spec F-8, N-6, D-5, U-1..U-8.

Run:
    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        AERIAL_DATA_ROOT=<data root> \
        <python> -m pytest retrieval/tests/test_export_html.py -v -s

Two families of test, deliberately kept separate (test_export_basis.py's own
precedent):

  * synthetic, tmp_path-rooted tests (small fixture raster, real embedder,
    no dependency on the production index) -- fast, and exercise "AOI is a
    parameter" (S5b brief) directly by using an AOI name that is not the
    demo one.
  * the production-fidelity tests, which build the real deliverable at its
    real, gitignored location (`index/export/X605_Y3388/export.html` --
    exactly what `export_basis.build_export(DEMO_AOI)` already does with no
    `index_root` override in `test_export_basis.py`'s own
    `test_pca_overlap_measured_per_scale_production`). This is not the "real
    index root" CLAUDE.md's D-3 incident is about -- that incident was a
    *data-root* sabotage test aimed at the wrong root; the export's actual
    designated output home is under `index/`, which is gitignored and safe
    to write, and is where the brief says the artifact belongs.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import config
import embed_index
import embedders
import export_basis
import export_html
import retrieve
import tiling

DEMO_AOI = "X605_Y3388"
LEB_AOI = "leb"

NODE = shutil.which("node") or shutil.which("nodejs")
CHROME = (
    shutil.which("google-chrome")
    or shutil.which("google-chrome-stable")
    or shutil.which("chromium")
    or shutil.which("chromium-browser")
)


def _dump_dom(path: Path, timeout: int = 60) -> str:
    """Render the export in headless Chrome and return the *live* DOM after
    its JS has run (`--dump-dom`). Results and chips are painted by
    client-side JS, not present in the static markup `_render_html` emits,
    so this is the only way to check what actually renders at rest -- the
    property S5c defect 1 is about ("opens as a wall of chips and nothing
    else")."""
    proc = subprocess.run(
        [
            CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
            "--virtual-time-budget=15000", "--dump-dom", f"file://{path.resolve()}",
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, f"chrome --dump-dom failed: {proc.stderr}"
    return proc.stdout


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_embedder():
    return embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)


def _make_fixture_raster(path: Path, size: int, seed: int = 0) -> Path:
    """Same recipe as test_retrieve.py's own fixture raster: real CRS/GSD
    (EPSG:32636 UTM, 0.1 m/px) so ground-extent maths needs no
    special-casing, non-degenerate random content."""
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
def synthetic_export(real_embedder, tmp_path_factory):
    """A small multi-scale index for a NON-demo AOI, built and exported
    entirely under tmp_path -- proves the exporter takes the AOI as a
    parameter rather than assuming X605_Y3388, and gives the fast
    (F-8/N-6) tests something cheap to build against."""
    data_root = tmp_path_factory.mktemp("data")
    index_root = tmp_path_factory.mktemp("index")
    rel = "synthetic_scene.tif"
    _make_fixture_raster(data_root / rel, size=896, seed=7)
    aoi = "synthetic_scene"
    embed_index.build_index(
        rel, scales=(448, 224, 112), batch_size=32,
        data_root=data_root, index_root=index_root, embedder=real_embedder,
    )
    report = export_html.build_export_html(
        aoi, index_root=index_root, data_root=data_root, embedder=real_embedder,
    )
    return {"aoi": aoi, "data_root": data_root, "index_root": index_root, "report": report}


@pytest.fixture(scope="module")
def production_export(real_embedder):
    """Builds the real deliverable against the real X605_Y3388 index -- see
    module docstring for why this is not the tmp_path-only rule's target."""
    report = export_html.build_export_html(DEMO_AOI, embedder=real_embedder)
    return report


@pytest.fixture(scope="module")
def production_export_leb(real_embedder):
    """Builds the real deliverable against the real `leb` index -- S5c
    defect 2. `leb` is two dates over one footprint, 41,888 tiles; at
    EXPORT_DIM 384 the int8 vectors alone are 16.1 MB, over the cap before
    any basemap exists, so this exercises the EXPORT_DIM fallback ladder
    for real rather than only on a synthetic fixture."""
    report = export_html.build_export_html(LEB_AOI, embedder=real_embedder)
    return report


# --------------------------------------------------------------------------
# F-8 / N-6 -- size cap and external refs. Run against the fast synthetic
# export so these do not each pay for a full production build.
# --------------------------------------------------------------------------


def test_export_is_single_file_under_cap(synthetic_export):
    path = Path(synthetic_export["report"]["out_path"])
    assert path.is_file()
    size = path.stat().st_size  # measured on disk, not the builder's return
    assert size == synthetic_export["report"]["size_bytes"]
    assert size <= export_html.EXPORT_SIZE_CAP_BYTES, (
        f"{path}: {size} bytes exceeds the {export_html.EXPORT_SIZE_CAP_BYTES} byte cap"
    )


class _RefCollector(HTMLParser):
    """Collects every src/href attribute value in the document, by parsing
    (F-8's own instruction), not by eyeballing or regexing."""

    def __init__(self):
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("src", "href") and value is not None:
                self.refs.append(value)


def _is_external(ref: str) -> bool:
    if ref.startswith("data:"):
        return False
    if ref.startswith("#"):
        return False
    # Any scheme (http, https, //cdn...) or bare host/path counts as
    # external/off-file; a standalone export has none of these at all.
    return True


def test_export_has_zero_external_refs(synthetic_export):
    """F-8: assert by parsing, not by eyeballing. Basemap images are never
    static `<img src=...>` markup at all -- they are created in JS from
    embedded `data:` URIs at runtime (`preloadBasemaps` in `UI_JS`) -- so a
    parse of the static document is expected to find zero src/href
    attributes, not merely zero external ones. To guard against the parser
    silently operating on an empty/truncated document instead, this checks
    the document actually has real content (a real element count) rather
    than asserting refs must be non-empty."""
    path = Path(synthetic_export["report"]["out_path"])
    html_text = path.read_text(encoding="utf-8")  # fresh read from disk
    assert len(html_text) > 1000, "export file looks empty/truncated"
    parser = _RefCollector()
    parser.feed(html_text)
    external = [r for r in parser.refs if _is_external(r)]
    assert external == [], f"external src/href found: {external}"


def test_export_size_fails_loudly(synthetic_export, tmp_path):
    """N-6: an impossible byte cap raises, names the computed size and tile
    count, and writes nothing -- even at the smallest tried resolution."""
    data = synthetic_export
    out_path = tmp_path / "should_not_exist" / "export.html"
    with pytest.raises(export_html.ExportSizeError) as excinfo:
        export_html.build_export_html(
            data["aoi"], index_root=data["index_root"], data_root=data["data_root"],
            max_bytes=100, out_path=out_path,
        )
    msg = str(excinfo.value)
    assert "100" in msg
    n_tiles_expected = data["report"]["n_tiles"]
    assert str(n_tiles_expected) in msg, f"expected tile count {n_tiles_expected} named in: {msg}"
    assert not out_path.exists(), "a truncated/partial file was written despite the size failure"
    assert not out_path.parent.exists() or not any(out_path.parent.iterdir())


# --------------------------------------------------------------------------
# S5c defect 2 -- EXPORT_DIM fallback ladder. Basemap reduction alone
# cannot save an AOI whose vectors already exceed the cap before any
# basemap exists; N-6 must still raise when even the smallest (dim,
# basemap) combination does not fit (covered above, unchanged).
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# S5c ADDENDUM -- "legibility has a floor; fidelity does not." The dim
# fallback (above) can starve the basemap to protect a fidelity number
# nobody looks at; these tests are the regression guard for the reordered
# ladder that funds the basemap floor by stepping EXPORT_DIM down first.
# --------------------------------------------------------------------------


def test_basemap_scale_floor(synthetic_export):
    """The finest-scale tile must render at >= MIN_RENDERED_TILE_PX real
    basemap pixels, or the shipped payload/footer must say the floor was
    breached and why (never a silent shortfall)."""
    path = Path(synthetic_export["report"]["out_path"])
    data = export_html.read_export_data(path)
    rendered = data["renderedFinestPx"]
    assert rendered is not None
    if data["basemapFloorBreached"]:
        assert rendered < export_html.MIN_RENDERED_TILE_PX
        html_text = path.read_text(encoding="utf-8")
        lowered = html_text.lower()
        assert "legibility floor" in lowered or "basemap resolution notice" in lowered, (
            "floor breached but the footer does not explain it"
        )
    else:
        assert rendered >= export_html.MIN_RENDERED_TILE_PX - 1e-6, (
            f"finest-scale tile renders at {rendered:.2f}px, below the "
            f"{export_html.MIN_RENDERED_TILE_PX}px floor, with no footer explanation"
        )
    # the report returned in memory must agree with what was written to disk
    report = synthetic_export["report"]
    assert report["rendered_finest_px"] == pytest.approx(rendered, abs=1e-6)
    assert report["basemap_floor_breached"] == data["basemapFloorBreached"]


def test_x605_unaffected_by_basemap_floor(production_export):
    """X605_Y3388 already sits at scale 0.400 (well above the floor) and
    fits at EXPORT_DIM 384 -- the ADDENDUM must not touch it: same dim, same
    top-of-ladder basemap resolution as before the floor was introduced."""
    report = production_export
    assert report["export_dim"] == export_html.EXPORT_DIM_LADDER[0]
    assert report["basemap_max_px"] == export_html.BASEMAP_RESOLUTION_LADDER[0]
    assert not report["basemap_floor_breached"]
    assert report["rendered_finest_px"] >= export_html.MIN_RENDERED_TILE_PX


def test_leb_basemap_meets_legibility_floor(production_export_leb):
    """The regression test for the actual bug: leb's 112px tiles shipped
    from 7 real px (scale 0.063) before this fix -- unreadable, rubble
    indistinguishable from an intact roof. The reordered ladder must fund a
    basemap that clears MIN_RENDERED_TILE_PX by stepping EXPORT_DIM down
    first, not by starving the basemap."""
    report = production_export_leb
    assert report["rendered_finest_px"] >= export_html.MIN_RENDERED_TILE_PX - 1e-6, (
        f"leb's finest-scale tiles render at {report['rendered_finest_px']:.2f}px, "
        f"still below the {export_html.MIN_RENDERED_TILE_PX}px floor"
    )
    assert not report["basemap_floor_breached"]

    path = Path(report["out_path"])
    data = export_html.read_export_data(path)
    assert len(data["sources"]) == 2, "both leb dates must each carry a basemap that meets the floor"
    for src in data["sources"]:
        finest = min(data["scales"])
        rendered = finest * min(src["basemapScaleX"], src["basemapScaleY"])
        assert rendered >= export_html.MIN_RENDERED_TILE_PX - 1e-6, (
            f"{src['relPath']} ({src['date']}): finest tile renders at {rendered:.2f}px"
        )


def test_export_dim_fallback(real_embedder, tmp_path):
    """An AOI that cannot fit at EXPORT_DIM 384 (even at the smallest
    basemap) steps down the dimension ladder and the report records which
    dimension was actually used; an AOI that already fits at 384 is
    unaffected -- it must not be downgraded just because a smaller
    dimension also exists in the ladder. Isolated tmp_path index/out_path so
    this cannot mutate the shared `synthetic_export` fixture other tests
    depend on."""
    data_root = tmp_path / "data"
    index_root = tmp_path / "index"
    data_root.mkdir()
    index_root.mkdir()
    rel = "dim_fallback_scene.tif"
    _make_fixture_raster(data_root / rel, size=896, seed=11)
    aoi = "dim_fallback_scene"
    embed_index.build_index(
        rel, scales=(448, 224, 112), batch_size=32,
        data_root=data_root, index_root=index_root, embedder=real_embedder,
    )

    unaffected = export_html.build_export_html(
        aoi, index_root=index_root, data_root=data_root, embedder=real_embedder,
        out_path=tmp_path / "unaffected.html",
    )
    assert unaffected["export_dim"] == export_html.EXPORT_DIM_LADDER[0], (
        "an AOI that already fits at the top of the dim ladder must not be downgraded: "
        f"got export_dim={unaffected['export_dim']}"
    )

    forced = export_html.build_export_html(
        aoi, index_root=index_root, data_root=data_root, embedder=real_embedder,
        out_path=tmp_path / "forced.html", max_bytes=350_000,
    )
    assert forced["export_dim"] in export_html.EXPORT_DIM_LADDER
    assert forced["export_dim"] < export_html.EXPORT_DIM_LADDER[0], (
        f"expected a step down from {export_html.EXPORT_DIM_LADDER[0]}, got {forced['export_dim']}"
    )
    forced_path = Path(forced["out_path"])
    assert forced_path.stat().st_size <= 350_000
    data = export_html.read_export_data(forced_path)  # fresh read from disk
    assert data["exportDim"] == forced["export_dim"], "the shipped file must name the dimension actually used"


def test_leb_export_builds_under_cap(production_export_leb):
    """The real `leb` AOI -- two dates, 41,888 tiles, 16.1 MB of vectors
    alone at 384-d -- must build under the cap via the dim fallback, and
    both dates must survive in the one file (splitting them destroys the
    2022->2025 destruction comparison that is leb's whole value)."""
    report = production_export_leb
    path = Path(report["out_path"])
    assert path.is_file()
    size = path.stat().st_size  # measured on disk, not the builder's return
    assert size == report["size_bytes"]
    assert size <= export_html.EXPORT_SIZE_CAP_BYTES, (
        f"{path}: {size} bytes exceeds the {export_html.EXPORT_SIZE_CAP_BYTES} byte cap"
    )
    assert report["export_dim"] < export_html.EXPORT_DIM_LADDER[0], (
        "leb is known to exceed the cap at EXPORT_DIM 384 -- the fallback must have stepped down"
    )

    data = export_html.read_export_data(path)  # fresh read from disk
    dates = sorted({s["date"] for s in data["sources"]})
    assert dates == ["2022-10-29", "2025-06-06"], f"expected both leb dates in one file, got {dates}"
    assert len(data["sources"]) == 2, "expected exactly one basemap per date"


# --------------------------------------------------------------------------
# S5c defect 1 -- opens as a wall of chips and nothing else. U-1's intent,
# not only its letter: results must be visible at rest, not only after a
# click, and the landing chip count must be bounded (the filter box still
# reaches the full precomputed set).
# --------------------------------------------------------------------------


@pytest.mark.skipif(CHROME is None, reason="a headless Chrome/Chromium binary is required to render the export")
def test_export_opens_with_results(synthetic_export):
    path = Path(synthetic_export["report"]["out_path"])
    dom = _dump_dom(path)
    assert 'class="result-card"' in dom, "no rendered result cards at rest -- the default query did not run on load"
    assert 'class="scale-row"' in dom, "no rendered scale-row section at rest"
    assert 'class="example-badge"' in dom, "the auto-run query must be labelled as an example, not a user search"
    # the empty placeholder from before the fix must be gone
    assert '<div id="results"></div>' not in dom


def test_export_chip_count_is_curated(synthetic_export):
    assert len(export_html.CURATED_QUERIES) <= 20
    full_texts = set(export_html.precomputed_query_texts())
    assert set(export_html.CURATED_QUERIES) <= full_texts, "curated queries must be a subset of the precomputed set"
    assert len(set(export_html.CURATED_QUERIES)) == len(export_html.CURATED_QUERIES), "duplicate curated query"

    path = Path(synthetic_export["report"]["out_path"])
    data = export_html.read_export_data(path)  # fresh read from disk
    assert len(data["curatedQueryIndices"]) == len(export_html.CURATED_QUERIES)
    assert len(data["curatedQueryIndices"]) <= 20
    # the full set is still shipped and reachable via the filter box / show-all
    # toggle -- only the *default* rendering is curated, not the data.
    assert len(data["queries"]) == len(export_html.PRECOMPUTED_QUERIES)


# --------------------------------------------------------------------------
# U-5 / D-5 -- the resolution caveat must be present verbatim-in-spirit,
# not only in project docs.
# --------------------------------------------------------------------------


def test_resolution_caveat_in_export(synthetic_export):
    path = Path(synthetic_export["report"]["out_path"])
    html_text = path.read_text(encoding="utf-8")
    assert "35" in html_text and "45" in html_text and "cm" in html_text
    assert "coarse class" in html_text.lower()
    assert "presence" in html_text.lower()
    assert "fine attribute" in html_text.lower()
    # D-5: never overstate to attribute-level discrimination.
    forbidden = ["vehicle colour", "vehicle color", "exact model", "license plate", "make and model"]
    lowered = html_text.lower()
    for phrase in forbidden:
        assert phrase not in lowered, f"forbidden attribute-level claim found: {phrase!r}"


# --------------------------------------------------------------------------
# U-1/U-2/U-3/U-6/U-8 structural smoke checks -- cheap, on the synthetic
# export.
# --------------------------------------------------------------------------


def test_no_free_text_affordance(synthetic_export):
    """F-8 amendment: never render a box that looks like it accepts
    arbitrary text, and no question box (a VLM cannot run here either)."""
    path = Path(synthetic_export["report"]["out_path"])
    html_text = path.read_text(encoding="utf-8")
    lowered = html_text.lower()
    # No question-answering affordance at all -- disabled or not (a disabled
    # affordance is worse than none, per the brief): no textarea anywhere,
    # and no input/button whose id/class names a question box. A prose
    # mention explaining that Q&A is local-app-only (honesty, U-5's spirit)
    # is fine and expected -- what must not exist is an interactive element.
    assert "<textarea" not in lowered
    for marker in ('id="question', "id='question", 'class="question', "id=\"qa-", 'id="ask-'):
        assert marker not in lowered, f"a question/QA-box element marker found: {marker!r}"
    # The one text input on the page must be the filter box, explicitly
    # scoped to the precomputed set in its own placeholder text.
    assert 'id="filter-input"' in html_text
    assert "precomputed" in lowered
    # No other free-standing <input type="text"> exists (e.g. a disguised
    # free-text query box or a disabled question box).
    assert html_text.count('type="text"') == 1


def test_query_set_covers_four_vocabularies():
    cats = {c for c, _ in export_html.PRECOMPUTED_QUERIES}
    assert cats == {"small_object", "structure", "terrain", "damage"}
    texts = [t for _, t in export_html.PRECOMPUTED_QUERIES]
    assert len(texts) == len(set(texts)), "duplicate precomputed query text"
    assert 150 <= len(texts) <= 260, f"expected roughly 200 queries, got {len(texts)}"


def test_export_data_has_no_tile_id_strings_but_reconstructs_them(synthetic_export):
    """Packed-array design check: tile ids are not shipped as strings (would
    cost ~hundreds of KB for no display value); `read_export_data` plus the
    JS `makeTileId` (exercised in test_export_query_matches_local) must be
    able to reconstruct one exactly from packed columns."""
    path = Path(synthetic_export["report"]["out_path"])
    data = export_html.read_export_data(path)
    for scale_key, cols in data["perScale"].items():
        assert "tileId" not in cols and "tile_id" not in cols
        assert cols["n"] > 0


# --------------------------------------------------------------------------
# The test that matters most: the export's own shipped JS, run under Node,
# must produce the same top-10-per-scale as `retrieve.rank_per_scale` fed
# the exact same (dequantised int8, PCA-projected) vectors -- i.e. the
# ranking `retrieve.py`'s trusted, already-tested algorithm would produce if
# it searched the exported basis instead of full precision.
# --------------------------------------------------------------------------

_NODE_HARNESS_HEADER = """
globalThis.atob = function (b64) {
  return Buffer.from(b64, "base64").toString("binary");
};
"""

_NODE_HARNESS_FOOTER = """
const fs = require("fs");
const DATA = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const decoded = decodeAllScales(DATA);
const queryIndex = parseInt(process.argv[3], 10);
const dateFilter = process.argv[4] === "__none__" ? null : process.argv[4];
const result = rankQuery(DATA, decoded, queryIndex, dateFilter);
process.stdout.write(JSON.stringify(result));
"""


def _run_js_rank_query(export_path: Path, data_path: Path, query_index: int, tmp_path: Path, date_filter=None):
    ranking_js = export_html.read_ranking_core_js(export_path)
    script = _NODE_HARNESS_HEADER + ranking_js + _NODE_HARNESS_FOOTER
    script_path = tmp_path / f"harness_{query_index}.js"
    script_path.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [NODE, str(script_path), str(data_path), str(query_index), date_filter or "__none__"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    return json.loads(proc.stdout)


def _python_reference_ranking(aoi: str, index_root, data_root, query_text: str, data: dict, background: dict):
    """`retrieve.rank_per_scale`, run over the exact exported (dequantised
    int8 PCA-384) vectors and the exact dequantised query vector the export
    shipped for `query_text` -- the "local app's exported-basis ranking"
    the brief names."""
    corpus = retrieve.load_corpus(aoi, index_root=index_root, data_root=data_root)
    loaded_export = export_basis.load_export(aoi, index_root=index_root)
    exported_f32 = export_basis.exported_vectors_f32(loaded_export)
    shadow = retrieve.Corpus(
        aoi=corpus.aoi, manifest=corpus.manifest, vectors=exported_f32,
        scale_indices=corpus.scale_indices, locations=corpus.locations,
        ground_extent_m=corpus.ground_extent_m,
    )
    q_idx = [t for _, t in export_html.PRECOMPUTED_QUERIES].index(query_text)
    q_entry = data["queries"][q_idx]
    q_i8 = np.frombuffer(base64.b64decode(q_entry["vectorB64"]), dtype=np.int8)
    qvec = q_i8.astype(np.float32) * data["queryScale"]
    return retrieve.rank_per_scale(shadow, qvec, top_k=export_html.TOP_K, background=background)


@pytest.mark.skipif(NODE is None, reason="node is required to execute the shipped ranking JS")
def test_export_query_matches_local(production_export, tmp_path):
    export_path = Path(production_export["out_path"])
    data = export_html.read_export_data(export_path)  # fresh read from disk
    data_path = tmp_path / "export_data.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")

    background = {int(k): v for k, v in data["background"].items()}

    sample_queries = ["a car", "a building with a flat roof", "sand", "palm trees", "rubble"]
    all_texts = [t for _, t in export_html.PRECOMPUTED_QUERIES]
    for q in sample_queries:
        assert q in all_texts, f"{q!r} must be one of the precomputed queries for this test to mean anything"

    mismatches = []
    for query_text in sample_queries:
        q_idx = all_texts.index(query_text)
        js_result = _run_js_rank_query(export_path, data_path, q_idx, tmp_path)
        py_result = _python_reference_ranking(
            DEMO_AOI, None, None, query_text, data, background,
        )
        for scale in data["scales"]:
            js_ids = [r["tileId"] for r in js_result["rankings"][str(scale)]["results"]]
            py_ids = [r["tile_id"] for r in py_result["rankings"][scale]["results"]]
            if js_ids != py_ids:
                mismatches.append((query_text, scale, js_ids, py_ids))

    assert not mismatches, (
        "export ranking (JS, run under Node) disagrees with the local app's "
        f"exported-basis ranking (retrieve.rank_per_scale over the same exported "
        f"vectors) for: {mismatches}"
    )


@pytest.mark.skipif(NODE is None, reason="node is required to execute the shipped ranking JS")
def test_js_confidence_band_matches_python():
    """Fast, isolated check of the ported `confidenceBand` (no export build
    needed): the JS band/message/percentile must agree with
    `retrieve.confidence_band` bit-for-bit at several sample points,
    including the two edges (33rd/67th percentile cut points)."""
    background = [0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.35]
    ranking_js = (
        export_html.RANKING_CORE_JS
        .replace("__CONF_LOW__", repr(retrieve.CONFIDENCE_LOW_MAX_PERCENTILE))
        .replace("__CONF_HIGH__", repr(retrieve.CONFIDENCE_HIGH_MIN_PERCENTILE))
        .replace("__TOP_K__", "10")
    )
    footer = """
const bg = JSON.parse(process.argv[2]);
const top = parseFloat(process.argv[3]);
process.stdout.write(JSON.stringify(confidenceBand(top, bg)));
"""
    script = ranking_js + footer
    tmp = Path(export_html.__file__).resolve().parent  # any writable dir works; use a temp file instead
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        script_path = Path(d) / "band.js"
        script_path.write_text(script, encoding="utf-8")
        for top_score in [0.05, 0.1, 0.19, 0.2, 0.29, 0.3, 0.4]:
            proc = subprocess.run(
                [NODE, str(script_path), json.dumps(background), str(top_score)],
                capture_output=True, text=True, timeout=30,
            )
            assert proc.returncode == 0, proc.stderr
            js_band = json.loads(proc.stdout)
            py_band = retrieve.confidence_band(top_score, background)
            assert js_band["band"] == py_band["band"], (top_score, js_band, py_band)
            assert js_band["message"] == py_band["message"]
            assert js_band["percentile"] == pytest.approx(py_band["percentile"], abs=1e-9)


@pytest.mark.skipif(NODE is None, reason="node is required to execute the shipped ranking JS")
def test_js_make_tile_id_matches_tiling():
    """`makeTileId` in the shipped JS must reconstruct the exact same id
    `tiling.make_tile_id` produces (D-4/F-10's round trip), since the export
    never ships tile_id strings -- only col/row-derivable pixel offsets."""
    ranking_js = (
        export_html.RANKING_CORE_JS
        .replace("__CONF_LOW__", "33.0")
        .replace("__CONF_HIGH__", "67.0")
        .replace("__TOP_K__", "10")
    )
    footer = """
process.stdout.write(makeTileId(process.argv[2], process.argv[3], process.argv[4], parseInt(process.argv[5],10), parseInt(process.argv[6],10), parseInt(process.argv[7],10)));
"""
    script = ranking_js + footer
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        script_path = Path(d) / "tileid.js"
        script_path.write_text(script, encoding="utf-8")
        cases = [
            ("X605_Y3388", "X605_Y3388.tif", "unknown", 448, 10, 0),
            ("X605_Y3388", "X605_Y3388.tif", "unknown", 112, 91, 91),
            ("leb", "leb/2022-10-29.tif", "2022-10-29", 224, 3, 400),
        ]
        for aoi, src, date, scale, col, row in cases:
            proc = subprocess.run(
                [NODE, str(script_path), aoi, src, date, str(scale), str(col), str(row)],
                capture_output=True, text=True, timeout=30,
            )
            assert proc.returncode == 0, proc.stderr
            js_id = proc.stdout
            py_id = tiling.make_tile_id(aoi, src, date, scale, col, row)
            assert js_id == py_id


# --------------------------------------------------------------------------
# AOI is a parameter (S5b brief), not a hardcoded string.
# --------------------------------------------------------------------------


def test_export_html_path_is_parametrized_by_aoi(tmp_path):
    p1 = export_html.export_html_path("aoi_one", index_root=tmp_path)
    p2 = export_html.export_html_path("aoi_two", index_root=tmp_path)
    assert p1 != p2
    assert "aoi_one" in str(p1)
    assert "aoi_two" in str(p2)


def test_build_export_html_does_not_hardcode_demo_aoi(synthetic_export):
    """The synthetic fixture already builds a non-demo AOI end to end; this
    just asserts the report reflects that AOI, not X605_Y3388."""
    assert synthetic_export["report"]["aoi"] == synthetic_export["aoi"]
    assert synthetic_export["aoi"] != DEMO_AOI
