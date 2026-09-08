"""Standalone offline HTML export -- brief S5b, spec F-8, N-6, D-5, U-1..U-8.

**F-8's amendment (read before touching this module):** RemoteCLIP's text
tower is 123.7 M params -- 247 MB fp16, 124 MB int8 -- against this file's
16 MB cap. 15.5x the whole budget; no compression closes it. So this export
carries **no text tower and no free-text search**. What it carries instead:

  * ~200 precomputed query VECTORS (`PRECOMPUTED_QUERIES` below), spanning
    the four vocabularies the owner named (small objects, structures,
    terrain, damage) -- embedded once, here, at build time, through the same
    RemoteCLIP text tower `retrieve.embed_query` uses.
  * the **whole AOI's tile corpus**, in the S4 export basis
    (`export_basis.py`: PCA to 384-d, int8) -- not just precomputed answers.

Ranking therefore still happens live, but entirely client-side, as a plain
dot product between one (fixed) query vector and the shipped tile vectors --
no model, no network, no server. This is why the file ships tile vectors at
all rather than only the 200 queries' top-10 answers: it is what lets U-6's
date filter actually re-rank within the filtered subset (mirroring
`retrieve.rank_per_scale`'s bbox-filter design) instead of merely trimming a
frozen list, and it is what `test_export_query_matches_local` exercises for
real -- the shipped JS, not a Python stand-in, computes the ranking that
test checks against `retrieve.rank_per_scale` run over the same exported
vectors.

**Never fake free text.** The page has no text input that reaches a model.
The filter box only ever narrows the fixed 200-phrase list by substring/word
overlap (`nearestQueries` in `RANKING_CORE_JS`) -- a lexical heuristic, which
is the honest option, since a semantic "nearest query" would itself require
the text tower this file cannot carry.

**Confidence band uses an export-space background, not `retrieve.py`'s
`background.json`.** `retrieve.build_background` measures each background
query's top-1 score against the **full-precision (768-d)** corpus; this
export's queries are scored against the **384-d PCA/int8** corpus instead,
and PCA truncation shrinks dot-product magnitude systematically (it discards
variance, so `||projected|| < ||full||` almost always). Comparing an
export-space top score against a full-precision background distribution
would read as chronically "low confidence" for a reason that has nothing to
do with the query -- a representation mismatch dressed up as a finding. So
`_build_export_background` below re-embeds `retrieve.BACKGROUND_QUERIES` (no
new list -- reuse, not duplication) and re-measures top-1 per scale in the
**same exported basis** the page actually searches. Percentile cut points
(`CONFIDENCE_LOW_MAX_PERCENTILE`/`CONFIDENCE_HIGH_MIN_PERCENTILE`) are
imported from `retrieve.py`, not re-declared, and are baked into the shipped
JS as literals (there is no Python at runtime to import them from).

**Basemap, not thumbnails (the brief's central design call).** One
downsampled JPEG per source scene, embedded as a `data:` URI; every result
tile is a client-side `<canvas>` crop of it. A tile id carries **grid
indices** (`tiling.make_tile_id`/`parse_tile_id`): pixel offset is
``col * scale`` / ``row * scale``, so the crop rectangle on the basemap is
``pixel_offset * basemap_scale`` -- computable from one image, never a
per-tile file. `tile_id` strings themselves are not shipped either (would
cost ~480 KB for 9,631 tiles of no display value) -- `makeTileId` in
`RANKING_CORE_JS` reconstructs them from ``col = pxOffsetX / scale`` for the
F-10 round-trip, mirroring `tiling.make_tile_id` exactly (same ``::``
join, same 5-digit zero pad).

**N-6 is enforced by measurement, not by budgeting on paper.** The whole
page is assembled in memory; only if its *actual* byte size is <= the cap is
anything written to disk (`build_export_html`'s ``max_bytes``). If it does
not fit, resolution degrades first (`BASEMAP_RESOLUTION_LADDER`) -- tiles and
precision are never dropped to make a number close, per the brief. If even
the smallest rung does not fit, `ExportSizeError` names the computed size and
the tile count, and nothing is written.

Layout (gitignored under the index root, alongside the S4 export basis)::

    <index_root>/export/<aoi>/export.html

Environment: every invocation must be prefixed inline, every time --

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 AERIAL_DATA_ROOT=<path> <python> ...

(see INSTRUCTIONS.md for the interpreter path on the dev machine).
"""

from __future__ import annotations

import base64
import datetime
import io
import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image

import config
import embed_index
import embedders
import export_basis
import retrieve

log = logging.getLogger(__name__)

EXPORT_HTML_NAME = "export.html"

#: N-6's hard cap.
EXPORT_SIZE_CAP_BYTES = 16 * 1024 * 1024

#: Degrade basemap resolution, not tiles or precision, when oversize (brief:
#: "it degrades gracefully; dropping tiles or precision does not"). Tried
#: largest-first; the first rung that fits under the cap wins.
BASEMAP_RESOLUTION_LADDER: tuple[int, ...] = (4096, 3200, 2560, 2048, 1600, 1280, 1024, 768, 512)

DEFAULT_JPEG_QUALITY = 82

#: How many results a client-side query shows per scale -- the same constant
#: as the local app's TOP_K (F-4's "top-TOP_K per scale"), so a row here
#: never shows a different depth than the local app would for the same rank.
TOP_K = retrieve.TOP_K


class ExportSizeError(RuntimeError):
    """N-6: the assembled export exceeds the cap even at the smallest tried
    basemap resolution. Raised before anything is written -- never a
    truncated file on disk."""


# --------------------------------------------------------------------------
# ~200 precomputed queries, spanning the four vocabularies CLAUDE.md names
# (problem statement: "small objects (tents, cars), structures (buildings,
# flat roofs), terrain (dirt road, sand, palm trees), and damage (rubble,
# debris, collapsed roof)"). Picked a priori for vocabulary coverage, not
# tuned against this AOI's scores -- some will genuinely have nothing to
# find on `X605_Y3388` (no damage vocabulary is answerable there; see
# CLAUDE.md's ratification note), and the confidence band is what is
# supposed to say so, honestly, not a curation pass that quietly drops them.
# --------------------------------------------------------------------------

_SMALL_OBJECTS: tuple[str, ...] = (
    "a car", "cars", "a parked car", "several parked cars", "a row of cars",
    "a truck", "a pickup truck", "a delivery van", "a van", "a bus",
    "a motorcycle", "a bicycle", "bicycles", "a boat", "boats",
    "a tent", "tents", "a cluster of tents", "a refugee camp",
    "a shipping container", "shipping containers", "a dumpster",
    "a water tank", "a fuel tank", "a generator", "a satellite dish",
    "an antenna", "a solar panel", "solar panels", "a swimming pool",
    "a small swimming pool", "a trampoline", "a picnic table", "an umbrella",
    "a playground", "playground equipment", "a shed", "a small shed",
    "a trailer", "a construction vehicle", "a bulldozer", "an excavator",
    "a crane", "a tower crane", "scaffolding", "a stack of pallets",
    "a pile of pipes", "a flagpole", "a light pole",
    "a group of people", "a herd of animals", "livestock", "a pile of tires",
    "a stack of shipping crates", "parked motorcycles", "a fire truck",
    "an ambulance", "a school bus",
)

_STRUCTURES: tuple[str, ...] = (
    "a building", "buildings", "a house", "houses", "a residential building",
    "an apartment building", "a flat roof", "a building with a flat roof",
    "a flat-roofed building", "a pitched roof", "a red tiled roof",
    "a warehouse", "an industrial building", "a factory", "a mosque",
    "a church", "a school", "a hospital", "a parking garage", "a parking lot",
    "an empty parking lot", "a stadium", "a bridge", "an overpass",
    "a water tower", "a silo", "grain silos", "a hangar",
    "an airport terminal", "a runway", "a taxiway", "a paved road",
    "a highway", "an intersection", "a roundabout", "a fence",
    "a perimeter fence", "a wall", "a stone wall", "a courtyard",
    "a driveway", "a gated compound", "a walled compound", "a greenhouse",
    "greenhouses", "a barn", "a garage", "a rooftop with air conditioning units",
    "a rooftop with mechanical equipment", "a construction site",
    "an unfinished building", "a building under construction",
    "scaffolded building", "a multi-story building", "a cemetery",
    "a graveyard", "a tower", "an office building", "a shopping mall",
)

_TERRAIN: tuple[str, ...] = (
    "a dirt road", "an unpaved road", "a gravel road",
    "sand", "a sandy area", "a beach", "dunes", "sand dunes", "a desert",
    "palm trees", "a palm grove", "trees", "a group of trees", "a forest",
    "a wooded area", "hedges", "a hedge row", "grass", "a grassy field",
    "farmland", "agricultural fields", "a field", "cultivated fields",
    "a vineyard", "an orchard", "a river", "a stream", "a lake", "a pond",
    "a reservoir", "a coastline", "a cliff", "a hillside", "a mountain",
    "mountains", "a rocky area", "bare soil", "exposed dirt",
    "a dry riverbed", "a wadi", "an olive grove", "shrubland", "wetlands",
    "a canal", "an irrigation channel", "terraced fields", "a quarry",
    "a mining site", "snow", "a snowy area", "sparse vegetation",
    "dense vegetation", "open ground", "a rocky coastline",
    "a barren landscape", "scrubland", "a salt flat",
)

_DAMAGE: tuple[str, ...] = (
    "rubble", "a pile of rubble", "debris", "scattered debris",
    "a collapsed building", "a collapsed roof", "a partially collapsed building",
    "a destroyed building", "destroyed buildings", "a demolished building",
    "bomb damage", "war damage", "a bomb crater", "a crater", "craters",
    "a damaged road", "a cratered road", "burned buildings",
    "a burned-out building", "scorch marks", "fire damage",
    "a damaged bridge", "a collapsed bridge", "wreckage", "a wrecked vehicle",
    "damaged vehicles", "debris beside a road", "rubble beside a road",
    "flattened buildings", "an area of destruction", "damaged rooftops",
    "structural damage", "a demolished house", "ruins", "a ruined structure",
    "an abandoned building", "a shelled building", "shrapnel damage",
    "a hole in a roof", "exposed rebar", "a broken wall", "a toppled wall",
    "a damaged fence", "an impact crater", "blast damage", "a razed area",
    "a flattened neighborhood", "piles of debris", "a demolition site",
    "earthquake damage", "a partially standing wall", "a roof with a hole",
)

#: (category, text) pairs, in a fixed order -- deterministic across builds
#: (no set, no dict iteration) so query index N always names the same phrase.
PRECOMPUTED_QUERIES: tuple[tuple[str, str], ...] = tuple(
    [("small_object", t) for t in _SMALL_OBJECTS]
    + [("structure", t) for t in _STRUCTURES]
    + [("terrain", t) for t in _TERRAIN]
    + [("damage", t) for t in _DAMAGE]
)

CATEGORY_LABELS = {
    "small_object": "Small objects",
    "structure": "Structures",
    "terrain": "Terrain",
    "damage": "Damage",
}


def precomputed_query_texts() -> list[str]:
    return [t for _, t in PRECOMPUTED_QUERIES]


def export_html_path(aoi: str, index_root: Path | None = None) -> Path:
    return export_basis.export_dir(aoi, index_root) / EXPORT_HTML_NAME


# --------------------------------------------------------------------------
# Query-vector quantisation -- same scheme as `export_basis._quantize_int8`
# (symmetric int8, one shared scale over the whole set), applied to the
# *projected* (384-d) precomputed query vectors rather than the tile corpus.
# A separate scale from the tile corpus's own `int8_scale` on purpose: query
# vectors and tile vectors are drawn from different distributions (tiles are
# renormalised photographic content; queries are short-phrase text
# embeddings), so sharing one scale would waste int8's range on whichever
# set has the smaller peak.
# --------------------------------------------------------------------------


def _quantize_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
    peak = float(np.abs(arr).max()) if arr.size else 0.0
    scale = peak / export_basis.INT8_MAX if peak > 0 else 1.0
    q = np.clip(np.round(arr / scale), -export_basis.INT8_MAX, export_basis.INT8_MAX).astype(np.int8)
    return q, scale


def _embed_texts_batch(texts: Sequence[str], embedder=None) -> np.ndarray:
    """Batch text-tower embedding for many short phrases at once -- the same
    unit-norm guarantee `retrieve.embed_query` enforces per call, checked
    vectorised here rather than by looping that function (this module embeds
    ~250 phrases at build time; a Python-level loop of 250 single-item
    forward passes is needless GPU round-trip overhead for no benefit)."""
    emb = embedder or embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)
    vecs = np.asarray(emb.embed_texts(list(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    bad = np.where(np.abs(norms - 1.0) > 1e-5)[0]
    if bad.size:
        i = int(bad[0])
        raise retrieve.UnitNormError(
            f"_embed_texts_batch: {bad.size} of {len(texts)} embeddings are not unit-norm "
            f"(first offender {texts[i]!r}: ||q|| = {norms[i]})"
        )
    return vecs


def build_query_vectors(components: np.ndarray, embedder=None) -> dict:
    """Embed `PRECOMPUTED_QUERIES` through the text tower once, project
    through the AOI's export basis (`export_basis.project_query` -- the one
    function both the S4 export and this module use, so "projects
    identically" holds structurally), quantise to int8 with one shared
    scale. Returns int8 vectors (Q, EXPORT_DIM), the scale, and the
    (category, text) pairs in the same row order."""
    texts = precomputed_query_texts()
    qvecs_768 = _embed_texts_batch(texts, embedder=embedder)
    projected = export_basis.project_query(qvecs_768, components).astype(np.float32)
    q_i8, scale = _quantize_int8(projected)
    return {"vectors_i8": q_i8, "scale": scale, "queries": list(PRECOMPUTED_QUERIES)}


# --------------------------------------------------------------------------
# U-3's confidence band, in the export's own coordinate system -- see the
# module docstring for why this does not reuse `retrieve.load_background`'s
# on-disk full-precision numbers verbatim.
# --------------------------------------------------------------------------


def _build_export_background(
    corpus: "retrieve.Corpus", exported_vectors: np.ndarray, components: np.ndarray, embedder=None
) -> dict[int, list[float]]:
    """Per scale: each of `retrieve.BACKGROUND_QUERIES`' top-1 score against
    the **exported** (384-d, dequantised) corpus -- the representation the
    shipped page actually ranks against, not the full-precision one
    `retrieve.build_background` measures."""
    qvecs_768 = _embed_texts_batch(retrieve.BACKGROUND_QUERIES, embedder=embedder)
    projected = export_basis.project_query(qvecs_768, components).astype(np.float32)
    out: dict[int, list[float]] = {}
    for scale in corpus.manifest["scales"]:
        idx = corpus.scale_indices[scale]
        vecs = exported_vectors[idx]
        scores = projected @ vecs.T  # (Q, n_scale)
        out[scale] = [float(s) for s in scores.max(axis=1)]
    return out


# --------------------------------------------------------------------------
# Per-tile payload -- packed typed arrays (brief: "consider a packed array,
# not per-tile JSON objects"), grouped by scale in the same row order as the
# exported int8 vectors for that scale, so index i in one array is index i
# in every other array for that scale.
# --------------------------------------------------------------------------


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _build_sources_and_tiles(corpus: "retrieve.Corpus", vectors_i8_full: np.ndarray) -> tuple[list[dict], dict]:
    """`sources` (one entry per distinct source_file: relPath, date) and, per
    scale, packed columns (vectors int8, srcIdx, pxOffsetX/Y int16, bbox
    float32 x4) in the export vector's own row order.

    `vectors_i8_full` is `export_basis.load_export(...)["vectors_i8"]` --
    already int8, already scaled by the ONE global `int8_scale` in
    `basis.json` -- sliced per scale here, never dequantised and
    requantised (that would throw away precision twice for nothing: the
    shipped bytes are exactly the S4 export's own bytes).
    """
    source_order: list[str] = []
    source_lookup: dict[str, int] = {}
    sources: list[dict] = []
    for t in corpus.manifest["tiles"]:
        loc = corpus.locations[t["tile_id"]]
        sf = loc["source_file"]
        if sf not in source_lookup:
            source_lookup[sf] = len(source_order)
            source_order.append(sf)
            sources.append({"relPath": sf, "date": loc["date"]})

    per_scale: dict[int, dict] = {}
    for scale in corpus.manifest["scales"]:
        idx = corpus.scale_indices[scale]
        n = idx.size
        tile_ids = [corpus.manifest["tiles"][i]["tile_id"] for i in idx]
        locs = [corpus.locations[tid] for tid in tile_ids]
        src_idx = np.array([source_lookup[l["source_file"]] for l in locs], dtype=np.uint8)
        px_x = np.array([l["px_offset_x"] for l in locs], dtype=np.int16)
        px_y = np.array([l["px_offset_y"] for l in locs], dtype=np.int16)
        bbox = np.array([l["bbox_lonlat"] for l in locs], dtype=np.float32)  # (n, 4)
        vecs_i8 = vectors_i8_full[idx]
        per_scale[scale] = {
            "n": int(n),
            "vectorsB64": _b64(vecs_i8),
            "srcIdxB64": _b64(src_idx),
            "pxOffsetXB64": _b64(px_x),
            "pxOffsetYB64": _b64(px_y),
            "bboxMinLonB64": _b64(bbox[:, 0]),
            "bboxMinLatB64": _b64(bbox[:, 1]),
            "bboxMaxLonB64": _b64(bbox[:, 2]),
            "bboxMaxLatB64": _b64(bbox[:, 3]),
        }
    return sources, per_scale


# --------------------------------------------------------------------------
# Basemap -- one downsampled JPEG per source scene (never per-tile
# thumbnails). Header+decimated read only; the raster is opened read-only
# (D-3) and never written to.
# --------------------------------------------------------------------------


def _build_basemap(rel_path: str, data_root: Path, max_px: int, jpeg_quality: int) -> dict:
    root = data_root or config.get_data_root()
    with rasterio.open(root / rel_path) as ds:
        width, height = ds.width, ds.height
        long_side = max(width, height)
        factor = min(1.0, max_px / long_side) if long_side else 1.0
        out_w = max(1, round(width * factor))
        out_h = max(1, round(height * factor))
        band_count = min(3, ds.count)
        arr = ds.read(
            indexes=list(range(1, band_count + 1)),
            out_shape=(band_count, out_h, out_w),
            resampling=Resampling.average,
        )
    arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if band_count == 1:
        img = Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
    else:
        img = Image.fromarray(arr[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    jpeg_bytes = buf.getvalue()
    return {
        "orig_width": width,
        "orig_height": height,
        "basemap_width": out_w,
        "basemap_height": out_h,
        "basemap_scale_x": out_w / width if width else 1.0,
        "basemap_scale_y": out_h / height if height else 1.0,
        "jpeg_bytes": len(jpeg_bytes),
        "data_url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii"),
    }


# --------------------------------------------------------------------------
# The shipped JS. Split in two, deliberately:
#
#   RANKING_CORE_JS -- pure functions, no `document`/`window` reference
#   anywhere in it. This is what `test_export_query_matches_local` runs
#   under plain Node (run directly, `atob` polyfilled by the test harness)
#   to get the *actual* shipped ranking
#   logic's answer, not a Python reimplementation's. It is embedded in the
#   page verbatim, inside its own <script id="ranking-core"> tag, so what
#   the test extracts and what the browser runs are byte-identical.
#
#   UI_JS -- wires RANKING_CORE_JS's functions to the DOM: renders the
#   example/filter list, draws canvas crops from the basemap, shows the
#   confidence band and the U-5 resolution caveat, handles the date filter.
#
# Placeholders (__UPPER_SNAKE__) are substituted by `_render_html` rather
# than via `str.format`, so JS's own `{}` never needs escaping.
# --------------------------------------------------------------------------

RANKING_CORE_JS = r"""
"use strict";

// ---- base64 -> typed array -----------------------------------------------
function b64ToUint8(b64) {
  var bin = atob(b64);
  var len = bin.length;
  var bytes = new Uint8Array(len);
  for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
function decodeInt8(b64) { return new Int8Array(b64ToUint8(b64).buffer); }
function decodeUint8(b64) { return b64ToUint8(b64); }
function decodeInt16(b64) { return new Int16Array(b64ToUint8(b64).buffer); }
function decodeFloat32(b64) { return new Float32Array(b64ToUint8(b64).buffer); }

// ---- D-4/F-10 tile id round trip: id = aoi::source::date::scale::col::row,
// pixel offset = index * scale (CLAUDE.md's own worked example of the
// trap). Mirrors tiling.make_tile_id exactly: 5-digit zero-padded col/row.
function zeroPad5(n) {
  var s = String(n);
  while (s.length < 5) s = "0" + s;
  return s;
}
function makeTileId(aoi, sourceFile, date, scale, col, row) {
  return aoi + "::" + sourceFile + "::" + date + "::" + scale + "::" + zeroPad5(col) + "::" + zeroPad5(row);
}

// ---- decode one scale's packed columns ------------------------------------
function decodeScaleData(scaleObj) {
  return {
    n: scaleObj.n,
    vectors: decodeInt8(scaleObj.vectorsB64),
    srcIdx: decodeUint8(scaleObj.srcIdxB64),
    pxOffsetX: decodeInt16(scaleObj.pxOffsetXB64),
    pxOffsetY: decodeInt16(scaleObj.pxOffsetYB64),
    bboxMinLon: decodeFloat32(scaleObj.bboxMinLonB64),
    bboxMinLat: decodeFloat32(scaleObj.bboxMinLatB64),
    bboxMaxLon: decodeFloat32(scaleObj.bboxMaxLonB64),
    bboxMaxLat: decodeFloat32(scaleObj.bboxMaxLatB64),
  };
}
function decodeAllScales(DATA) {
  var out = {};
  for (var i = 0; i < DATA.scales.length; i++) {
    var scale = DATA.scales[i];
    out[scale] = decodeScaleData(DATA.perScale[String(scale)]);
  }
  return out;
}

// ---- F-4: exact brute-force cosine, one ranking per scale (int8 tile
// vectors dequantised via one shared DATA.tileScale, the query vector via
// its own DATA.queryScale -- combine both scales once per query rather than
// dequantising 9,631 vectors to float on every click).
function scoreScale(sd, qVec, tileScale, queryScale) {
  var n = sd.n, dim = qVec.length;
  var scores = new Float64Array(n);
  var vectors = sd.vectors;
  var combined = tileScale * queryScale;
  for (var i = 0; i < n; i++) {
    var acc = 0;
    var base = i * dim;
    for (var d = 0; d < dim; d++) acc += vectors[base + d] * qVec[d];
    scores[i] = acc * combined;
  }
  return scores;
}

function topKIndices(scores, k, filterFn) {
  var idxs = [];
  for (var i = 0; i < scores.length; i++) {
    if (!filterFn || filterFn(i)) idxs.push(i);
  }
  idxs.sort(function (a, b) { return scores[b] - scores[a]; });
  return idxs.slice(0, k);
}

// ---- U-3: calibrated confidence BAND, never a present/absent boolean --
// mirrors retrieve.confidence_band exactly, including its wording. Cut
// points are baked in from retrieve.py's own constants (see _render_html),
// not re-tuned here.
function confidenceBand(topScore, backgroundScores) {
  if (!backgroundScores || backgroundScores.length === 0) {
    return { percentile: null, band: "unknown", message: "no background reference built for this AOI -- run build_background" };
  }
  var countLE = 0;
  for (var i = 0; i < backgroundScores.length; i++) if (backgroundScores[i] <= topScore) countLE++;
  var percentile = (countLE / backgroundScores.length) * 100;
  var band, message;
  if (percentile < __CONF_LOW__) {
    band = "low";
    message = "score is unremarkable against the background set -- may not be present";
  } else if (percentile < __CONF_HIGH__) {
    band = "medium";
    message = "score is inconclusive against the background set -- inspect the tiles directly";
  } else {
    band = "high";
    message = "score stands out against the background set -- not a presence guarantee";
  }
  return { percentile: percentile, band: band, message: message };
}

// ---- F-4 + F-6 + F-10 + U-3, all together: rank one precomputed query
// against the shipped corpus. `decoded` is `decodeAllScales(DATA)`'s
// result (decode once at load, reuse on every click). `dateFilter`, if not
// null, restricts the *candidates* a scale's top-K is drawn from -- but
// never the confidence band, which is always computed over the scale's
// full, unfiltered set (retrieve.rank_per_scale's own rule, mirrored here).
function rankQuery(DATA, decoded, queryIndex, dateFilter) {
  var q = DATA.queries[queryIndex];
  var qVec = decodeInt8(q.vectorB64);
  var rankings = {};
  for (var s = 0; s < DATA.scales.length; s++) {
    var scale = DATA.scales[s];
    var sd = decoded[scale];
    var scores = scoreScale(sd, qVec, DATA.tileScale, DATA.queryScale);

    var topScoreFull = -Infinity;
    for (var i = 0; i < scores.length; i++) if (scores[i] > topScoreFull) topScoreFull = scores[i];
    var bg = (DATA.background && DATA.background[String(scale)]) || null;
    var confidence = confidenceBand(topScoreFull, bg);

    var filterFn = null;
    if (dateFilter) {
      filterFn = (function (sdInner) {
        return function (i) { return DATA.sources[sdInner.srcIdx[i]].date === dateFilter; };
      })(sd);
    }
    var top = topKIndices(scores, __TOP_K__, filterFn);
    var results = [];
    for (var r = 0; r < top.length; r++) {
      var idx = top[r];
      var src = DATA.sources[sd.srcIdx[idx]];
      var col = Math.round(sd.pxOffsetX[idx] / scale);
      var row = Math.round(sd.pxOffsetY[idx] / scale);
      results.push({
        rank: r + 1,
        score: scores[idx],
        scale: scale,
        tileId: makeTileId(DATA.aoi, src.relPath, src.date, scale, col, row),
        sourceFile: src.relPath,
        date: src.date,
        srcIdx: sd.srcIdx[idx],
        pxOffsetX: sd.pxOffsetX[idx],
        pxOffsetY: sd.pxOffsetY[idx],
        bbox: [sd.bboxMinLon[idx], sd.bboxMinLat[idx], sd.bboxMaxLon[idx], sd.bboxMaxLat[idx]],
      });
    }
    rankings[scale] = {
      scale: scale,
      groundExtentM: DATA.groundExtentM[String(scale)],
      results: results,
      confidence: confidence,
      candidateCount: (dateFilter ? results.length : sd.n),
    };
  }
  return { query: q.text, category: q.category, rankings: rankings };
}

// ---- F-8 amendment: type-to-filter over the fixed set, never free text.
// A lexical heuristic (substring + shared-word overlap) -- not a semantic
// "nearest embedding", which would need the text tower this file cannot
// carry. Used only to suggest phrases when a typed query matches none.
function matchesFilter(text, needle) {
  return text.toLowerCase().indexOf(needle) !== -1;
}
function similarityScore(needle, candidate) {
  var q = needle.toLowerCase().trim();
  var c = candidate.toLowerCase();
  if (!q) return 0;
  var score = 0;
  if (c.indexOf(q) !== -1) score += 10;
  var qWords = q.split(/\s+/).filter(function (w) { return w.length > 0; });
  var cWords = c.split(/\s+/);
  for (var i = 0; i < qWords.length; i++) {
    for (var j = 0; j < cWords.length; j++) {
      if (cWords[j] === qWords[i]) score += 3;
      else if (cWords[j].indexOf(qWords[i]) !== -1 || qWords[i].indexOf(cWords[j]) !== -1) score += 0.5;
    }
  }
  return score;
}
function nearestQueries(DATA, needle, limit) {
  var scored = DATA.queries.map(function (q, i) { return { i: i, s: similarityScore(needle, q.text) }; });
  scored.sort(function (a, b) { return b.s - a.s; });
  var out = [];
  for (var k = 0; k < scored.length && out.length < limit; k++) {
    if (scored[k].s > 0) out.push(DATA.queries[scored[k].i].text);
  }
  return out;
}
"""


UI_JS = r"""
(function () {
  var DATA = JSON.parse(document.getElementById("export-data").textContent);
  var decoded = decodeAllScales(DATA);

  var elFilter = document.getElementById("filter-input");
  var elChips = document.getElementById("query-chips");
  var elNoMatch = document.getElementById("no-match");
  var elResults = document.getElementById("results");
  var elDateSelect = document.getElementById("date-filter");
  var elFilterBar = document.getElementById("filter-bar");
  var elFilterBarText = document.getElementById("filter-bar-text");
  var elClearDate = document.getElementById("clear-date-filter");
  var elModal = document.getElementById("tile-modal");
  var elModalBody = document.getElementById("tile-modal-body");
  var elModalClose = document.getElementById("tile-modal-close");
  var basemapImgs = {};
  var currentQueryIndex = null;

  function fmtScore(s) {
    var v = s.toFixed(4);
    if (s >= 0) v = " " + v;
    return v;
  }

  function uniqueDates() {
    var seen = {};
    var out = [];
    for (var i = 0; i < DATA.sources.length; i++) {
      var d = DATA.sources[i].date;
      if (!seen[d]) { seen[d] = true; out.push(d); }
    }
    return out;
  }

  function preloadBasemaps(cb) {
    var sources = DATA.sources;
    var remaining = sources.length;
    if (remaining === 0) { cb(); return; }
    for (var i = 0; i < sources.length; i++) {
      (function (idx) {
        var img = new Image();
        img.onload = function () { basemapImgs[idx] = img; if (--remaining === 0) cb(); };
        img.onerror = function () { if (--remaining === 0) cb(); };
        img.src = sources[idx].dataUrl;
      })(i);
    }
  }

  function drawCrop(canvas, result, destPx) {
    var img = basemapImgs[result.srcIdx];
    canvas.width = destPx;
    canvas.height = destPx;
    var ctx = canvas.getContext("2d");
    if (!img) { ctx.fillStyle = "#ddd"; ctx.fillRect(0, 0, destPx, destPx); return; }
    var src = DATA.sources[result.srcIdx];
    var sx = result.pxOffsetX * src.basemapScaleX;
    var sy = result.pxOffsetY * src.basemapScaleY;
    var sw = result.scale * src.basemapScaleX;
    var sh = result.scale * src.basemapScaleY;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, destPx, destPx);
  }

  function renderChips(filterText) {
    elChips.innerHTML = "";
    elNoMatch.innerHTML = "";
    var needle = (filterText || "").toLowerCase().trim();
    var byCategory = {};
    var anyMatch = false;
    for (var i = 0; i < DATA.queries.length; i++) {
      var q = DATA.queries[i];
      if (needle && !matchesFilter(q.text, needle)) continue;
      anyMatch = true;
      if (!byCategory[q.category]) byCategory[q.category] = [];
      byCategory[q.category].push(i);
    }
    if (needle && !anyMatch) {
      var nearest = nearestQueries(DATA, needle, 6);
      var msg = document.createElement("div");
      msg.className = "no-match-msg";
      var label = document.createElement("p");
      label.textContent = "\"" + filterText + "\" is not in the precomputed query set.";
      msg.appendChild(label);
      if (nearest.length) {
        var label2 = document.createElement("p");
        label2.textContent = "Nearest available queries:";
        msg.appendChild(label2);
        var chipsWrap = document.createElement("div");
        chipsWrap.className = "chip-row";
        for (var n = 0; n < nearest.length; n++) chipsWrap.appendChild(makeChip(nearest[n], DATA.queries.map(function(q){return q.text;}).indexOf(nearest[n])));
        msg.appendChild(chipsWrap);
      } else {
        var label3 = document.createElement("p");
        label3.textContent = "No similar phrase found either -- try a shorter or more general word.";
        msg.appendChild(label3);
      }
      elNoMatch.appendChild(msg);
      return;
    }
    var order = ["small_object", "structure", "terrain", "damage"];
    for (var c = 0; c < order.length; c++) {
      var cat = order[c];
      if (!byCategory[cat] || !byCategory[cat].length) continue;
      var section = document.createElement("div");
      section.className = "chip-section";
      var h = document.createElement("h3");
      h.textContent = DATA.categoryLabels[cat] + " (" + byCategory[cat].length + ")";
      section.appendChild(h);
      var row = document.createElement("div");
      row.className = "chip-row";
      for (var k = 0; k < byCategory[cat].length; k++) {
        row.appendChild(makeChip(DATA.queries[byCategory[cat][k]].text, byCategory[cat][k]));
      }
      section.appendChild(row);
      elChips.appendChild(section);
    }
  }

  function makeChip(text, queryIndex) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = text;
    if (queryIndex === currentQueryIndex) b.classList.add("chip-active");
    b.addEventListener("click", function () { runQuery(queryIndex); });
    return b;
  }

  function currentDateFilter() {
    return elDateSelect.value === "__all__" ? null : elDateSelect.value;
  }

  function updateFilterBar() {
    var d = currentDateFilter();
    if (!d) { elFilterBar.hidden = true; return; }
    elFilterBar.hidden = false;
    var count = 0;
    if (currentQueryIndex !== null) {
      var r = rankQuery(DATA, decoded, currentQueryIndex, d);
      for (var s = 0; s < DATA.scales.length; s++) count += r.rankings[DATA.scales[s]].results.length;
    }
    elFilterBarText.textContent = "Date filter: " + d + (currentQueryIndex !== null ? " -- " + count + " result(s) shown" : "");
  }

  function runQuery(queryIndex) {
    currentQueryIndex = queryIndex;
    elResults.innerHTML = "<div class=\"searching\">Searching...</div>";
    updateFilterBar();
    renderChips(elFilter.value);
    window.setTimeout(function () { renderResults(queryIndex); }, 0);
  }

  function renderResults(queryIndex) {
    var t0 = performance.now();
    var result = rankQuery(DATA, decoded, queryIndex, currentDateFilter());
    var elapsed = performance.now() - t0;
    elResults.innerHTML = "";
    var heading = document.createElement("div");
    heading.className = "results-heading";
    heading.textContent = "\"" + result.query + "\"" + (elapsed > 1 ? " (" + elapsed.toFixed(0) + " ms)" : "");
    elResults.appendChild(heading);

    for (var s = 0; s < DATA.scales.length; s++) {
      var scale = DATA.scales[s];
      var row = result.rankings[scale];
      var section = document.createElement("section");
      section.className = "scale-row";
      var head = document.createElement("div");
      head.className = "scale-row-head";
      var extent = row.groundExtentM ? row.groundExtentM.toFixed(1) : "?";
      head.innerHTML = "<span class=\"scale-label\">" + scale + " px tiles &mdash; " + extent + " m ground extent</span>";
      var band = document.createElement("span");
      band.className = "conf-badge conf-" + row.confidence.band;
      band.textContent = row.confidence.band === "unknown" ? "confidence: unknown" : "confidence: " + row.confidence.band + " (" + row.confidence.percentile.toFixed(0) + "th pct)";
      band.title = row.confidence.message;
      head.appendChild(band);
      section.appendChild(head);
      var msgEl = document.createElement("div");
      msgEl.className = "conf-message";
      msgEl.textContent = row.confidence.message;
      section.appendChild(msgEl);

      if (!row.results.length) {
        var empty = document.createElement("div");
        empty.className = "empty-row";
        empty.textContent = currentDateFilter()
          ? "No tiles at this scale match the current date filter."
          : "No tiles at this scale.";
        section.appendChild(empty);
      } else {
        var strip = document.createElement("div");
        strip.className = "result-strip";
        for (var i = 0; i < row.results.length; i++) {
          strip.appendChild(makeResultCard(row.results[i]));
        }
        section.appendChild(strip);
      }
      elResults.appendChild(section);
    }
  }

  function makeResultCard(result) {
    var card = document.createElement("div");
    card.className = "result-card";
    var canvas = document.createElement("canvas");
    canvas.className = "result-thumb";
    drawCrop(canvas, result, 96);
    card.appendChild(canvas);
    var score = document.createElement("div");
    score.className = "result-score";
    score.textContent = fmtScore(result.score);
    card.appendChild(score);
    var date = document.createElement("div");
    date.className = "result-date";
    date.textContent = result.date;
    card.appendChild(date);
    card.addEventListener("click", function () { openTileModal(result); });
    return card;
  }

  function openTileModal(result) {
    elModalBody.innerHTML = "";
    var canvas = document.createElement("canvas");
    canvas.className = "modal-canvas";
    drawCrop(canvas, result, 360);
    elModalBody.appendChild(canvas);
    var info = document.createElement("dl");
    info.className = "modal-info";
    function row(label, value) {
      var dt = document.createElement("dt"); dt.textContent = label;
      var dd = document.createElement("dd"); dd.textContent = value;
      info.appendChild(dt); info.appendChild(dd);
    }
    row("Score", fmtScore(result.score) + " (compare only within its own scale row)");
    row("Scale", result.scale + " px (" + (DATA.groundExtentM[String(result.scale)] || 0).toFixed(1) + " m ground extent)");
    row("Date", result.date);
    row("Source", result.sourceFile);
    row("Lat/lon (bbox)", result.bbox.map(function (v) { return v.toFixed(6); }).join(", "));
    row("Tile id", result.tileId);
    elModalBody.appendChild(info);
    var note = document.createElement("p");
    note.className = "modal-note";
    note.textContent = "Magnified crop of the downsampled basemap, not new pixel detail -- see the resolution notice above.";
    elModalBody.appendChild(note);
    elModal.hidden = false;
  }

  elModalClose.addEventListener("click", function () { elModal.hidden = true; });
  elModal.addEventListener("click", function (ev) { if (ev.target === elModal) elModal.hidden = true; });

  elFilter.addEventListener("input", function () { renderChips(elFilter.value); });
  elDateSelect.addEventListener("change", function () { updateFilterBar(); if (currentQueryIndex !== null) renderResults(currentQueryIndex); });
  elClearDate.addEventListener("click", function () { elDateSelect.value = "__all__"; updateFilterBar(); if (currentQueryIndex !== null) renderResults(currentQueryIndex); });

  (function initDateFilter() {
    var dates = uniqueDates();
    var optAll = document.createElement("option");
    optAll.value = "__all__"; optAll.textContent = "All dates";
    elDateSelect.appendChild(optAll);
    for (var i = 0; i < dates.length; i++) {
      var opt = document.createElement("option");
      opt.value = dates[i]; opt.textContent = dates[i];
      elDateSelect.appendChild(opt);
    }
    elDateSelect.value = "__all__";
    if (dates.length <= 1) elDateSelect.parentElement.hidden = true;
  })();

  renderChips("");
  preloadBasemaps(function () {});
})();
"""

PAGE_CSS = r"""
:root { color-scheme: light; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; max-width: 100%; overflow-x: hidden; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #f7f7f5; color: #1b1b1b; font-size: 14px; line-height: 1.4;
}
header { padding: 16px; background: #1b1b1b; color: #fff; }
header h1 { margin: 0 0 4px 0; font-size: 18px; }
header p { margin: 0; font-size: 12px; color: #cfcfcf; }
.caveat {
  background: #fff4d6; border-bottom: 2px solid #caa53d; color: #4a3b00;
  padding: 10px 16px; font-size: 12.5px;
}
.caveat strong { display: block; margin-bottom: 2px; }
main { padding: 12px 16px 40px; max-width: 100%; }
.search-wrap { margin-bottom: 12px; }
.search-wrap input {
  width: 100%; max-width: 480px; padding: 8px 10px; font-size: 14px;
  border: 1px solid #bbb; border-radius: 6px;
}
.filter-bar {
  display: flex; align-items: center; gap: 10px; background: #eef; border: 1px solid #ccd;
  border-radius: 6px; padding: 6px 10px; margin-bottom: 12px; font-size: 12.5px; flex-wrap: wrap;
}
.filter-bar button { border: none; background: #445; color: #fff; border-radius: 4px; padding: 3px 8px; cursor: pointer; font-size: 12px; }
.date-filter-wrap { margin-bottom: 12px; font-size: 12.5px; }
.date-filter-wrap select { padding: 4px 6px; font-size: 13px; }
.chip-section { margin-bottom: 10px; }
.chip-section h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #555; margin: 6px 0 4px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; max-width: 100%; }
.chip {
  border: 1px solid #ccc; background: #fff; border-radius: 14px; padding: 5px 11px;
  font-size: 12.5px; cursor: pointer; white-space: nowrap;
}
.chip:hover { background: #eef1ff; border-color: #99a; }
.chip-active { background: #2c3e91; color: #fff; border-color: #2c3e91; }
.no-match-msg { background: #fdecec; border: 1px solid #e5b3b3; border-radius: 6px; padding: 10px; margin-bottom: 10px; }
.no-match-msg p { margin: 0 0 6px; font-size: 13px; }
.results-heading { font-size: 15px; font-weight: 600; margin: 14px 0 8px; }
.searching { font-size: 13px; color: #777; padding: 10px 0; }
.scale-row { background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 10px; margin-bottom: 12px; max-width: 100%; overflow: hidden; }
.scale-row-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.scale-label { font-weight: 600; font-size: 13px; }
.conf-badge { font-size: 11.5px; padding: 2px 8px; border-radius: 10px; border: 1px solid transparent; }
.conf-low { background: #fdecec; color: #8a2020; border-color: #e5b3b3; }
.conf-medium { background: #fff6dc; color: #6b5300; border-color: #e3ce8f; }
.conf-high { background: #e7f6ea; color: #1d5c2e; border-color: #a9d8b3; }
.conf-unknown { background: #eee; color: #555; border-color: #ccc; }
.conf-message { font-size: 11.5px; color: #666; margin-top: 4px; }
.empty-row { font-size: 12.5px; color: #777; padding: 10px 0; }
.result-strip { display: flex; gap: 10px; overflow-x: auto; padding: 8px 2px; max-width: 100%; }
.result-card { flex: 0 0 auto; width: 96px; cursor: pointer; text-align: center; }
.result-thumb { width: 96px; height: 96px; border-radius: 4px; border: 1px solid #ddd; background: #ddd; image-rendering: -webkit-optimize-contrast; }
.result-score { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12px; margin-top: 4px; white-space: pre; }
.result-date { font-size: 10.5px; color: #777; }
.tile-modal {
  position: fixed; inset: 0; background: rgba(0,0,0,0.55); display: flex;
  align-items: center; justify-content: center; padding: 16px; z-index: 10;
}
.tile-modal[hidden] { display: none; }
.tile-modal-inner { background: #fff; border-radius: 8px; padding: 16px; max-width: 480px; width: 100%; max-height: 90vh; overflow-y: auto; position: relative; }
.tile-modal-close { position: absolute; top: 8px; right: 10px; border: none; background: none; font-size: 20px; cursor: pointer; line-height: 1; }
.modal-canvas { width: 100%; max-width: 360px; height: auto; border: 1px solid #ddd; border-radius: 4px; display: block; margin: 0 auto 12px; image-rendering: -webkit-optimize-contrast; }
.modal-info { display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; font-size: 12.5px; margin: 0; }
.modal-info dt { font-weight: 600; color: #444; }
.modal-info dd { margin: 0; word-break: break-word; }
.modal-note { font-size: 11.5px; color: #777; margin-top: 10px; }
footer { padding: 12px 16px; font-size: 11px; color: #888; }
@media (max-width: 480px) {
  .result-card, .result-thumb { width: 78px; height: 78px; }
  header h1 { font-size: 16px; }
}
"""


def _render_html(payload: dict) -> str:
    """Assemble the final single-file page. `payload` carries everything
    already computed (JSON-safe); this function only does string assembly
    and the __PLACEHOLDER__ substitutions into `RANKING_CORE_JS`."""
    ranking_js = (
        RANKING_CORE_JS
        .replace("__CONF_LOW__", repr(retrieve.CONFIDENCE_LOW_MAX_PERCENTILE))
        .replace("__CONF_HIGH__", repr(retrieve.CONFIDENCE_HIGH_MIN_PERCENTILE))
        .replace("__TOP_K__", str(TOP_K))
    )
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    aoi = payload["aoi"]
    generated_at = payload["generatedAt"]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aerial tile retrieval -- {aoi} (offline export)</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<header>
  <h1>Aerial tile retrieval &mdash; {aoi}</h1>
  <p>Offline export, generated {generated_at}. No server, no network, no free text.</p>
</header>
<div class="caveat">
  <strong>Resolution notice</strong>
  This imagery's effective optical resolution is approximately 35&ndash;45 cm on the ground
  (the 10&ndash;12 cm pixel grid is upsampled 3&ndash;4x). The system can indicate presence and a
  coarse class of object &mdash; it cannot support fine attribute search (colour, make, model,
  exact count). Treat every result as "something like this may be here", not a confirmed
  identification.
</div>
<main>
  <div class="search-wrap">
    <input id="filter-input" type="text" placeholder="Filter the {len(PRECOMPUTED_QUERIES)} precomputed queries (e.g. 'roof', 'road', 'debris')&hellip;" autocomplete="off">
  </div>
  <div class="date-filter-wrap">
    Date filter: <select id="date-filter"></select>
  </div>
  <div id="filter-bar" class="filter-bar" hidden>
    <span id="filter-bar-text"></span>
    <button id="clear-date-filter" type="button">Clear</button>
  </div>
  <div id="no-match"></div>
  <div id="query-chips"></div>
  <div id="results"></div>
</main>
<footer>
  This file answers only the {len(PRECOMPUTED_QUERIES)} precomputed queries above &mdash; it carries
  no text model and cannot answer arbitrary text. Free-text search and tile question-answering
  are available only in the local application. Model: {payload.get("modelId", "")} (revision {payload.get("revision", "")}).
</footer>
<div id="tile-modal" class="tile-modal" hidden>
  <div class="tile-modal-inner">
    <button id="tile-modal-close" class="tile-modal-close" type="button" aria-label="Close">&times;</button>
    <div id="tile-modal-body"></div>
  </div>
</div>
<script type="application/json" id="export-data">{data_json}</script>
<script id="ranking-core">{ranking_js}</script>
<script id="ui-logic">{UI_JS}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _ensure_export_basis(aoi: str, index_root: Path | None) -> dict:
    """Load the S4 export basis, rebuilding it if missing or if its tile
    order has drifted from the current index (the same staleness check
    `export_basis.measure_overlap` makes before trusting an export)."""
    manifest_tile_ids = None
    try:
        loaded = export_basis.load_export(aoi, index_root=index_root)
    except FileNotFoundError:
        loaded = None
    if loaded is not None:
        current = embed_index.load_index(aoi, index_root=index_root)
        manifest_tile_ids = [t["tile_id"] for t in current["manifest"]["tiles"]]
        if loaded["basis"]["tile_ids"] == manifest_tile_ids:
            return loaded
        log.warning("export_html: %s export basis is stale against the current index -- rebuilding", aoi)
    export_basis.build_export(aoi, index_root=index_root)
    return export_basis.load_export(aoi, index_root=index_root)


def _assemble_payload(
    aoi: str,
    corpus: "retrieve.Corpus",
    loaded_export: dict,
    query_bundle: dict,
    background: dict[int, list[float]],
    sources: list[dict],
    per_scale: dict[int, dict],
    basemap_max_px: int,
    jpeg_quality: int,
    data_root: Path | None,
) -> tuple[dict, dict]:
    """Build one candidate payload at `basemap_max_px`. Returns (payload,
    component_sizes) -- the caller checks the encoded size and, on the
    resolution ladder, tries the next rung rather than reusing a half-built
    payload."""
    sources_out = []
    basemap_bytes_total = 0
    for src in sources:
        bm = _build_basemap(src["relPath"], data_root, basemap_max_px, jpeg_quality)
        basemap_bytes_total += bm["jpeg_bytes"]
        sources_out.append(
            {
                "relPath": src["relPath"],
                "date": src["date"],
                "dataUrl": bm["data_url"],
                "basemapScaleX": bm["basemap_scale_x"],
                "basemapScaleY": bm["basemap_scale_y"],
                "basemapWidth": bm["basemap_width"],
                "basemapHeight": bm["basemap_height"],
                "origWidth": bm["orig_width"],
                "origHeight": bm["orig_height"],
            }
        )

    per_scale_out = {str(scale): {k: v for k, v in cols.items()} for scale, cols in per_scale.items()}
    ground_extent_m = {str(s): corpus.ground_extent_m.get(s) for s in corpus.manifest["scales"]}

    queries_out = []
    for (category, text), vec_i8 in zip(query_bundle["queries"], query_bundle["vectors_i8"]):
        queries_out.append({"text": text, "category": category, "vectorB64": _b64(vec_i8)})

    payload = {
        "aoi": aoi,
        "modelId": corpus.manifest["model_id"],
        "revision": corpus.manifest["revision"],
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scales": list(corpus.manifest["scales"]),
        "groundExtentM": ground_extent_m,
        "tileScale": loaded_export["basis"]["int8_scale"],
        "queryScale": query_bundle["scale"],
        "categoryLabels": CATEGORY_LABELS,
        "queries": queries_out,
        "perScale": per_scale_out,
        "sources": sources_out,
        "background": {str(k): v for k, v in background.items()},
    }
    component_sizes = {
        "tile_vectors_raw_bytes": sum(cols["n"] * export_basis.EXPORT_DIM for cols in per_scale.values()),
        "query_vectors_raw_bytes": query_bundle["vectors_i8"].size,
        "basemap_jpeg_bytes": basemap_bytes_total,
        "n_queries": len(queries_out),
        "n_tiles": sum(cols["n"] for cols in per_scale.values()),
        "basemap_max_px": basemap_max_px,
    }
    return payload, component_sizes


def build_export_html(
    aoi: str,
    index_root: Path | None = None,
    data_root: Path | None = None,
    embedder=None,
    basemap_ladder: Sequence[int] = BASEMAP_RESOLUTION_LADDER,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    max_bytes: int = EXPORT_SIZE_CAP_BYTES,
    out_path: Path | None = None,
) -> dict:
    """Build the standalone export for one AOI (F-8). Degrades basemap
    resolution (never tile count or vector precision) until the assembled
    file fits `max_bytes`; raises `ExportSizeError` naming the computed size
    and tile count if even the smallest rung does not (N-6). Nothing is
    written to disk unless the final size is within the cap.
    """
    corpus = retrieve.load_corpus(aoi, index_root=index_root, data_root=data_root)
    loaded_export = _ensure_export_basis(aoi, index_root)
    if loaded_export["basis"]["tile_ids"] != [t["tile_id"] for t in corpus.manifest["tiles"]]:
        raise RuntimeError(f"{aoi}: export basis tile order does not match the loaded corpus after rebuild")

    emb = embedder
    query_bundle = build_query_vectors(loaded_export["components"], embedder=emb)
    exported_f32 = export_basis.exported_vectors_f32(loaded_export)
    background = _build_export_background(corpus, exported_f32, loaded_export["components"], embedder=emb)
    sources, per_scale = _build_sources_and_tiles(corpus, loaded_export["vectors_i8"])

    last_size = None
    last_component_sizes = None
    for basemap_max_px in basemap_ladder:
        payload, component_sizes = _assemble_payload(
            aoi, corpus, loaded_export, query_bundle, background, sources, per_scale,
            basemap_max_px, jpeg_quality, data_root,
        )
        html = _render_html(payload)
        html_bytes = html.encode("utf-8")
        size = len(html_bytes)
        last_size, last_component_sizes = size, component_sizes
        log.info(
            "export_html: %s at basemap_max_px=%d -> %.2f MB (cap %.0f MB)",
            aoi, basemap_max_px, size / 1e6, max_bytes / 1e6,
        )
        if size <= max_bytes:
            path = out_path or export_html_path(aoi, index_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(html_bytes)
            os.replace(tmp, path)
            return {
                "aoi": aoi,
                "out_path": str(path),
                "size_bytes": size,
                "basemap_max_px": basemap_max_px,
                "n_tiles": component_sizes["n_tiles"],
                "n_queries": component_sizes["n_queries"],
                "component_sizes": component_sizes,
            }

    n_tiles = last_component_sizes["n_tiles"] if last_component_sizes else 0
    raise ExportSizeError(
        f"{aoi}: export does not fit under the {max_bytes} byte cap even at the smallest "
        f"tried basemap resolution ({basemap_ladder[-1]} px): computed size = {last_size} bytes "
        f"({last_size / 1e6:.2f} MB) for {n_tiles} tiles. Nothing was written."
    )


# --------------------------------------------------------------------------
# Fresh-process verification helpers -- read the file back from disk rather
# than trusting `build_export_html`'s in-memory return (this project's own
# test discipline: a stage has twice reported numbers computed in memory
# while the file on disk was wrong).
# --------------------------------------------------------------------------

_DATA_SCRIPT_OPEN = '<script type="application/json" id="export-data">'
_RANKING_SCRIPT_OPEN = '<script id="ranking-core">'
_SCRIPT_CLOSE = "</script>"


def _extract_tagged_block(html_text: str, open_tag: str) -> str:
    start = html_text.index(open_tag) + len(open_tag)
    end = html_text.index(_SCRIPT_CLOSE, start)
    return html_text[start:end]


def read_export_data(path: Path) -> dict:
    """Re-parse the embedded `DATA` object from a built export file on
    disk."""
    html_text = path.read_text(encoding="utf-8")
    return json.loads(_extract_tagged_block(html_text, _DATA_SCRIPT_OPEN))


def read_ranking_core_js(path: Path) -> str:
    """Re-extract the exact, DOM-free ranking JS shipped in a built export
    file on disk -- what `test_export_query_matches_local` actually runs
    (via Node), not a Python reimplementation of it."""
    html_text = path.read_text(encoding="utf-8")
    return _extract_tagged_block(html_text, _RANKING_SCRIPT_OPEN)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    p = argparse.ArgumentParser(description="Build the standalone offline HTML export for one AOI.")
    p.add_argument("--aoi", required=True)
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    p.add_argument("--max-bytes", type=int, default=EXPORT_SIZE_CAP_BYTES)
    args = p.parse_args()
    report = build_export_html(args.aoi, jpeg_quality=args.jpeg_quality, max_bytes=args.max_bytes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

