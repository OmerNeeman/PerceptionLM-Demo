"""The local interactive query app -- brief S5, spec U-1..U-8, D-5, F-8's
amendment (free text lives locally, never in the export).

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 AERIAL_DATA_ROOT=<path> \
        <python> app.py

(or, equivalently, via uvicorn's own CLI with the `--factory` flag --
`create_app` is a callable that builds the app, not a module-level app
object, so a bare `app:app` target would not work here:

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 AERIAL_DATA_ROOT=<path> \
        <python> -m uvicorn app:create_app --factory --host 127.0.0.1 --port 8420

see INSTRUCTIONS.md Sec.9 for the interpreter path and the exact command).

This module is deliberately split into two layers so the acceptance tests
(brief S5.md: "test the app's logic, not pixel layout") never need an HTTP
client or a browser:

  * pure logic  -- ``Engine``, ``answer_query``, ``render_index_html``,
    ``format_score``, ``filter_corpus_by_date`` -- takes/returns plain dicts
    and strings, reuses ``retrieve.py`` for every ranking/scoring/geo
    decision (F-3/F-4/F-5/F-6/F-10/U-3), never reimplements it.
  * FastAPI wiring -- ``app``, the ``/`` and ``/api/*`` routes -- thin
    adapters that call the logic layer and serialise its result.

U-3's amendment (S4_fix Fix 2, `notes.md#s4-review`): the confidence band is
shown **alongside** results, never gates or reorders them, and a low band
reads "may not be present" -- never "is not present". `retrieve.confidence_band`
already enforces the wording; this module only displays it.

U-8's amendment (`notes.md#s4-green`) gives the local app a tile Q&A box, but
this brief scopes S5 to the grid + detail view only -- **the question box is
F-9/S5a**. The extension point is the comment directly above the ``_JS``
string: nothing here renders a disabled or placeholder control, because an
affordance that cannot work is worse than its absence (the export's own U-8
rationale, and no less true locally while the backend does not exist yet).

Environment: every invocation must be prefixed inline, every time --

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 AERIAL_DATA_ROOT=<path> <python> ...

(see INSTRUCTIONS.md for the interpreter path on the dev machine).
"""

from __future__ import annotations

import dataclasses
import html as html_lib
import io
import logging
import threading
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

import config
import embed_index
import embedders
import retrieve
import tiling

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# U-1 -- example queries, seeded from the four vocabularies CLAUDE.md names
# (damage is excluded on purpose: the ratified problem statement says damage
# is not answerable on this demo AOI, X605_Y3388 -- offering it here would be
# exactly the overstatement U-5 exists to prevent).
# --------------------------------------------------------------------------

EXAMPLE_QUERIES: tuple[str, ...] = (
    "tents",
    "a car",
    "a building with a flat roof",
    "dirt road",
    "sand",
    "palm trees",
)

#: Auto-run once at page load so the app opens already showing a populated,
#: located, banded result grid -- U-1's "never a blank box" read as a
#: requirement on the *state* the app opens into, not only on the input
#: field. Also doubles as the U-4 first-query warm-up demonstration.
DEFAULT_QUERY = EXAMPLE_QUERIES[0]

#: U-5 / D-5 -- the honesty requirement CLAUDE.md calls the most important
#: UX item in the spec. Rendered verbatim in the page (see
#: `test_resolution_caveat_in_page`) -- never only in a README.
RESOLUTION_CAVEAT = (
    "Effective optical resolution here is about 35–45 cm, not the 10 cm "
    "pixel grid — the imagery has been upsampled roughly 3–4x. That "
    "supports presence and coarse class only: “a car” and “a "
    "white car” retrieve the same crops, and the colour word makes "
    "results measurably worse. This tool does not do attribute-level search "
    "(colour, make, condition) and no result here should be read as "
    "supporting one."
)

#: U-3 -- shown next to every confidence band so "not a verdict" is stated,
#: not just implied by wording elsewhere.
CONFIDENCE_EXPLAINER = (
    "The band shows where this query's top score falls against a background "
    "of ~30 unrelated queries, per scale. It is not a presence/absence "
    "verdict: on this corpus no statistic reliably separates present from "
    "absent queries — a low band can still be a real hit, and a high "
    "band is not a guarantee. Results are never hidden or reordered by it."
)

#: F-4's owner clarification, restated for the UI -- U-2's "must not invite
#: cross-row comparison".
CROSS_ROW_CAVEAT = "Scores are only meaningful within one scale's row — never compare scores across rows."

DEMO_AOI = "X605_Y3388"

MAX_QUERY_CHARS = 2000  # generous; the 500-char abuse case is well inside this
MAX_THUMB_PX = 512
MIN_THUMB_PX = 16
DEFAULT_THUMB_PX = 160


# --------------------------------------------------------------------------
# Suppress open_clip's benign "initialized randomly" warning in the app's own
# startup output (brief S5.md: "the owner will see this console and it reads
# like a failure"). Scoped to exactly the embedder-load call, not the whole
# process, so a genuine later warning is never silenced by accident.
# --------------------------------------------------------------------------


class _DropBenignRandomInitWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "No pretrained weights loaded" not in record.getMessage()


class _quiet_open_clip_warning:
    def __enter__(self):
        self._filter = _DropBenignRandomInitWarning()
        logging.getLogger().addFilter(self._filter)
        return self

    def __exit__(self, *exc):
        logging.getLogger().removeFilter(self._filter)
        return False


# --------------------------------------------------------------------------
# Engine -- the loaded, queryable state. Corpus + background load eagerly
# (F-5: < 5 s, no GPU needed); the embedder loads lazily on the first query
# so the process itself starts fast and U-4's "first query, ~9 s" is an
# honest, observable event rather than something hidden behind a slow
# process launch.
# --------------------------------------------------------------------------


class Engine:
    """Holds one AOI's loaded corpus/background/embedder.

    A single `threading.Lock` serialises every embedder call (text
    tokenize + GPU forward pass): FastAPI's sync routes run in a thread
    pool, and this project has never verified concurrent GPU calls through
    this exact loader are safe -- serialising is cheap for a local,
    single-user demo and removes the question entirely (this is also what
    keeps "rapid repeated submits" -- one of the brief's own abuse cases --
    from racing a half-initialised `self.embedder`).
    """

    def __init__(self, aoi: str = DEMO_AOI, index_root: Path | None = None, data_root: Path | None = None):
        self.aoi = aoi
        self.index_root = index_root
        self.data_root = data_root
        self._lock = threading.Lock()
        self.corpus: retrieve.Corpus = retrieve.load_corpus(aoi, index_root=index_root, data_root=data_root)
        try:
            self.background: dict[int, Sequence[float]] | None = retrieve.load_background(
                aoi, index_root=index_root
            )["scores_by_scale"]
        except FileNotFoundError:
            log.warning("app: no background reference set found for %s -- confidence bands will read 'unknown'", aoi)
            self.background = None
        self.embedder = None  # lazy -- see class docstring
        self._available_dates: list[str] = sorted(
            {loc["date"] for loc in self.corpus.locations.values() if loc["date"] != "unknown"}
        )

    def available_dates(self) -> list[str]:
        return list(self._available_dates)

    def get_embedder(self) -> tuple[object, bool, float]:
        """Returns (embedder, was_just_loaded, load_ms). Thread-safe,
        idempotent -- a second caller while another thread is mid-load
        simply waits on the lock and then sees the already-loaded embedder
        (load_ms 0.0, was_just_loaded False)."""
        with self._lock:
            if self.embedder is not None:
                return self.embedder, False, 0.0
            t0 = time.perf_counter()
            with _quiet_open_clip_warning():
                self.embedder = embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)
            load_ms = (time.perf_counter() - t0) * 1000.0
            log.info("app: embedder loaded in %.1f ms", load_ms)
            return self.embedder, True, load_ms

    def run_query(self, text: str, *, bbox=None, date=None, top_k: int = retrieve.TOP_K) -> dict:
        with self._lock:
            embedder, warm_up, model_load_ms = (self.embedder, False, 0.0)
        if embedder is None:
            embedder, warm_up, model_load_ms = self.get_embedder()

        corpus = filter_corpus_by_date(self.corpus, date)
        t0 = time.perf_counter()
        with self._lock:
            qvec = retrieve.embed_query(text, embedder=embedder)
        embed_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        out = retrieve.rank_per_scale(corpus, qvec, top_k=top_k, bbox=bbox, background=self.background)
        search_ms = (time.perf_counter() - t1) * 1000.0

        return {
            "query": text,
            "rankings": out["rankings"],
            "timing": {
                "model_load_ms": round(model_load_ms, 1),
                "embed_ms": round(embed_ms, 1),
                "search_ms": round(search_ms, 1),
                "total_ms": round(model_load_ms + embed_ms + search_ms, 1),
                "warm_up": warm_up,
            },
        }


# --------------------------------------------------------------------------
# Pure logic -- date filtering. Reuses retrieve.rank_per_scale's own ranking
# maths unmodified: this only narrows the candidate index arrays it consumes
# (exactly analogous to what `bbox` already does *inside* rank_per_scale, for
# the axis rank_per_scale itself does not filter on), so it is a candidate
# restriction, not a reimplementation of the ranking (brief: "reuse, do not
# reimplement retrieval").
# --------------------------------------------------------------------------


def filter_corpus_by_date(corpus: retrieve.Corpus, date: str | None) -> retrieve.Corpus:
    """F-7's date filter, generalised to any AOI's corpus in memory.

    `date` of `None`/``""``/``"any"`` is "no filter" (identity). Otherwise
    only tiles whose recorded date equals `date` survive -- an
    ``unknown``-dated tile is *excluded* under an active date filter, per
    F-7's amendment, never silently included.
    """
    if not date or date == "any":
        return corpus
    new_scale_indices: dict[int, np.ndarray] = {}
    for scale, idx in corpus.scale_indices.items():
        tile_ids = [corpus.manifest["tiles"][i]["tile_id"] for i in idx]
        keep = np.fromiter(
            (corpus.locations[tid]["date"] == date for tid in tile_ids), dtype=bool, count=len(tile_ids)
        )
        new_scale_indices[scale] = idx[keep]
    return dataclasses.replace(corpus, scale_indices=new_scale_indices)


def format_score(score: float) -> str:
    """Fixed-width score formatting (U-2): always a sign and 4 decimals, so
    a column of scores is scannable -- '+0.2650', '-0.0421', both 7 chars."""
    return f"{score:+.4f}"


def tile_center_lonlat(bbox_lonlat: Sequence[float]) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0


def enrich_result(res: dict) -> dict:
    """Adds display-only fields (formatted score, centre lat/lon) to one
    `rank_per_scale` result record, without dropping anything F-10 requires
    (source_file, date, px_offset_x/y, bbox_lonlat all pass through)."""
    lon, lat = tile_center_lonlat(res["bbox_lonlat"])
    return {
        **res,
        "score_display": format_score(res["score"]),
        "lon": lon,
        "lat": lat,
    }


def answer_query(engine: Engine, text: str, *, bbox=None, date=None, top_k: int = retrieve.TOP_K) -> dict:
    """The full app-level answer to one query: validated empty/whitespace
    handling, then `Engine.run_query`, then display enrichment. Never raises
    on an empty/whitespace query -- returns a guidance response instead
    (mirrors U-3's "empty state guides" spirit for the query box itself)."""
    stripped = text.strip()
    if not stripped:
        return {
            "query": text,
            "empty": True,
            "message": "Type a description, or click one of the examples above.",
            "rankings": {},
            "timing": {"model_load_ms": 0.0, "embed_ms": 0.0, "search_ms": 0.0, "total_ms": 0.0, "warm_up": False},
        }
    truncated = stripped[:MAX_QUERY_CHARS]
    out = engine.run_query(truncated, bbox=bbox, date=date, top_k=top_k)
    rankings = {}
    for scale, ranking in out["rankings"].items():
        rankings[scale] = {
            **ranking,
            "results": [enrich_result(r) for r in ranking["results"]],
        }
    out["rankings"] = rankings
    out["empty"] = False
    out["truncated"] = len(stripped) > MAX_QUERY_CHARS
    return out


# --------------------------------------------------------------------------
# Thumbnails -- read one tile's own pixels off the (read-only) source
# raster, at query time, for display. Reuses tiling's own grid arithmetic
# (parse_tile_id, PAD_VALUE) rather than a second parse; never opens the
# raster for write (D-3).
# --------------------------------------------------------------------------


def render_tile_thumbnail(tile_id: str, size: int, data_root: Path | None = None) -> bytes:
    try:
        rec = tiling.parse_tile_id(tile_id)
    except Exception as exc:
        raise ValueError(f"not a valid tile id: {tile_id!r} ({exc})") from exc

    root = data_root or config.get_data_root()
    scale = rec["scale"]
    x0, y0 = rec["col"] * scale, rec["row"] * scale
    path = root / rec["source_file"]
    if not path.exists():
        raise FileNotFoundError(f"source raster not found: {path}")

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    with rasterio.open(path) as ds:  # read-only (D-3)
        vw = max(0, min(scale, ds.width - x0))
        vh = max(0, min(scale, ds.height - y0))
        if vw <= 0 or vh <= 0:
            raise ValueError(f"tile {tile_id!r} falls entirely outside {rec['source_file']}")
        arr = ds.read((1, 2, 3), window=Window(x0, y0, vw, vh))
    out = np.full((scale, scale, 3), tiling.PAD_VALUE, dtype=arr.dtype)
    out[:vh, :vw, :] = np.moveaxis(arr, 0, -1)
    img = Image.fromarray(out).resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# U-1/U-2/U-3/U-5/U-6/U-7 -- the single-page HTML. No external CDN, no
# network calls at runtime (CLAUDE.md: every external fetch is a trust
# boundary; F-8's amendment: this is the side that keeps a server, but the
# offline expectation still holds -- the owner may run this with no network
# at all). CSS/JS are inlined; nothing here references an external origin.
# --------------------------------------------------------------------------


def _example_chips_html() -> str:
    chips = "\n".join(
        f'<button type="button" class="example-chip" data-query="{html_lib.escape(q)}">{html_lib.escape(q)}</button>'
        for q in EXAMPLE_QUERIES
    )
    return chips


def render_index_html(engine: Engine | None = None) -> str:
    dates = engine.available_dates() if engine is not None else []
    date_options = "".join(f'<option value="{html_lib.escape(d)}">{html_lib.escape(d)}</option>' for d in dates)
    date_disabled = "" if dates else "disabled"
    date_note = "" if dates else "<p class=\"muted small\">No dated imagery indexed for this AOI.</p>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aerial Tile Retrieval</title>
<style>
{_CSS}
</style>
</head>
<body>
<header>
  <h1>Aerial Tile Retrieval</h1>
  <p class="subtitle">Type a description of what you're looking for in the imagery.</p>
</header>

<section id="caveat" class="caveat" role="note">
  <strong>Resolution limit:</strong> {html_lib.escape(RESOLUTION_CAVEAT)}
</section>

<section id="search">
  <form id="query-form" autocomplete="off">
    <input id="query-input" name="text" type="text" maxlength="{MAX_QUERY_CHARS}"
           placeholder="e.g. a car, dirt road, a building with a flat roof"
           value="{html_lib.escape(DEFAULT_QUERY)}">
    <button type="submit" id="search-btn">Search</button>
  </form>
  <div id="examples">
    <span class="muted small">Try:</span>
    {_example_chips_html()}
  </div>
  <p id="status" class="status muted small" aria-live="polite"></p>
</section>

<section id="filters">
  <div class="filter-row">
    <fieldset>
      <legend>AOI bbox (min_lon, min_lat, max_lon, max_lat)</legend>
      <input id="bbox-input" type="text" placeholder="leave empty for no bbox filter">
      <button type="button" id="bbox-apply">Apply</button>
    </fieldset>
    <fieldset>
      <legend>Date</legend>
      <select id="date-select" {date_disabled}>
        <option value="">(any)</option>
        {date_options}
      </select>
      {date_note}
    </fieldset>
    <button type="button" id="clear-filters" class="secondary">Clear filters</button>
  </div>
  <div id="active-filters" class="active-filters muted small" aria-live="polite"></div>
</section>

<section id="notes" class="muted small">
  <p>{html_lib.escape(CROSS_ROW_CAVEAT)}</p>
  <p>{html_lib.escape(CONFIDENCE_EXPLAINER)}</p>
</section>

<section id="results" aria-live="polite"></section>

<div id="tile-modal" class="modal hidden" role="dialog" aria-modal="true">
  <div class="modal-content">
    <button type="button" id="modal-close" class="modal-close" aria-label="Close">&times;</button>
    <img id="modal-img" alt="tile detail">
    <dl id="modal-meta"></dl>
  </div>
</div>

<script>
{_JS}
</script>
</body>
</html>"""


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1rem; max-width: 1100px; margin-inline: auto;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: #fafafa; color: #1a1a1a;
}
@media (prefers-color-scheme: dark) { body { background: #16181c; color: #eee; } }
header h1 { margin: 0 0 0.25rem 0; font-size: 1.4rem; }
.subtitle { margin: 0 0 0.75rem 0; color: #666; }
.caveat {
  background: #fff3cd; border: 1px solid #e6c200; border-radius: 6px;
  padding: 0.6rem 0.8rem; margin-bottom: 1rem; font-size: 0.92rem; line-height: 1.4;
}
@media (prefers-color-scheme: dark) { .caveat { background: #3a3410; border-color: #7a6a10; } }
#query-form { display: flex; gap: 0.5rem; flex-wrap: wrap; }
#query-input { flex: 1 1 240px; min-width: 0; padding: 0.5rem; font-size: 1rem; }
#search-btn { padding: 0.5rem 1rem; }
#examples { margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
.example-chip {
  border: 1px solid #999; background: transparent; border-radius: 999px;
  padding: 0.2rem 0.7rem; cursor: pointer; font-size: 0.85rem;
}
.example-chip:hover { background: #e0e0e0; }
.muted { color: #666; }
.small { font-size: 0.85rem; }
.status { min-height: 1.2em; }
#filters { margin-top: 1rem; }
.filter-row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-end; }
fieldset { border: 1px solid #ccc; border-radius: 6px; padding: 0.5rem; }
#bbox-input { width: 20rem; max-width: 100%; }
.active-filters { margin-top: 0.4rem; }
.filter-chip {
  display: inline-block; background: #e6e6e6; border-radius: 999px;
  padding: 0.15rem 0.6rem; margin-right: 0.4rem;
}
@media (prefers-color-scheme: dark) { .filter-chip { background: #333; } }
#results { margin-top: 1.5rem; }
.scale-row { margin-bottom: 2rem; }
.scale-row h2 { font-size: 1.05rem; margin-bottom: 0.15rem; }
.scale-caption { color: #666; font-size: 0.85rem; margin: 0 0 0.5rem 0; }
.confidence-band {
  display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px;
  font-size: 0.82rem; margin-bottom: 0.4rem;
}
.confidence-band.low { background: #dbe4f0; }
.confidence-band.medium { background: #b9c9e6; }
.confidence-band.high { background: #8fa8d6; }
.confidence-band.unknown { background: #e0e0e0; }
.confidence-explainer { font-size: 0.78rem; color: #666; margin: 0.2rem 0 0.6rem 0; max-width: 60ch; }
.tile-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.75rem;
}
.tile-card {
  border: 1px solid #ccc; border-radius: 6px; overflow: hidden; cursor: pointer;
  background: white; display: flex; flex-direction: column;
}
@media (prefers-color-scheme: dark) { .tile-card { background: #222; border-color: #444; } }
.tile-card img { width: 100%; display: block; aspect-ratio: 1 / 1; object-fit: cover; background: #ddd; }
.tile-meta { padding: 0.35rem 0.5rem; font-size: 0.78rem; font-variant-numeric: tabular-nums; }
.tile-meta .score { font-weight: 600; }
.empty-row { color: #666; font-style: italic; }
.modal.hidden { display: none; }
.modal {
  position: fixed; inset: 0; background: rgba(0,0,0,0.6);
  display: flex; align-items: center; justify-content: center; padding: 1rem; z-index: 10;
}
.modal-content {
  background: white; border-radius: 8px; padding: 1rem; max-width: 90vw; max-height: 90vh;
  overflow: auto; position: relative;
}
@media (prefers-color-scheme: dark) { .modal-content { background: #222; color: #eee; } }
.modal-content img { max-width: 100%; display: block; margin-bottom: 0.75rem; }
.modal-close {
  position: absolute; top: 0.4rem; right: 0.6rem; border: none; background: none;
  font-size: 1.4rem; cursor: pointer; line-height: 1;
}
@media (max-width: 375px) {
  body { padding: 0.5rem; }
  .filter-row { flex-direction: column; align-items: stretch; }
  #bbox-input { width: 100%; }
}
"""

# Extension point for F-9/S5a (owner-amended U-8): a tile-detail Q&A box.
# Deliberately not rendered here -- S5's brief scopes the question box out of
# this stage, and "leave the hook, render no placeholder" means exactly
# that: no disabled input, no greyed-out button. When S5a lands, it adds a
# form to #modal-meta's sibling and a POST to a new /api/ask endpoint; no
# restructuring of the grid/detail code above should be needed.
_JS = """
const state = { bbox: null, date: "", lastQuery: document.getElementById('query-input').value };

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

async function runQuery(text) {
  state.lastQuery = text;
  const statusEl = document.getElementById('status');
  const resultsEl = document.getElementById('results');
  statusEl.textContent = 'Searching… (first search can take ~10s while the model loads)';
  document.getElementById('search-btn').disabled = true;
  try {
    const body = { text, top_k: 10 };
    if (state.bbox) body.bbox = state.bbox;
    if (state.date) body.date = state.date;
    const resp = await fetch('/api/query', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.textContent = data.detail || 'Something went wrong with that query.';
      resultsEl.innerHTML = '';
      return;
    }
    renderResults(data);
    if (data.empty) {
      statusEl.textContent = data.message || '';
    } else {
      const t = data.timing;
      statusEl.textContent = `Query "${data.query}" — ${t.total_ms.toFixed(0)} ms` +
        (t.warm_up ? ` (includes ${t.model_load_ms.toFixed(0)} ms model warm-up, first query only)` : '');
    }
  } catch (err) {
    statusEl.textContent = 'Request failed — is the server still running?';
  } finally {
    document.getElementById('search-btn').disabled = false;
  }
}

function renderActiveFilters(resultCounts) {
  const el = document.getElementById('active-filters');
  const chips = [];
  if (state.bbox) {
    const total = resultCounts ? Object.values(resultCounts).reduce((a,b) => a+b, 0) : 0;
    chips.push(`<span class="filter-chip">bbox [${state.bbox.map(v => v.toFixed(4)).join(', ')}] — ${total} result(s)</span>`);
  }
  if (state.date) {
    chips.push(`<span class="filter-chip">date = ${state.date}</span>`);
  }
  el.innerHTML = chips.join(' ');
}

function renderResults(data) {
  const el = document.getElementById('results');
  el.innerHTML = '';
  if (data.empty) return;
  const counts = {};
  const scales = Object.keys(data.rankings).map(Number).sort((a, b) => b - a);
  for (const scale of scales) {
    const row = data.rankings[scale];
    counts[scale] = row.results.length;
    const section = document.createElement('div');
    section.className = 'scale-row';
    const extent = row.ground_extent_m != null ? row.ground_extent_m.toFixed(1) : '?';
    const conf = row.confidence || { band: 'unknown', message: '' };
    section.innerHTML = `
      <h2>${scale}px tile &mdash; ${extent} m true ground extent</h2>
      <p class="scale-caption">${data.rankings[scale].results.length} result(s) at this scale. Scores are only comparable within this row.</p>
      <span class="confidence-band ${conf.band}">confidence: ${conf.band}${conf.percentile != null ? ' (' + conf.percentile.toFixed(0) + 'th pct.)' : ''}</span>
      <p class="confidence-explainer">${conf.message}</p>
      <div class="tile-grid" id="grid-${scale}"></div>
    `;
    el.appendChild(section);
    const grid = section.querySelector(`#grid-${scale}`);
    if (row.results.length === 0) {
      grid.innerHTML = '<p class="empty-row">No results at this scale for the current filters.</p>';
      continue;
    }
    for (const r of row.results) {
      const card = document.createElement('div');
      card.className = 'tile-card';
      card.innerHTML = `
        <img loading="lazy" src="/api/thumb?tile_id=${encodeURIComponent(r.tile_id)}&size=160" alt="tile thumbnail">
        <div class="tile-meta">
          <div class="score">${r.score_display}</div>
          <div>${r.date} · ${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}</div>
        </div>
      `;
      card.addEventListener('click', () => openTileModal(r, scale, extent));
      grid.appendChild(card);
    }
  }
  renderActiveFilters(counts);
}

function openTileModal(r, scale, extent) {
  const modal = document.getElementById('tile-modal');
  document.getElementById('modal-img').src = `/api/thumb?tile_id=${encodeURIComponent(r.tile_id)}&size=480`;
  document.getElementById('modal-meta').innerHTML = `
    <dt>Score</dt><dd>${r.score_display} (row: ${scale}px / ${extent} m — not comparable across rows)</dd>
    <dt>Date</dt><dd>${r.date}</dd>
    <dt>Location</dt><dd>${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}</dd>
    <dt>Source</dt><dd>${r.source_file}</dd>
  `;
  modal.classList.remove('hidden');
}

document.getElementById('modal-close').addEventListener('click', () => {
  document.getElementById('tile-modal').classList.add('hidden');
});
document.getElementById('tile-modal').addEventListener('click', (e) => {
  if (e.target.id === 'tile-modal') e.target.classList.add('hidden');
});

document.getElementById('query-form').addEventListener('submit', (e) => {
  e.preventDefault();
  runQuery(document.getElementById('query-input').value);
});

for (const chip of document.querySelectorAll('.example-chip')) {
  chip.addEventListener('click', () => {
    document.getElementById('query-input').value = chip.dataset.query;
    runQuery(chip.dataset.query);
  });
}

document.getElementById('bbox-apply').addEventListener('click', () => {
  const raw = document.getElementById('bbox-input').value.trim();
  if (!raw) { state.bbox = null; }
  else {
    const parts = raw.split(',').map(s => parseFloat(s.trim()));
    if (parts.length === 4 && parts.every(v => !isNaN(v))) { state.bbox = parts; }
    else { document.getElementById('status').textContent = 'bbox must be 4 comma-separated numbers: min_lon, min_lat, max_lon, max_lat'; return; }
  }
  runQuery(state.lastQuery);
});

document.getElementById('date-select').addEventListener('change', (e) => {
  state.date = e.target.value;
  runQuery(state.lastQuery);
});

document.getElementById('clear-filters').addEventListener('click', () => {
  state.bbox = null; state.date = '';
  document.getElementById('bbox-input').value = '';
  document.getElementById('date-select').value = '';
  document.getElementById('active-filters').innerHTML = '';
  runQuery(state.lastQuery);
});

// U-1: open in a working state -- run the default query immediately.
runQuery(state.lastQuery);
"""


# --------------------------------------------------------------------------
# FastAPI wiring
# --------------------------------------------------------------------------


class QueryRequest(BaseModel):
    text: str = ""
    bbox: list[float] | None = None
    date: str | None = None
    top_k: int = retrieve.TOP_K


def create_app(engine: Engine | None = None) -> FastAPI:
    """Build the FastAPI app around one `Engine`. A factory (not a bare
    module-level `app = FastAPI()` wired to a module-level engine) so tests
    can point it at a synthetic index/engine instead of the real production
    one, the same isolation `test_retrieve.py`'s own `synthetic_index`
    fixture already relies on."""
    eng = engine or Engine()
    fastapi_app = FastAPI(title="Aerial Tile Retrieval")

    @fastapi_app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_index_html(eng)

    @fastapi_app.get("/health")
    def health() -> dict:
        return {"status": "ok", "aoi": eng.aoi, "n_tiles": int(eng.corpus.vectors.shape[0])}

    @fastapi_app.post("/api/query")
    def query(req: QueryRequest) -> dict:
        if req.bbox is not None and len(req.bbox) != 4:
            raise HTTPException(422, detail="bbox must have exactly 4 numbers: min_lon, min_lat, max_lon, max_lat")
        top_k = max(1, min(req.top_k, 200))
        try:
            return answer_query(eng, req.text, bbox=req.bbox, date=req.date, top_k=top_k)
        except retrieve.UnitNormError as exc:
            log.exception("app: embedder returned a non-unit query vector")
            raise HTTPException(500, detail="the embedder returned an invalid vector for that query") from exc
        except Exception as exc:  # never a raw traceback to the client (S5 abuse-case requirement)
            log.exception("app: unexpected error answering query %r", req.text)
            raise HTTPException(500, detail=f"unexpected error answering that query: {exc.__class__.__name__}") from exc

    @fastapi_app.get("/api/thumb")
    def thumb(tile_id: str, size: int = DEFAULT_THUMB_PX) -> Response:
        size = max(MIN_THUMB_PX, min(size, MAX_THUMB_PX))
        try:
            png = render_tile_thumbnail(tile_id, size, data_root=eng.data_root)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("app: unexpected error rendering thumbnail for %r", tile_id)
            raise HTTPException(500, detail="unexpected error rendering that thumbnail") from exc
        return Response(content=png, media_type="image/png")

    @fastapi_app.exception_handler(Exception)
    async def _catch_all(request, exc: Exception):  # last-resort net -- see module docstring
        log.exception("app: unhandled exception on %s", request.url)
        return JSONResponse(status_code=500, content={"detail": "internal error -- see server log"})

    fastapi_app.state.engine = eng
    return fastapi_app


# No module-level `app = FastAPI()`: building one means loading the real
# production corpus (`Engine()`'s default), and that must never happen as a
# side effect of `import app` during test collection -- tests call
# `create_app(engine=...)` themselves, pointed at whatever corpus (synthetic
# or production) that test needs. `create_app` with no arguments (the
# default) is what both `main()` below and uvicorn's own `--factory` CLI
# flag use to build the real app on demand.
def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    fastapi_app = create_app()
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8420, log_level="info")


if __name__ == "__main__":
    main()
