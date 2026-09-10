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
import json
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

#: S11b -- F-1a's UI constraint, stated for the compare view the same way
#: CROSS_ROW_CAVEAT states it for scales: different models occupy different
#: embedding spaces at different scales, so a score from one model's column
#: means nothing next to a score from another model's column.
CROSS_MODEL_CAVEAT = (
    "In compare view, scores are only meaningful within one model's own column "
    "— RemoteCLIP's and PE Core's scores are not on the same scale and were "
    "never meant to be compared to each other."
)

DEMO_AOI = "X605_Y3388"

#: Part 3 (brief S6) -- the AOI-selector sentinel meaning "search every
#: indexed AOI at once", never a real AOI name (`embed_index.list_indexed_aois`
#: only ever returns real ones, so this can never collide with one).
ALL_AOIS = "all"

# --------------------------------------------------------------------------
# S11b -- model selector + compare view. Spec F-1a expressed in the UI:
# scores from two models occupy different spaces and are never comparable,
# so this app never merges, interleaves or jointly ranks results from two
# models, and never renders a "winner" computed from raw scores. See
# `Engine.run_compare` / `answer_compare` for where that constraint is
# actually enforced in code, not just in this comment.
# --------------------------------------------------------------------------

#: Models this app can query or compare, in display/default order. Index 0
#: is the shipped default (RemoteCLIP -- S0's measured winner, CLAUDE.md);
#: a default single-model view has to start somewhere, which is a UX
#: convenience, not a "which model is better" claim -- that claim is exactly
#: what this stage refuses to make (see the module-level note above).
COMPARE_MODELS: tuple[str, ...] = (embed_index.DEFAULT_MODEL_ID, "PE-Core-L14-336")


def model_label(model_id: str) -> str:
    """`RemoteCLIP-ViT-L-14 · 768-d` -- dimensionality read from
    `embedders.CANDIDATES`, never a hardcoded literal (the same "derive it,
    don't type it" discipline `Engine.ground_extent_note` already uses for
    true ground extent)."""
    dim = embedders.CANDIDATES[model_id].dim
    return f"{model_id} · {dim}-d"

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
    """Holds every model's loaded corpora/backgrounds for this process, plus
    lazily-loaded embedders per model.

    A single `threading.Lock` serialises every embedder call (text
    tokenize + GPU forward pass) *and* every model's lazy load, across every
    model: FastAPI's sync routes run in a thread pool, and this project has
    never verified concurrent GPU calls through this exact loader are safe --
    serialising is cheap for a local, single-user demo and removes the
    question entirely (this is also what keeps "rapid repeated submits" --
    one of the brief's own abuse cases -- from racing a half-initialised
    embedder). Two models never run a forward pass at once as a result, which
    is a latency cost the compare view's own N-1 measurement accounts for
    (see `run_compare`'s docstring), not a correctness one.
    """

    def __init__(
        self,
        aoi: str = DEMO_AOI,
        index_root: Path | None = None,
        data_root: Path | None = None,
        models: Sequence[str] = COMPARE_MODELS,
    ):
        """`aoi` is the *default* selected AOI (Part 3: "defaulting to one
        AOI rather than everything, so results stay interpretable") -- not
        the only one this Engine can serve. `models` (S11b) is every model
        this Engine can query or compare; `models[0]` is the default model
        (S0's shipped choice).

        For the **default model**, every AOI with a complete on-disk index
        under `index_root` (`embed_index.list_indexed_aois`) is discovered
        and its corpus/background loaded eagerly here, alongside `aoi`
        itself if for some reason it is not otherwise discovered (e.g. a
        fixture index that predates `list_indexed_aois`, or `aoi` was
        renamed on disk) -- this keeps the pre-S6 single-AOI call signature
        (`Engine(aoi=..., index_root=..., data_root=...)`) working unchanged,
        including its exact failure mode (`FileNotFoundError` propagates
        uncaught if `aoi` itself has no default-model index -- never
        silently skipped) since discovery then finds exactly that one AOI
        and every existing caller/test still gets the same behaviour.

        For every **other** model (S11b), AOI discovery and loading is
        independent and forgiving: a model can legitimately lack an index
        for an AOI the default model has (or have extra AOIs the default
        model lacks) -- see `briefs/S11b.md`'s "switch to a model whose
        index is missing for that AOI" abuse case. A missing (model, AOI)
        index is logged and skipped here, never raised at startup; it
        surfaces later, cleanly, from `get_corpus`/`get_background` only if
        that exact combination is actually requested.

        F-5's < 5 s budget covers this whole eager load, not just one AOI's
        (measured on the real 4-AOI production index: well under a second
        total -- see briefs/S6_result.md). Eagerly loading a second model's
        corpora roughly doubles the vector memory (measured, not assumed --
        see `briefs/S11b_result.md`'s peak-RSS figure) but stays well within
        that budget; embedders themselves (the GPU-resident part) remain
        lazy per model exactly as before, so opening the app never pays for
        a model the user never selects.
        """
        self.aoi = aoi
        self.index_root = index_root
        self.data_root = data_root
        self.models: tuple[str, ...] = tuple(models)
        self.default_model: str = self.models[0]
        self._lock = threading.Lock()

        discovered = embed_index.list_indexed_aois(index_root, model_id=self.default_model)
        self.available_aois: list[str] = sorted(set(discovered) | {aoi})

        # Keyed (model_id, aoi) -- see class docstring for why the default
        # model's loop is strict (propagates FileNotFoundError) while every
        # other model's is forgiving.
        self._corpora: dict[tuple[str, str], retrieve.Corpus] = {}
        self._backgrounds: dict[tuple[str, str], dict[int, Sequence[float]] | None] = {}
        self._model_aois: dict[str, list[str]] = {self.default_model: list(self.available_aois)}

        for a in self.available_aois:
            self._corpora[(self.default_model, a)] = retrieve.load_corpus(
                a, index_root=index_root, data_root=data_root, model_id=self.default_model
            )
            self._backgrounds[(self.default_model, a)] = self._try_load_background(
                a, self.default_model
            )

        for m in self.models[1:]:
            found: list[str] = []
            for a in sorted(embed_index.list_indexed_aois(index_root, model_id=m)):
                try:
                    self._corpora[(m, a)] = retrieve.load_corpus(
                        a, index_root=index_root, data_root=data_root, model_id=m
                    )
                except FileNotFoundError:
                    log.warning("app: no on-disk index for model=%s aoi=%s -- skipping (not an error)", m, a)
                    continue
                found.append(a)
                self._backgrounds[(m, a)] = self._try_load_background(a, m)
            self._model_aois[m] = found

        # Kept as plain attributes (not methods) for backward compatibility --
        # every pre-S6 caller/test reads `engine.corpus` / `engine.background`
        # directly, meaning "the default model's default AOI's corpus/
        # background". They are fixed at construction time, never mutated by
        # a later per-query `aoi`/`model_id` argument (concurrent requests
        # may select different AOIs/models; Engine itself carries no
        # "currently selected" mutable state).
        self.corpus: retrieve.Corpus = self._corpora[(self.default_model, aoi)]
        self.background: dict[int, Sequence[float]] | None = self._backgrounds[(self.default_model, aoi)]

        self._all_corpus: dict[str, retrieve.Corpus] = {}  # built lazily per model -- see get_corpus
        self.embedders: dict[str, object] = {}  # lazy per model -- see class docstring
        self._available_dates_by_aoi: dict[str, list[str]] = {
            a: sorted({loc["date"] for loc in c.locations.values() if loc["date"] != "unknown"})
            for a, c in ((a, self._corpora[(self.default_model, a)]) for a in self.available_aois)
        }

    @property
    def embedder(self):
        """Backward-compat (pre-S11b): the *default* model's lazily-loaded
        embedder, or `None` if not yet loaded. `self.embedders` (S11b) is the
        per-model dict this now actually tracks; every pre-S11b caller reads
        this singular property and means "the one model this Engine had"."""
        return self.embedders.get(self.default_model)

    def _known_aois(self) -> set[str]:
        """Every AOI known to *any* model this Engine has -- used by
        `run_compare` to tell "this specific model lacks this AOI" (a
        legitimate per-model gap) apart from "this AOI does not exist at
        all" (a real error, same as an unknown AOI on a single-model
        query)."""
        known: set[str] = set()
        for aois in self._model_aois.values():
            known.update(aois)
        return known

    def _try_load_background(self, aoi: str, model_id: str) -> dict[int, Sequence[float]] | None:
        try:
            return retrieve.load_background(aoi, index_root=self.index_root, model_id=model_id)["scores_by_scale"]
        except FileNotFoundError:
            log.warning(
                "app: no background reference set found for model=%s aoi=%s -- confidence bands will read 'unknown'",
                model_id, aoi,
            )
            return None

    def get_corpus(self, aoi: str | None, model_id: str | None = None) -> retrieve.Corpus:
        """The corpus for one selected (model, AOI) pair, or the `ALL_AOIS`
        sentinel for every indexed AOI *of that model* concatenated (Part 3).
        The combined corpus is built once per model, lazily, and cached --
        most sessions never ask for it, and building it eagerly for every
        model would spend F-5's budget on a view most queries do not use.

        `model_id` (S11b) defaults to `self.default_model`, so every pre-S11b
        caller (which never passed a second argument) keeps reading exactly
        the default model's corpus it always did.
        """
        aoi = aoi or self.aoi
        model_id = model_id or self.default_model
        if model_id not in self.models:
            raise ValueError(f"unknown model {model_id!r} -- available: {list(self.models)}")
        if aoi == ALL_AOIS:
            if model_id not in self._all_corpus:
                with self._lock:
                    if model_id not in self._all_corpus:
                        self._all_corpus[model_id] = retrieve.load_corpus_multi(
                            self._model_aois.get(model_id, []),
                            index_root=self.index_root, data_root=self.data_root, model_id=model_id,
                        )
            return self._all_corpus[model_id]
        key = (model_id, aoi)
        if key not in self._corpora:
            raise ValueError(
                f"no index for model={model_id!r} aoi={aoi!r} -- AOIs indexed for this model: "
                f"{self._model_aois.get(model_id, [])}"
            )
        return self._corpora[key]

    def get_background(self, aoi: str | None, model_id: str | None = None) -> dict[int, Sequence[float]] | None:
        """Confidence-band background for one selected (model, AOI) pair.
        `ALL_AOIS` has no combined background reference set (unspecified by
        the brief; building one would need its own calibration, not just a
        concatenation) -- `retrieve.confidence_band` already handles `None`
        by reporting `band: "unknown"` rather than raising, so this is an
        honest omission, not a broken path. Likewise for a (model, AOI) pair
        whose background was never built."""
        aoi = aoi or self.aoi
        model_id = model_id or self.default_model
        if aoi == ALL_AOIS:
            return None
        return self._backgrounds.get((model_id, aoi))

    def available_dates(self, aoi: str | None = None) -> list[str]:
        aoi = aoi or self.aoi
        if aoi == ALL_AOIS:
            combined: set[str] = set()
            for dates in self._available_dates_by_aoi.values():
                combined |= set(dates)
            return sorted(combined)
        return list(self._available_dates_by_aoi.get(aoi, []))

    def ground_extent_note(self) -> str | None:
        """Part 3: "note in the UI that a 448 px tile is 46.91 m in leb and
        44.80 m in the UTM scenes" -- computed from each loaded AOI's own
        `ground_extent_m` (itself sourced from `geo.py`, never hardcoded
        here), not typed as a literal. Returns None when every indexed AOI
        happens to share the same true GSD (e.g. a single-AOI fixture index),
        since there is then nothing to caveat.

        Purely geometric (CRS/GSD), not a function of which model embedded
        the tiles -- so this reads only the default model's corpora, exactly
        as before S11b; a second model over the same on-disk tiles would
        report identical ground extents and add nothing here."""
        scale = tiling.SCALES[0]
        extents = {a: self._corpora[(self.default_model, a)].ground_extent_m.get(scale) for a in self.available_aois}
        distinct = sorted({round(v, 2) for v in extents.values() if v is not None})
        if len(distinct) <= 1:
            return None
        lo, hi = distinct[0], distinct[-1]
        pct = (hi - lo) / lo * 100.0
        lo_aois = sorted(a for a, v in extents.items() if v is not None and round(v, 2) == lo)
        hi_aois = sorted(a for a, v in extents.items() if v is not None and round(v, 2) == hi)
        return (
            f"A {scale}px tile is {lo:.2f} m across in {', '.join(lo_aois)} and "
            f"{hi:.2f} m in {', '.join(hi_aois)} ({pct:.1f}% difference) — cross-AOI "
            f"ranking is still valid (one embedder, one embedding space), but tile "
            f"footprints are not identical ground area."
        )

    def get_embedder(self, model_id: str | None = None) -> tuple[object, bool, float]:
        """Returns (embedder, was_just_loaded, load_ms) for one model.
        Thread-safe, idempotent -- a second caller while another thread is
        mid-load simply waits on the lock and then sees the already-loaded
        embedder (load_ms 0.0, was_just_loaded False).

        `model_id` (S11b) defaults to `self.default_model`, so every pre-S11b
        caller keeps loading exactly the model it always did."""
        model_id = model_id or self.default_model
        if model_id not in self.models:
            raise ValueError(f"unknown model {model_id!r} -- available: {list(self.models)}")
        with self._lock:
            if model_id in self.embedders:
                return self.embedders[model_id], False, 0.0
            t0 = time.perf_counter()
            with _quiet_open_clip_warning():
                emb = embedders.load_embedder(model_id)
            load_ms = (time.perf_counter() - t0) * 1000.0
            self.embedders[model_id] = emb
            log.info("app: embedder %s loaded in %.1f ms", model_id, load_ms)
            return emb, True, load_ms

    def run_query(
        self,
        text: str,
        *,
        aoi: str | None = None,
        bbox=None,
        date=None,
        top_k: int = retrieve.TOP_K,
        model_id: str | None = None,
    ) -> dict:
        model_id = model_id or self.default_model
        embedder, warm_up, model_load_ms = self.get_embedder(model_id)  # raises ValueError on an unknown model

        selected_aoi = aoi or self.aoi
        base_corpus = self.get_corpus(
            selected_aoi, model_id=model_id
        )  # raises ValueError on an unknown AOI or missing (model, AOI) index -- never silently ignored
        corpus = filter_corpus_by_date(base_corpus, date)
        background = self.get_background(selected_aoi, model_id=model_id)
        t0 = time.perf_counter()
        with self._lock:
            qvec = retrieve.embed_query(text, embedder=embedder)
        embed_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        out = retrieve.rank_per_scale(corpus, qvec, top_k=top_k, bbox=bbox, background=background)
        search_ms = (time.perf_counter() - t1) * 1000.0

        return {
            "query": text,
            "aoi": selected_aoi,
            "model_id": model_id,
            "model_label": model_label(model_id),
            "rankings": out["rankings"],
            "timing": {
                "model_load_ms": round(model_load_ms, 1),
                "embed_ms": round(embed_ms, 1),
                "search_ms": round(search_ms, 1),
                "total_ms": round(model_load_ms + embed_ms + search_ms, 1),
                "warm_up": warm_up,
            },
        }

    def run_compare(
        self,
        text: str,
        *,
        aoi: str | None = None,
        bbox=None,
        date=None,
        top_k: int = retrieve.TOP_K,
        models: Sequence[str] | None = None,
    ) -> dict:
        """S11b -- the same query, same AOI/bbox/date filters, run
        **independently** against every model in `models` (default:
        `self.models`, i.e. every model this Engine knows). Returns
        ``{"query": text, "aoi": ..., "models": {model_id: <run_query's own
        result dict, or an "unavailable" stub>, ...}}``.

        This is the one place F-1a's UI constraint (module docstring: never
        merge, interleave or jointly rank two models' results) has to be
        actively honoured, not just assumed -- and the way it is honoured is
        structural: each model's ranking comes from its own, independent
        `run_query` call, kept in its own bucket of the returned dict. No
        step here ever concatenates, sorts, or compares two models' `results`
        lists or `score` values against each other -- see
        `test_no_cross_model_merging`.

        A model with no index for `aoi`, where `aoi` is otherwise real (known
        to at least one model this Engine has), does not fail the whole
        comparison -- its entry reports ``{"available": False, ...}``
        instead, so one model's missing data never 500s the other model's
        genuinely-available column (the compare view's own version of
        "switch to a model whose index is missing for that AOI", handled
        without a stack trace).

        An `aoi` unknown to *every* model, however, still raises `ValueError`
        up front -- exactly like `get_corpus` does for a single-model query
        -- rather than quietly reporting every model as "unavailable"; a
        typo'd AOI is a real error, not a legitimate per-model gap, and
        deserves the same clean 422 a single-model query already gets for it.
        """
        selected_aoi = aoi or self.aoi
        if selected_aoi != ALL_AOIS and selected_aoi not in self._known_aois():
            raise ValueError(
                f"unknown AOI {selected_aoi!r} -- available: {sorted(self._known_aois())} (or {ALL_AOIS!r})"
            )
        chosen = tuple(models) if models else self.models
        out: dict[str, dict] = {}
        for m in chosen:
            try:
                out[m] = self.run_query(text, aoi=aoi, bbox=bbox, date=date, top_k=top_k, model_id=m)
                out[m]["available"] = True
            except ValueError as exc:
                out[m] = {
                    "available": False,
                    "model_id": m,
                    "model_label": model_label(m),
                    "error": str(exc),
                }
        return {"query": text, "aoi": aoi or self.aoi, "models": out}


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


def answer_query(
    engine: Engine,
    text: str,
    *,
    aoi: str | None = None,
    bbox=None,
    date=None,
    top_k: int = retrieve.TOP_K,
    model_id: str | None = None,
) -> dict:
    """The full app-level answer to one query: validated empty/whitespace
    handling, then `Engine.run_query`, then display enrichment. Never raises
    on an empty/whitespace query -- returns a guidance response instead
    (mirrors U-3's "empty state guides" spirit for the query box itself).

    `aoi` (Part 3): `None` means the engine's own default AOI; a real AOI
    name restricts candidates to it; `ALL_AOIS` spans every indexed AOI.
    An unknown AOI name still raises (via `Engine.get_corpus`) rather than
    silently falling back to the default -- a filter that silently does
    nothing is worse than one that fails loudly.

    `model_id` (S11b): `None` means the engine's own default model
    (RemoteCLIP); a real model id switches which index answers this one
    query. An unknown model id, or a model with no index for `aoi`, raises
    the same way an unknown AOI does (via `Engine.get_corpus`/`get_embedder`)."""
    stripped = text.strip()
    if not stripped:
        return {
            "query": text,
            "aoi": aoi or engine.aoi,
            "model_id": model_id or engine.default_model,
            "model_label": model_label(model_id or engine.default_model),
            "empty": True,
            "message": "Type a description, or click one of the examples above.",
            "rankings": {},
            "timing": {"model_load_ms": 0.0, "embed_ms": 0.0, "search_ms": 0.0, "total_ms": 0.0, "warm_up": False},
        }
    truncated = stripped[:MAX_QUERY_CHARS]
    out = engine.run_query(truncated, aoi=aoi, bbox=bbox, date=date, top_k=top_k, model_id=model_id)
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


def answer_compare(
    engine: Engine,
    text: str,
    *,
    aoi: str | None = None,
    bbox=None,
    date=None,
    top_k: int = retrieve.TOP_K,
    models: Sequence[str] | None = None,
) -> dict:
    """The full app-level answer to one compare-view query: the same query,
    ranked independently through every model in `models` (default: every
    model the engine knows), enriched for display exactly like
    `answer_query` does per model -- **never merged, interleaved, or
    re-ranked across models** (F-1a's UI constraint; see `Engine.run_compare`
    and `test_no_cross_model_merging`).

    Shape: ``{"query": ..., "aoi": ..., "empty": bool, "models": {model_id:
    {"available": bool, "model_id", "model_label", "rankings": {scale: {...,
    "results": [enriched...]}}, "timing": {...}} | {"available": False,
    "model_id", "model_label", "error"}}}``.
    """
    stripped = text.strip()
    if not stripped:
        return {
            "query": text,
            "aoi": aoi or engine.aoi,
            "empty": True,
            "message": "Type a description, or click one of the examples above.",
            "models": {},
        }
    truncated = stripped[:MAX_QUERY_CHARS]
    out = engine.run_compare(truncated, aoi=aoi, bbox=bbox, date=date, top_k=top_k, models=models)
    models_out: dict[str, dict] = {}
    for model_id, result in out["models"].items():
        if not result.get("available", False):
            models_out[model_id] = result
            continue
        rankings = {}
        for scale, ranking in result["rankings"].items():
            rankings[scale] = {**ranking, "results": [enrich_result(r) for r in ranking["results"]]}
        models_out[model_id] = {**result, "rankings": rankings}
    out["models"] = models_out
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
    default_aoi = engine.aoi if engine is not None else DEMO_AOI
    available_aois = engine.available_aois if engine is not None else [default_aoi]
    models = engine.models if engine is not None else COMPARE_MODELS
    default_model = engine.default_model if engine is not None else COMPARE_MODELS[0]
    model_options = "".join(
        f'<option value="{html_lib.escape(m)}"{" selected" if m == default_model else ""}>'
        f'{html_lib.escape(model_label(m))}</option>'
        for m in models
    )
    dates = engine.available_dates(default_aoi) if engine is not None else []
    date_options = "".join(f'<option value="{html_lib.escape(d)}">{html_lib.escape(d)}</option>' for d in dates)
    date_disabled = "" if dates else "disabled"
    date_note = "" if dates else "No dated imagery indexed for this AOI."

    # Part 3: single-AOI default, explicit "all AOIs" option -- never opens
    # already spanning everything (that would make results uninterpretable).
    aoi_options = "".join(
        f'<option value="{html_lib.escape(a)}"{" selected" if a == default_aoi else ""}>{html_lib.escape(a)}</option>'
        for a in available_aois
    )
    aoi_options += f'<option value="{ALL_AOIS}">All AOIs</option>'

    # Per-AOI available dates, so the date <select> can be repopulated
    # client-side when the AOI selection changes, with no extra round trip --
    # embedded as a JSON data island (never string-concatenated JS) rather
    # than one <option> soup per AOI, so JS owns the DOM update, not this
    # function guessing which AOI is active.
    dates_by_aoi = (
        {a: engine.available_dates(a) for a in available_aois} if engine is not None else {default_aoi: []}
    )
    dates_by_aoi[ALL_AOIS] = engine.available_dates(ALL_AOIS) if engine is not None else []
    # `<script>` content is HTML's "raw text" model -- entities are never
    # decoded inside it, so `html_lib.escape` here would hand JSON.parse a
    # literal "&quot;" and break it. The only real risk is a literal
    # "</script" substring prematurely closing the tag; escape just that
    # (the standard "JSON inside a script tag" mitigation), not the quotes.
    dates_by_aoi_json = json.dumps(dates_by_aoi).replace("</", "<\\/")

    extent_note = engine.ground_extent_note() if engine is not None else None
    extent_note_html = f'<p class="muted small">{html_lib.escape(extent_note)}</p>' if extent_note else ""

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
      <legend>AOI</legend>
      <select id="aoi-select">
        {aoi_options}
      </select>
    </fieldset>
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
      <p id="date-note" class="muted small">{date_note}</p>
    </fieldset>
    <fieldset>
      <legend>Model</legend>
      <select id="model-select">
        {model_options}
      </select>
    </fieldset>
    <fieldset>
      <legend>Compare</legend>
      <label><input type="checkbox" id="compare-toggle"> Compare models side by side</label>
    </fieldset>
    <button type="button" id="clear-filters" class="secondary">Clear filters</button>
  </div>
  <div id="active-filters" class="active-filters muted small" aria-live="polite"></div>
</section>

<script id="aoi-dates-data" type="application/json">{dates_by_aoi_json}</script>

<section id="notes" class="muted small">
  <p>{html_lib.escape(CROSS_ROW_CAVEAT)}</p>
  <p>{html_lib.escape(CROSS_MODEL_CAVEAT)}</p>
  <p>{html_lib.escape(CONFIDENCE_EXPLAINER)}</p>
  {extent_note_html}
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
.compare-columns {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: start;
}
@media (max-width: 700px) { .compare-columns { grid-template-columns: 1fr; } }
.compare-col { border: 1px solid #ccc; border-radius: 6px; padding: 0.6rem; min-width: 0; }
.compare-col h3 { margin: 0 0 0.2rem 0; font-size: 0.95rem; }
@media (prefers-color-scheme: dark) { .compare-col { border-color: #444; } }
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
const datesByAoi = JSON.parse(document.getElementById('aoi-dates-data').textContent);
const state = {
  bbox: null, date: "",
  aoi: document.getElementById('aoi-select').value,
  modelId: document.getElementById('model-select').value,
  compare: false,
  lastQuery: document.getElementById('query-input').value,
};

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

async function runQuery(text) {
  state.lastQuery = text;
  const statusEl = document.getElementById('status');
  const resultsEl = document.getElementById('results');
  statusEl.textContent = state.compare
    ? 'Comparing… (first search for a given model can take ~10s while that model loads)'
    : 'Searching… (first search for a given model can take ~10s while that model loads)';
  document.getElementById('search-btn').disabled = true;
  try {
    const body = { text, top_k: 10, aoi: state.aoi };
    if (state.bbox) body.bbox = state.bbox;
    if (state.date) body.date = state.date;
    const endpoint = state.compare ? '/api/compare' : '/api/query';
    if (!state.compare) body.model_id = state.modelId;
    const resp = await fetch(endpoint, {
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
    } else if (state.compare) {
      const parts = Object.values(data.models).map((m) => {
        if (m.available === false) return `${m.model_label}: no index for this AOI`;
        return `${m.model_label}: ${m.timing.total_ms.toFixed(0)} ms`;
      });
      statusEl.textContent = `Comparing "${data.query}" — ` + parts.join(' · ');
    } else {
      const t = data.timing;
      statusEl.textContent = `Query "${data.query}" [${data.model_label}] — ${t.total_ms.toFixed(0)} ms` +
        (t.warm_up ? ` (includes ${t.model_load_ms.toFixed(0)} ms model warm-up, first query only)` : '');
    }
  } catch (err) {
    statusEl.textContent = 'Request failed — is the server still running?';
  } finally {
    document.getElementById('search-btn').disabled = false;
  }
}

// U-6 -- active filters (AOI, bbox, date, model/compare) are always legible,
// with the result count each produced, in one place. `modelChips` is
// pre-built by the single-model / compare renderer below, since only they
// know how many results *each model* actually produced (compare never
// collapses two models' counts into one number -- that would already be a
// small step toward the "which won" verdict this app must never show).
function renderActiveFilters(modelChips) {
  const el = document.getElementById('active-filters');
  const chips = [`<span class="filter-chip">AOI: ${state.aoi === 'all' ? 'All AOIs' : state.aoi}</span>`, ...modelChips];
  if (state.bbox) {
    chips.push(`<span class="filter-chip">bbox [${state.bbox.map(v => v.toFixed(4)).join(', ')}]</span>`);
  }
  if (state.date) {
    chips.push(`<span class="filter-chip">date = ${state.date}</span>`);
  }
  el.innerHTML = chips.join(' ');
}

// Part 3 -- the date control is only ever live for an AOI that actually has
// dated imagery indexed (today: leb). Switching AOI must repopulate (not
// just enable/disable) the date <select>, since "unknown" AOIs offer no
// dates at all and a stale option from a previous AOI would silently filter
// on a date that AOI's tiles can never carry (F-7: excluded, never raises --
// but a UI offering a dead option is its own kind of dishonesty, U-5's
// spirit applied to the control itself, not just the copy).
function updateDateOptionsForAoi(aoi) {
  const dates = datesByAoi[aoi] || [];
  const select = document.getElementById('date-select');
  const note = document.getElementById('date-note');
  const current = state.date;
  select.innerHTML = '<option value="">(any)</option>' +
    dates.map(d => `<option value="${d}">${d}</option>`).join('');
  select.disabled = dates.length === 0;
  note.textContent = dates.length === 0 ? 'No dated imagery indexed for this AOI.' : '';
  if (!dates.includes(current)) {
    state.date = '';
  }
  select.value = state.date;
}

function renderResults(data) {
  const el = document.getElementById('results');
  el.innerHTML = '';
  if (data.empty) return;
  // S11b: dispatch on payload shape (`data.models` only exists on an
  // /api/compare response) rather than on `state.compare` -- the renderer
  // reflects what the server actually answered, not what the toggle happens
  // to say right now (avoids ever painting a compare response into the
  // single-column layout, or vice versa, if a stale response lands after a
  // fast toggle).
  if (data.models) { renderCompareResults(data); return; }
  renderSingleResults(data);
}

function renderSingleResults(data) {
  const el = document.getElementById('results');
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
      <p class="scale-caption">${row.results.length} result(s) at this scale. Scores are only comparable within this row.</p>
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
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  renderActiveFilters([`<span class="filter-chip">Model: ${data.model_label} — ${total} result(s)</span>`]);
}

// S11b -- the compare view: one query, both models, results in two columns
// per scale row (never merged, interleaved or jointly ranked -- each
// column's tile-grid is built from that model's own `rankings` only, and no
// code path here ever reads one model's `results`/`score` while iterating
// the other's).
function renderCompareResults(data) {
  const el = document.getElementById('results');
  const modelIds = Object.keys(data.models);
  const scaleSet = new Set();
  for (const mid of modelIds) {
    const m = data.models[mid];
    if (m.available === false) continue;
    Object.keys(m.rankings || {}).forEach(s => scaleSet.add(Number(s)));
  }
  const scales = [...scaleSet].sort((a, b) => b - a);
  const perModelCounts = {};
  for (const mid of modelIds) perModelCounts[mid] = 0;

  for (const scale of scales) {
    let extent = null;
    for (const mid of modelIds) {
      const m = data.models[mid];
      if (m.available !== false && m.rankings[scale]) { extent = m.rankings[scale].ground_extent_m; break; }
    }
    const extentTxt = extent != null ? extent.toFixed(1) : '?';
    const section = document.createElement('div');
    section.className = 'scale-row';
    section.innerHTML = `
      <h2>${scale}px tile &mdash; ${extentTxt} m true ground extent</h2>
      <p class="scale-caption">Same query, same scale, two independent models -- scores are only comparable within one column.</p>
      <div class="compare-columns" id="compare-${scale}"></div>
    `;
    el.appendChild(section);
    const wrap = section.querySelector(`#compare-${scale}`);
    for (const mid of modelIds) {
      const m = data.models[mid];
      const col = document.createElement('div');
      col.className = 'compare-col';
      if (m.available === false) {
        col.innerHTML = `<h3>${m.model_label}</h3><p class="empty-row">No index for the selected AOI with this model.</p>`;
        wrap.appendChild(col);
        continue;
      }
      const row = m.rankings[scale] || { results: [], confidence: { band: 'unknown', message: '' } };
      perModelCounts[mid] += row.results.length;
      const conf = row.confidence || { band: 'unknown', message: '' };
      col.innerHTML = `
        <h3>${m.model_label}</h3>
        <p class="scale-caption">${row.results.length} result(s).</p>
        <span class="confidence-band ${conf.band}">confidence: ${conf.band}${conf.percentile != null ? ' (' + conf.percentile.toFixed(0) + 'th pct.)' : ''}</span>
        <p class="confidence-explainer">${conf.message}</p>
        <div class="tile-grid" id="grid-${mid}-${scale}"></div>
      `;
      wrap.appendChild(col);
      const grid = col.querySelector(`#grid-${mid}-${scale}`);
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
        card.addEventListener('click', () => openTileModal(r, scale, extentTxt));
        grid.appendChild(card);
      }
    }
  }
  const modelChips = modelIds.map(mid => {
    const m = data.models[mid];
    const n = m.available === false ? 'no index' : `${perModelCounts[mid]} result(s)`;
    return `<span class="filter-chip">${m.model_label} — ${n}</span>`;
  });
  renderActiveFilters(modelChips);
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

document.getElementById('aoi-select').addEventListener('change', (e) => {
  state.aoi = e.target.value;
  updateDateOptionsForAoi(state.aoi);
  runQuery(state.lastQuery);
});

// S11b -- switching the model re-runs the current query against that
// model's index (brief: "switching re-runs the current query against that
// model's index").
document.getElementById('model-select').addEventListener('change', (e) => {
  state.modelId = e.target.value;
  runQuery(state.lastQuery);
});

// S11b -- while comparing, the single-model selector does not apply (both
// models answer); disabling it (not hiding it) keeps U-6's "state is always
// legible" true of the control itself, not just the results.
document.getElementById('compare-toggle').addEventListener('change', (e) => {
  state.compare = e.target.checked;
  document.getElementById('model-select').disabled = state.compare;
  runQuery(state.lastQuery);
});

const defaultAoi = state.aoi;
const defaultModelId = state.modelId;
document.getElementById('clear-filters').addEventListener('click', () => {
  state.bbox = null; state.date = ''; state.aoi = defaultAoi;
  state.modelId = defaultModelId; state.compare = false;
  document.getElementById('bbox-input').value = '';
  document.getElementById('aoi-select').value = defaultAoi;
  document.getElementById('model-select').value = defaultModelId;
  document.getElementById('model-select').disabled = false;
  document.getElementById('compare-toggle').checked = false;
  updateDateOptionsForAoi(defaultAoi);
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
    aoi: str | None = None
    bbox: list[float] | None = None
    date: str | None = None
    top_k: int = retrieve.TOP_K
    model_id: str | None = None  # S11b -- None means the engine's default model


class CompareRequest(BaseModel):
    """S11b -- no `model_id`: a compare request always answers with every
    model the engine knows (`Engine.run_compare`'s default), never a single
    selected one."""

    text: str = ""
    aoi: str | None = None
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
            return answer_query(
                eng, req.text, aoi=req.aoi, bbox=req.bbox, date=req.date, top_k=top_k, model_id=req.model_id
            )
        except retrieve.UnitNormError as exc:
            log.exception("app: embedder returned a non-unit query vector")
            raise HTTPException(500, detail="the embedder returned an invalid vector for that query") from exc
        except ValueError as exc:  # unknown AOI/model (Engine.get_corpus/get_embedder) -- a clean 422, not a 500
            raise HTTPException(422, detail=str(exc)) from exc
        except Exception as exc:  # never a raw traceback to the client (S5 abuse-case requirement)
            log.exception("app: unexpected error answering query %r", req.text)
            raise HTTPException(500, detail=f"unexpected error answering that query: {exc.__class__.__name__}") from exc

    @fastapi_app.post("/api/compare")
    def compare(req: CompareRequest) -> dict:
        """S11b -- the same query, ranked independently through every model,
        never merged (see `answer_compare` / `Engine.run_compare`)."""
        if req.bbox is not None and len(req.bbox) != 4:
            raise HTTPException(422, detail="bbox must have exactly 4 numbers: min_lon, min_lat, max_lon, max_lat")
        top_k = max(1, min(req.top_k, 200))
        try:
            return answer_compare(eng, req.text, aoi=req.aoi, bbox=req.bbox, date=req.date, top_k=top_k)
        except retrieve.UnitNormError as exc:
            log.exception("app: embedder returned a non-unit query vector")
            raise HTTPException(500, detail="the embedder returned an invalid vector for that query") from exc
        except ValueError as exc:  # unknown AOI (Engine.get_corpus) -- a clean 422, not a 500
            raise HTTPException(422, detail=str(exc)) from exc
        except Exception as exc:  # never a raw traceback to the client (S5 abuse-case requirement)
            log.exception("app: unexpected error comparing query %r", req.text)
            raise HTTPException(500, detail=f"unexpected error comparing that query: {exc.__class__.__name__}") from exc

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
