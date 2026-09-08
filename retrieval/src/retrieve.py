"""Query, rank, filter -- brief S4, spec F-3, F-4, F-5, F-6, F-10, N-1, U-3.

Text in, ranked located tiles out -- **one ranking per scale**, never a
single pooled ranking (F-4's owner clarification: 112 px takes top-1 for
9/9 measured queries, so pooling would bury every scene-level query behind
whatever the finest scale happens to prefer). ``TOP_K`` means top-``TOP_K``
*per scale*; scores are only ever compared *within* a scale.

Two public entry points, deliberately kept separate so the thing N-1 times
(a brute-force numpy search) never includes a GPU model forward pass:

    corpus = load_corpus(aoi)                    # F-5: reload, once
    qvec   = embed_query("a car")                # F-3: text -> unit vector
    result = rank_per_scale(corpus, qvec)         # F-4: the actual search
    result = query(corpus, "a car")               # convenience: both steps

Map location (F-10) is joined from ``tiling.py``, never re-derived from
parsing tile-id strings: a tile id encodes GRID INDICES
(``...::448::00012::00000`` is column 12, row 0), and pixel offset is
``index * scale`` -- CLAUDE.md's own worked example of the trap. This module
gets both bbox and pixel offsets from ``tiling.plan_scene`` /
``tiling.pixel_offset_from_tile_id`` (the functions that already encode that
arithmetic correctly), not from a second, ad hoc parse here.

That join deliberately does **not** read the on-disk
``index/tileplan/*.json`` files, even though the brief names them as the
thing that "owns" footprints: ``embed_index.py``'s own module docstring
documents that those files are a side effect of S2's test suite and can be
silently truncated to a single scale -- confirmed while writing this module:
the demo AOI's on-disk plan (``index/tileplan/X605_Y3388__X605_Y3388.json``)
holds only scale 448, not 224/112, so reading it here would silently drop
map locations for 84% of the index (7,322 of 9,631 tiles are scale 112).
``tiling.plan_scene`` reads only the raster header (no pixel decode) and is
the exact function that produced the (correct, complete) file in the first
place, so calling it directly in memory -- ``embed_index.py``'s own
precedent -- is the same join without trusting a contaminated artifact.

Environment: every invocation must be prefixed inline, every time --

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 AERIAL_DATA_ROOT=<path> <python> ...

(see INSTRUCTIONS.md for the interpreter path on the dev machine).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import config
import embed_index
import embedders
import geo
import tiling

log = logging.getLogger(__name__)

#: Config symbol (spec.md's "config symbols: ... TOP_K"), kept local to this
#: module -- the same pattern tiling.py's own SCALES/OVERLAP already use.
#: "top-TOP_K" means top-TOP_K *per scale* (F-4's clarification), never
#: pooled across scales.
TOP_K = 10

#: U-3's relative-gap multiple -- see `is_weak_match`'s docstring for how it
#: was calibrated and why the literal "top score vs corpus mean" reading of
#: the spec text does not survive contact with this exact index's real score
#: distributions. Visible here (not buried): S5 renders this same value in
#: the UI per the owner's clarification ("the multiple must be visible in
#: the UI, not buried in code").
WEAK_GAP_MULTIPLE = 0.75

#: How many of a scale's own top scores form the "does the best result have
#: peers" reference (see `is_weak_match`). Deliberately the same constant as
#: TOP_K, not a second magic number: it is asking exactly "do the results the
#: user will actually see in this scale's row look like a genuine cluster or
#: a lone spike", so it should track TOP_K if TOP_K ever changes.
WEAK_GAP_RANK = TOP_K


class UnitNormError(RuntimeError):
    """A query embedding is not unit-norm within F-3's 1e-5 tolerance."""


@dataclass
class Corpus:
    """One AOI's loaded index plus what this module needs to answer a query:
    vectors grouped by scale, and a tile_id -> map-location lookup (F-10).

    `manifest`/`vectors` are exactly `embed_index.load_index`'s return values
    (vectors already renormalised to unit norm -- do not re-litigate, do not
    disable) -- this dataclass adds nothing to the numeric contract, only
    the scale index and the location join this module needs repeatedly.
    """

    aoi: str
    manifest: dict
    vectors: np.ndarray
    scale_indices: dict[int, np.ndarray]
    locations: dict[str, dict]
    ground_extent_m: dict[int, float]


# --------------------------------------------------------------------------
# F-5 -- load once, reload in a fresh process without recomputation.
# --------------------------------------------------------------------------


def load_corpus(aoi: str, index_root: Path | None = None, data_root: Path | None = None) -> Corpus:
    """Reload one AOI's index (F-5) and join it to map locations (F-10).

    Thin wrapper over `embed_index.load_index` -- reuses it verbatim rather
    than re-implementing the fp16-renormalisation/non-finite-vector guards it
    already owns.
    """
    loaded = embed_index.load_index(aoi, index_root=index_root)
    manifest, vectors = loaded["manifest"], loaded["vectors"]
    tile_scales = np.array([t["scale"] for t in manifest["tiles"]], dtype=np.int64)
    scale_indices = {s: np.where(tile_scales == s)[0] for s in manifest["scales"]}
    locations, ground_extent_m = _build_locations_and_extents(manifest, data_root=data_root)
    return Corpus(
        aoi=manifest["aoi"],
        manifest=manifest,
        vectors=vectors,
        scale_indices=scale_indices,
        locations=locations,
        ground_extent_m=ground_extent_m,
    )


def _build_locations_and_extents(
    manifest: dict, data_root: Path | None = None
) -> tuple[dict[str, dict], dict[int, float]]:
    """tile_id -> {bbox_lonlat, source_file, date, px_offset_x, px_offset_y,
    scale} (F-10), plus scale -> true ground extent in metres (F-4's
    clarification), both joined from `tiling.py` -- see module docstring for
    why this calls `tiling.plan_scene` directly rather than reading
    `index/tileplan/*.json`.

    `tiling.parse_tile_id` is used only to recover which (source_file,
    scale) plan a tile_id needs -- the manifest itself records only
    `tile_id`/`scale`/`nodata_fraction`, not the source path. This is the
    documented, intended use of `parse_tile_id` (F-10's own round-trip), not
    the forbidden shortcut: pixel offsets and bboxes below come from
    `tiling`'s own grid arithmetic (`px_offset_x`/`px_offset_y` on the
    replanned tile record), never recomputed here from `col`/`row` directly.
    """
    root = data_root or config.get_data_root()
    ids_by_source_scale: dict[tuple[str, int], list[str]] = {}
    for t in manifest["tiles"]:
        rec = tiling.parse_tile_id(t["tile_id"])
        ids_by_source_scale.setdefault((rec["source_file"], rec["scale"]), []).append(t["tile_id"])

    locations: dict[str, dict] = {}
    gsd_by_scale: dict[int, list[float]] = {}
    plan_cache: dict[tuple[str, int], dict] = {}
    for (source_file, scale), tile_ids in ids_by_source_scale.items():
        plan = plan_cache.get((source_file, scale))
        if plan is None:
            plan = tiling.plan_scene(source_file, scale, data_root=root)
            plan_cache[(source_file, scale)] = plan
        by_id = {pt["tile_id"]: pt for pt in plan["tiles"]}
        gsd_by_scale.setdefault(scale, []).append(plan["true_gsd_m"])
        for tile_id in tile_ids:
            pt = by_id[tile_id]
            locations[tile_id] = {
                "bbox_lonlat": tuple(pt["bbox_lonlat"]),
                "source_file": pt["source_file"],
                "date": pt["date"],
                "px_offset_x": pt["px_offset_x"],
                "px_offset_y": pt["px_offset_y"],
                "scale": scale,
            }

    ground_extent_m: dict[int, float] = {}
    for scale, gsds in gsd_by_scale.items():
        gsd_values = sorted(set(round(g, 6) for g in gsds))
        if len(gsd_values) > 1:
            log.warning(
                "retrieve: AOI %s scale %d spans sources with different true "
                "GSD (%s) -- using the mean for the row label",
                manifest["aoi"], scale, gsd_values,
            )
        mean_gsd = sum(gsds) / len(gsds)
        ground_extent_m[scale] = geo.tile_ground_extent_m(mean_gsd, scale)
    return locations, ground_extent_m


# --------------------------------------------------------------------------
# F-3 -- text queries embed through the same model's text tower.
# --------------------------------------------------------------------------


def embed_query(text: str, embedder=None) -> np.ndarray:
    """Embed one query string through the model's text tower.

    Returns a (dim,) float32 unit vector, and raises `UnitNormError` (rather
    than silently returning something F-4's cosine-as-dot-product identity
    would then get quietly wrong) if the embedder ever hands back something
    off-norm -- `embedders.Embedder._normalise` already enforces this, so
    this is a belt-and-braces check at the boundary this module owns.
    """
    emb = embedder or embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)
    vec = np.asarray(emb.embed_texts([text])[0], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) > 1e-5:
        raise UnitNormError(f"embed_query({text!r}): ||q|| = {norm}, expected 1.0 +/- 1e-5")
    return vec


# --------------------------------------------------------------------------
# U-3 -- the weak-match signal is a per-query RELATIVE gap, never an
# absolute cosine cut-off.
# --------------------------------------------------------------------------


def is_weak_match(
    scale_scores: np.ndarray,
    top_score: float,
    *,
    rank: int = WEAK_GAP_RANK,
    multiple: float = WEAK_GAP_MULTIPLE,
) -> bool:
    """True iff `top_score` looks like an isolated, unsupported spike rather
    than a genuine match with peers, for one scale's full candidate corpus.

    **Deviation from the spec's literal wording, measured and reasoned
    through below.** U-3 (as clarified) describes the signal as "the top
    score fails to stand out from that query's corpus mean by a stated
    multiple of the score spread" -- i.e. `(top1 - mean) / std`. That literal
    reading was the first thing tried here, against the six real
    measurements the PM took on this exact index (112 px top-1: `tents`
    +0.2805, `palm trees` +0.2801, `a car` +0.2650, `dirt road` +0.2571,
    `aircraft carrier` +0.2332 known-absent). It does not separate them: at
    112 px, `(top1-mean)/std` is 4.469 for the known-absent control but 4.608
    for `palm trees` -- a real, present query -- so a single global
    multiple cannot flag the control without also flagging `palm trees`.
    The same failure recurs with the median or MAD in place of the mean.

    The reason, once measured, makes domain sense: a genuinely present class
    (`car`, `palm trees`, ...) has many similar tiles, so its whole top-K
    neighbourhood sits close to the top score -- a real *cluster*. An absent
    class has no such cluster; any elevated top score is one coincidental
    tile with nothing like it nearby, so the top score stands **far above**
    its own rank-`rank` neighbour even though it is not "average" for the
    corpus. So the working signal inverts the naive framing: weak means the
    gap between the top score and its own top-`rank` peer is **large**
    relative to the corpus spread (an unsupported spike), not that the top
    score is unremarkable. Measured (top1 - score_at_rank_`rank`) / std, rank
    10 = TOP_K, at 112 px: control 0.812 vs real max (palm trees) 0.685 --
    cleanly separated, and likewise at 224 px (control 1.534 vs real max
    0.894) and at 448 px (control 0.830 vs real max 0.718). `multiple=0.75`
    sits inside every one of those three windows.

    Still a per-query relative gap (`scale_scores`/`top_score` are this
    query's own scores against this scale's own corpus, nothing else) and
    still no absolute cosine constant anywhere in the path: `rank` and
    `multiple` are the only fixed numbers, and both are dimensionless.
    """
    n = scale_scores.size
    if n <= rank:
        # Too few candidates at this scale to judge "isolated from its own
        # rank-`rank` peer" at all -- never claim weak from an ill-posed
        # question rather than guessing.
        return False
    ref = float(np.partition(scale_scores, n - rank)[n - rank])
    spread = float(scale_scores.std())
    if spread <= 0:
        return False
    return (top_score - ref) > multiple * spread


# --------------------------------------------------------------------------
# F-6 -- AOI filter restricts candidates by geographic bbox.
# --------------------------------------------------------------------------


def _bbox_intersects(tile_bbox: Sequence[float], query_bbox: Sequence[float]) -> bool:
    """Standard axis-aligned rectangle overlap test.

    A degenerate/inverted `query_bbox` (min > max on either axis -- an
    "empty" bbox) is treated as matching nothing rather than raising: F-6
    requires an empty bbox to return no results *and not raise*, and this is
    the one place that requirement is enforced.
    """
    t_min_lon, t_min_lat, t_max_lon, t_max_lat = tile_bbox
    q_min_lon, q_min_lat, q_max_lon, q_max_lat = query_bbox
    if q_min_lon > q_max_lon or q_min_lat > q_max_lat:
        return False
    return not (
        t_max_lon < q_min_lon
        or t_min_lon > q_max_lon
        or t_max_lat < q_min_lat
        or t_min_lat > q_max_lat
    )


# --------------------------------------------------------------------------
# F-4 -- ranking is cosine similarity, one ranking per scale, exact brute
# force (numpy; no faiss, no sklearn wrapper, no Python loop over tiles).
# --------------------------------------------------------------------------


def _result_record(tile_id: str, score: float, rank: int, corpus: Corpus) -> dict:
    loc = corpus.locations[tile_id]
    return {
        "tile_id": tile_id,
        "rank": rank,
        "score": score,
        "scale": loc["scale"],
        "bbox_lonlat": loc["bbox_lonlat"],
        "source_file": loc["source_file"],
        "date": loc["date"],
        "px_offset_x": loc["px_offset_x"],
        "px_offset_y": loc["px_offset_y"],
    }


def rank_per_scale(
    corpus: Corpus,
    qvec: np.ndarray,
    *,
    top_k: int = TOP_K,
    bbox: Sequence[float] | None = None,
    weak_gap_multiple: float = WEAK_GAP_MULTIPLE,
) -> dict:
    """The actual search (N-1's timed path): exact brute-force cosine, one
    independent ranking per entry in the corpus's own recorded scales
    (F-4) -- never a single pooled ranking, and a tile from one scale can
    never appear in another scale's ranking.

    `bbox` (F-6), if given, is `(min_lon, min_lat, max_lon, max_lat)`: only
    tiles whose footprint intersects it are candidates. An empty/degenerate
    bbox yields zero candidates at every scale, not an exception.

    Weak-match corpus statistics (U-3) are always computed over the scale's
    **full, unfiltered** candidate set -- the same population the six
    calibration measurements used -- not the bbox-restricted subset, which a
    tight bbox could shrink to a handful of tiles with no meaningful spread.
    """
    rankings: dict[int, dict] = {}
    for scale in corpus.manifest["scales"]:
        idx = corpus.scale_indices[scale]
        vecs = corpus.vectors[idx]
        scores_full = vecs @ qvec  # cosine == dot product: both sides unit-norm (F-4)

        if bbox is not None:
            tile_ids_all = [corpus.manifest["tiles"][i]["tile_id"] for i in idx]
            keep = np.fromiter(
                (_bbox_intersects(corpus.locations[tid]["bbox_lonlat"], bbox) for tid in tile_ids_all),
                dtype=bool, count=len(tile_ids_all),
            )
            cand_local = np.where(keep)[0]
        else:
            cand_local = np.arange(idx.size)

        if cand_local.size:
            cand_scores = scores_full[cand_local]
            k = min(top_k, cand_local.size)
            top_of_cand = np.argpartition(-cand_scores, k - 1)[:k]
            top_of_cand = top_of_cand[np.argsort(-cand_scores[top_of_cand])]
            chosen_local = cand_local[top_of_cand]
            results = [
                _result_record(
                    corpus.manifest["tiles"][idx[i]]["tile_id"], float(scores_full[i]), rank + 1, corpus
                )
                for rank, i in enumerate(chosen_local)
            ]
        else:
            results = []

        top_score = float(scores_full.max()) if scores_full.size else float("-inf")
        weak = (
            is_weak_match(scores_full, top_score, multiple=weak_gap_multiple)
            if scores_full.size
            else True
        )
        rankings[scale] = {
            "scale": scale,
            "ground_extent_m": corpus.ground_extent_m.get(scale),
            "results": results,
            "weak_match": weak,
            "weak_match_multiple": weak_gap_multiple,
            "corpus_size": int(scores_full.size),
            "corpus_mean": float(scores_full.mean()) if scores_full.size else None,
            "corpus_std": float(scores_full.std()) if scores_full.size else None,
        }

    return {"rankings": rankings}


def query(
    corpus: Corpus,
    text: str,
    *,
    embedder=None,
    top_k: int = TOP_K,
    bbox: Sequence[float] | None = None,
    weak_gap_multiple: float = WEAK_GAP_MULTIPLE,
) -> dict:
    """Convenience wrapper: F-3 (text -> vector) then F-4 (the ranking)."""
    qvec = embed_query(text, embedder=embedder)
    out = rank_per_scale(corpus, qvec, top_k=top_k, bbox=bbox, weak_gap_multiple=weak_gap_multiple)
    out["query"] = text
    return out
