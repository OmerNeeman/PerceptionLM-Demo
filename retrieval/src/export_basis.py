"""PCA export basis + int8 quantisation, and its measured accuracy cost --
brief S4, spec F-2a.

The local index (`embed_index.py`) keeps full precision (768-d float16 on
disk, renormalised to unit norm on load). This module builds the *separate*
export artifact S5 will ship: every indexed vector projected down to
`EXPORT_DIM` by PCA fit on the AOI's own corpus, then quantised to int8. The
basis (components + the int8 scale) is stored alongside the quantised
vectors so a **query** vector can be projected identically at query time --
`project_query` below is the one function both the export build and any
later query path must call, so "identically" is not a promise kept by two
independent implementations.

**`EXPORT_DIM` raised from 128 to 384 -- S4_fix Fix 3, owner decision from
the PM's measured dim-vs-accuracy curve (retrieval@10 overlap, int8, 8
queries, this index):** 128d -> 0.713 mean overlap (1.23 MB), 256d -> 0.800
(2.47 MB), **384d -> 0.833 (3.70 MB, chosen)**, 384d fp16 -> 0.871 (7.40 MB),
768d fp16 -> 1.000 (lossless -- the local index is already fp16, see below).
`measure_overlap`'s own module-level report/tests reproduce this module's
share of that curve at 384d.

**Deliberately NOT mean-centered -- measured, not a stylistic choice.** The
textbook PCA (subtract the corpus mean, eigendecompose the *covariance*
matrix) was the first thing tried here, and it is wrong for this use case in
a way that is easy to miss: this corpus's mean vector has norm 0.906 (the
768-d unit vectors are packed tightly around one dominant direction -- a
generic "aerial tile" component), so centering does not remove noise, it
subtracts most of the signal. Concretely, for a fixed query `q`, centered
similarity is `(q-mean)-(v-mean) = q.v - q.mean - v.mean + mean.mean`; the
first, third and fourth terms are constant across candidates `v` for that
query and do not affect ranking, but `-v.mean` is *not* constant -- it
penalises whichever tiles happen to be more aligned with the corpus mean,
which has nothing to do with relevance to `q`. Measured effect: retrieval@10
overlap against the full-precision ranking (six queries, same as
`is_weak_match`'s calibration set) was **0.00-0.10** at every scale with
centered PCA -- a false "the export budget is unusable" finding, not a real
one. Switching to the **uncentered** second-moment matrix (`X.T @ X / n`,
i.e. PCA through the origin -- the formulation that maximises retained
`sum ||Xw||^2`, not retained variance-from-centroid) recovered mean
retrieval@10 overlap of **0.70-0.73 per scale** on the same six queries. That
is the real number (see `measure_overlap`'s module-level report); the
centered version was a modelling bug, reported here rather than silently
fixed with no trace, because the failure mode (a real-looking measurement
that is actually testing a bug) is exactly what this project's test
discipline exists to catch.

PCA is fit once per AOI, over the **whole** indexed corpus pooled across
scales (one embedding space, not one per scale) -- via `numpy.linalg.eigh`
on the (dim x dim) uncentered second-moment matrix, not
`sklearn.decomposition.PCA`: the 768x768 eigendecomposition is cheap
regardless (no need for a randomised solver over 9,631 rows), and this
project's own measured reason to avoid a `scikit-learn` wrapper on the
*retrieval* path (68x slower than numpy brute force, CLAUDE.md/N-1) is one
more reason not to add the dependency's other wrapper here either.
Deterministic: `eigh` on a real symmetric matrix has no random seed to fix.

**The accuracy cost is measured, not assumed (F-2a):** `measure_overlap`
reports retrieval@10 overlap between the full-precision ranking and the
exported (PCA-128 + int8) ranking, **per scale** -- a single pooled number
would hide a scale-specific loss, and 112 px is where the small objects
live (S0/F-4). See `retrieve.py` for TOP_K and the per-scale ranking this
mirrors.

Layout, one directory per AOI, gitignored under the index root::

    <index_root>/export/<aoi>/basis.json        -- provenance, int8 scale,
                                                     tile_id/scale per row
    <index_root>/export/<aoi>/components.npy     -- (EXPORT_DIM, dim) float32
    <index_root>/export/<aoi>/vectors_int8.npy   -- (n, EXPORT_DIM) int8

Environment: every invocation must be prefixed inline, every time --

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 AERIAL_DATA_ROOT=<path> <python> ...

(see INSTRUCTIONS.md for the interpreter path on the dev machine).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

import config
import embed_index
import embedders

log = logging.getLogger(__name__)

#: Config symbol (spec.md's "EXPORT_DIM (PCA target, default 128)" -- raised
#: to 384 by S4_fix Fix 3, an owner decision from the measured
#: dim-vs-accuracy curve in this module's docstring; still int8).
EXPORT_DIM = 384

#: int8 covers [-127, 127] symmetrically (not -128) so quantise/dequantise
#: round-trips through a single scale factor with no off-by-one at either end.
INT8_MAX = 127

BASIS_NAME = "basis.json"
COMPONENTS_NAME = "components.npy"
VECTORS_INT8_NAME = "vectors_int8.npy"


def export_dir(aoi: str, index_root: Path | None = None) -> Path:
    root = index_root or config.get_index_root()
    return root / "export" / aoi


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# PCA -- fit + project. One function for both the export build and query
# time, so "projects identically" (F-2a's expectation) is structural, not a
# promise two call sites happen to keep in sync.
# --------------------------------------------------------------------------


def fit_pca(vectors: np.ndarray, export_dim: int = EXPORT_DIM) -> np.ndarray:
    """Top-`export_dim` principal directions of `vectors` (n, dim), through
    the **origin** -- see module docstring for why this is not the textbook
    mean-centered PCA. Via eigendecomposition of the (dim, dim) uncentered
    second-moment matrix `X.T @ X / n` -- cheap here regardless of n, and
    exact/deterministic (no randomised solver, no seed to fix).
    """
    x = vectors.astype(np.float64)  # fp64 accumulation for a stabler second-moment matrix
    n = x.shape[0]
    gram = (x.T @ x) / max(n, 1)
    eigvals, eigvecs = np.linalg.eigh(gram)  # ascending order
    order = np.argsort(eigvals)[::-1][:export_dim]
    components = eigvecs[:, order].T.astype(np.float32)  # (export_dim, dim)
    return components


def project_query(vec: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project one (or many) full-precision vector(s) through a fitted
    basis (no centering -- see module docstring). The single function both
    `build_export` and any later query path must call, so a query embeds
    into the same coordinates the stored export vectors were quantised in.
    """
    return vec @ components.T


def _quantize_int8(arr: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.round(arr / scale), -INT8_MAX, INT8_MAX).astype(np.int8)


def _dequantize_int8(arr_i8: np.ndarray, scale: float) -> np.ndarray:
    return arr_i8.astype(np.float32) * scale


# --------------------------------------------------------------------------
# Build + load the export artifact.
# --------------------------------------------------------------------------


def build_export(aoi: str, index_root: Path | None = None, export_dim: int = EXPORT_DIM) -> dict:
    """Fit PCA on the AOI's indexed corpus, quantise to int8, write the
    basis under `export_dir` (never the local index directory -- a separate
    artifact). Returns a small report dict; does not mutate the local index.
    """
    loaded = embed_index.load_index(aoi, index_root=index_root)  # full precision, renormalised
    manifest, vectors = loaded["manifest"], loaded["vectors"]

    components = fit_pca(vectors, export_dim)
    projected = project_query(vectors, components).astype(np.float32)
    peak = float(np.abs(projected).max()) if projected.size else 0.0
    scale = peak / INT8_MAX if peak > 0 else 1.0
    vectors_i8 = _quantize_int8(projected, scale)

    out_dir = export_dir(aoi, index_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / COMPONENTS_NAME, components)
    np.save(out_dir / VECTORS_INT8_NAME, vectors_i8)
    basis = {
        "aoi": manifest["aoi"],
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "dim": manifest["dim"],
        "export_dim": export_dim,
        "int8_scale": scale,
        "tile_ids": [t["tile_id"] for t in manifest["tiles"]],
        "scales": [t["scale"] for t in manifest["tiles"]],
    }
    _atomic_write_json(out_dir / BASIS_NAME, basis)
    return {
        "aoi": aoi,
        "out_dir": str(out_dir),
        "n": int(vectors_i8.shape[0]),
        "export_dim": export_dim,
        "int8_scale": scale,
    }


def load_export(aoi: str, index_root: Path | None = None) -> dict:
    """Reload one AOI's export artifact from disk (the fresh-process
    verification path this brief's test discipline requires -- do not trust
    what `build_export` returned in memory)."""
    out_dir = export_dir(aoi, index_root)
    basis = json.loads((out_dir / BASIS_NAME).read_text())
    components = np.load(out_dir / COMPONENTS_NAME)
    vectors_i8 = np.load(out_dir / VECTORS_INT8_NAME)
    if vectors_i8.shape[0] != len(basis["tile_ids"]):
        raise RuntimeError(
            f"{out_dir}: basis lists {len(basis['tile_ids'])} tile ids but "
            f"{VECTORS_INT8_NAME} has {vectors_i8.shape[0]} rows -- inconsistent export"
        )
    return {"basis": basis, "components": components, "vectors_i8": vectors_i8}


def exported_vectors_f32(loaded_export: dict) -> np.ndarray:
    """Dequantised (n, EXPORT_DIM) float32 -- what a search against the
    export would actually score against."""
    return _dequantize_int8(loaded_export["vectors_i8"], loaded_export["basis"]["int8_scale"])


# --------------------------------------------------------------------------
# F-2a -- the measurement the owner asked for: retrieval@10 overlap between
# full-precision and exported embeddings, per scale.
# --------------------------------------------------------------------------


def measure_overlap(
    aoi: str,
    queries: Sequence[str],
    *,
    index_root: Path | None = None,
    top_k: int = 10,
    embedder=None,
) -> dict[int, dict]:
    """Per scale: mean retrieval@`top_k` overlap (Jaccard-style: fraction of
    the full-precision top-`top_k` tile ids also present in the exported
    top-`top_k`) across `queries`, plus the per-query breakdown.

    Reloads both the full-precision index and the export from disk (not the
    in-memory arrays `build_export` just computed) -- this brief's "verify
    what you actually wrote, in a fresh process" discipline.
    """
    loaded = embed_index.load_index(aoi, index_root=index_root)
    manifest, vectors = loaded["manifest"], loaded["vectors"]
    exp = load_export(aoi, index_root=index_root)
    if exp["basis"]["tile_ids"] != [t["tile_id"] for t in manifest["tiles"]]:
        raise RuntimeError(
            f"{aoi}: export basis tile order does not match the current index -- "
            "rebuild the export before measuring overlap against it"
        )
    exported_vectors = exported_vectors_f32(exp)
    components = exp["components"]

    tile_scales = np.array([t["scale"] for t in manifest["tiles"]], dtype=np.int64)
    emb = embedder or embedders.load_embedder(embed_index.DEFAULT_MODEL_ID)
    qvecs = np.asarray(emb.embed_texts(list(queries)), dtype=np.float32)

    report: dict[int, dict] = {}
    for scale in manifest["scales"]:
        idx = np.where(tile_scales == scale)[0]
        full_v = vectors[idx]
        exp_v = exported_vectors[idx]
        k = min(top_k, idx.size)
        per_query = []
        for text, qv in zip(queries, qvecs):
            full_scores = full_v @ qv
            q_proj = project_query(qv, components)
            exp_scores = exp_v @ q_proj
            full_top = set(idx[np.argpartition(-full_scores, k - 1)[:k]].tolist())
            exp_top = set(idx[np.argpartition(-exp_scores, k - 1)[:k]].tolist())
            overlap = len(full_top & exp_top) / k if k else 0.0
            per_query.append({"query": text, "overlap": overlap})
        report[scale] = {
            "n": int(idx.size),
            "top_k": k,
            "mean_overlap": float(np.mean([p["overlap"] for p in per_query])) if per_query else None,
            "per_query": per_query,
        }
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    p = argparse.ArgumentParser(description="Build the PCA/int8 export basis for one AOI.")
    p.add_argument("--aoi", required=True)
    p.add_argument("--export-dim", type=int, default=EXPORT_DIM)
    p.add_argument(
        "--measure", nargs="*", default=None,
        help="also measure retrieval@10 overlap for these query strings (at least one "
        "recommended; omit to skip the measurement)",
    )
    args = p.parse_args()
    report = build_export(args.aoi, export_dim=args.export_dim)
    print(json.dumps(report, indent=2))
    if args.measure:
        overlap = measure_overlap(args.aoi, args.measure)
        print(json.dumps(overlap, indent=2))


if __name__ == "__main__":
    main()
