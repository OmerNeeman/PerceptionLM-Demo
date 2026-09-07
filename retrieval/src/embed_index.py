"""Embedding-index build pipeline -- brief S3, spec F-1a, F-2, N-2, N-4.

Turns one scene's tile pyramid (S2's ``tiling.py``) into a reloadable
on-disk index of unit-normalised, model-tagged vectors, using the S0-chosen
embedder (``embedders.py``) through its public ``embed_images`` path only --
pooled output, never patch tokens (settled, see spec F-2 and CLAUDE.md).

Layout, one directory per AOI, everything gitignored under the index root::

    <index_root>/emb/<aoi>/manifest.json   -- one model id, one revision,
                                               one record per embedded tile
                                               (tile_id, scale, nodata_fraction)
    <index_root>/emb/<aoi>/vectors.npy     -- (n, dim) float16, row i is
                                               manifest["tiles"][i]'s vector

The tile plan itself is **not** read from the on-disk
``index/tileplan/*.json`` files -- those are a shared side effect of S2's own
test suite (one of its tests calls ``plan_all(scales=(448,))`` last and
overwrites every AOI's plan file down to a single scale), so depending on
their content here would be fragile. Instead this module calls
``tiling.plan_scene`` directly, in memory, once per scale -- exactly the
pattern S2's own test suite (``test_tiling.py``'s ``all_plans`` fixture)
already uses, not a new way of doing it.

Resumability (N-2): a build reads the existing manifest, skips every
tile_id already recorded there, and checkpoints (vectors.npy, then
manifest.json, both via write-tmp-then-rename) after every batch. Vectors
are written before the manifest that references them, so a kill at any
point can only ever leave vectors.npy with *more* rows than the manifest
lists (harmless -- truncated away on the next load), never fewer.

Determinism (N-4) and the "one embedder" guarantee (F-1a) are both the
embedder's and this module's job respectively: ``embedders.py`` fixes seeds
and cudnn determinism; this module refuses (raises ``MixedEmbedderError``)
to append vectors from a different model id or revision onto an existing
index.

A known, measured numerical fact this module works around -- read before
touching the F-2 norm check: an fp32 unit vector cast to float16 and back
does **not** generally satisfy ``||v|| = 1.0 +/- 1e-5``. Measured on 300 real
RemoteCLIP embeddings of real crops from the demo AOI: max deviation
2.26e-4, mean 6.2e-5, 266/300 (89%) exceed 1e-5 -- fp16's 10-bit mantissa is
the limit, not a bug, and no per-vector rescaling before storage changes it
(quantisation is idempotent once a value is on the fp16 grid). Vectors are
still stored as fp16 (F-2 requires it; fp32 would double the size for no
reason at this corpus scale). ``load_index`` is where this is resolved: it
renormalises the fp32-upcast vectors on load (an O(n*d) division, negligible
cost) so that what later stages actually receive satisfies 1e-5 -- see
``load_index``'s docstring for the sanity bound that still catches a
genuinely broken (not just fp16-quantised) vector.

Environment: every invocation must be prefixed inline, every time --

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 AERIAL_DATA_ROOT=<path> <python> ...

(see INSTRUCTIONS.md for the interpreter path on the dev machine).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from PIL import Image

import config
import embedders
import tiling

log = logging.getLogger(__name__)

#: The settled S0 choice (CLAUDE.md, spec F-0) -- callers may still name a
#: different candidate explicitly (that is exactly what F-1a's own test
#: needs to do), this is only the default.
DEFAULT_MODEL_ID = "RemoteCLIP-ViT-L-14"

#: Checkpoint / GPU-call granularity. Independent of embedders.py's own
#: internal sub-batching (its `embed_images` default is 32) -- this is how
#: many tiles this module reads off disk and checkpoints together, not the
#: model's forward-pass batch size.
DEFAULT_BATCH_SIZE = 256

MANIFEST_NAME = "manifest.json"
VECTORS_NAME = "vectors.npy"

#: F-2: everything downstream of fp16 storage tolerates this much norm
#: deviation from fp16 quantisation alone (measured up to 2.26e-4 on real
#: data). Anything past this is not quantisation noise -- it is a real bug
#: (wrong dtype, unnormalised vector, patch tokens) and must raise.
RAW_NORM_SANITY_ATOL = 1e-2


class MixedEmbedderError(RuntimeError):
    """A build would append vectors from a different model id or revision
    onto an existing index -- F-1a. Raised, never worked around."""


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def emb_dir(aoi: str, index_root: Path | None = None) -> Path:
    root = index_root or config.get_index_root()
    return root / "emb" / aoi


def _manifest_paths(aoi_dir: Path) -> tuple[Path, Path]:
    return aoi_dir / MANIFEST_NAME, aoi_dir / VECTORS_NAME


# --------------------------------------------------------------------------
# Atomic on-disk state -- write-tmp-then-rename, vectors before manifest.
# --------------------------------------------------------------------------


def _load_state(manifest_path: Path, vectors_path: Path):
    """Existing (manifest dict, vectors array) or (None, None) if this is a
    fresh build. Vectors are truncated to `len(manifest["tiles"])` rows --
    defensive against a kill landing between the two atomic writes of one
    checkpoint, which (by the vectors-then-manifest write order below) can
    only ever leave vectors.npy with *extra*, not missing, rows."""
    if not manifest_path.exists():
        return None, None
    manifest = json.loads(manifest_path.read_text())
    n = len(manifest["tiles"])
    if vectors_path.exists():
        vectors = np.load(vectors_path)
        # S3-fix, Fix 2: the write-ordering argument above says a kill can
        # only ever leave vectors.npy with MORE rows than the manifest lists
        # (truncated away below, harmless) -- never fewer. That argument was
        # never checked at runtime: the reviewer hand-built a manifest
        # listing 16 tiles against a 10-row vectors.npy and `build_index`
        # computed `todo` as empty (every tile_id already counted "done"),
        # returning a false `{"embedded_total": 16}` success report. Raise
        # here, before any caller decides what work remains, rather than
        # trusting an argument that holds only until it doesn't.
        if vectors.shape[0] < n:
            raise RuntimeError(
                f"{manifest_path.parent}: manifest lists {n} tiles but "
                f"vectors.npy has only {vectors.shape[0]} rows -- inconsistent "
                f"index, refusing to resume from it (a build must never report "
                f"success without validating what it resumed from)"
            )
        vectors = vectors[:n]
    else:
        vectors = np.zeros((0, manifest.get("dim") or 0), dtype=np.float16)
    return manifest, vectors


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _atomic_save_vectors(path: Path, arr: np.ndarray) -> None:
    # tmp already ends in .npy so np.save does not append a second one.
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _checkpoint(manifest_path: Path, vectors_path: Path, manifest: dict, vectors: np.ndarray) -> None:
    """Vectors first, then the manifest that references them (see module
    docstring): a kill between the two writes can only leave vectors.npy
    with rows the manifest does not yet reference (harmless), never the
    reverse (a manifest entry with no backing vector)."""
    _atomic_save_vectors(vectors_path, vectors)
    _atomic_write_json(manifest_path, manifest)


# --------------------------------------------------------------------------
# Reading one tile's pixels off the source raster.
# --------------------------------------------------------------------------


def _open_raster(path: Path):
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    return rasterio.open(path)  # mode defaults to "r" -- data dir is read-only


def _read_tile_array(ds, tile: dict) -> np.ndarray:
    """One tile's RGB crop, padded to `scale x scale` with tiling.PAD_VALUE
    when it spills off the raster edge. `valid_width`/`valid_height` are
    geometric, computed at plan time from raster dimensions alone (S2) --
    no extra pixel read is needed to know the pad shape. Returns HWC uint8,
    ready for `PIL.Image.fromarray`."""
    scale = tile["scale"]
    vw, vh = tile["valid_width"], tile["valid_height"]
    x0, y0 = tile["px_offset_x"], tile["px_offset_y"]
    window = Window(x0, y0, vw, vh)
    arr = ds.read((1, 2, 3), window=window)  # (3, vh, vw)
    out = np.full((scale, scale, 3), tiling.PAD_VALUE, dtype=arr.dtype)
    out[:vh, :vw, :] = np.moveaxis(arr, 0, -1)
    return out


def _nodata_fraction(tile_arr: np.ndarray) -> float:
    """Fraction of pixels that are the mosaics' zero-fill sentinel (all of
    bands 1-3 == 0) -- the same definition `tiling._nodata_mask` uses.
    Computed over the *padded* `scale x scale` tile: PAD_VALUE (0) matches
    the sentinel by construction (tiling.py's own docstring), so a padded
    edge tile and a genuinely nodata interior tile read identically."""
    mask = (tile_arr[..., 0] == 0) & (tile_arr[..., 1] == 0) & (tile_arr[..., 2] == 0)
    return float(mask.mean())


def _ordered_tile_plan(rel_path: str, scales, data_root: Path | None, limit: int | None) -> list[dict]:
    """Every tile of `rel_path` across `scales`, scale-major then row-major
    within each scale (`tiling.plan_scene`'s own order) -- one fixed,
    deterministic order every build agrees on, so `manifest["tiles"][i]`
    always means the same tile across runs. `limit` (testing/calibration
    only, never used in the production build) truncates the plan itself,
    not just what gets embedded."""
    tiles: list[dict] = []
    for s in scales:
        plan = tiling.plan_scene(rel_path, s, data_root=data_root)
        tiles.extend(plan["tiles"])
    if limit is not None:
        tiles = tiles[:limit]
    return tiles


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_index(
    rel_path: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    scales=tiling.SCALES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    data_root: Path | None = None,
    index_root: Path | None = None,
    embedder=None,
    limit: int | None = None,
    batch_sleep_s: float = 0.0,
) -> dict:
    """Embed one scene's tile pyramid into a reloadable on-disk index.

    `batch_sleep_s` (default 0, never set by the production build): sleep
    this long after each checkpoint. Exists solely to make
    ``test_resume_does_not_reembed`` able to land a real, uncontrolled kill
    reliably between two checkpoints instead of racing GPU throughput --
    it changes timing only, never what gets written or in what order.

    F-1a: raises `MixedEmbedderError` (naming both the recorded and the
    requested model id/revision) rather than silently appending vectors from
    a different embedder onto an existing index.

    N-2: resumable. Tiles already present in the manifest are never re-read
    or re-embedded; every batch is checkpointed atomically so a kill at any
    point leaves a valid, reloadable index and a restart continues rather
    than starting over. Consistency between the manifest and vectors.npy is
    validated (S3-fix, Fix 2) before any todo list is computed: a manifest
    listing more tiles than vectors.npy actually holds raises, naming both
    counts, rather than silently reporting the manifest's count as done.

    Nodata (brief's decision, not the planner's stricter >50% rule): every
    tile with at least one real pixel is embedded; only a 100%-nodata tile is
    skipped, and every embedded tile records its `nodata_fraction`.

    N-4 amendment: an index is only reproducible at a fixed `batch_size`
    (GPU fp16 GEMM kernel selection depends on batch shape -- up to 2.574e-4
    per-vector deviation measured between batch-of-1 and batch-of-32, not a
    logic bug). The manifest records the `batch_size` an index was built
    with; resuming with a different one is allowed (never blocks a build)
    but logs a warning naming both values.

    Returns a report dict with planned/embedded counts, measured tiles/sec
    and wall time for the tiles this call actually embedded, and the model
    id/revision used.
    """
    root = data_root or config.get_data_root()
    aoi = tiling.scene_aoi(rel_path)
    aoi_dir = emb_dir(aoi, index_root)
    aoi_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, vectors_path = _manifest_paths(aoi_dir)

    ordered_tiles = _ordered_tile_plan(rel_path, scales, root, limit)
    planned_count = len(ordered_tiles)

    existing_manifest, existing_vectors = _load_state(manifest_path, vectors_path)

    if embedder is not None:
        model_id = embedder.model_id

    if existing_manifest is not None and existing_manifest["model_id"] != model_id:
        raise MixedEmbedderError(
            f"{aoi_dir}: index already built with model_id={existing_manifest['model_id']!r} "
            f"(revision={existing_manifest['revision']!r}); refusing to append "
            f"model_id={model_id!r} -- F-1a: exactly one embedder serves the whole index."
        )

    emb = embedder or embedders.load_embedder(model_id)

    if existing_manifest is not None and existing_manifest["revision"] != emb.revision:
        raise MixedEmbedderError(
            f"{aoi_dir}: index already built with model_id={existing_manifest['model_id']!r} "
            f"revision={existing_manifest['revision']!r}; refusing to append "
            f"revision={emb.revision!r} of the same model -- F-1a: exactly one "
            f"embedder revision serves the whole index."
        )

    if existing_manifest is None:
        manifest = {
            "aoi": aoi,
            "model_id": emb.model_id,
            "revision": emb.revision,
            "dim": emb.dim,
            "dtype": "float16",
            "scales": list(scales),
            "planned_count": planned_count,
            "batch_size": batch_size,
            "tiles": [],
            "skipped_tile_ids": [],
        }
        vectors = np.zeros((0, emb.dim), dtype=np.float16)
    else:
        manifest = existing_manifest
        manifest["planned_count"] = planned_count
        # Backward-compat for a manifest written before skipped tiles were
        # tracked by id (an int counter alone can't be de-duplicated across
        # resumes -- that was exactly the bug: every resume re-examined the
        # same 100%-nodata tiles and re-incremented it).
        manifest.setdefault("skipped_tile_ids", [])
        manifest.pop("skipped_all_nodata", None)
        # N-4 amendment: the same tile embedded alone vs. inside a batch of
        # 32 differs by up to 2.574e-4 (GPU fp16 GEMM non-associativity --
        # kernel selection depends on batch shape, not a logic bug). An
        # index is only reproducible at a fixed batch size, so record which
        # one built it, and warn loudly -- rather than silently -- if a
        # resume would use a different one.
        recorded_batch_size = manifest.setdefault("batch_size", batch_size)
        if recorded_batch_size != batch_size:
            log.warning(
                "embed_index: %s resuming with batch_size=%d but this index was "
                "built with batch_size=%d -- batch composition affects fp16 GEMM "
                "kernel selection (up to ~2.574e-4 per-vector deviation measured), "
                "so vectors embedded in this run will not exactly reproduce a run "
                "at the original batch size",
                aoi, batch_size, recorded_batch_size,
            )
        vectors = existing_vectors

    # A tile is "done" -- excluded from every future run's todo list -- once
    # it has either been embedded (has a vector, in manifest["tiles"]) or
    # been positively identified as 100%-nodata (manifest["skipped_tile_ids"]).
    # Tracking skips by id (not just a running count) is what makes a
    # skipped tile stay skipped across resumes instead of being re-read and
    # re-counted every single run.
    done_ids = {t["tile_id"] for t in manifest["tiles"]} | set(manifest["skipped_tile_ids"])
    todo = [t for t in ordered_tiles if t["tile_id"] not in done_ids]

    n_embedded_this_run = 0
    n_skipped_nodata_this_run = 0
    t_start = time.perf_counter()

    if todo:
        with _open_raster(root / rel_path) as ds:
            for i in range(0, len(todo), batch_size):
                batch = todo[i : i + batch_size]
                images, keep = [], []
                for t in batch:
                    arr = _read_tile_array(ds, t)
                    frac = _nodata_fraction(arr)
                    if frac >= 1.0:
                        n_skipped_nodata_this_run += 1
                        manifest["skipped_tile_ids"].append(t["tile_id"])
                        continue
                    images.append(Image.fromarray(arr))
                    keep.append((t, frac))
                if images:
                    new_vecs = emb.embed_images(images).astype(np.float16)
                    for t, frac in keep:
                        manifest["tiles"].append(
                            {"tile_id": t["tile_id"], "scale": t["scale"], "nodata_fraction": frac}
                        )
                    vectors = np.concatenate([vectors, new_vecs], axis=0)
                    n_embedded_this_run += len(keep)
                _checkpoint(manifest_path, vectors_path, manifest, vectors)
                log.info(
                    "embed_index: %s checkpoint -- %d/%d tiles recorded",
                    aoi, len(manifest["tiles"]), planned_count,
                )
                if batch_sleep_s > 0:
                    time.sleep(batch_sleep_s)

    wall_s = time.perf_counter() - t_start
    tiles_per_sec = n_embedded_this_run / wall_s if wall_s > 0 else float("inf")
    report = {
        "aoi": aoi,
        "planned_count": planned_count,
        "embedded_total": len(manifest["tiles"]),
        "embedded_this_run": n_embedded_this_run,
        "skipped_all_nodata_total": len(manifest["skipped_tile_ids"]),
        "skipped_all_nodata_this_run": n_skipped_nodata_this_run,
        "wall_s": wall_s,
        "tiles_per_sec": tiles_per_sec,
        "index_dir": str(aoi_dir),
        "model_id": emb.model_id,
        "revision": emb.revision,
    }
    log.info(
        "embed_index: %s done -- %d/%d embedded total (%d this run), %d skipped "
        "(100%% nodata), %.2f tiles/sec, %.1fs wall",
        aoi, report["embedded_total"], planned_count, n_embedded_this_run,
        report["skipped_all_nodata_total"], tiles_per_sec, wall_s,
    )
    return report


# --------------------------------------------------------------------------
# Reload -- the only path later stages (and this stage's own tests) should
# use to get vectors back. See module docstring for why renormalisation
# happens here rather than at storage time.
# --------------------------------------------------------------------------


def load_index(aoi: str, index_root: Path | None = None, renormalize: bool = True) -> dict:
    """Reload one AOI's index from disk -- the fresh-process verification
    path (F-2, F-5). Raises if the manifest and vectors.npy disagree on row
    count (an inconsistent index must never be loaded silently).

    `renormalize` (default True): fp16 storage of an fp32 unit vector does
    not generally satisfy ||v|| = 1.0 +/- 1e-5 on reload -- measured up to
    2.26e-4 deviation on real embeddings (module docstring). Every raw
    stored norm is still checked against a generous
    `RAW_NORM_SANITY_ATOL` (1e-2) that fp16 quantisation alone cannot
    breach, so a genuinely broken vector (wrong dtype, missed normalisation,
    patch tokens) still raises here rather than being silently corrected.
    What is returned, by default, is renormalised to satisfy the strict
    1e-5 tolerance -- because that is what later stages (and F-4's
    cosine-as-dot-product identity) actually need.

    Non-finite values (NaN or +/-Inf, S3-fix Fix 1) are checked explicitly
    and first, before the ATOL comparison: `abs(nan - 1.0) > ATOL` is always
    False in NumPy, so a NaN vector would otherwise defeat the sanity check
    entirely rather than merely miss its threshold, and get renormalised
    into a NaN-poisoned "successfully loaded" vector with no exception. A
    non-finite component always raises, naming the offending row indices and
    what was found -- never silently dropped, repaired, or renormalised.
    """
    aoi_dir = emb_dir(aoi, index_root)
    manifest_path, vectors_path = _manifest_paths(aoi_dir)
    manifest = json.loads(manifest_path.read_text())
    vectors_raw = np.load(vectors_path)
    n = len(manifest["tiles"])
    if vectors_raw.shape[0] != n:
        raise RuntimeError(
            f"{aoi_dir}: manifest lists {n} tiles but vectors.npy has "
            f"{vectors_raw.shape[0]} rows -- inconsistent index, refusing to load"
        )
    vectors_f32 = vectors_raw.astype(np.float32)
    raw_norms = np.linalg.norm(vectors_f32, axis=1) if n else np.zeros(0, dtype=np.float32)

    # S3-fix, Fix 1: NumPy comparisons against NaN are always False, so
    # `abs(nan - 1.0) > ATOL` never fires -- a NaN component is not merely
    # outside the 1e-2 window, it defeats the comparison's domain entirely,
    # and the pre-fix code returned a NaN-poisoned "successfully loaded"
    # vector with no exception. Non-finite (NaN or +/-Inf, in either the
    # vector itself or its computed norm) must be checked explicitly and
    # first, and must always raise -- never be silently dropped, repaired,
    # or (worse) renormalised, because a NaN/Inf vector means something
    # upstream is broken and the PM must see it.
    non_finite_vec = ~np.isfinite(vectors_f32).all(axis=1) if n else np.zeros(0, dtype=bool)
    non_finite_norm = ~np.isfinite(raw_norms)
    non_finite = non_finite_vec | non_finite_norm
    if non_finite.any():
        idx = np.argwhere(non_finite)[:5].ravel().tolist()
        kinds = [
            "NaN" if (np.isnan(vectors_f32[i]).any() or np.isnan(raw_norms[i])) else "Inf"
            for i in idx
        ]
        raise RuntimeError(
            f"{aoi_dir}: {int(non_finite.sum())} stored vectors contain non-finite "
            f"values -- refusing to load (a NaN/Inf vector means something upstream "
            f"is broken, not something to renormalise around); first offending row "
            f"indices {idx}, kinds {kinds}"
        )

    bad = np.abs(raw_norms - 1.0) > RAW_NORM_SANITY_ATOL
    if bad.any():
        idx = np.argwhere(bad)[:5].ravel().tolist()
        raise RuntimeError(
            f"{aoi_dir}: {int(bad.sum())} stored vectors have norm far from 1.0 "
            f"(> {RAW_NORM_SANITY_ATOL} off -- fp16 quantisation alone cannot "
            f"explain this); first offending indices {idx}, norms {raw_norms[idx].tolist()}"
        )
    if renormalize and n:
        vectors = vectors_f32 / raw_norms[:, None]
    else:
        vectors = vectors_f32
    return {"manifest": manifest, "vectors": vectors, "raw_norms": raw_norms}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    p = argparse.ArgumentParser(description="Build the tile-embedding index for one scene.")
    p.add_argument("--source", required=True, help="scene path relative to AERIAL_DATA_ROOT")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--scales", default=",".join(str(s) for s in tiling.SCALES))
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--limit", type=int, default=None,
        help="process only the first N planned tiles (testing/calibration only)",
    )
    p.add_argument(
        "--batch-sleep-s", type=float, default=0.0,
        help="sleep this long after each checkpoint (testing only, default 0)",
    )
    args = p.parse_args()
    scales = tuple(int(s) for s in args.scales.split(","))
    report = build_index(
        args.source,
        model_id=args.model_id,
        scales=scales,
        batch_size=args.batch_size,
        limit=args.limit,
        batch_sleep_s=args.batch_sleep_s,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
