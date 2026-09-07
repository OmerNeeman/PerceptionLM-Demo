"""Tile planner -- spec F-1 (coverage), F-10 (map location), D-4 (stable ids).

Two disjoint concepts, kept as separate, separately-named fields throughout
this module (briefs/S2.md's addendum -- do not conflate them):

  PLANNED   -- ``ceil(W/s) x ceil(H/s)``, every grid cell, nodata ignored.
               This is what F-1's 108,542 counts and what ``plan_scene``
               below produces. Edge cells are padded to full size, never
               dropped (F-1: dropping the 376 remainder rows at 448 px on a
               10240-px scene would lose 3.8% of scene height).
  COMPLETE  -- ``floor(W/s) x floor(H/s)``: only cells wholly inside the
               raster, no padding. This is the base of the OTHER count that
               appears in docs/DATA.md's "448 tiles (valid)" column.
  VALID     -- COMPLETE, further reduced by dropping any complete tile whose
               pixels are mostly the mosaic's zero-fill sentinel (measured,
               not assumed -- see ``radiometric_valid_counts``). This is the
               full docs/DATA.md convention: floor, *and* nodata-reduced.

A tile's *planned* record always carries its geometric valid (unpadded) pixel
region -- ``valid_width``/``valid_height``/``is_edge`` below -- computed from
raster dimensions alone, no pixel read required. That is a third, purely
geometric notion (an edge cell can be "geometrically partial" without having
any nodata in it at all, and vice versa); it must not be confused with the
radiometric VALID count above either.

The planner itself (``plan_scene``, ``plan_all``) reads **no pixels** -- only
raster headers (CRS, transform, width, height), which is why planning all 8
indexable scenes at all 3 scales is cheap. Pixel reads enter only in
``radiometric_valid_counts`` (nodata fraction) and in ``cog.py``.

Reuses geo.py (CRS classification, true ground resolution) and inventory.py
(the D-1 source/indexable classification) as-is -- neither is reimplemented
here. Where this module needs a capability neither exposes (date-from-path,
AOI-from-path, corner-grid reprojection for tile bboxes, nodata measurement)
it adds a new function of its own rather than changing either module's
signatures.

Environment: every invocation must be prefixed inline, every time --

    env PYTHONNOUSERSITE=1 AERIAL_DATA_ROOT=<path> <python> ...

(see INSTRUCTIONS.md for the interpreter path on the dev machine).
"""

from __future__ import annotations

import json
import logging
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform as warp_transform

import config
import geo
import inventory

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Fixed project constants (spec.md's "config symbols": SCALES, OVERLAP).
# config.py deliberately carries only path/device settings (S1/S1a); these
# are the tiling-specific ones, kept local to this module -- the same
# pattern src/calibrate.py (S0) already uses for its own local SCALES.
# --------------------------------------------------------------------------

#: Finest-to-coarsest is irrelevant; order fixed here so every emitted plan
#: and every "per scale" report iterates scales in one stable order.
SCALES: tuple[int, ...] = (448, 224, 112)

#: This stage implements exactly OVERLAP=0.0 (the spec default, and the only
#: value the acceptance criteria exercise: "adjacent tiles' bboxes are
#: contiguous with no overlap at OVERLAP=0"). A nonzero overlap would change
#: the grid stride and is out of scope here.
OVERLAP: float = 0.0

#: Pixel value a later embedding stage pads a geometrically-partial edge
#: tile up to `scale x scale` with (F-1's "state what pad value is used").
#: Zero matches the mosaics' own zero-fill sentinel (see
#: `radiometric_valid_counts`), so a padded edge tile and a genuinely
#: nodata-filled interior tile read identically to anything downstream.
PAD_VALUE = 0

#: A complete (floor-grid) tile counts as radiometrically valid iff strictly
#: less than this fraction of its pixels are the zero-fill sentinel.
NODATA_TILE_THRESHOLD = 0.5

WGS84 = "EPSG:4326"

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.tiff?$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Identity: AOI and date from the path -- new functions, not filename-based
# source/derived classification (that stays in inventory.py, untouched).
# --------------------------------------------------------------------------


def scene_aoi(rel_path: str) -> str:
    """The AOI name a scene belongs to.

    The first path segment if the scene sits in a subdirectory
    (``leb/2022-10-29.tif`` -> ``leb``), else the file's own stem for a
    "loose" scene directly under the data root (``X605_Y3388.tif`` ->
    ``X605_Y3388`` -- docs/DATA.md's demo AOI, listed there as "(loose)").
    """
    p = config.posix_key(rel_path)
    parts = p.split("/")
    if len(parts) > 1:
        return parts[0]
    return Path(parts[0]).stem


def scene_date(rel_path: str) -> str:
    """Acquisition date encoded in the filename, or ``"unknown"``.

    Only `leb`'s two scenes carry a date at all (`2022-10-29.tif`,
    `2025-06-06.tif`) -- docs/DATA.md: five of the eight indexed footprints
    have no date in the file. Matched against the basename only, so a date
    substring elsewhere in a directory name can never leak in.
    """
    name = Path(config.posix_key(rel_path)).name
    m = _DATE_RE.search(name)
    return m.group(1) if m else "unknown"


# --------------------------------------------------------------------------
# The grid -- pure arithmetic, no I/O.
# --------------------------------------------------------------------------


def grid_dims(width: int, height: int, scale: int) -> tuple[int, int]:
    """Padded planned grid: ``(ceil(W/s), ceil(H/s))``. Edge cells padded,
    never dropped -- F-1."""
    return math.ceil(width / scale), math.ceil(height / scale)


def complete_grid_dims(width: int, height: int, scale: int) -> tuple[int, int]:
    """The OTHER grid: ``(floor(W/s), floor(H/s))``, no padding -- the base
    of docs/DATA.md's "448 tiles (valid)" convention. Never used for F-1's
    coverage assertion; kept separate on purpose."""
    return width // scale, height // scale


def valid_pixel_region(width: int, height: int, scale: int, col: int, row: int) -> tuple[int, int]:
    """Geometric (not radiometric) unpadded pixel extent of tile (col, row)
    at this scale: how much of the nominal ``scale x scale`` window actually
    falls inside the raster. Equal to ``(scale, scale)`` for every interior
    tile; smaller on the last column/row when W or H is not a multiple of
    `scale`. No pixel read -- purely `width`/`height` arithmetic."""
    x0, y0 = col * scale, row * scale
    return max(0, min(scale, width - x0)), max(0, min(scale, height - y0))


def make_tile_id(aoi: str, source_rel_path: str, date: str, scale: int, col: int, row: int) -> str:
    """D-4: id derived from (aoi, source_file, date, col, row, scale) alone.

    No absolute path, no timestamp, no filesystem-iteration-order dependence
    -- every input is either a config-free string or an integer the caller
    already had in hand. ``::`` cannot appear in any of aoi/path/date, so the
    id round-trips through `parse_tile_id` with a plain split.
    """
    src = config.posix_key(source_rel_path)
    for part in (aoi, src, date):
        if "::" in part:
            raise ValueError(f"tile id component contains the '::' separator: {part!r}")
    return f"{aoi}::{src}::{date}::{scale}::{col:05d}::{row:05d}"


def parse_tile_id(tile_id: str) -> dict:
    """Inverse of `make_tile_id` -- F-10's round-trip starts here."""
    aoi, src, date, scale, col, row = tile_id.split("::")
    return {
        "aoi": aoi,
        "source_file": src,
        "date": date,
        "scale": int(scale),
        "col": int(col),
        "row": int(row),
    }


def pixel_offset_from_tile_id(tile_id: str) -> tuple[int, int]:
    """F-10's round-trip: tile id -> pixel offsets, exactly."""
    rec = parse_tile_id(tile_id)
    s = rec["scale"]
    return rec["col"] * s, rec["row"] * s


# --------------------------------------------------------------------------
# Geo round-trip: tile corners -> lon/lat, via rasterio/GDAL only (pyproj.CRS
# is broken in this env -- CLAUDE.md trap, geo.py's own docstring). Uses
# rasterio.warp.transform directly rather than reaching into geo.py's private
# `_centre_lonlat`/`_geodesic_pixel_m` (those are for a raster's centre pixel,
# not for an arbitrary tile-corner grid) -- a new function, not a modified
# signature, per the brief.
# --------------------------------------------------------------------------


def _corner_grid_lonlat(crs, transform, cols: int, rows: int, scale: int):
    """lon/lat for every grid-line intersection of the padded planned grid,
    as two (rows+1, cols+1) arrays. Computed once per (scene, scale) and
    reused by every tile: two tiles sharing an edge look up the *same* grid
    point, so their bboxes are bit-identical at that edge -- no floating
    seam, no overlap, by construction rather than by tolerance.

    Vectorised over the affine's own coefficients (a, b, c, d, e, f: x = a*col
    + b*row + c, y = d*col + e*row + f) rather than looping `Affine.__mul__`
    per point -- same maths, applied to the whole corner grid at once.
    """
    col_idx = np.arange(cols + 1, dtype="float64") * scale
    row_idx = np.arange(rows + 1, dtype="float64") * scale
    cc, rr = np.meshgrid(col_idx, row_idx)  # shape (rows+1, cols+1)
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    xs = a * cc + b * rr + c
    ys = d * cc + e * rr + f
    lons, lats = warp_transform(crs, WGS84, xs.ravel().tolist(), ys.ravel().tolist())
    lons = np.asarray(lons).reshape(rows + 1, cols + 1)
    lats = np.asarray(lats).reshape(rows + 1, cols + 1)
    return lons, lats


def _tile_bbox(lons, lats, col: int, row: int) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) from the four corner-grid points
    of tile (col, row). Min/max over all four rather than assuming an axis
    order, so it holds regardless of transform sign conventions."""
    corner_lons = (lons[row, col], lons[row, col + 1], lons[row + 1, col], lons[row + 1, col + 1])
    corner_lats = (lats[row, col], lats[row, col + 1], lats[row + 1, col], lats[row + 1, col + 1])
    return min(corner_lons), min(corner_lats), max(corner_lons), max(corner_lats)


# --------------------------------------------------------------------------
# Scene-level metadata -- header reads only (D-3: read-only, mode "r").
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneMeta:
    rel_path: str  # posix, relative to the data root
    aoi: str
    date: str
    width: int
    height: int
    crs: str
    true_gsd_m: float


def scene_meta(rel_path: str, data_root: Path | None = None) -> tuple[SceneMeta, "CRS", "Affine"]:
    """Header-only read (CRS, transform, width, height) -- no pixel decode.
    Returns `(SceneMeta, crs, transform)` -- the raw CRS/transform are
    returned alongside the plain-data `SceneMeta` because `plan_scene` needs
    them for `_corner_grid_lonlat`, and re-opening the raster a second time
    just to get them back would defeat the "reads no pixels" property by
    doing the header read twice.
    `geo.ground_resolution` is reused verbatim for the true GSD; it may raise
    `GsdGuardError`, and that is not caught here -- a raised guard is a
    finding to report, not an exception to swallow (S2 brief)."""
    root = data_root or config.get_data_root()
    p = root / rel_path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(p) as ds:  # mode defaults to "r" -- D-3
            res = geo.ground_resolution(ds.crs, ds.transform, ds.width, ds.height)
            if res is None:
                raise ValueError(f"{rel_path}: not georeferenced, cannot be tiled")
            return SceneMeta(
                rel_path=config.posix_key(rel_path),
                aoi=scene_aoi(rel_path),
                date=scene_date(rel_path),
                width=ds.width,
                height=ds.height,
                crs=str(ds.crs),
                true_gsd_m=res.true_gsd_m,
            ), ds.crs, ds.transform


def load_indexable_scenes(inventory_path: Path | None = None) -> list[str]:
    """The D-1 indexable set, reused from S1's inventory -- never
    reclassified here. Loads the artifact S1 already wrote; builds it once
    if it is missing (e.g. a fresh checkout) rather than silently returning
    an empty plan."""
    path = inventory_path or (config.get_index_root() / "inventory.json")
    if not path.exists():
        log.info("tiling: %s missing, building inventory once via inventory.py", path)
        inventory.write_inventory(path=path)
    inv = json.loads(path.read_text())
    return inventory.indexable_paths(inv)


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


def plan_scene(rel_path: str, scale: int, data_root: Path | None = None) -> dict:
    """Every tile of one scene at one scale. Reads only the raster header.

    Returns a dict with the scene metadata and the full list of tile
    records (each a plain dict, JSON-ready).
    """
    meta, crs, transform = scene_meta(rel_path, data_root)
    cols, rows = grid_dims(meta.width, meta.height, scale)
    lons, lats = _corner_grid_lonlat(crs, transform, cols, rows, scale)

    tiles = []
    for row in range(rows):
        for col in range(cols):
            vw, vh = valid_pixel_region(meta.width, meta.height, scale, col, row)
            min_lon, min_lat, max_lon, max_lat = _tile_bbox(lons, lats, col, row)
            tile_id = make_tile_id(meta.aoi, meta.rel_path, meta.date, scale, col, row)
            tiles.append(
                {
                    "tile_id": tile_id,
                    "aoi": meta.aoi,
                    "source_file": meta.rel_path,
                    "date": meta.date,
                    "scale": scale,
                    "col": col,
                    "row": row,
                    "px_offset_x": col * scale,
                    "px_offset_y": row * scale,
                    "pixel_size_m": meta.true_gsd_m,
                    "valid_width": vw,
                    "valid_height": vh,
                    "is_edge": (vw, vh) != (scale, scale),
                    "pad_value": PAD_VALUE,
                    "bbox_lonlat": [min_lon, min_lat, max_lon, max_lat],
                    "crs": meta.crs,
                }
            )
    return {
        "aoi": meta.aoi,
        "source_file": meta.rel_path,
        "date": meta.date,
        "scale": scale,
        "width": meta.width,
        "height": meta.height,
        "crs": meta.crs,
        "true_gsd_m": meta.true_gsd_m,
        "grid_cols": cols,
        "grid_rows": rows,
        "planned_count": cols * rows,
        "tiles": tiles,
    }


def plan_all(scales=SCALES, inventory_path: Path | None = None, data_root: Path | None = None) -> dict:
    """Plan every indexable scene at every scale. Returns
    ``{rel_path: {scale: plan_dict}}``."""
    scenes = load_indexable_scenes(inventory_path)
    out: dict[str, dict[int, dict]] = {}
    for rel_path in scenes:
        out[rel_path] = {s: plan_scene(rel_path, s, data_root) for s in scales}
    return out


def write_tile_plans(plans: dict, out_dir: Path | None = None) -> list[Path]:
    """One JSON per scene under `index/tileplan/` -- never elsewhere (D-3).
    All three scales for a scene share one file, keyed by scale."""
    index_dir = config.get_index_root().resolve()
    out_dir = (out_dir or (config.get_index_root() / "tileplan")).resolve()
    if not (out_dir == index_dir or index_dir in out_dir.parents):
        raise ValueError(f"refusing to write outside {index_dir}: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for rel_path, per_scale in sorted(plans.items()):
        aoi = scene_aoi(rel_path)
        stem = Path(config.posix_key(rel_path)).stem
        fname = f"{aoi}__{stem}.json".replace("/", "_")
        out_path = out_dir / fname
        doc = {str(scale): plan for scale, plan in sorted(per_scale.items())}
        out_path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
        written.append(out_path)
    return written


# --------------------------------------------------------------------------
# Radiometric validity -- the OTHER count (docs/DATA.md's "448 tiles
# (valid)"). This is the only place in this module that reads pixels.
# --------------------------------------------------------------------------


def _nodata_mask(rel_path: str, data_root: Path | None = None) -> np.ndarray:
    """Whole-raster boolean mask, True where a pixel is the mosaics' zero-fill
    sentinel (all of bands 1-3 == 0). One sequential full read per scene --
    cheap even for `leb` (~592 MB, uncompressed) because it is sequential,
    not the small-random-window access pattern that costs 45x on `leb`'s
    un-tiled layout (that cost is `cog.py`'s subject, not this one). Never
    written anywhere; the raster is opened read-only (D-3)."""
    root = data_root or config.get_data_root()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(root / rel_path) as ds:
            arr = ds.read((1, 2, 3))
    return (arr[0] == 0) & (arr[1] == 0) & (arr[2] == 0)


def radiometric_valid_counts(rel_path: str, scales=SCALES, data_root: Path | None = None) -> dict:
    """Per scale: the docs/DATA.md-style count -- floor grid, tiles with
    >= NODATA_TILE_THRESHOLD zero-fill dropped -- plus the overall nodata
    fraction and the padded planned count, so all three numbers this stage
    must never conflate sit side by side for one scene.

    Distinct from `plan_scene`: this is the only function that reads pixels,
    and it is a report, not part of the plan artifact.
    """
    mask = _nodata_mask(rel_path, data_root)
    height, width = mask.shape
    out = {"nodata_fraction": float(mask.mean()), "by_scale": {}}
    for s in scales:
        cols_c, rows_c = complete_grid_dims(width, height, s)
        cols_p, rows_p = grid_dims(width, height, s)
        if cols_c > 0 and rows_c > 0:
            cropped = mask[: rows_c * s, : cols_c * s]
            frac = cropped.reshape(rows_c, s, cols_c, s).mean(axis=(1, 3))
            valid = int((frac < NODATA_TILE_THRESHOLD).sum())
        else:
            valid = 0
        out["by_scale"][s] = {
            "planned": cols_p * rows_p,
            "complete": cols_c * rows_c,
            "valid": valid,
        }
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    plans = plan_all()
    total = sum(p[s]["planned_count"] for p in plans.values() for s in SCALES)
    log.info("tiling: planned %d scenes x %d scales = %d tiles total", len(plans), len(SCALES), total)
    written = write_tile_plans(plans)
    log.info("tiling: wrote %d plan files under %s", len(written), config.get_index_root() / "tileplan")

    print(f"{'scene':42s} " + " ".join(f"{s:>18d}(plan/valid/nodata%)" for s in SCALES))
    for rel_path in sorted(plans):
        rv = radiometric_valid_counts(rel_path)
        cells = []
        for s in SCALES:
            b = rv["by_scale"][s]
            cells.append(f"{b['planned']:>6d}/{b['valid']:>6d}/{rv['nodata_fraction']*100:5.1f}%")
        print(f"{rel_path:42s} " + " ".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
