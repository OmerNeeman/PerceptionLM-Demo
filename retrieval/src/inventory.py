"""Source-vs-derived raster classification and the scene inventory.

Spec D-1 (only source imagery is indexed), D-3 (the data directory is
read-only), D-4 (the artifact is reproducible). See `briefs/S1.md`.

Why this module exists
----------------------
Of the 266 rasters under the data root, 8 are indexable aerial scenes. The
rest are binary masks, class rasters, softmax outputs and fused
change-detection products -- plus real imagery in the wrong resolution regime,
real imagery with no georeference, and byte-identical copies. Indexing a mask
as if it were imagery produces a retrieval index that looks entirely healthy
and means nothing, so the classification is the load-bearing part of the
stage, not the plumbing.

The D-1 rule, measured (docs/DATA.md, spec D-1)
-----------------------------------------------
A raster is **source imagery** iff

    1. it has >= 3 bands, AND
    2. sampled pixel values show > 64 distinct levels in each of bands 1-3.

**Never by filename.** `leb/2022-10-29.tif` (imagery) and
`leb/cache/2022-10-29_ST_output.tiff` (a 4-band softmax raster) share a
prefix; `leb/leb_crop_x4096_y3072_1024.tif` looks derived and is real
imagery. `tests/test_inventory.py::test_source_classifier` runs the classifier
over deliberately renamed symlinks precisely so a filename fast path cannot
survive.

Being source imagery is necessary but not sufficient to be **indexable**. The
distinct rejection reasons, in the order they are applied:

    too_few_bands               < 3 bands
    derived_raster              <= 64 distinct levels in some of bands 1-3
    no_georeference             no CRS or no geotransform (e.g. gaza.tiff)
    duplicate_of                same pixels as an already-indexed scene
    resolution_regime_excluded  true GSD outside the ~10 cm regime

Sampling
--------
A fixed 8x8 grid of 64x64 windows -- 32,768 pixels per band, deterministic,
no RNG. Rasters here reach 6.44 gigapixels (`sin/Sini_Oct_Det_2025.tif`), so
nothing is read whole. The strategy and its parameters are written into the
artifact because D-4 requires two runs to agree.

Read-only (spec D-3)
--------------------
Every `rasterio.open` in this module and in `geo.py` uses the default mode
"r". Nothing is ever written under the data root; every output path resolves
inside `retrieval/index/`.

Run:
    env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python \\
        retrieval/src/inventory.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.windows import Window

import geo

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths. The data root is READ-ONLY; every output goes under retrieval/index/.
# --------------------------------------------------------------------------

DATA_ROOT = Path("/home/omer/PycharmProjects/Dynamic-Terrain/data")

SRC_DIR = Path(__file__).resolve().parent
RETRIEVAL_DIR = SRC_DIR.parent
INDEX_DIR = RETRIEVAL_DIR / "index"
INVENTORY_JSON = INDEX_DIR / "inventory.json"

# --------------------------------------------------------------------------
# The rule. Constants, not magic numbers buried in a branch.
# --------------------------------------------------------------------------

SOURCE_MIN_BANDS = 3
SOURCE_MIN_DISTINCT_LEVELS = 64  # strict: > 64
CLASSIFY_BANDS = 3  # bands 1-3 decide; extra bands are recorded, not used
MAX_RECORDED_BANDS = 8

#: True ground GSD band, in metres, that the 448 px embedding space assumes.
#: docs/DATA.md: the 8 indexed scenes are 10.00-10.47 cm. `sin` is 415.7 cm
#: and Teheran/iran are 50 cm -- different regimes, admissible later only as
#: separate coarse tiers, never in the same embedding space.
INDEXABLE_GSD_RANGE_M = (0.05, 0.15)

TILE_PX = 448

# Deterministic sampling: a fixed stride grid, no RNG at all.
SAMPLE_GRID = 8
SAMPLE_WINDOW_PX = 64

# Duplicate content verification: a second, smaller fixed grid.
DEDUP_GRID = 4
DEDUP_WINDOW_PX = 8
DEDUP_PX_TOLERANCE = 1e-6

REASONS = frozenset(
    {
        "not_a_raster",
        "open_failed",
        "too_few_bands",
        "derived_raster",
        "no_georeference",
        "duplicate_of",
        "resolution_regime_excluded",
    }
)

_UNSUPPORTED_FORMAT = "not recognized as being in a supported file format"


# --------------------------------------------------------------------------
# The pure rule
# --------------------------------------------------------------------------


def passes_pixel_rule(band_count: int, distinct_per_band) -> bool:
    """D-1, with no I/O: >= 3 bands and > 64 distinct levels in bands 1-3."""
    if band_count < SOURCE_MIN_BANDS:
        return False
    head = list(distinct_per_band)[:CLASSIFY_BANDS]
    if len(head) < CLASSIFY_BANDS:
        return False
    return all(d > SOURCE_MIN_DISTINCT_LEVELS for d in head)


def _decide_source(path, band_count: int, distinct_per_band) -> bool:
    """D-1's decision, made on the pixels of `path` and nothing else.

    `path` is accepted only so a failure can name the file; it is deliberately
    never inspected. Reintroducing a filename test here is caught by
    tests/test_inventory.py::test_source_classifier, which runs the classifier
    over symlinks whose names contradict their contents.
    """
    del path
    return passes_pixel_rule(band_count, distinct_per_band)


def pixel_rule_reason(band_count: int, distinct_per_band) -> str | None:
    """Which half of the pixel rule rejected this raster, or None if neither."""
    if band_count < SOURCE_MIN_BANDS:
        return "too_few_bands"
    if not passes_pixel_rule(band_count, distinct_per_band):
        return "derived_raster"
    return None


# --------------------------------------------------------------------------
# Deterministic sampling
# --------------------------------------------------------------------------


def sample_windows(width: int, height: int, grid: int = SAMPLE_GRID, win: int = SAMPLE_WINDOW_PX):
    """A fixed stride grid of read windows. Deterministic, no RNG."""
    w, h = min(win, width), min(win, height)

    def origins(extent, size):
        if extent <= size:
            return [0]
        if grid == 1:
            return [(extent - size) // 2]
        return sorted({round(i * (extent - size) / (grid - 1)) for i in range(grid)})

    return [(x, y, w, h) for y in origins(height, h) for x in origins(width, w)]


def distinct_levels(ds, grid: int = SAMPLE_GRID, win: int = SAMPLE_WINDOW_PX) -> list[int]:
    """Distinct value count per band, pooled over the sampling grid."""
    nb = min(ds.count, MAX_RECORDED_BANDS)
    bands = list(range(1, nb + 1))
    seen: list[set] = [set() for _ in range(nb)]
    for x, y, w, h in sample_windows(ds.width, ds.height, grid, win):
        block = ds.read(bands, window=Window(x, y, w, h))
        for i in range(nb):
            seen[i].update(np.unique(block[i]).tolist())
    return [len(s) for s in seen]


# --------------------------------------------------------------------------
# Probing one raster
# --------------------------------------------------------------------------


def probe_raster(path: str | Path) -> dict:
    """Open a raster READ-ONLY and measure everything D-1/D-2 need.

    Returns a record with `opened=False` and an `error` string if GDAL cannot
    read the file. Never raises for an unreadable file, never silences it: the
    failure is logged and reported in the artifact.
    """
    p = Path(path)
    rec: dict = {
        "opened": False,
        "error": None,
        "driver": None,
        "width": None,
        "height": None,
        "band_count": None,
        "dtype": None,
        "nodata": None,
        "crs": None,
        "crs_kind": None,
        "has_geotransform": None,
        "projected_px_x_m": None,
        "projected_px_y_m": None,
        "centre_lon": None,
        "centre_lat": None,
        "scale_correction": None,
        "true_gsd_m": None,
        "true_gsd_cm": None,
        "geodesic_gsd_cm": None,
        "tile_ground_m": None,
        "distinct_per_band": None,
        "is_source_imagery": None,
    }
    try:
        with warnings.catch_warnings():
            # A missing geotransform is recorded below, not printed as noise.
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(p) as ds:  # mode defaults to "r" -- D-3
                rec.update(
                    opened=True,
                    driver=ds.driver,
                    width=ds.width,
                    height=ds.height,
                    band_count=ds.count,
                    dtype=str(ds.dtypes[0]),
                    nodata=None if ds.nodata is None else float(ds.nodata),
                    crs=str(ds.crs) if ds.crs else None,
                    crs_kind=geo.crs_kind(ds.crs),
                    has_geotransform=geo.is_georeferenced(ds.crs, ds.transform),
                    distinct_per_band=distinct_levels(ds),
                )
                res = geo.ground_resolution(ds.crs, ds.transform, ds.width, ds.height)
                if res is not None:
                    rec.update(
                        projected_px_x_m=round(res.projected_px_x_m, 9),
                        projected_px_y_m=round(res.projected_px_y_m, 9),
                        centre_lon=round(res.centre_lon, 7),
                        centre_lat=round(res.centre_lat, 7),
                        scale_correction=(
                            None if res.scale_correction is None else round(res.scale_correction, 9)
                        ),
                        true_gsd_m=round(res.true_gsd_m, 9),
                        true_gsd_cm=round(res.true_gsd_m * 100.0, 4),
                        geodesic_gsd_cm=(
                            None
                            if res.geodesic_gsd_m is None
                            else round(res.geodesic_gsd_m * 100.0, 4)
                        ),
                        tile_ground_m=round(res.tile_extent_m(TILE_PX), 4),
                    )
    except RasterioIOError as exc:
        rec["error"] = str(exc)
        # A sidecar or vector file GDAL has no raster driver for is expected,
        # and recorded in the artifact's `non_rasters`. A raster that GDAL
        # *should* read and cannot is a real problem, so it is louder.
        if _UNSUPPORTED_FORMAT in str(exc):
            log.debug("not a raster: %s", p)
        else:
            log.warning("could not open %s: %s", p, exc)
        return rec
    except Exception as exc:  # reported, never swallowed
        rec["error"] = f"{type(exc).__name__}: {exc}"
        log.error("unexpected failure reading %s: %s", p, exc)
        return rec

    rec["is_source_imagery"] = _decide_source(p, rec["band_count"], rec["distinct_per_band"])
    return rec


def is_source_imagery(path: str | Path) -> bool:
    """D-1 for a single raster, by pixels. Works on any path, symlinks included."""
    rec = probe_raster(path)
    if not rec["opened"]:
        raise RasterioIOError(f"{path}: {rec['error']}")
    return bool(rec["is_source_imagery"])


# --------------------------------------------------------------------------
# Read-only guarantee (D-3)
# --------------------------------------------------------------------------


def tree_snapshot(root: Path = DATA_ROOT) -> dict[str, tuple[int, int]]:
    """relative path -> (size, mtime_ns) for every file under `root`."""
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() or p.is_symlink():
            st = p.lstat()
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def tree_digest(root: Path = DATA_ROOT) -> str:
    h = hashlib.sha256()
    for rel, (size, mtime) in sorted(tree_snapshot(root).items()):
        h.update(f"{rel}\0{size}\0{mtime}\n".encode())
    return "sha256:" + h.hexdigest()


def output_paths() -> list[Path]:
    """Every path this module can write. All must resolve inside index/."""
    return [
        INDEX_DIR,
        INVENTORY_JSON,
        INDEX_DIR / "inventory_run_a.json",
        INDEX_DIR / "inventory_run_b.json",
        INDEX_DIR / "symlink_probe",
    ]


# --------------------------------------------------------------------------
# Duplicate detection: same ground, same pixels
# --------------------------------------------------------------------------


def _bounds(rec) -> tuple[float, float, float, float]:
    px, py = rec["projected_px_x_m"], rec["projected_px_y_m"]
    return (0.0, 0.0, rec["width"] * px, rec["height"] * py)


def _same_pixels(cand_path: Path, cont_path: Path) -> bool:
    """Compare a deterministic pixel sample of `cand` against the matching
    window of `cont`. Two different-date scenes over one footprint (leb 2022
    vs 2025) have identical bounds, so bounds alone can never decide this."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(cand_path) as a, rasterio.open(cont_path) as b:
            if a.count != b.count:
                return False
            ta, tb = a.transform, b.transform
            col = (ta.c - tb.c) / tb.a
            row = (ta.f - tb.f) / tb.e
            if abs(col - round(col)) > 1e-3 or abs(row - round(row)) > 1e-3:
                return False
            col, row = int(round(col)), int(round(row))
            if col < 0 or row < 0 or col + a.width > b.width or row + a.height > b.height:
                return False
            for x, y, w, h in sample_windows(a.width, a.height, DEDUP_GRID, DEDUP_WINDOW_PX):
                pa = a.read(window=Window(x, y, w, h))
                pb = b.read(window=Window(col + x, row + y, w, h))
                if not np.array_equal(pa, pb):
                    return False
    return True


def _find_duplicates(records: list[dict], root: Path) -> None:
    """Annotate `duplicate_of` / `duplicate_kind` in place.

    Candidates must share a CRS, a pixel size and a band count, and the
    smaller footprint must sit inside the larger one on an integer pixel
    offset. Content is then verified -- containment is never enough.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if not r["is_source_imagery"] or not r["has_geotransform"]:
            continue
        key = (r["crs"], r["projected_px_x_m"], r["projected_px_y_m"], r["band_count"])
        groups.setdefault(key, []).append(r)

    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        # Largest first; identical footprints tie-break on path so the choice
        # of canonical scene is stable across runs (D-4).
        members = sorted(members, key=lambda r: (-(r["width"] * r["height"]), r["path"]))
        for i, cand in enumerate(members):
            for cont in members[:i]:
                if cont.get("duplicate_of"):
                    continue  # never chain onto a duplicate
                if cand["width"] > cont["width"] or cand["height"] > cont["height"]:
                    continue
                if _same_pixels(root / cand["path"], root / cont["path"]):
                    cand["duplicate_of"] = cont["path"]
                    cand["duplicate_kind"] = (
                        "identical_footprint"
                        if (cand["width"], cand["height"]) == (cont["width"], cont["height"])
                        else "contained_crop"
                    )
                    log.info(
                        "duplicate: %s == %s (%s)",
                        cand["path"],
                        cont["path"],
                        cand["duplicate_kind"],
                    )
                    break


# --------------------------------------------------------------------------
# Indexability
# --------------------------------------------------------------------------


def _indexability_reason(rec: dict) -> str | None:
    """The single most specific reason this raster is not an indexable scene."""
    if not rec["is_source_imagery"]:
        # Which half of the pixel rule rejected it -- too few bands, or too
        # few distinct levels. These must never collapse into one reason.
        return pixel_rule_reason(rec["band_count"], rec["distinct_per_band"]) or "derived_raster"
    if not rec["has_geotransform"]:
        return "no_georeference"
    if rec.get("duplicate_of"):
        return "duplicate_of"
    lo, hi = INDEXABLE_GSD_RANGE_M
    gsd = rec["true_gsd_m"]
    if gsd is None or not (lo <= gsd <= hi):
        return "resolution_regime_excluded"
    return None


# --------------------------------------------------------------------------
# The inventory
# --------------------------------------------------------------------------


def build_inventory(root: Path = DATA_ROOT) -> dict:
    """Walk `root`, classify every file, and return the inventory document.

    Every regular file is handed to GDAL -- there is no extension filter,
    because an extension filter is a filename pre-filter (spec D-1).
    """
    root = Path(root)
    log.info("inventory: start root=%s sampling=%dx%d grid", root, SAMPLE_GRID, SAMPLE_WINDOW_PX)

    rasters: list[dict] = []
    non_rasters: list[str] = []
    unreadable: list[dict] = []

    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        rel = str(p.relative_to(root))
        rec = probe_raster(p)
        if not rec["opened"]:
            if _UNSUPPORTED_FORMAT in (rec["error"] or ""):
                non_rasters.append(rel)
            else:
                log.error("unreadable raster %s: %s", rel, rec["error"])
                unreadable.append({"path": rel, "error": rec["error"]})
            continue
        rec = {"path": rel, "size_bytes": p.stat().st_size, **rec}
        rec.pop("error")
        rec["duplicate_of"] = None
        rec["duplicate_kind"] = None
        rasters.append(rec)

    _find_duplicates(rasters, root)

    for rec in rasters:
        reason = _indexability_reason(rec)
        rec["reason"] = reason
        rec["is_indexable"] = reason is None

    rasters.sort(key=lambda r: r["path"])

    src_distinct = [
        min(r["distinct_per_band"][:CLASSIFY_BANDS]) for r in rasters if r["is_source_imagery"]
    ]
    non_src_distinct = [
        max(r["distinct_per_band"][:CLASSIFY_BANDS]) for r in rasters if not r["is_source_imagery"]
    ]

    counts = {
        "files_scanned": len(rasters) + len(non_rasters) + len(unreadable),
        "rasters": len(rasters),
        "non_rasters": len(non_rasters),
        "unreadable": len(unreadable),
        "leb_rasters": sum(1 for r in rasters if r["path"].startswith("leb/")),
        "source_imagery": sum(1 for r in rasters if r["is_source_imagery"]),
        "indexable": sum(1 for r in rasters if r["is_indexable"]),
        "min_distinct_among_source": min(src_distinct) if src_distinct else None,
        "max_distinct_among_non_source": max(non_src_distinct) if non_src_distinct else None,
    }
    for reason in sorted(REASONS):
        n = sum(1 for r in rasters if r["reason"] == reason)
        if n:
            counts[reason] = n

    inv = {
        "schema_version": 1,
        "stage": "S1",
        "data_root": str(root),
        "proves": ["D-1", "D-2", "D-3", "D-4"],
        "rules": {
            "source_min_bands": SOURCE_MIN_BANDS,
            "source_min_distinct_levels_exclusive": SOURCE_MIN_DISTINCT_LEVELS,
            "classify_bands": CLASSIFY_BANDS,
            "indexable_gsd_range_m": list(INDEXABLE_GSD_RANGE_M),
            "tile_px": TILE_PX,
            "note": "classification is by pixel value distribution, never by filename",
        },
        "sampling": {
            "strategy": "fixed_stride_window_grid",
            "grid": SAMPLE_GRID,
            "window_px": SAMPLE_WINDOW_PX,
            "pixels_per_band": SAMPLE_GRID * SAMPLE_GRID * SAMPLE_WINDOW_PX * SAMPLE_WINDOW_PX,
            "seed": None,
            "note": "no RNG: window origins are round(i*(extent-win)/(grid-1))",
            "dedup_grid": DEDUP_GRID,
            "dedup_window_px": DEDUP_WINDOW_PX,
        },
        "counts": counts,
        "rasters": rasters,
        "non_rasters": sorted(non_rasters),
        "unreadable": sorted(unreadable, key=lambda d: d["path"]),
    }
    # Computed last so it reflects the tree *after* the run (D-3).
    inv["data_tree_digest"] = tree_digest(root)
    inv["summary"] = summary_line(inv)
    log.info("inventory: %s", inv["summary"])
    return inv


def summary_line(inv: dict) -> str:
    """The one line a human reads first when an index build looks wrong."""
    c = inv["counts"]
    parts = [f"{c['indexable']} indexable"]
    for reason in sorted(REASONS):
        if c.get(reason):
            parts.append(f"{c[reason]} {reason}")
    parts.append(f"{c['rasters']} rasters of {c['files_scanned']} files scanned")
    if c["unreadable"]:
        parts.append(f"{c['unreadable']} UNREADABLE")
    return ", ".join(parts)


def source_imagery_paths(inv: dict, subdir: str | None = None) -> list[str]:
    """Paths passing the D-1 pixel rule -- source pixels, indexable or not."""
    pre = f"{subdir.rstrip('/')}/" if subdir else ""
    return sorted(
        r["path"] for r in inv["rasters"] if r["is_source_imagery"] and r["path"].startswith(pre)
    )


def indexable_paths(inv: dict, subdir: str | None = None) -> list[str]:
    """Paths that are source imagery AND admissible as scenes to index."""
    pre = f"{subdir.rstrip('/')}/" if subdir else ""
    return sorted(r["path"] for r in inv["rasters"] if r["is_indexable"] and r["path"].startswith(pre))


def write_inventory(inv: dict | None = None, path: Path | None = None) -> Path:
    """Write the inventory as sorted-key JSON under retrieval/index/ (D-3, D-4)."""
    inv = build_inventory() if inv is None else inv
    path = Path(INVENTORY_JSON if path is None else path)
    index_dir = INDEX_DIR.resolve()
    resolved = path.resolve()
    if not (resolved == index_dir or index_dir in resolved.parents):
        raise ValueError(f"refusing to write outside {index_dir}: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inv, sort_keys=True, indent=2) + "\n")
    return path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    # GDAL logs one INFO line per sidecar/vector file it has no raster driver
    # for -- 50 of them here. They are recorded in the artifact's
    # `non_rasters`, so they are not also shouted at the reader.
    logging.getLogger("rasterio._env").setLevel(logging.WARNING)
    inv = build_inventory()
    out = write_inventory(inv)
    print(inv["summary"])
    print(f"indexable scenes ({inv['counts']['indexable']}):")
    for r in inv["rasters"]:
        if r["is_indexable"]:
            print(
                f"  {r['path']:42s} {r['width']}x{r['height']} {r['band_count']}b "
                f"{r['crs']:11s} {r['true_gsd_cm']:.2f} cm/px  "
                f"{TILE_PX}px tile = {r['tile_ground_m']:.2f} m"
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
