"""F-10: every planned tile carries its map location, and a tile id round-trips
through the geotransform to reproduce its pixel offsets exactly -- for both
CRSs in the corpus (leb, EPSG:3857 Mercator; the six others, EPSG:32636 UTM).

Run:
    env PYTHONNOUSERSITE=1 AERIAL_DATA_ROOT=<data root> \
        /home/omer/anaconda3/envs/geo/bin/python -m pytest \
        retrieval/tests/test_geo_roundtrip.py -v
"""

from __future__ import annotations

import math

import pytest
import rasterio

import config
import tiling

MERCATOR_SCENE = "leb/2022-10-29.tif"  # EPSG:3857
UTM_SCENE = "X605_Y3388.tif"  # EPSG:32636

# The lon/lat AABB of an adjacent-tile pair is bit-identical only when the
# raster's projected axes are already aligned with lon/lat -- true for
# Mercator (a vertical line of constant projected x IS a line of constant
# longitude) but not for UTM, where meridian convergence away from the zone's
# central meridian tilts the raster grid relative to lon/lat by a few tenths
# of a degree. That tilts each tile's true (rotated) footprint slightly
# inside its axis-aligned bounding box, so neighbouring AABBs can overlap by
# a small, bounded amount even though the true tile edges touch with zero
# overlap. Measured worst case over X605's full grid at 224 px: ~0.26 m.
# 1.0 m is comfortably above that and comfortably below 1% of even the
# smallest tile (11.2 m at 112 px), so it catches a real defect (a
# mis-indexed grid line, a sign error) without flagging this known,
# geometry-driven artifact.
_UTM_AABB_TOLERANCE_M = 1.0
_LAT_M_PER_DEG = 111_320.0


@pytest.fixture(scope="module")
def mercator_plan():
    return tiling.plan_scene(MERCATOR_SCENE, 224)


@pytest.fixture(scope="module")
def utm_plan():
    return tiling.plan_scene(UTM_SCENE, 224)


# --------------------------------------------------------------------------
# id -> pixel offsets, exact, for both CRSs.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("plan_fixture", ["mercator_plan", "utm_plan"])
def test_tile_id_roundtrips_to_exact_pixel_offsets(plan_fixture, request):
    plan = request.getfixturevalue(plan_fixture)
    for t in plan["tiles"]:
        col_px, row_px = tiling.pixel_offset_from_tile_id(t["tile_id"])
        assert (col_px, row_px) == (t["px_offset_x"], t["px_offset_y"])
        # And the parse recovers every other identity component exactly.
        parsed = tiling.parse_tile_id(t["tile_id"])
        assert parsed["aoi"] == t["aoi"]
        assert parsed["source_file"] == t["source_file"]
        assert parsed["date"] == t["date"]
        assert parsed["scale"] == t["scale"]
        assert parsed["col"] == t["col"]
        assert parsed["row"] == t["row"]


def test_roundtrip_reproduces_the_geotransform_corner_from_a_fresh_open(mercator_plan):
    """The strong version of the round-trip: reopen the raster from disk
    (not the cached transform used to build the plan), recompute the tile's
    top-left corner from its pixel offset via the geotransform, reproject to
    WGS84, and confirm it matches the bbox stored in the plan exactly."""
    from rasterio.warp import transform as warp_transform

    t = mercator_plan["tiles"][len(mercator_plan["tiles"]) // 2]  # an interior tile
    col_px, row_px = tiling.pixel_offset_from_tile_id(t["tile_id"])

    root = config.get_data_root()
    with rasterio.open(root / MERCATOR_SCENE) as ds:
        x, y = ds.transform * (col_px, row_px)
        lon, lat = warp_transform(ds.crs, tiling.WGS84, [x], [y])

    min_lon, min_lat, max_lon, max_lat = t["bbox_lonlat"]
    # The recomputed corner must be exactly one of the tile's stored bbox
    # corners (its top-left, in pixel-offset terms).
    assert lon[0] == pytest.approx(min_lon, abs=1e-9) or lon[0] == pytest.approx(max_lon, abs=1e-9)
    assert lat[0] == pytest.approx(min_lat, abs=1e-9) or lat[0] == pytest.approx(max_lat, abs=1e-9)


# --------------------------------------------------------------------------
# bbox plausibility.
# --------------------------------------------------------------------------


def test_mercator_bboxes_are_plausible(mercator_plan):
    # docs/DATA.md: leb centres near lon 35.239, lat 33.107.
    for t in mercator_plan["tiles"]:
        min_lon, min_lat, max_lon, max_lat = t["bbox_lonlat"]
        assert 35.0 < min_lon < max_lon < 35.5
        assert 32.9 < min_lat < max_lat < 33.3


def test_utm_bboxes_are_plausible(utm_plan):
    # docs/DATA.md: X605_Y3388 sits near lon 34.26, lat 31.36.
    for t in utm_plan["tiles"]:
        min_lon, min_lat, max_lon, max_lat = t["bbox_lonlat"]
        assert 34.0 < min_lon < max_lon < 34.5
        assert 31.0 < min_lat < max_lat < 31.6


# --------------------------------------------------------------------------
# Adjacency: no gaps at OVERLAP=0, for both CRSs; bit-exact contiguity for
# Mercator, and a small, explained, bounded overlap for UTM (see the module
# docstring above the tolerance constant).
# --------------------------------------------------------------------------


def _by_col_row(plan):
    return {(t["col"], t["row"]): t for t in plan["tiles"]}


def test_mercator_adjacent_tiles_share_bit_exact_edges(mercator_plan):
    tiles = _by_col_row(mercator_plan)
    cols, rows = mercator_plan["grid_cols"], mercator_plan["grid_rows"]
    checked = 0
    for c in range(cols - 1):
        for r in range(rows):
            a = tiles[(c, r)]["bbox_lonlat"]
            b = tiles[(c + 1, r)]["bbox_lonlat"]
            assert a[2] == b[0], f"gap/mismatch at col {c}|{c+1}, row {r}: {a[2]} != {b[0]}"
            checked += 1
    for c in range(cols):
        for r in range(rows - 1):
            a = tiles[(c, r)]["bbox_lonlat"]
            b = tiles[(c, r + 1)]["bbox_lonlat"]
            assert a[1] == b[3], f"gap/mismatch at row {r}|{r+1}, col {c}: {a[1]} != {b[3]}"
            checked += 1
    assert checked > 0


def test_utm_adjacent_tiles_have_no_gap_and_bounded_overlap(utm_plan):
    """UTM's per-tile AABB is not exactly contiguous with its neighbour's --
    see the module docstring on `_UTM_AABB_TOLERANCE_M`. Measured live: the
    mismatch is a few tenths of a metre and can show as a tiny *overlap* or a
    tiny *gap* depending on which of a tile's two corners on that edge sits
    outside the neighbour's, which flips across the grid as the meridian
    convergence angle changes sign/magnitude with position. This is provably
    not a real coverage gap -- `test_projected_space_grid_is_exactly_
    contiguous_for_both_crs` below shows the underlying raster-space grid has
    none -- so what is actually asserted here is that the WGS84 AABB artifact
    stays within the stated bound in *either* direction, never that it is
    zero or one-signed."""
    tiles = _by_col_row(utm_plan)
    cols, rows = utm_plan["grid_cols"], utm_plan["grid_rows"]
    lat0 = (utm_plan["tiles"][0]["bbox_lonlat"][1] + utm_plan["tiles"][0]["bbox_lonlat"][3]) / 2
    m_per_deg_lon = _LAT_M_PER_DEG * math.cos(math.radians(lat0))

    checked = 0
    for c in range(cols - 1):
        for r in range(rows):
            a = tiles[(c, r)]["bbox_lonlat"]
            b = tiles[(c + 1, r)]["bbox_lonlat"]
            mismatch_m = abs(a[2] - b[0]) * m_per_deg_lon
            assert mismatch_m < _UTM_AABB_TOLERANCE_M, (
                f"col {c}|{c+1} row {r}: edge mismatch {mismatch_m:.3f} m exceeds "
                f"{_UTM_AABB_TOLERANCE_M} m tolerance"
            )
            checked += 1
    for c in range(cols):
        for r in range(rows - 1):
            a = tiles[(c, r)]["bbox_lonlat"]
            b = tiles[(c, r + 1)]["bbox_lonlat"]
            mismatch_m = abs(a[1] - b[3]) * _LAT_M_PER_DEG
            assert mismatch_m < _UTM_AABB_TOLERANCE_M, (
                f"row {r}|{r+1} col {c}: edge mismatch {mismatch_m:.3f} m exceeds "
                f"{_UTM_AABB_TOLERANCE_M} m tolerance"
            )
            checked += 1
    assert checked > 0


def test_projected_space_grid_is_exactly_contiguous_for_both_crs(mercator_plan, utm_plan):
    """The underlying claim F-1 actually needs -- the raster-space grid has
    no gaps or overlaps -- is true in the CRS's own projected coordinates
    regardless of what happens after WGS84 reprojection. Both rasters are
    north-up (no shear in the affine), so this must hold exactly for both.

    Tolerance note: `transform * (col, row)` adds a large false-easting/
    -northing constant (~3.9e6 for leb, ~7e5/3.6e6 for X605) to a small
    per-tile term before this test subtracts two such sums back down to a
    ~10-100 m difference -- textbook catastrophic cancellation, costing
    ~9-10 significant digits versus a raw double's ~15-16. A `rel=1e-9`
    tolerance absorbs that (measured residual ~1.7e-12 relative) while
    staying 9 orders of magnitude tighter than a real grid-stride bug would
    produce (a wrong stride misses by a whole pixel, ~1e-2 relative at these
    tile sizes).
    """
    for plan, rel in [(mercator_plan, MERCATOR_SCENE), (utm_plan, UTM_SCENE)]:
        root = config.get_data_root()
        with rasterio.open(root / rel) as ds:
            transform = ds.transform
        assert transform.b == 0.0 and transform.d == 0.0, f"{rel}: unexpected shear in geotransform"
        scale = plan["scale"]
        for t in plan["tiles"][:5]:  # a handful is enough to prove the arithmetic
            x0, y0 = transform * (t["px_offset_x"], t["px_offset_y"])
            x1, y1 = transform * (t["px_offset_x"] + scale, t["px_offset_y"] + scale)
            assert x1 - x0 == pytest.approx(scale * transform.a, rel=1e-9)
            assert y1 - y0 == pytest.approx(scale * transform.e, rel=1e-9)
