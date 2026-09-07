"""F-1 (coverage), D-4 (stable ids), D-3 (data dir read-only) for the tile
planner. See briefs/S2.md for the arithmetic this pins -- it was independently
recomputed by the PM from docs/DATA.md's scene dimensions and is asserted
here exactly, not re-derived.

Run:
    env PYTHONNOUSERSITE=1 AERIAL_DATA_ROOT=<data root> \
        /home/omer/anaconda3/envs/geo/bin/python -m pytest \
        retrieval/tests/test_tiling.py -v
"""

from __future__ import annotations

import json

import pytest

import cog
import config
import inventory
import tiling

# From briefs/S2.md -- exact, PM-verified, not to be re-derived.
EXPECTED_PER_SCALE_TOTAL = {448: 5198, 224: 20704, 112: 82640}
EXPECTED_GRAND_TOTAL = 108542
LEB_448_SINGLE_SCENE = 1012
DEMO_AOI = "X605_Y3388.tif"
DEMO_AOI_TOTAL = 11109


@pytest.fixture(scope="module")
def indexable_scenes():
    return tiling.load_indexable_scenes()


@pytest.fixture(scope="module")
def all_plans(indexable_scenes):
    return {rel: {s: tiling.plan_scene(rel, s) for s in tiling.SCALES} for rel in indexable_scenes}


# --------------------------------------------------------------------------
# F-1 -- exact counts and zero coverage gaps.
# --------------------------------------------------------------------------


def test_exactly_eight_indexable_scenes(indexable_scenes):
    assert len(indexable_scenes) == 8, indexable_scenes


def test_tiling_covers_extent_exact_counts(all_plans):
    per_scale_totals = {s: 0 for s in tiling.SCALES}
    for rel, per_scale in all_plans.items():
        for s, plan in per_scale.items():
            assert plan["planned_count"] == len(plan["tiles"])
            assert plan["planned_count"] == plan["grid_cols"] * plan["grid_rows"]
            per_scale_totals[s] += plan["planned_count"]

    assert per_scale_totals == EXPECTED_PER_SCALE_TOTAL
    assert sum(per_scale_totals.values()) == EXPECTED_GRAND_TOTAL


def test_one_leb_scene_at_448_is_1012(all_plans):
    plan = all_plans["leb/2022-10-29.tif"][448]
    assert plan["planned_count"] == LEB_448_SINGLE_SCENE
    assert (plan["grid_cols"], plan["grid_rows"]) == (46, 22)


def test_demo_aoi_pyramid(all_plans):
    total = sum(all_plans[DEMO_AOI][s]["planned_count"] for s in tiling.SCALES)
    assert total == DEMO_AOI_TOTAL


@pytest.mark.parametrize("scale", tiling.SCALES)
def test_zero_coverage_gaps(all_plans, scale):
    """Every pixel of every raster falls in >= 1 tile at this scale -- not
    just a correct count (an off-by-one origin can pass the count and still
    leave a seam). Checked two ways: (a) the last row/column of the grid
    still has a nonzero valid region (padding never degenerates to an empty
    tile), and (b) the padded grid's total footprint
    (grid_cols*scale, grid_rows*scale) is >= the raster's own (width,
    height) on both axes -- i.e. the grid cannot possibly stop short.
    """
    for rel, per_scale in all_plans.items():
        plan = per_scale[scale]
        w, h = plan["width"], plan["height"]
        cols, rows = plan["grid_cols"], plan["grid_rows"]

        assert cols * scale >= w, f"{rel}@{scale}: grid width {cols*scale} < raster width {w}"
        assert rows * scale >= h, f"{rel}@{scale}: grid height {rows*scale} < raster height {h}"
        # ceil() must not overshoot by a whole tile either -- that would mean
        # a phantom all-padding row/column with zero real content.
        assert (cols - 1) * scale < w, f"{rel}@{scale}: an entirely-padding column exists"
        assert (rows - 1) * scale < h, f"{rel}@{scale}: an entirely-padding row exists"

        by_cr = {(t["col"], t["row"]): t for t in plan["tiles"]}
        for c in range(cols):
            last_row_tile = by_cr[(c, rows - 1)]
            assert last_row_tile["valid_height"] > 0, f"{rel}@{scale}: empty tile at col {c}"
        for r in range(rows):
            last_col_tile = by_cr[(cols - 1, r)]
            assert last_col_tile["valid_width"] > 0, f"{rel}@{scale}: empty tile at row {r}"


def test_edge_tiles_are_padded_not_dropped(all_plans):
    """F-1: dropping the incomplete edge tiles would lose real content
    (376 rows / 3.8% of scene height at 448 px on a 10240 px scene). The
    planned count must equal the CEIL grid, never the FLOOR grid."""
    plan = all_plans["AYOSH/X693_Y3500.tif"][448]
    floor_cols, floor_rows = tiling.complete_grid_dims(plan["width"], plan["height"], 448)
    assert plan["planned_count"] > floor_cols * floor_rows
    assert (plan["grid_cols"], plan["grid_rows"]) == (23, 23)
    assert (floor_cols, floor_rows) == (22, 22)

    edge_tiles = [t for t in plan["tiles"] if t["is_edge"]]
    assert len(edge_tiles) > 0
    for t in edge_tiles:
        assert 0 < t["valid_width"] <= 448
        assert 0 < t["valid_height"] <= 448
        assert t["valid_width"] < 448 or t["valid_height"] < 448
        assert t["pad_value"] == tiling.PAD_VALUE


def test_stated_pad_value(all_plans):
    """F-1: 'state ... what pad value is used'."""
    assert tiling.PAD_VALUE == 0
    for per_scale in all_plans.values():
        for plan in per_scale.values():
            assert all(t["pad_value"] == 0 for t in plan["tiles"])


# --------------------------------------------------------------------------
# D-4 -- tile identity is stable and reproducible.
# --------------------------------------------------------------------------


def test_tile_id_stable_across_two_runs():
    rel = "leb/2022-10-29.tif"
    plan_a = tiling.plan_scene(rel, 224)
    plan_b = tiling.plan_scene(rel, 224)
    ids_a = [t["tile_id"] for t in plan_a["tiles"]]
    ids_b = [t["tile_id"] for t in plan_b["tiles"]]
    assert ids_a == ids_b
    assert len(ids_a) == len(set(ids_a)), "duplicate tile ids within one plan"
    assert plan_a["planned_count"] == plan_b["planned_count"]


def test_tile_ids_carry_no_machine_specific_or_run_dependent_value():
    import re

    root = str(config.get_data_root())
    plan = tiling.plan_scene("leb/2022-10-29.tif", 448)
    # A build-time value (wall clock, run counter) would show up as a long
    # digit run with no counterpart in the inputs -- col/row/scale are all
    # small, zero-padded fields, so a >= 8-digit run is foreign to this format.
    timestamp_like = re.compile(r"\d{8,}")
    for t in plan["tiles"]:
        assert root not in t["tile_id"]
        assert "/home/" not in t["tile_id"]
        assert not timestamp_like.search(t["tile_id"]), t["tile_id"]
        assert not t["source_file"].startswith("/"), "absolute path leaked into tile id"


def test_two_date_unknown_scenes_get_disjoint_ids():
    """The trap this stage must not fall into: five of the eight scenes carry
    date: unknown. A tile id scheme keyed only on (aoi, date, col, row) would
    collapse AYOSH's two same-AOI, same-date, same-dimensions scenes into one
    id space. source_file is part of the id precisely to prevent that."""
    a = tiling.plan_scene("AYOSH/X693_Y3500.tif", 448)
    b = tiling.plan_scene("AYOSH/X693_Y3501.tif", 448)
    assert a["date"] == b["date"] == "unknown"
    assert a["aoi"] == b["aoi"] == "AYOSH"

    ids_a = {t["tile_id"] for t in a["tiles"]}
    ids_b = {t["tile_id"] for t in b["tiles"]}
    assert ids_a.isdisjoint(ids_b), "two date:unknown scenes collapsed into one id space"
    # Same (col, row) must still differ, since only source_file separates them.
    same_cr = tiling.make_tile_id("AYOSH", "AYOSH/X693_Y3500.tif", "unknown", 448, 3, 3)
    other = tiling.make_tile_id("AYOSH", "AYOSH/X693_Y3501.tif", "unknown", 448, 3, 3)
    assert same_cr != other


def test_tile_id_depends_on_full_identity_tuple():
    base = tiling.make_tile_id("leb", "leb/2022-10-29.tif", "2022-10-29", 448, 1, 2)
    assert tiling.make_tile_id("leb", "leb/2022-10-29.tif", "2022-10-29", 448, 1, 3) != base  # row
    assert tiling.make_tile_id("leb", "leb/2022-10-29.tif", "2022-10-29", 448, 2, 2) != base  # col
    assert tiling.make_tile_id("leb", "leb/2022-10-29.tif", "2022-10-29", 224, 1, 2) != base  # scale
    assert tiling.make_tile_id("leb", "leb/2025-06-06.tif", "2025-06-06", 448, 1, 2) != base  # date+file
    assert tiling.make_tile_id("aoi2", "leb/2022-10-29.tif", "2022-10-29", 448, 1, 2) != base  # aoi
    # source_file alone, everything else held equal -- this is the exact
    # shape of the AYOSH X693_Y3500/X693_Y3501 collision this stage must
    # avoid (both date: unknown, both AOI "AYOSH"), isolated from any
    # simultaneous date change so a regression that drops source_file from
    # the id cannot hide behind date also differing.
    assert tiling.make_tile_id("AYOSH", "AYOSH/X693_Y3500.tif", "unknown", 448, 1, 2) != (
        tiling.make_tile_id("AYOSH", "AYOSH/X693_Y3501.tif", "unknown", 448, 1, 2)
    )


# --------------------------------------------------------------------------
# D-3 -- the data directory is read-only; every output resolves inside index/.
# --------------------------------------------------------------------------


def test_tile_plan_output_paths_resolve_inside_index(tmp_path):
    out_dir = config.get_index_root() / "tileplan"
    plans = {"leb/2022-10-29.tif": {448: tiling.plan_scene("leb/2022-10-29.tif", 448)}}
    written = tiling.write_tile_plans(plans, out_dir=out_dir)
    index_root = config.get_index_root().resolve()
    assert written
    for p in written:
        assert index_root in p.resolve().parents or p.resolve() == index_root


def test_write_tile_plans_refuses_to_write_outside_index(tmp_path):
    """The guard must reject ANY path outside the index root -- exercised
    against a throwaway tmp_path, deliberately never against the real data
    root: if this guard were ever disabled, pointing the target at the data
    root here would actually write under it, which is the one outcome this
    whole stage exists to prevent."""
    plans = {"leb/2022-10-29.tif": {448: tiling.plan_scene("leb/2022-10-29.tif", 448)}}
    with pytest.raises(ValueError):
        tiling.write_tile_plans(plans, out_dir=tmp_path / "escape")
    assert not (tmp_path / "escape").exists(), "guard raised too late -- directory was already created"


def test_data_dir_digest_check_would_actually_catch_a_change(monkeypatch):
    """Proves the assertion in the test below has teeth, without ever
    mutating the real (read-only) data tree to do it: fakes tree_digest
    returning two different values across the two calls the real test makes,
    and confirms that shape is detected as a failure."""
    calls = iter(["sha256:before", "sha256:after-a-real-change"])
    monkeypatch.setattr(inventory, "tree_digest", lambda *a, **k: next(calls))
    before = inventory.tree_digest()
    after = inventory.tree_digest()
    with pytest.raises(AssertionError):
        assert before == after, "the read-only data tree changed during a plan + COG build"


def test_data_dir_unchanged_across_a_full_plan_and_cog_build():
    """D-3: recursive mtime/checksum of the data tree is unchanged across a
    full index build -- here, a full tile plan (all 8 scenes x 3 scales,
    written to disk) plus the leb COG build."""
    before = inventory.tree_digest()

    plans = tiling.plan_all()
    tiling.write_tile_plans(plans)
    cog.convert_leb()

    after = inventory.tree_digest()
    assert before == after, "the read-only data tree changed during a plan + COG build"


def test_tile_plan_files_are_valid_json_and_round_trip():
    plans = tiling.plan_all(scales=(448,))
    written = tiling.write_tile_plans(plans, out_dir=config.get_index_root() / "tileplan")
    total_from_disk = 0
    for p in written:
        doc = json.loads(p.read_text())
        for scale_key, plan in doc.items():
            total_from_disk += plan["planned_count"]
    assert total_from_disk == EXPECTED_PER_SCALE_TOTAL[448]
