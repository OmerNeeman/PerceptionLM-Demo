"""Tests for plm_sam2_tracker.tiled_grounding.

Pure stdlib, no model and no pixels: the "grounder" is a closure that
returns whatever part of a canned target falls inside the tile it is
handed - i.e. a perfect detector on a crop. Run with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plm_sam2_tracker.tiled_grounding import (  # noqa: E402
    Tile, ground_tiled, iou, merge_boxes, plan_tiles,
)


def fake_grounder(targets, min_visible=0.0, score=None, calls=None):
    """Perfect detector on a crop: target ∩ tile, in tile-local coords.

    min_visible skips targets with less than that fraction of their area in
    the tile (a real model misses a sliver); calls, if given, collects the
    tiles it was handed.
    """
    def ground(tile):
        if calls is not None:
            calls.append(tile)
        out = []
        for t in targets:
            x1, y1 = max(t[0], tile.x0), max(t[1], tile.y0)
            x2, y2 = min(t[2], tile.x1), min(t[3], tile.y1)
            if x2 <= x1 or y2 <= y1:
                continue
            visible = ((x2 - x1) * (y2 - y1)
                       / ((t[2] - t[0]) * (t[3] - t[1])))
            if visible < min_visible:
                continue
            box = [x1 - tile.x0, y1 - tile.y0, x2 - tile.x0, y2 - tile.y0]
            out.append(box + ([score] if score is not None else []))
        return out
    return ground


class BoxAssertions(unittest.TestCase):
    def assertBoxes(self, actual, expected, tol=1.0):
        """Compare box lists as sets of rounded corners, order-insensitive."""
        self.assertEqual(len(actual), len(expected),
                         f"expected {len(expected)} box(es), got {actual}")
        remaining = list(expected)
        for box in actual:
            hit = next((e for e in remaining
                        if all(abs(box[k] - e[k]) <= tol for k in range(4))),
                       None)
            self.assertIsNotNone(hit, f"unexpected box {box}, want {remaining}")
            remaining.remove(hit)


class TestPlanTiles(BoxAssertions):
    SIZES = [(3840, 2160), (1920, 1080), (1000, 500), (896, 448), (449, 449),
             (448, 448), (200, 150), (1, 1)]

    def test_covers_frame_without_gaps(self):
        for w, h in self.SIZES:
            for overlap in (0.0, 0.2, 0.5):
                tiles = plan_tiles(w, h, tile=448, overlap=overlap)
                cols, rows = set(), set()
                for t in tiles:
                    cols.update(range(t.x0, t.x1))
                    rows.update(range(t.y0, t.y1))
                self.assertEqual(cols, set(range(w)), f"{w}x{h} ov={overlap}")
                self.assertEqual(rows, set(range(h)), f"{w}x{h} ov={overlap}")

    def test_tiles_stay_inside_the_frame(self):
        for w, h in self.SIZES:
            for t in plan_tiles(w, h, tile=448, overlap=0.2):
                self.assertGreaterEqual(t.x0, 0)
                self.assertGreaterEqual(t.y0, 0)
                self.assertLessEqual(t.x1, w)
                self.assertLessEqual(t.y1, h)

    def test_every_tile_is_tile_sized(self):
        tiles = plan_tiles(1000, 500, tile=448, overlap=0.2)
        for t in tiles:
            self.assertEqual((t.width, t.height), (448, 448))
        self.assertEqual(min(t.x0 for t in tiles), 0)
        self.assertEqual(max(t.x1 for t in tiles), 1000)   # clamped to the edge
        self.assertEqual(max(t.y1 for t in tiles), 500)

    def test_frame_smaller_than_a_tile_is_one_tile(self):
        tiles = plan_tiles(200, 150, tile=448, overlap=0.2)
        self.assertEqual(tiles, [Tile(0, 0, 200, 150, 0, 0)])

    def test_exact_multiple_and_zero_overlap_abut(self):
        tiles = plan_tiles(896, 448, tile=448, overlap=0.0)
        self.assertEqual([(t.x0, t.x1) for t in tiles], [(0, 448), (448, 896)])

    def test_overlap_steps_by_one_minus_overlap(self):
        tiles = plan_tiles(2000, 448, tile=448, overlap=0.2)
        xs = [t.x0 for t in tiles]
        self.assertEqual(xs[0], 0)
        self.assertEqual(tiles[-1].x1, 2000)
        self.assertTrue(all(b - a <= 358 for a, b in zip(xs, xs[1:])), xs)

    def test_row_major_grid_indices(self):
        tiles = plan_tiles(1000, 500, tile=448, overlap=0.2)
        self.assertEqual([(t.col, t.row) for t in tiles],
                         [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)])

    def test_bad_arguments(self):
        for args in [(0, 100), (100, -1)]:
            with self.assertRaises(ValueError):
                plan_tiles(*args)
        with self.assertRaises(ValueError):
            plan_tiles(100, 100, tile=0)
        for overlap in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                plan_tiles(100, 100, overlap=overlap)


class TestTileMapping(BoxAssertions):
    def test_local_box_translates_to_frame(self):
        tile = Tile(276, 52, 724, 500, 1, 1)
        self.assertEqual(tile.to_frame([10, 20, 30, 40]), [286, 72, 306, 92])

    def test_extras_ride_along(self):
        tile = Tile(100, 200, 548, 648)
        self.assertEqual(tile.to_frame([0, 0, 10, 10, 0.9]),
                         [100, 200, 110, 210, 0.9])

    def test_box_is_clipped_to_the_tile(self):
        tile = Tile(276, 52, 724, 500)
        self.assertEqual(tile.to_frame([-20, -5, 999, 999]),
                         [276, 52, 724, 500])

    def test_mapping_round_trips_the_fake_grounder(self):
        # A target the grounder reports tile-locally maps back to itself.
        target = [300, 200, 340, 240]
        ground = fake_grounder([target])
        for tile in plan_tiles(1000, 500):
            local = ground(tile)
            if not local:
                continue
            mapped = tile.to_frame(local[0])
            self.assertEqual(mapped, [max(target[0], tile.x0),
                                      max(target[1], tile.y0),
                                      min(target[2], tile.x1),
                                      min(target[3], tile.y1)])


class TestBoxMath(BoxAssertions):
    def test_iou(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    def test_merge_collapses_duplicates(self):
        boxes = [[10, 10, 50, 50], [11, 9, 49, 51], [200, 200, 240, 240]]
        self.assertBoxes(merge_boxes(boxes), [[10, 10, 50, 50],
                                              [200, 200, 240, 240]], tol=2)

    def test_merge_keeps_the_highest_score_and_orders_by_it(self):
        boxes = [[10, 10, 50, 50, 0.4], [11, 11, 51, 51, 0.9],
                 [200, 200, 240, 240, 0.6]]
        kept = merge_boxes(boxes)
        self.assertEqual(kept, [[11, 11, 51, 51, 0.9],
                                [200, 200, 240, 240, 0.6]])

    def test_merge_prefers_the_larger_box_on_a_score_tie(self):
        # Unscored boxes tie at 1.0, so the most complete box wins.
        kept = merge_boxes([[12, 12, 48, 48], [10, 10, 50, 50]])
        self.assertEqual(kept, [[10, 10, 50, 50]])

    def test_merge_respects_the_threshold(self):
        boxes = [[0, 0, 10, 10], [5, 0, 15, 10]]        # IoU = 1/3
        self.assertEqual(len(merge_boxes(boxes, iou_thresh=0.5)), 2)
        self.assertEqual(len(merge_boxes(boxes, iou_thresh=0.3)), 1)

    def test_merge_empty(self):
        self.assertEqual(merge_boxes([]), [])


class TestGroundTiled(BoxAssertions):
    def test_calls_the_grounder_once_per_tile(self):
        calls = []
        ground_tiled(1000, 500, fake_grounder([], calls=calls))
        self.assertEqual(calls, plan_tiles(1000, 500))

    def test_no_detections(self):
        self.assertEqual(ground_tiled(1000, 500, fake_grounder([])), [])
        self.assertEqual(ground_tiled(1000, 500, lambda t: None), [])

    def test_duplicate_across_a_seam_merges_to_one(self):
        # A 40px car sitting in the overlap zone is seen by four tiles.
        target = [300, 200, 340, 240]
        seen = [t for t in plan_tiles(1000, 500)
                if t.x0 <= target[0] and target[2] <= t.x1
                and t.y0 <= target[1] and target[3] <= t.y1]
        self.assertGreater(len(seen), 1, "test needs a target in >1 tile")
        self.assertBoxes(ground_tiled(1000, 500, fake_grounder([target])),
                         [target])

    def test_target_split_by_a_seam_is_stitched(self):
        # Straddles the x=448 boundary with overlap=0: two half boxes.
        target = [430, 200, 470, 240]
        self.assertBoxes(
            ground_tiled(896, 448, fake_grounder([target]), overlap=0.0),
            [target])

    def test_target_wider_than_a_tile(self):
        target = [100, 100, 800, 300]           # 700px wide vs a 448px tile
        self.assertBoxes(ground_tiled(1000, 500, fake_grounder([target])),
                         [target])

    def test_target_larger_than_a_tile_in_both_axes(self):
        target = [100, 50, 900, 700]
        self.assertBoxes(ground_tiled(1200, 800, fake_grounder([target])),
                         [target])

    def test_slivers_are_dropped_when_a_neighbour_saw_the_whole_object(self):
        # The model only reports a target when most of it is in the tile, so
        # the seam-side sliver never appears - the whole box must still win.
        target = [440, 200, 500, 240]
        self.assertBoxes(
            ground_tiled(1000, 500, fake_grounder([target], min_visible=0.5)),
            [target])

    def test_tiny_frame_passes_boxes_straight_through(self):
        target = [20, 30, 60, 90]
        out = ground_tiled(200, 150, fake_grounder([target]))
        self.assertEqual(len(plan_tiles(200, 150)), 1)
        self.assertBoxes(out, [target])

    def test_several_targets_survive_separately(self):
        targets = [[80, 60, 120, 100], [300, 200, 340, 240],
                   [905, 410, 960, 470], [500, 30, 560, 70]]
        self.assertBoxes(ground_tiled(1000, 500, fake_grounder(targets)),
                         targets)

    def test_neighbours_along_a_seam_are_not_stitched_together(self):
        # Two distinct cars either side of the x=448 seam, 60px apart.
        targets = [[380, 200, 430, 240], [490, 200, 540, 240]]
        self.assertBoxes(
            ground_tiled(896, 448, fake_grounder(targets), overlap=0.0),
            targets)

    def test_scores_are_preserved(self):
        out = ground_tiled(1000, 500,
                           fake_grounder([[300, 200, 340, 240]], score=0.75))
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 5)
        self.assertAlmostEqual(out[0][4], 0.75)

    def test_boxes_stay_inside_the_frame(self):
        # A grounder that overshoots its tile must not escape the frame.
        out = ground_tiled(1000, 500, lambda t: [[-50, -50, 999, 999]])
        for box in out:
            self.assertGreaterEqual(min(box[0], box[1]), 0)
            self.assertLessEqual(box[2], 1000)
            self.assertLessEqual(box[3], 500)

    def test_degenerate_boxes_are_dropped(self):
        self.assertEqual(ground_tiled(1000, 500, lambda t: [[10, 10, 10, 40]]),
                         [])

    def test_stitching_can_be_switched_off(self):
        target = [430, 200, 470, 240]
        out = ground_tiled(896, 448, fake_grounder([target]), overlap=0.0,
                           stitch=False)
        self.assertEqual(len(out), 2)          # the two halves, unstitched


if __name__ == "__main__":
    unittest.main()
