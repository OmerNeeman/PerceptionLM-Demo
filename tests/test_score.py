"""Unit tests for the eval scorer's metric math.

    python3 -m unittest discover -s tests

Stdlib only. Fixtures are two 5-frame ground-truth tracks in separate lanes,
so every expected count can be checked by hand.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("score", ROOT / "eval" / "score.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)


def box(frame: int, lane: int = 0, dx: float = 0.0) -> list[float]:
    """A 10x10 box that steps 20px right each frame, one lane per object."""
    x = 20 * frame + dx
    return [x, 100 * lane, x + 10, 100 * lane + 10]


def clip(tracks: list[dict], labeled: list[int] | None = None, **meta) -> object:
    payload = {"meta": {"fps": 10.0, **meta}, "tracks": tracks}
    if labeled is not None:
        payload["meta"]["labeled_frames"] = labeled
    return score.Clip("<test>", payload)


def track(tid: int, lane: int, frames, dx: float = 0.0, **extra) -> dict:
    rec = {"id": tid, "label": f"obj{tid}", "acquired_frame": min(frames),
           "boxes": {str(f): box(f, lane, dx) for f in frames}}
    rec.update(extra)
    return rec


def gt_two_tracks() -> object:
    """Two objects, both present on labeled frames 0-4. 10 GT boxes."""
    return clip([track(1, 0, range(5)), track(2, 1, range(5))],
                labeled=list(range(5)))


class TestIoU(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(score.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint(self):
        self.assertEqual(score.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_touching_edges_is_zero(self):
        self.assertEqual(score.iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)

    def test_half_overlap(self):
        # 5x10 intersection, 150 union.
        self.assertAlmostEqual(score.iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)

    def test_contained(self):
        self.assertAlmostEqual(score.iou([0, 0, 10, 10], [0, 0, 5, 10]), 0.5)

    def test_degenerate_box(self):
        self.assertEqual(score.iou([0, 0, 0, 0], [0, 0, 10, 10]), 0.0)


class TestAssignment(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(score.hungarian([]), [])
        self.assertEqual(score.hungarian([[]]), [])

    def test_square_optimal(self):
        # Greedy would take (0,0) at -0.9 and strand row 1; the optimum is
        # -0.6 + -0.85 = -1.45.
        pairs = score.hungarian([[-0.9, -0.6], [-0.85, 0.0]])
        self.assertEqual(sorted(pairs), [(0, 1), (1, 0)])

    def test_rectangular_both_orientations(self):
        wide = score.hungarian([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        self.assertEqual(sorted(wide), [(0, 0), (1, 2)])
        tall = score.hungarian([[-1.0, 0.0], [0.0, 0.0], [0.0, -1.0]])
        self.assertEqual(sorted(tall), [(0, 0), (2, 1)])

    def test_match_boxes_drops_below_threshold(self):
        rows = [[0, 0, 10, 10]]
        cols = [[7, 0, 17, 10]]           # IoU 3/17 ~ 0.176
        self.assertEqual(score.match_boxes(rows, cols, 0.5), [])
        self.assertEqual(score.match_boxes(rows, cols, 0.1), [(0, 0)])

    def test_hungarian_beats_greedy_on_matched_count(self):
        # G0 overlaps both preds; G1 only overlaps P0. Greedy grabs the best
        # pair first and finds one match, Hungarian finds two.
        rows = [[0, 0, 10, 10], [4, 0, 14, 10]]
        cols = [[4, 0, 14, 10], [0, 0, 10, 10]]
        self.assertEqual(len(score.match_boxes(rows, cols, 0.4, greedy=False)), 2)


class TestPerfectPrediction(unittest.TestCase):
    def setUp(self):
        self.gt = gt_two_tracks()
        self.pred = clip([track(1, 0, range(5)), track(2, 1, range(5))])
        self.res = score.score(self.gt, self.pred)

    def test_detection_perfect(self):
        det = self.res["detection"]
        self.assertEqual((det["tp"], det["fp"], det["fn"]), (10, 0, 0))
        self.assertEqual((det["precision"], det["recall"], det["f1"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(det["mean_iou"], 1.0)

    def test_tracking_perfect(self):
        trk = self.res["tracking"]
        self.assertAlmostEqual(trk["mota"], 1.0)
        self.assertAlmostEqual(trk["idf1"], 1.0)
        self.assertAlmostEqual(trk["motp"], 1.0)
        self.assertEqual(trk["id_switches"], 0)

    def test_grounding_perfect(self):
        gr = self.res["grounding"]
        self.assertEqual((gr["grounded"], gr["total"]), (2, 2))
        self.assertEqual(gr["recall"], 1.0)
        self.assertEqual(gr["median_latency_frames"], 0)

    def test_greedy_matches_hungarian_here(self):
        greedy = score.score(self.gt, self.pred, greedy=True)
        self.assertEqual(greedy["detection"], self.res["detection"])
        self.assertEqual(greedy["tracking"]["idf1"], self.res["tracking"]["idf1"])

    def test_report_renders(self):
        text = score.format_report(self.res)
        self.assertIn("GROUNDING", text)
        self.assertIn("MOTA", text)
        self.assertEqual(text.count("\n") + 1, len(text.splitlines()))


class TestKnownWrongPrediction(unittest.TestCase):
    """One dropped box (FN) plus one spurious detection (FP) out of 10 GT."""

    def setUp(self):
        spurious = {"id": 3, "label": "ghost", "acquired_frame": 0,
                    "boxes": {"0": [500, 500, 510, 510]}}
        self.res = score.score(gt_two_tracks(), clip([
            track(1, 0, range(5)),
            track(2, 1, [0, 1, 3, 4]),      # frame 2 missing
            spurious,
        ]))

    def test_detection_counts(self):
        det = self.res["detection"]
        self.assertEqual((det["tp"], det["fp"], det["fn"]), (9, 1, 1))
        self.assertAlmostEqual(det["precision"], 0.9)
        self.assertAlmostEqual(det["recall"], 0.9)
        self.assertAlmostEqual(det["f1"], 0.9)

    def test_mota(self):
        trk = self.res["tracking"]
        self.assertAlmostEqual(trk["mota"], 1 - 2 / 10)     # 1 fp + 1 fn
        self.assertEqual(trk["id_switches"], 0)

    def test_idf1(self):
        trk = self.res["tracking"]
        self.assertEqual((trk["idtp"], trk["idfp"], trk["idfn"]), (9, 1, 1))
        self.assertAlmostEqual(trk["idf1"], 2 * 9 / (2 * 9 + 1 + 1))

    def test_gap_does_not_count_as_id_switch(self):
        # Track 2 disappears for a frame and comes back under the same id.
        self.assertEqual(self.res["tracking"]["id_switches"], 0)


class TestIdentitySwitches(unittest.TestCase):
    def test_handover_counts_one_switch(self):
        """One GT object handed from pred #1 to pred #3 mid-clip: 1 switch."""
        res = score.score(gt_two_tracks(), clip([
            track(1, 0, [0, 1, 2]),
            track(3, 0, [3, 4]),            # same object, new id
            track(2, 1, range(5)),
        ]))
        trk = res["tracking"]
        self.assertEqual(trk["id_switches"], 1)
        self.assertEqual((trk["tp"], trk["fp"], trk["fn"]), (10, 0, 0))
        self.assertAlmostEqual(trk["mota"], 1 - 1 / 10)
        # IDF1 pairs GT1 with the longer half only: 3 + 5 of 10 boxes.
        self.assertEqual(trk["idtp"], 8)
        self.assertAlmostEqual(trk["idf1"], 2 * 8 / (2 * 8 + 2 + 2))

    def test_mutual_swap_counts_two_switches(self):
        """Both objects trade ids at frame 3 - each GT object switches once."""
        pred = clip([
            {"id": 1, "label": "a", "acquired_frame": 0,
             "boxes": {**{str(f): box(f, 0) for f in (0, 1, 2)},
                       **{str(f): box(f, 1) for f in (3, 4)}}},
            {"id": 2, "label": "b", "acquired_frame": 0,
             "boxes": {**{str(f): box(f, 1) for f in (0, 1, 2)},
                       **{str(f): box(f, 0) for f in (3, 4)}}},
        ])
        trk = score.score(gt_two_tracks(), pred)["tracking"]
        self.assertEqual(trk["id_switches"], 2)
        self.assertEqual((trk["tp"], trk["fp"], trk["fn"]), (10, 0, 0))
        self.assertAlmostEqual(trk["mota"], 1 - 2 / 10)

    def test_stable_ids_after_occlusion(self):
        """Coming back with the original id after a gap is not a switch."""
        trk = score.score(gt_two_tracks(), clip([
            track(1, 0, [0, 1, 4]),
            track(2, 1, range(5)),
        ]))["tracking"]
        self.assertEqual(trk["id_switches"], 0)
        self.assertEqual(trk["fn"], 2)


class TestGrounding(unittest.TestCase):
    def test_missed_target_halves_recall(self):
        res = score.score(gt_two_tracks(), clip([track(1, 0, range(5))]))
        gr = res["grounding"]
        self.assertEqual((gr["grounded"], gr["total"]), (1, 2))
        self.assertEqual(gr["recall"], 0.5)
        self.assertFalse(gr["per_track"][1]["grounded"])

    def test_latency_and_window(self):
        """A target found 3 frames late counts inside the window, not outside."""
        gt = clip([track(1, 0, range(10))], labeled=list(range(10)))
        pred = clip([track(1, 0, range(3, 10))])
        inside = score.score(gt, pred, ground_window=5)["grounding"]
        self.assertTrue(inside["per_track"][0]["grounded"])
        self.assertEqual(inside["per_track"][0]["latency_frames"], 3)
        outside = score.score(gt, pred, ground_window=2)["grounding"]
        self.assertFalse(outside["per_track"][0]["grounded"])

    def test_acquisition_must_hit_the_target(self):
        """Acquiring the right frame with a badly placed box is not grounding."""
        gt = clip([track(1, 0, range(5))], labeled=list(range(5)))
        off = clip([track(1, 0, range(5), dx=9)])       # IoU 1/19 at every frame
        self.assertEqual(score.score(gt, off)["grounding"]["recall"], 0.0)

    def test_late_entry_is_reported_separately(self):
        gt = clip([track(1, 0, range(10)), track(2, 1, range(6, 10))],
                  labeled=list(range(10)))
        pred = clip([track(1, 0, range(10)), track(2, 1, range(6, 10))])
        gr = score.score(gt, pred, ground_window=3)["grounding"]
        self.assertEqual((gr["initial_total"], gr["initial_grounded"]), (1, 1))
        self.assertEqual((gr["late_total"], gr["late_grounded"]), (1, 1))

    def test_one_prediction_cannot_ground_two_targets(self):
        gt = clip([track(1, 0, range(5)), track(2, 0, range(5))],
                  labeled=list(range(5)))
        gr = score.score(gt, clip([track(1, 0, range(5))]))["grounding"]
        self.assertEqual(gr["grounded"], 1)


class TestIgnoreRegions(unittest.TestCase):
    def test_ignored_prediction_is_not_a_false_positive(self):
        gt = clip([track(1, 0, range(5)),
                   track(9, 2, range(5), ignore=True)],
                  labeled=list(range(5)))
        pred = clip([track(1, 0, range(5)), track(2, 2, range(5))])
        res = score.score(gt, pred)
        det = res["detection"]
        self.assertEqual((det["tp"], det["fp"], det["fn"]), (5, 0, 0))
        self.assertEqual(res["clip"]["pred_boxes_suppressed"], 5)
        self.assertAlmostEqual(res["tracking"]["mota"], 1.0)

    def test_ignore_track_is_not_a_scoring_target(self):
        gt = clip([track(1, 0, range(5)),
                   track(9, 2, range(5), ignore=True)],
                  labeled=list(range(5)))
        res = score.score(gt, clip([track(1, 0, range(5))]))
        self.assertEqual(res["clip"]["gt_tracks"], 1)
        self.assertEqual(res["grounding"]["total"], 1)
        self.assertEqual(res["detection"]["fn"], 0)

    def test_ignore_does_not_excuse_a_real_miss(self):
        """A prediction on a real target still counts, ignore region or not."""
        gt = clip([track(1, 0, range(5)),
                   track(9, 0, range(5), ignore=True)],
                  labeled=list(range(5)))
        det = score.score(gt, clip([track(1, 0, range(5))]))["detection"]
        self.assertEqual((det["tp"], det["fp"]), (5, 0))


class TestLabeledFrames(unittest.TestCase):
    def test_unlabeled_frames_are_not_scored(self):
        """Predictions on frames nobody labeled must not become false positives."""
        gt = clip([track(1, 0, [0, 2, 4])], labeled=[0, 2, 4])
        res = score.score(gt, clip([track(1, 0, range(5))]))
        det = res["detection"]
        self.assertEqual((det["tp"], det["fp"], det["fn"]), (3, 0, 0))
        self.assertEqual(res["clip"]["labeled_frames"], 3)

    def test_default_labeled_frames_are_the_annotated_ones(self):
        gt = clip([track(1, 0, [0, 2, 4])])
        self.assertEqual(gt.labeled_frames(), [0, 2, 4])

    def test_empty_frame_is_still_scored(self):
        """A labeled frame with no GT object turns predictions into FPs."""
        gt = clip([track(1, 0, [0])], labeled=[0, 1])
        det = score.score(gt, clip([track(1, 0, [0, 1])]))["detection"]
        self.assertEqual((det["tp"], det["fp"], det["fn"]), (1, 1, 0))


class TestEndToEnd(unittest.TestCase):
    """The mock scene through the real CLI: proves the harness runs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        run = subprocess.run(
            [sys.executable, str(ROOT / "run_demo.py"), "--mock",
             "--out", cls.tmp.name],
            capture_output=True, text=True, cwd=str(ROOT))
        if run.returncode != 0:
            raise unittest.SkipTest(f"mock demo failed: {run.stderr[-400:]}")
        cls.pred_path = os.path.join(cls.tmp.name, "tracks.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cli_json_output_on_the_mock_scene(self):
        run = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "score.py"),
             "--gt", str(ROOT / "eval" / "example_gt.json"),
             "--pred", self.pred_path, "--json"],
            capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(run.returncode, 0, run.stderr)
        res = json.loads(run.stdout)
        self.assertEqual(res["grounding"]["recall"], 1.0)
        self.assertGreater(res["detection"]["f1"], 0.9)
        self.assertGreater(res["tracking"]["mota"], 0.85)
        self.assertGreater(res["tracking"]["idf1"], 0.9)
        self.assertEqual(res["tracking"]["id_switches"], 0)

    def test_cli_reports_a_table(self):
        run = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "score.py"),
             "--gt", str(ROOT / "eval" / "example_gt.json"),
             "--pred", self.pred_path],
            capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(run.returncode, 0, run.stderr)
        for header in ("GROUNDING", "DETECTION", "TRACKING",
                       "PER GROUND-TRUTH TRACK"):
            self.assertIn(header, run.stdout)

    def test_missing_file_exits_nonzero(self):
        run = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "score.py"),
             "--gt", str(ROOT / "eval" / "does_not_exist.json"),
             "--pred", self.pred_path],
            capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(run.returncode, 2)


if __name__ == "__main__":
    unittest.main()
