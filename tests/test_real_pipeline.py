"""Tests for real_pipeline's dependency-free helpers.

The pipeline itself needs a GPU, torch, transformers and sam2, but the box
decoding and IoU maths are pure Python and are exactly where a silent bug
costs the most: a misdecoded box seeds SAM 2 on the wrong object and the
run reports confident nonsense rather than failing.
"""

from __future__ import annotations

import unittest

from plm_sam2_tracker.real_pipeline import _iou, parse_boxes


class TestParseBoxes(unittest.TestCase):
    """PLM emits zero-padded fixed point: [040,482,...] == 0.040, 0.482, ..."""

    def test_metas_documented_example(self):
        # image_grounding.ipynb: raw [040,482,112,576] on a 480x640 image
        (box,) = parse_boxes("The region is [040,482,112,576].", 480, 640)
        self.assertEqual([round(v, 1) for v in box],
                         [19.2, 308.5, 53.8, 368.6])

    def test_scales_with_frame_size(self):
        """The same digits must land proportionally, whatever the frame."""
        (small,) = parse_boxes("[250,250,750,750]", 100, 100)
        (large,) = parse_boxes("[250,250,750,750]", 4000, 4000)
        self.assertEqual(small, [25.0, 25.0, 75.0, 75.0])
        self.assertEqual(large, [1000.0, 1000.0, 3000.0, 3000.0])

    def test_leading_zeros_are_significant(self):
        """float() on "040" would give 40, not 0.040 - the original bug."""
        (box,) = parse_boxes("[040,040,960,960]", 1000, 1000)
        self.assertEqual(box, [40.0, 40.0, 960.0, 960.0])
        (tiny,) = parse_boxes("[004,004,996,996]", 1000, 1000)
        self.assertEqual(tiny, [4.0, 4.0, 996.0, 996.0])

    def test_boxes_stay_inside_the_frame(self):
        """Fixed point is 0-1, so no decode can escape the frame."""
        for digits in ("[999,999,999,999]", "[000,000,999,999]"):
            for box in parse_boxes(digits, 1920, 1080):
                self.assertLessEqual(box[2], 1920)
                self.assertLessEqual(box[3], 1080)

    def test_requires_brackets(self):
        """Bare numbers in prose are not boxes."""
        self.assertEqual(parse_boxes("about 40, 482, 112, 576 px", 640, 480), [])

    def test_rejects_degenerate_boxes(self):
        self.assertEqual(parse_boxes("[100,100,101,101]", 1000, 1000), [])
        self.assertEqual(parse_boxes("[500,500,500,500]", 1000, 1000), [])

    def test_multiple_boxes(self):
        boxes = parse_boxes("[100,100,300,300] and [600,600,900,900]", 1000, 1000)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0], [100.0, 100.0, 300.0, 300.0])
        self.assertEqual(boxes[1], [600.0, 600.0, 900.0, 900.0])

    def test_no_match_returns_empty(self):
        self.assertEqual(parse_boxes("I cannot find that object.", 640, 480), [])


class TestIou(unittest.TestCase):

    def test_identical(self):
        self.assertAlmostEqual(_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint(self):
        self.assertEqual(_iou([0, 0, 10, 10], [50, 50, 60, 60]), 0.0)

    def test_touching_edges_do_not_overlap(self):
        self.assertEqual(_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)

    def test_half_overlap(self):
        # union 300, intersection 100
        self.assertAlmostEqual(_iou([0, 0, 20, 10], [10, 0, 30, 10]), 100 / 300)

    def test_degenerate_box_is_safe(self):
        self.assertEqual(_iou([5, 5, 5, 5], [0, 0, 10, 10]), 0.0)


if __name__ == "__main__":
    unittest.main()
