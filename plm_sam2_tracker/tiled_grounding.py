"""Tiled grounding for large aerial frames.

PerceptionLM looks at a frame a few hundred pixels a side, so a 4K frame
handed over whole loses anything car-sized before the model ever sees it.
The fix is to ground tile by tile: crop overlapping tile-sized windows,
ground each one, map the boxes back to full-frame coordinates, and
reconcile the duplicates that overlapping tiles inevitably produce.

This module owns only the bookkeeping:

    plan_tiles    frame size              -> tile rectangles covering it
    ground_tiled  frame size + a callable -> merged full-frame boxes
    merge_boxes   boxes                   -> greedy-NMS survivors

It never touches pixels - the caller crops and runs the model - so it stays
pure stdlib and unit-testable on a machine with no torch, no cv2, no GPU:

    from plm_sam2_tracker.tiled_grounding import ground_tiled
    h, w = image.shape[:2]
    boxes = ground_tiled(w, h, lambda t: grounder.ground(
        image[t.y0:t.y1, t.x0:t.x1], query))

Boxes are [x1, y1, x2, y2] with an optional 5th element, the score
(default 1.0); anything after that rides along untouched. Tile rectangles
are half-open, crop-style: img[y0:y1, x0:x1].
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

Box = list[float]

DEFAULT_TILE = 448        # PLM's native vision tile
DEFAULT_OVERLAP = 0.2     # fraction of a tile shared with each neighbour


@dataclass(frozen=True)
class Tile:
    """One crop rectangle. x1/y1 are exclusive: img[y0:y1, x0:x1]."""
    x0: int
    y0: int
    x1: int
    y1: int
    col: int = 0
    row: int = 0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def to_frame(self, box: Box) -> Box:
        """Tile-local box -> full-frame box, clipped to the tile."""
        x1 = min(max(float(box[0]), 0.0), float(self.width))
        y1 = min(max(float(box[1]), 0.0), float(self.height))
        x2 = min(max(float(box[2]), 0.0), float(self.width))
        y2 = min(max(float(box[3]), 0.0), float(self.height))
        return [x1 + self.x0, y1 + self.y0,
                x2 + self.x0, y2 + self.y0] + list(box[4:])


def plan_tiles(width: int, height: int, tile: int = DEFAULT_TILE,
               overlap: float = DEFAULT_OVERLAP) -> list[Tile]:
    """Overlapping tiles covering the whole frame, row-major.

    Tiles are tile x tile and step by (1 - overlap) * tile. The origins are
    spread evenly along each axis so the last row/column lands exactly on
    the frame edge: coverage is gap-free and no tile hangs off the frame.
    A frame smaller than one tile gives a single frame-sized tile.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame must be non-empty, got {width}x{height}")
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    step = max(1, int(round(tile * (1.0 - overlap))))
    tw, th = min(tile, width), min(tile, height)
    xs, ys = _starts(width, tw, step), _starts(height, th, step)
    return [Tile(x, y, x + tw, y + th, col, row)
            for row, y in enumerate(ys) for col, x in enumerate(xs)]


def _starts(extent: int, tile: int, step: int) -> list[int]:
    """Tile origins along one axis, spread so the last one ends on the edge."""
    if extent <= tile:
        return [0]
    n = 1 + -(-(extent - tile) // step)      # ceil division -> tiles needed
    return [round(i * (extent - tile) / (n - 1)) for i in range(n)]


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(boxes: Sequence[Box], iou_thresh: float = 0.5) -> list[Box]:
    """Greedy NMS: one survivor per cluster of overlapping boxes.

    Ranked by score (5th element, default 1.0) and then by area, so equal-
    confidence duplicates collapse onto the most complete box. Returned in
    that same descending order.
    """
    order = sorted(range(len(boxes)),
                   key=lambda i: (-_score(boxes[i]), -_area(boxes[i])))
    kept: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < iou_thresh for j in kept):
            kept.append(i)
    return [list(boxes[i]) for i in kept]


def ground_tiled(width: int, height: int,
                 ground_fn: Callable[[Tile], Sequence[Box] | None],
                 tile: int = DEFAULT_TILE, overlap: float = DEFAULT_OVERLAP,
                 iou_thresh: float = 0.5, *, seam_margin: float = 2.0,
                 contain_thresh: float = 0.6, stitch: bool = True) -> list[Box]:
    """Ground every tile of a frame and return merged full-frame boxes.

    `ground_fn(tile)` is called once per tile and must return boxes in
    TILE-LOCAL coordinates (the caller crops with tile.x0/y0/x1/y1). The
    results are translated to full-frame coordinates and reconciled in
    three passes:

      1. stitch  - fragments of one object cut by a shared seam (a target
                   wider than a tile) are unioned back together;
      2. seam    - a box hugging an interior tile edge is dropped when a
                   neighbouring tile saw the same object whole;
      3. merge   - NMS at `iou_thresh` collapses the plain duplicates that
                   overlapping tiles produce.

    seam_margin is how close (px) a box must sit to a tile edge to count as
    cut off; contain_thresh is how much of such a box a neighbour's box must
    swallow before it is dropped.
    """
    tiles = plan_tiles(width, height, tile, overlap)
    dets: list[_Det] = []
    for idx, t in enumerate(tiles):
        for raw in ground_fn(t) or ():
            box = t.to_frame(raw)
            if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
                continue                    # empty once clipped to the tile
            dets.append(_Det(box, idx, _seams(box, t, width, height,
                                              seam_margin)))
    if stitch:
        dets = _stitch(dets, seam_margin)
    dets = _drop_seam_duplicates(dets, contain_thresh)
    return merge_boxes([d.box for d in dets], iou_thresh)


@dataclass
class _Det:
    """A detection in full-frame coordinates, plus where it came from."""
    box: Box
    tile: int                              # index into the tile list
    seams: tuple[bool, bool, bool, bool]   # left/top/right/bottom cut off?


def _seams(box: Box, t: Tile, width: int, height: int,
           margin: float) -> tuple[bool, bool, bool, bool]:
    """Which sides of a box sit on an interior tile edge (so are cut off)."""
    return (t.x0 > 0 and box[0] - t.x0 <= margin,
            t.y0 > 0 and box[1] - t.y0 <= margin,
            t.x1 < width and t.x1 - box[2] <= margin,
            t.y1 < height and t.y1 - box[3] <= margin)


def _stitch(dets: list[_Det], margin: float) -> list[_Det]:
    """Union pairs that look like two halves of one seam-split object.

    Greedy and restarted after every merge, so an object spanning three
    tiles reassembles in three steps. Quadratic, but a frame yields tens of
    detections, not thousands.
    """
    dets = list(dets)
    merged = True
    while merged:
        merged = False
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i], dets[j]
                if _fragments(a, b, margin):
                    dets[i] = _union(a, b)
                elif _fragments(b, a, margin):
                    dets[i] = _union(b, a)
                else:
                    continue
                del dets[j]
                merged = True
                break
            if merged:
                break
    return dets


def _fragments(a: _Det, b: _Det, margin: float) -> bool:
    """True if b continues a past a tile seam (a on the near side)."""
    if a.tile == b.tile:
        return False
    for axis in (0, 1):                     # 0 = split by a vertical seam
        lo, hi = axis, axis + 2
        if not (a.seams[hi] and b.seams[lo]):
            continue                        # facing sides are not both cut
        if a.box[lo] > b.box[lo] or a.box[hi] > b.box[hi]:
            continue                        # b does not carry on past a
        if b.box[lo] - a.box[hi] > margin:
            continue                        # a real gap, so two objects
        if _span_overlap(a.box, b.box, 1 - axis) >= 0.5:
            return True
    return False


def _union(a: _Det, b: _Det) -> _Det:
    """Merge two fragments; a side stays "cut" only if it survives the union."""
    box = [min(a.box[0], b.box[0]), min(a.box[1], b.box[1]),
           max(a.box[2], b.box[2]), max(a.box[3], b.box[3])]
    src = a if _score(a.box) >= _score(b.box) else b
    seams = tuple((a.seams[k] and a.box[k] == box[k])
                  or (b.seams[k] and b.box[k] == box[k]) for k in range(4))
    return _Det(box + list(src.box[4:]), src.tile, seams)


def _drop_seam_duplicates(dets: list[_Det], contain_thresh: float) -> list[_Det]:
    """Drop cut-off boxes that a neighbouring tile saw whole."""
    return [d for i, d in enumerate(dets)
            if not (any(d.seams) and any(
                o.tile != d.tile and _area(o.box) > _area(d.box)
                and _covered(d.box, o.box) >= contain_thresh
                for j, o in enumerate(dets) if j != i))]


def _inter(a: Box, b: Box) -> float:
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _score(b: Box) -> float:
    return float(b[4]) if len(b) > 4 else 1.0


def _covered(inner: Box, outer: Box) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    area = _area(inner)
    return _inter(inner, outer) / area if area > 0 else 0.0


def _span_overlap(a: Box, b: Box, axis: int) -> float:
    """Overlap along one axis, as a fraction of the shorter of the two spans."""
    lo, hi = axis, axis + 2
    shared = min(a[hi], b[hi]) - max(a[lo], b[lo])
    shortest = min(a[hi] - a[lo], b[hi] - b[lo])
    return shared / shortest if shortest > 0 else 0.0
