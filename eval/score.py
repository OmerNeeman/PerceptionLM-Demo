#!/usr/bin/env python3
"""Score a tracker prediction against a labeled aerial clip.

    python3 eval/score.py --gt eval/example_gt.json --pred output/tracks.json
    python3 eval/score.py --gt gt/clip_007.json --pred out/clip_007.json --json

Ground truth and prediction share the demo's `tracks.json` conventions:
boxes are `[x1, y1, x2, y2]` in pixels, frames are 0-based integer keys.
Only frames listed in the ground truth's `meta.labeled_frames` are scored -
see eval/schema.md.

Metrics: grounding recall (did acquisition ever find the target), per-frame
detection precision/recall/F1, and CLEAR MOT / IDF1 tracking scores. Pure
stdlib; the assignment problem is solved exactly with a Jonker-Volgenant
Hungarian (--greedy swaps in descending-IoU greedy matching instead).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

Box = list[float]
Match = list[tuple[int, int]]

DEFAULT_IOU = 0.5
DEFAULT_GROUND_WINDOW = 30


# --------------------------------------------------------------------------
# geometry + assignment
# --------------------------------------------------------------------------

def iou(a: Box, b: Box) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def hungarian(cost: list[list[float]]) -> Match:
    """Min-cost assignment (Jonker-Volgenant, O(n^2 m)) -> (row, col) pairs."""
    if not cost or not cost[0]:
        return []
    n, m = len(cost), len(cost[0])
    transposed = n > m
    if transposed:
        cost = [[cost[i][j] for i in range(n)] for j in range(m)]
        n, m = m, n
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    parent = [0] * (m + 1)   # parent[j] = 1-based row assigned to column j
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        parent[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = parent[j0], inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[parent[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if parent[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            parent[j0] = parent[j1]
            j0 = j1
    pairs = [(parent[j] - 1, j - 1) for j in range(1, m + 1) if parent[j]]
    return [(j, i) for i, j in pairs] if transposed else pairs


def match_boxes(rows: list[Box], cols: list[Box], thr: float,
                greedy: bool = False) -> Match:
    """One-to-one (row, col) pairs with IoU >= thr, maximizing total IoU."""
    if not rows or not cols:
        return []
    ious = [[iou(r, c) for c in cols] for r in rows]
    if greedy:
        cand = sorted(((ious[i][j], i, j)
                       for i in range(len(rows)) for j in range(len(cols))
                       if ious[i][j] >= thr), key=lambda t: -t[0])
        taken_r: set[int] = set()
        taken_c: set[int] = set()
        out = []
        for _, i, j in cand:
            if i not in taken_r and j not in taken_c:
                taken_r.add(i)
                taken_c.add(j)
                out.append((i, j))
        return sorted(out)
    # Below-threshold pairs cost 0 so they are never preferred over a real
    # match; they are filtered out after the assignment.
    cost = [[-ious[i][j] if ious[i][j] >= thr else 0.0 for j in range(len(cols))]
            for i in range(len(rows))]
    return sorted((i, j) for i, j in hungarian(cost) if ious[i][j] >= thr)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

class Clip:
    """A ground-truth or prediction file, normalized for scoring."""

    def __init__(self, path: str, payload: dict) -> None:
        self.path = path
        self.meta: dict = payload.get("meta", {}) or {}
        self.tracks: list[dict] = []
        for raw in payload.get("tracks", []):
            boxes = {int(f): [float(v) for v in box]
                     for f, box in (raw.get("boxes") or {}).items()}
            self.tracks.append({
                "id": raw.get("id"),
                "label": raw.get("label", ""),
                "ignore": bool(raw.get("ignore", False)),
                "acquired_frame": raw.get("acquired_frame",
                                          min(boxes, default=0)),
                "boxes": boxes,
            })

    @property
    def scored(self) -> list[dict]:
        return [t for t in self.tracks if not t["ignore"]]

    @property
    def ignored(self) -> list[dict]:
        return [t for t in self.tracks if t["ignore"]]

    def labeled_frames(self) -> list[int]:
        """Frames the annotator declared exhaustively labeled."""
        frames = self.meta.get("labeled_frames")
        if frames:
            return sorted({int(f) for f in frames})
        return sorted({f for t in self.tracks for f in t["boxes"]})

    def fps(self) -> float:
        try:
            return float(self.meta.get("fps") or 0.0) or 0.0
        except (TypeError, ValueError):
            return 0.0


def load_clip(path: str) -> Clip:
    """Read a tracks.json-style file."""
    with open(path, encoding="utf-8") as f:
        return Clip(path, json.load(f))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1}


def suppress_ignored(gt: Clip, pred: Clip, frames: list[int], thr: float,
                     greedy: bool) -> dict[int, set]:
    """Predicted ids per frame that fall on an ignore region and are dropped.

    A prediction is only suppressed if it matches no scored GT box - a box
    that tracks a real target is never excused by a nearby ignore region.
    """
    suppressed: dict[int, set] = {}
    ignore_tracks = gt.ignored
    if not ignore_tracks:
        return {f: set() for f in frames}
    for f in frames:
        g_ids, g_boxes = _boxes_at(gt.scored, f)
        p_ids, p_boxes = _boxes_at(pred.tracks, f)
        matched = {j for _, j in match_boxes(g_boxes, p_boxes, thr, greedy)}
        _, ig_boxes = _boxes_at(ignore_tracks, f)
        drop = set()
        for j, pid in enumerate(p_ids):
            if j in matched:
                continue
            if any(iou(p_boxes[j], ig) >= thr for ig in ig_boxes):
                drop.add(pid)
        suppressed[f] = drop
    return suppressed


def _boxes_at(tracks: list[dict], frame: int) -> tuple[list, list[Box]]:
    ids, boxes = [], []
    for t in tracks:
        box = t["boxes"].get(frame)
        if box is not None:
            ids.append(t["id"])
            boxes.append(box)
    return ids, boxes


def _visible(tracks: list[dict], frame: int, skip: set) -> tuple[list, list[Box]]:
    ids, boxes = _boxes_at(tracks, frame)
    keep = [i for i, tid in enumerate(ids) if tid not in skip]
    return [ids[i] for i in keep], [boxes[i] for i in keep]


def grounding_recall(gt: Clip, pred: Clip, frames: list[int], thr: float,
                     window: int) -> dict:
    """Did acquisition ever find each target, and how late?

    A predicted track acquires a GT track when it is acquired no later than
    `window` frames after the target's first labeled frame, and its box
    overlaps the target by >= `thr` on the first labeled frame at or after
    acquisition where both have a box. Each prediction is credited once.
    """
    frame_set = set(frames)
    per_track = []
    claimed: set = set()
    for g in sorted(gt.scored, key=lambda t: min(t["boxes"], default=0)):
        g_frames = sorted(f for f in g["boxes"] if f in frame_set)
        entry = {"gt_id": g["id"], "label": g["label"], "boxes": len(g_frames),
                 "first_frame": g_frames[0] if g_frames else None,
                 "late_entry": bool(g_frames) and g_frames[0] > window,
                 "grounded": False, "pred_id": None, "latency_frames": None,
                 "iou": 0.0}
        if not g_frames:
            per_track.append(entry)
            continue
        best = None
        for p in pred.tracks:
            if p["id"] in claimed or p["acquired_frame"] > g_frames[0] + window:
                continue
            check = next((f for f in g_frames
                          if f >= p["acquired_frame"] and f in p["boxes"]), None)
            if check is None:
                continue
            overlap = iou(g["boxes"][check], p["boxes"][check])
            if overlap >= thr and (best is None or overlap > best[0]):
                best = (overlap, p["id"], p["acquired_frame"])
        if best:
            claimed.add(best[1])
            entry.update(grounded=True, pred_id=best[1], iou=best[0],
                         latency_frames=max(0, best[2] - g_frames[0]))
        per_track.append(entry)
    hits = [e for e in per_track if e["grounded"]]
    early = [e for e in per_track if not e["late_entry"]]
    late = [e for e in per_track if e["late_entry"]]
    lat = [e["latency_frames"] for e in hits]
    return {
        "total": len(per_track), "grounded": len(hits),
        "recall": len(hits) / len(per_track) if per_track else 0.0,
        "initial_total": len(early),
        "initial_grounded": sum(1 for e in early if e["grounded"]),
        "late_total": len(late),
        "late_grounded": sum(1 for e in late if e["grounded"]),
        "median_latency_frames": statistics.median(lat) if lat else None,
        "max_latency_frames": max(lat) if lat else None,
        "per_track": per_track,
    }


def detection(gt: Clip, pred: Clip, frames: list[int], thr: float,
              suppressed: dict[int, set], greedy: bool) -> dict:
    """Per-frame detection counts, pooled over every labeled frame."""
    tp = fp = fn = 0
    overlaps = []
    for f in frames:
        _, g_boxes = _boxes_at(gt.scored, f)
        _, p_boxes = _visible(pred.tracks, f, suppressed.get(f, set()))
        pairs = match_boxes(g_boxes, p_boxes, thr, greedy)
        tp += len(pairs)
        fp += len(p_boxes) - len(pairs)
        fn += len(g_boxes) - len(pairs)
        overlaps += [iou(g_boxes[i], p_boxes[j]) for i, j in pairs]
    out = _prf(tp, fp, fn)
    out["mean_iou"] = statistics.fmean(overlaps) if overlaps else 0.0
    return out


def clear_mot(gt: Clip, pred: Clip, frames: list[int], thr: float,
              suppressed: dict[int, set], greedy: bool) -> dict:
    """CLEAR MOT: MOTA, MOTP and ID switches.

    Associations from the previous frame are kept whenever they still clear
    the IoU threshold, so a stable tracker scores no switches; the remaining
    boxes are assigned fresh. A GT object re-matched to a different predicted
    id than it last carried counts one ID switch.
    """
    last: dict = {}          # gt id -> most recent pred id
    tp = fp = fn = idsw = 0
    overlaps = []
    for f in frames:
        g_ids, g_boxes = _boxes_at(gt.scored, f)
        p_ids, p_boxes = _visible(pred.tracks, f, suppressed.get(f, set()))
        gi = {tid: i for i, tid in enumerate(g_ids)}
        pi = {tid: i for i, tid in enumerate(p_ids)}
        matched: dict = {}
        for g_id, p_id in last.items():
            if g_id in gi and p_id in pi and p_id not in matched.values():
                if iou(g_boxes[gi[g_id]], p_boxes[pi[p_id]]) >= thr:
                    matched[g_id] = p_id
        rest_g = [t for t in g_ids if t not in matched]
        rest_p = [t for t in p_ids if t not in matched.values()]
        pairs = match_boxes([g_boxes[gi[t]] for t in rest_g],
                            [p_boxes[pi[t]] for t in rest_p], thr, greedy)
        for i, j in pairs:
            g_id, p_id = rest_g[i], rest_p[j]
            if g_id in last and last[g_id] != p_id:
                idsw += 1
            matched[g_id] = p_id
        for g_id, p_id in matched.items():
            overlaps.append(iou(g_boxes[gi[g_id]], p_boxes[pi[p_id]]))
        last.update(matched)
        tp += len(matched)
        fn += len(g_ids) - len(matched)
        fp += len(p_ids) - len(matched)
    n_gt = tp + fn
    return {"mota": 1.0 - (fn + fp + idsw) / n_gt if n_gt else 0.0,
            "motp": statistics.fmean(overlaps) if overlaps else 0.0,
            "id_switches": idsw, "tp": tp, "fp": fp, "fn": fn, "num_gt": n_gt}


def idf1(gt: Clip, pred: Clip, frames: list[int], thr: float,
         suppressed: dict[int, set], greedy: bool) -> dict:
    """IDF1: one global GT-id <-> pred-id assignment maximizing matched frames."""
    g_tracks, p_tracks = gt.scored, pred.tracks
    n_gt = n_pred = 0
    counts = [[0] * len(p_tracks) for _ in g_tracks]
    for f in frames:
        skip = suppressed.get(f, set())
        n_gt += sum(1 for t in g_tracks if f in t["boxes"])
        n_pred += sum(1 for t in p_tracks
                      if f in t["boxes"] and t["id"] not in skip)
        for i, g in enumerate(g_tracks):
            gb = g["boxes"].get(f)
            if gb is None:
                continue
            for j, p in enumerate(p_tracks):
                pb = p["boxes"].get(f)
                if pb is not None and p["id"] not in skip and iou(gb, pb) >= thr:
                    counts[i][j] += 1
    if greedy:
        cand = sorted(((counts[i][j], i, j)
                       for i in range(len(g_tracks))
                       for j in range(len(p_tracks)) if counts[i][j] > 0),
                      key=lambda t: -t[0])
        rows: set[int] = set()
        cols: set[int] = set()
        pairs = []
        for _, i, j in cand:
            if i not in rows and j not in cols:
                rows.add(i)
                cols.add(j)
                pairs.append((i, j))
    else:
        cost = [[float(-c) for c in row] for row in counts]
        pairs = [(i, j) for i, j in hungarian(cost) if counts[i][j] > 0]
    idtp = sum(counts[i][j] for i, j in pairs)
    idfn, idfp = n_gt - idtp, n_pred - idtp
    denom = 2 * idtp + idfp + idfn
    return {"idf1": 2 * idtp / denom if denom else 0.0,
            "idtp": idtp, "idfp": idfp, "idfn": idfn,
            "idp": idtp / n_pred if n_pred else 0.0,
            "idr": idtp / n_gt if n_gt else 0.0,
            "pairs": {g_tracks[i]["id"]: (p_tracks[j]["id"], counts[i][j])
                      for i, j in pairs}}


def score(gt: Clip, pred: Clip, iou_thr: float = DEFAULT_IOU,
          ground_iou: float = DEFAULT_IOU,
          ground_window: int = DEFAULT_GROUND_WINDOW,
          greedy: bool = False) -> dict:
    """Run every metric over the ground truth's labeled frames."""
    frames = gt.labeled_frames()
    frame_set = set(frames)
    suppressed = suppress_ignored(gt, pred, frames, iou_thr, greedy)
    n_pred_boxes = sum(1 for t in pred.tracks for f in t["boxes"] if f in frame_set)
    n_suppressed = sum(len(s) for s in suppressed.values())
    fps = gt.fps() or pred.fps()
    return {
        "gt_file": gt.path, "pred_file": pred.path,
        "settings": {"iou": iou_thr, "grounding_iou": ground_iou,
                     "grounding_window_frames": ground_window,
                     "matching": "greedy" if greedy else "hungarian"},
        "clip": {
            "clip_id": gt.meta.get("clip_id", ""),
            "query": pred.meta.get("query") or gt.meta.get("query", ""),
            "fps": fps, "num_frames": gt.meta.get("num_frames"),
            "labeled_frames": len(frames),
            "gt_tracks": len(gt.scored), "ignore_tracks": len(gt.ignored),
            "gt_boxes": sum(1 for t in gt.scored for f in t["boxes"]
                            if f in frame_set),
            "pred_tracks": len(pred.tracks),
            "pred_boxes": n_pred_boxes - n_suppressed,
            "pred_boxes_suppressed": n_suppressed,
        },
        "grounding": grounding_recall(gt, pred, frames, ground_iou, ground_window),
        "detection": detection(gt, pred, frames, iou_thr, suppressed, greedy),
        "tracking": {**clear_mot(gt, pred, frames, iou_thr, suppressed, greedy),
                     **idf1(gt, pred, frames, iou_thr, suppressed, greedy)},
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _frames_and_seconds(frames: float | None, fps: float) -> str:
    if frames is None:
        return "-"
    return f"{frames:g} fr" + (f" ({frames / fps:.1f}s)" if fps else "")


def format_report(res: dict) -> str:
    """Human-readable summary table."""
    clip, gr, det = res["clip"], res["grounding"], res["detection"]
    trk, cfg = res["tracking"], res["settings"]
    fps = clip["fps"] or 0.0
    thr, gthr = cfg["iou"], cfg["grounding_iou"]
    out = []
    add = out.append
    name = clip["clip_id"] or res["gt_file"]
    add(f"clip {name}   query {clip['query']!r}   matching {cfg['matching']}")
    add(f"  gt    {clip['gt_tracks']} tracks / {clip['gt_boxes']} boxes over "
        f"{clip['labeled_frames']} labeled frames"
        + (f" (+{clip['ignore_tracks']} ignore)" if clip["ignore_tracks"] else ""))
    add(f"  pred  {clip['pred_tracks']} tracks / {clip['pred_boxes']} boxes on "
        f"those frames"
        + (f" ({clip['pred_boxes_suppressed']} dropped on ignore regions)"
           if clip["pred_boxes_suppressed"] else ""))
    if not clip["gt_boxes"]:
        add("  note  negative clip (no targets labeled) - only the false "
            "positive count below is meaningful")
    add("")
    add(f"GROUNDING @ IoU {gthr:.2f}  "
        f"(acquired within {cfg['grounding_window_frames']} frames)")
    add(f"  recall            {gr['recall']:.3f}   "
        f"{gr['grounded']}/{gr['total']} targets acquired")
    add(f"    at clip start   {_ratio(gr['initial_grounded'], gr['initial_total'])}"
        f"   entering later   {_ratio(gr['late_grounded'], gr['late_total'])}")
    add(f"  latency           median {_frames_and_seconds(gr['median_latency_frames'], fps)}"
        f"   worst {_frames_and_seconds(gr['max_latency_frames'], fps)}")
    add("")
    add(f"DETECTION @ IoU {thr:.2f}")
    add(f"  precision         {det['precision']:.3f}   "
        f"recall {det['recall']:.3f}   f1 {det['f1']:.3f}")
    add(f"  counts            tp {det['tp']}   fp {det['fp']}   fn {det['fn']}"
        f"   mean IoU {det['mean_iou']:.3f}")
    add("")
    add(f"TRACKING @ IoU {thr:.2f}")
    add(f"  MOTA              {trk['mota']:.3f}   "
        f"(fp {trk['fp']} + fn {trk['fn']} + idsw {trk['id_switches']} "
        f"over {trk['num_gt']} gt boxes)")
    add(f"  IDF1              {trk['idf1']:.3f}   "
        f"idtp {trk['idtp']}   idfp {trk['idfp']}   idfn {trk['idfn']}")
    add(f"  MOTP              {trk['motp']:.3f}   (mean IoU of matches)")
    add(f"  ID switches       {trk['id_switches']}")
    add("")
    add("PER GROUND-TRUTH TRACK")
    add(f"  {'id':>3}  {'label':<22} {'boxes':>5} {'entry':>6} "
        f"{'grounded':>8} {'latency':>9} {'pred':>5} {'idtp':>5}")
    pairs = trk["pairs"]
    for e in gr["per_track"]:
        pid, idtp = pairs.get(e["gt_id"], (None, 0))
        add(f"  {e['gt_id']:>3}  {e['label'][:22]:<22} {e['boxes']:>5} "
            f"{('late' if e['late_entry'] else 'start'):>6} "
            f"{('yes' if e['grounded'] else 'NO'):>8} "
            f"{_frames_and_seconds(e['latency_frames'], fps):>12} "
            f"{('#' + str(pid) if pid is not None else '-'):>5} {idtp:>5}")
    return "\n".join(out)


def _ratio(hit: int, total: int) -> str:
    return f"{hit}/{total}" if total else "n/a"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="ground-truth clip JSON")
    p.add_argument("--pred", required=True, help="prediction (tracks.json)")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU,
                   help=f"IoU threshold for detection/tracking (default {DEFAULT_IOU})")
    p.add_argument("--ground-iou", type=float, default=DEFAULT_IOU,
                   help=f"IoU threshold for grounding recall (default {DEFAULT_IOU})")
    p.add_argument("--ground-window", type=int, default=DEFAULT_GROUND_WINDOW,
                   help="frames after a target appears in which acquisition still "
                        f"counts (default {DEFAULT_GROUND_WINDOW})")
    p.add_argument("--greedy", action="store_true",
                   help="greedy descending-IoU matching instead of Hungarian")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    try:
        gt, pred = load_clip(args.gt), load_clip(args.pred)
    except (OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    if not gt.labeled_frames():
        print(f"[error] {args.gt} has no labeled frames", file=sys.stderr)
        return 2

    res = score(gt, pred, args.iou, args.ground_iou, args.ground_window, args.greedy)
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        print(format_report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
