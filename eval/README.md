# Aerial evaluation set

Mock mode exercises the pipeline *logic*; it says nothing about whether
PerceptionLM can actually find a 30-pixel truck in real drone footage. This
directory is the harness for finding that out. The scorer, schema and a
worked example are here — **the footage and labels are not**, and that is the
only thing standing between us and a real number.

Two questions the set has to answer:

1. **Does zero-shot PLM grounding work on aerial frames at all?**
   → grounding recall @ IoU 0.5. If this is under ~0.6 the whole
   PLM-grounds-then-SAM2-propagates design needs rethinking (fine-tune, or a
   detector front-end) before any tracking work is worth doing.
2. **Given a good first box, how well does the clip track end to end?**
   → MOTA / IDF1 / ID switches.

## The set

**30 clips** (20 is the floor, past ~50 the labeling cost stops paying for
itself). **8-12 seconds each**, extracted at 10 fps → 80-120 frames per clip.
Label **every 10th frame**, so ~10 labeled frames and 20-60 boxes per clip.
That is roughly 30-45 minutes of labeling per clip and a couple of days of
work for the whole set.

Short clips beat long ones: more independent grounding attempts per hour of
labeling, and grounding is the thing we are actually measuring.

### Picking clips

Spread them over the axes that we expect to break things, and record the
axis values in `meta` so the scorer's output can be sliced:

| Axis | Values | Target |
|---|---|---|
| Altitude (`meta.altitude`) | `low` (target > 60 px), `mid` (20-60 px), `high` (< 20 px) | ~10 clips each |
| Clutter (`meta.clutter`) | `low` (open ground), `medium` (roads, buildings), `high` (urban, car parks, crowds) | ~10 clips each |
| Targets | 1 target / 2-5 targets / 6+ targets | ~10 / ~13 / ~7 |
| Motion | steady cruise, hard camera pan, target stops, target occluded | at least 5 each |

Plus, deliberately:

- **3-5 "entering" clips** — a target that appears mid-clip. This is the only
  thing that tests re-acquisition, and re-acquisition is where the pipeline
  is most likely to be silently broken.
- **2-3 negative clips** — the query names something that is *not* in the
  clip ("boat" over a desert road). Only the false-positive count is
  meaningful on these; they measure grounding hallucination, which is the
  failure mode that quietly ruins precision.
- **A few near-miss queries** — "white pickup truck" in a clip that also has
  a white van. Tests whether grounding discriminates or just finds *a*
  vehicle.

Avoid: clips where you cannot decide what the query means, targets you
personally cannot see at 100% zoom, and footage you cannot redistribute.

### What to label

One JSON file per clip, `eval/gt/<clip_id>.json` — full field-by-field spec
in [`schema.md`](schema.md), working example in
[`example_gt.json`](example_gt.json). The short version:

- Boxes are `[x1, y1, x2, y2]` in **pixels**, frames are **0-based indices
  into the extracted frames** — the same conventions as `output/tracks.json`,
  so ground truth and predictions are directly comparable.
- Only frames in `meta.labeled_frames` are scored, and on those frames the
  labeling must be **exhaustive**: every object matching `meta.query` has a
  box. Include frames where nothing is visible.
- One `id` per physical object, held across occlusions, never reused. ID
  switches are measured against exactly this, so it is the field worth being
  fussy about.
- Anything you cannot fairly ask the tracker about — out-of-query objects, a
  car park too dense to label, targets too small to place a box you would
  defend — gets its own track with `"ignore": true` rather than being
  deleted. Deleting turns a reasonable detection into a false positive.
- Tight boxes on the visible extent. Boxes may run past the frame edge; drop
  a target once its centre leaves the frame.

Suggested layout:

```
eval/
  README.md  schema.md  score.py  example_gt.json
  gt/        <clip_id>.json          <- ground truth, committed
  clips/     <clip_id>/frame_*.jpg   <- extracted frames, NOT committed
  runs/      <clip_id>.json          <- predictions, NOT committed
```

## Running the scorer

Pure stdlib, no install:

```bash
python3 eval/score.py --gt eval/gt/<clip_id>.json --pred eval/runs/<clip_id>.json
python3 eval/score.py --gt ... --pred ... --json          # machine-readable
python3 eval/score.py --gt ... --pred ... --iou 0.75       # stricter box quality
```

| Flag | Default | Meaning |
|---|---|---|
| `--iou` | 0.5 | IoU threshold for detection and tracking metrics |
| `--ground-iou` | 0.5 | IoU threshold for grounding recall |
| `--ground-window` | 30 | frames after a target appears in which acquisition still counts (set it to one re-acquisition period: `--reacquire` x fps) |
| `--greedy` | off | greedy descending-IoU matching instead of the exact Hungarian |
| `--json` | off | dump the full result dict |

### Smoke test

The mock scene has ground truth in `example_gt.json`, so the harness can be
checked end to end without any footage:

```bash
python3 run_demo.py --mock
python3 eval/score.py --gt eval/example_gt.json --pred output/tracks.json
```

```
clip mock-all-vehicles   query 'all vehicles'   matching hungarian
  gt    3 tracks / 38 boxes over 20 labeled frames (+1 ignore)
  pred  3 tracks / 38 boxes on those frames

GROUNDING @ IoU 0.50  (acquired within 30 frames)
  recall            1.000   3/3 targets acquired
    at clip start   2/2   entering later   1/1
  latency           median 0 fr (0.0s)   worst 20 fr (2.0s)

DETECTION @ IoU 0.50
  precision         0.947   recall 0.947   f1 0.947
  counts            tp 36   fp 2   fn 2   mean IoU 0.872

TRACKING @ IoU 0.50
  MOTA              0.895   (fp 2 + fn 2 + idsw 0 over 38 gt boxes)
  IDF1              0.947   idtp 36   idfp 2   idfn 2
  MOTP              0.872   (mean IoU of matches)
  ID switches       0

PER GROUND-TRUTH TRACK
   id  label                  boxes  entry grounded   latency  pred  idtp
    1  white pickup truck        19  start      yes  0 fr (0.0s)    #1    19
    2  dark sedan                 6  start      yes  0 fr (0.0s)    #2     6
    3  box truck                 13   late      yes 20 fr (2.0s)    #3    11
```

Those residuals are real and worth understanding, because they are the same
ones real clips will show: the 2 false positives are frames where the mock
tracker holds a box on a vehicle whose centre has left the frame (so it is
not labeled), and the 2 misses plus the 2.0 s latency are the box truck,
which enters at t=7.0 s but is not picked up until the next re-acquisition
pass at t=9.0 s.

The pedestrian is `"ignore": true` in the example, since it is out of scope
for the `all vehicles` query. To see that path do something, run the mock
against `--query "person walking"` and note that the 20 pedestrian
predictions produce **0** false positives rather than 20.

### Whole set

```bash
for gt in eval/gt/*.json; do
  id=$(basename "$gt" .json)
  python3 eval/score.py --gt "$gt" --pred "eval/runs/$id.json" --json > "eval/runs/$id.score.json"
done
python3 - <<'PY'
import glob, json
rows = [json.load(open(p)) for p in sorted(glob.glob("eval/runs/*.score.json"))]
g = [r["grounding"] for r in rows]
print("grounding recall (pooled): %.3f" % (sum(x["grounded"] for x in g) / sum(x["total"] for x in g)))
print("mean MOTA %.3f   mean IDF1 %.3f   total ID switches %d" % (
    sum(r["tracking"]["mota"] for r in rows) / len(rows),
    sum(r["tracking"]["idf1"] for r in rows) / len(rows),
    sum(r["tracking"]["id_switches"] for r in rows)))
PY
```

Pool grounding recall over targets (not clips) — a clip with six vehicles
says six times as much about grounding as a clip with one.

## What the metrics mean

- **Grounding recall @ IoU 0.5** — fraction of ground-truth targets that some
  predicted track acquired: acquisition no later than `--ground-window`
  frames after the target's first labeled frame, and IoU >= 0.5 at the first
  labeled frame at or after acquisition. Split into targets present at clip
  start vs. those entering later, which separates first-frame grounding from
  re-acquisition. **This is the number that decides whether zero-shot PLM is
  viable.**
- **Detection P/R/F1** — per-frame box quality pooled over all labeled
  frames, identity ignored. Recall dropping while grounding recall stays high
  means SAM 2 propagation is losing targets, not that PLM failed to find them.
- **MOTA** = `1 - (FP + FN + IDSW) / GT boxes`. Can go negative. Dominated by
  FP/FN, so read it next to the detection counts.
- **IDF1** — one global GT-id <-> pred-id assignment maximizing matched
  frames. Low IDF1 with high MOTA means fragmented identities.
- **MOTP** — mean IoU over matched pairs; box tightness, not accuracy.
- **ID switches** — a GT object matched to a different predicted id than it
  last carried. Counted across gaps, so a target reacquired under a new id
  after an occlusion costs one switch.

Matching is an exact Jonker-Volgenant Hungarian assignment (pure stdlib);
`--greedy` swaps in descending-IoU greedy matching if you want to compare.

## Adding a clip

1. Extract frames at the fps you will label at (<=10) — same rate the tracker
   will run at, or every frame index is off.
2. Label the keyframes into `eval/gt/<clip_id>.json` per `schema.md`.
3. Run the tracker with `meta.query` and save to `eval/runs/<clip_id>.json`.
4. Score it. Sanity-check the `gt N tracks / M boxes` header against what you
   think you labeled — most schema mistakes show up there as a wall of false
   positives or an all-zero row.

Tests for the metric math: `python3 -m unittest discover -s tests`.
