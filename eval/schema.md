# Ground-truth clip schema

One JSON file per clip. It deliberately mirrors the demo's `tracks.json`
(see `plm_sam2_tracker/mock_scene.py`) so ground truth and predictions are
directly comparable: **same box convention, same frame indexing, same
`tracks` list shape**. `eval/example_gt.json` is a working instance.

```json
{
  "meta": { ... },
  "tracks": [ { ... }, { ... } ]
}
```

## Conventions (shared with `tracks.json`)

| | |
|---|---|
| Box | `[x1, y1, x2, y2]`, **pixels**, top-left origin, x right / y down |
| | `x1 < x2`, `y1 < y2`; ints or floats, both fine |
| Frame index | **0-based**, counting extracted frames, not source-video frames |
| Frame key | a **string** in the `boxes` object (`"0"`, `"7"`), JSON has no int keys |
| Missing box | just omit the key — a track has boxes only on frames where it is there |

Frame indices refer to the frames the pipeline actually sees. Real mode
extracts at <=10 fps with a stride, so **frame 7 means "the 8th extracted
frame", not "the 8th frame of the mp4"**. Record the fps you labeled at in
`meta.fps` and extract at that same rate when you run the tracker
(`run_demo.py` clamps to 10 fps by default).

Boxes may extend past the image edge for a partly-visible object — the
tracker emits unclipped boxes and the scorer does not clip either. Label the
extent you can actually see; a target whose centre has left the frame is
normally not labeled at all (that is the rule `example_gt.json` uses).

## `meta`

| Field | Type | Req. | Meaning |
|---|---|---|---|
| `clip_id` | string | yes | short unique id, e.g. `dji-market-0031`. Shown in the report |
| `source` | string | yes | where the footage came from + timecode, e.g. `flightA.mp4 @ 04:12-04:22` |
| `query` | string | yes | the text prompt the tracker must be run with |
| `fps` | number | yes | frames per second **of the extracted frames** the indices count |
| `num_frames` | int | yes | total extracted frames in the clip |
| `width`, `height` | int | yes | frame size in pixels; boxes are in this space |
| `labeled_frames` | int[] | yes | frames that are **exhaustively** labeled — see below |
| `altitude` | string | no | `low` / `mid` / `high` — for slicing results |
| `clutter` | string | no | `low` / `medium` / `high` |
| `notes` | string | no | anything a reader of the scores needs to know |

### `labeled_frames` — the important one

Scoring happens **only** on these frames, and on them the labels must be
complete: every object matching `query` has a box. This is what lets you
label every 5th or 10th frame instead of all 200 — predictions on unlabeled
frames are simply not scored, rather than counted as false positives.

If you omit the field, the scorer falls back to the sorted union of all
frames that appear in any track's `boxes`. That is only right when you
labeled densely; a frame where the correct answer is "nothing here" can never
be expressed that way, so prefer listing the frames explicitly.

Include frames where nothing is visible — they are where false positives
show up.

## `tracks[]`

One entry per object identity, for the whole clip.

| Field | Type | Req. | Meaning |
|---|---|---|---|
| `id` | int | yes | identity, unique within the clip, stable across frames |
| `label` | string | yes | what it is, in plain words: `white pickup truck` |
| `category` | string | no | coarse class for slicing: `vehicle`, `person`, ... |
| `boxes` | object | yes | `{"<frame>": [x1,y1,x2,y2]}` |
| `ignore` | bool | no | default `false`; see below |
| `attributes` | object | no | free-form, e.g. `{"occluded": [12, 13], "size_px": 34}` |
| `notes` | string | no | free-form |

`id` is the identity the tracker is scored against — reuse the same `id`
after an occlusion if it is the same physical object, and never reuse an id
for a different object. ID switches are measured against exactly this.

Prediction files additionally carry `color`, `acquired_frame` and `analysis`.
Ground truth does not need them; `acquired_frame` is read **from the
prediction** and is what the grounding metric times.

### `ignore: true`

An ignore track is scored as "don't care": it is not a target (it never
counts as a miss), and a predicted box that lands on it and matches no real
target is dropped instead of counted as a false positive.

Use it for anything you cannot fairly ask the tracker about:

- objects outside the clip's query (a pedestrian in an `all vehicles` clip)
- a car park or convoy too dense to label box by box — one big ignore box
- targets so small or blurred you cannot place a box you would defend

It is better to mark something ignore than to delete it: deleting turns a
reasonable detection into a false positive.

## Minimal valid file

```json
{
  "meta": {
    "clip_id": "demo-0001",
    "source": "flightA.mp4 @ 00:31-00:36",
    "query": "all vehicles",
    "fps": 10.0, "num_frames": 50,
    "width": 1920, "height": 1080,
    "labeled_frames": [0, 10, 20, 30, 40]
  },
  "tracks": [
    {
      "id": 1, "label": "white pickup truck", "category": "vehicle",
      "boxes": {
        "0":  [812, 440, 861, 466],
        "10": [864, 441, 913, 467],
        "20": [917, 442, 966, 468]
      }
    },
    {
      "id": 2, "label": "parked cars in the lot", "ignore": true,
      "boxes": {"0": [100, 600, 520, 780], "10": [100, 600, 520, 780]}
    }
  ]
}
```

## Validating

The scorer is the validator — run it against any prediction and read the
counts back:

```bash
python3 eval/score.py --gt eval/gt/demo-0001.json --pred out/demo-0001.json
```

`gt N tracks / M boxes over K labeled frames` at the top must match what you
think you labeled. Common mistakes it exposes: frame indices at source-video
rate (nothing matches anything), y/x swapped (nothing matches), boxes in
`[x, y, w, h]` (IoU is nonsense), `labeled_frames` listing frames you never
actually checked (phantom false positives).
