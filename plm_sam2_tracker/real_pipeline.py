"""Real pipeline: PerceptionLM grounding + SAM 2 mask propagation.

Reference implementation for a GPU Linux machine. Heavy deps are imported
lazily so mock mode never touches them. Verify API details against the
current facebookresearch/perception_models and facebookresearch/sam2 repos —
both move fast.

Flow (offline analysis):
  1. Extract frames from the input video (SAM 2's video predictor wants a
     frame directory).
  2. PLM grounds the text query on frame 0 (the image path tiles up to
     36x448px, so small aerial targets survive; never ground on the video path).
  3. Each returned box seeds a SAM 2 masklet; propagate through the video.
  4. Re-ground every N seconds; boxes with low IoU vs live masklets become
     new tracks (objects entering the scene).
  5. Optional: PLM analyzes a crop clip around each finished track.
Output: the same tracks.json schema as mock mode, plus sampled JPEG frames
embedded for the HTML viewer.
"""

from __future__ import annotations

import base64
import os
import re

PLM_CHECKPOINT = os.environ.get("PLM_CHECKPOINT", "facebook/Perception-LM-8B")
SAM2_CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_large.pt")

_BOX_RE = re.compile(r"[\[\(<]?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)"
                     r"\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)>]?")


def _require(module: str, hint: str):
    try:
        return __import__(module)
    except ImportError as e:
        raise SystemExit(
            f"Real mode needs '{module}' ({hint}). "
            f"Install with: pip install -r requirements.txt  "
            f"and see README.md for the sam2 checkout."
        ) from e


def extract_frames(video_path: str, out_dir: str, max_fps: float = 10.0) -> tuple[int, float, int, int]:
    """Decode video to JPEG frames (SAM 2 input). Returns (n, fps, w, h)."""
    cv2 = _require("cv2", "opencv-python")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / max_fps))
    n, idx = 0, 0
    w = h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            cv2.imwrite(os.path.join(out_dir, f"{n:05d}.jpg"), frame)
            n += 1
        idx += 1
    cap.release()
    return n, src_fps / stride, w, h


def parse_boxes(text: str, img_w: int, img_h: int) -> list[list[float]]:
    """Parse "[x1, y1, x2, y2]" box strings out of PLM's text output.

    PLM variants emit either pixel or 0-1000-normalized coordinates;
    rescale when values exceed the image size.
    """
    boxes = []
    for m in _BOX_RE.finditer(text):
        x1, y1, x2, y2 = (float(g) for g in m.groups())
        if max(x1, x2) > img_w or max(y1, y2) > img_h:
            x1, x2 = x1 / 1000 * img_w, x2 / 1000 * img_w
            y1, y2 = y1 / 1000 * img_h, y2 / 1000 * img_h
        if x2 - x1 > 2 and y2 - y1 > 2:
            boxes.append([x1, y1, x2, y2])
    return boxes


class PLMGrounder:
    """Text query -> bounding boxes on one frame, via PerceptionLM."""

    def __init__(self, checkpoint: str = PLM_CHECKPOINT):
        _require("torch", "pytorch")
        transformers = _require("transformers", "huggingface transformers")
        self.processor = transformers.AutoProcessor.from_pretrained(checkpoint)
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            checkpoint, device_map="cuda", torch_dtype="bfloat16")

    def ground(self, image, query: str) -> list[list[float]]:
        h, w = image.shape[:2]
        conversation = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",
             "text": f"Provide the bounding box coordinates [x1, y1, x2, y2] "
                     f"of every instance of: {query}."},
        ]}]
        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = self.processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_boxes(text, w, h)


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / area if area > 0 else 0.0


def run_real_pipeline(video_path: str, query: str, work_dir: str,
                      reacquire_every_s: float = 5.0,
                      max_frames_embedded: int = 120) -> dict:
    """PLM grounding + SAM 2 propagation -> tracks.json payload."""
    cv2 = _require("cv2", "opencv-python")
    _require("torch", "pytorch")
    sam2_mod = _require("sam2", "facebookresearch/sam2, pip install -e .")
    from sam2.build_sam import build_sam2_video_predictor  # noqa: deferred

    frames_dir = os.path.join(work_dir, "frames")
    n_frames, fps, w, h = extract_frames(video_path, frames_dir)
    print(f"[frames] {n_frames} frames @ {fps:.1f} fps ({w}x{h})")

    grounder = PLMGrounder()
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
    state = predictor.init_state(video_path=frames_dir)

    def frame(idx: int):
        img = cv2.imread(os.path.join(frames_dir, f"{idx:05d}.jpg"))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initial acquisition
    next_id = 1
    labels: dict[int, str] = {}
    for box in grounder.ground(frame(0), query):
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=next_id, box=box)
        labels[next_id] = f"{query} #{next_id}"
        next_id += 1
    print(f"[ground] frame 0: {next_id - 1} target(s) for query '{query}'")
    if next_id == 1:
        raise SystemExit("PLM found no match for the query on frame 0 - "
                         "try a more literal description, or ground on a "
                         "different frame with --ground-frame.")

    # Propagate + periodic re-acquisition
    boxes_per_track: dict[int, dict[str, list[float]]] = {i: {} for i in labels}
    reacquire_every = max(1, int(reacquire_every_s * fps))
    for f_idx, obj_ids, masks in predictor.propagate_in_video(state):
        live = {}
        for oid, mask in zip(obj_ids, masks):
            m = (mask[0] > 0.0).cpu().numpy()
            ys, xs = m.nonzero()
            if len(xs) == 0:
                continue
            box = [float(xs.min()), float(ys.min()),
                   float(xs.max()), float(ys.max())]
            boxes_per_track.setdefault(oid, {})[str(f_idx)] = box
            live[oid] = box
        if f_idx > 0 and f_idx % reacquire_every == 0:
            for box in grounder.ground(frame(f_idx), query):
                if all(_iou(box, lb) < 0.3 for lb in live.values()):
                    predictor.add_new_points_or_box(
                        state, frame_idx=f_idx, obj_id=next_id, box=box)
                    labels[next_id] = f"{query} #{next_id}"
                    boxes_per_track[next_id] = {}
                    next_id += 1

    # Embed a subsample of frames so the HTML viewer is standalone
    stride = max(1, n_frames // max_frames_embedded)
    embedded = []
    for i in range(0, n_frames, stride):
        img = cv2.imread(os.path.join(frames_dir, f"{i:05d}.jpg"))
        scale = min(1.0, 960 / img.shape[1])
        img = cv2.resize(img, None, fx=scale, fy=scale)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        embedded.append("data:image/jpeg;base64," +
                        base64.b64encode(buf).decode())

    palette = ["#37B58C", "#D98E3B", "#5B8DD9", "#C25B5B", "#8E6BC2", "#B5B537"]
    return {
        "meta": {"mode": "real", "query": query, "fps": fps,
                 "num_frames": n_frames, "width": w, "height": h,
                 "frame_stride": stride, "video": os.path.basename(video_path)},
        "frames": embedded,
        "tracks": [
            {"id": oid, "label": labels.get(oid, f"track {oid}"),
             "color": palette[(oid - 1) % len(palette)],
             "acquired_frame": min((int(k) for k in boxes), default=0),
             "boxes": boxes, "analysis": ""}
            for oid, boxes in sorted(boxes_per_track.items()) if boxes
        ],
    }
