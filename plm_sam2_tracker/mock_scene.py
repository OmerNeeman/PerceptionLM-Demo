"""Synthetic aerial scene for mock mode.

Simulates a small overhead world (roads, a compound, moving vehicles and a
pedestrian) and mock versions of the three pipeline stages:

- grounding  (stands in for PerceptionLM text -> box)
- tracking   (stands in for SAM 2 box -> per-frame masklet/box)
- analysis   (stands in for PLM RDCap-style per-track description)

Everything here is pure stdlib so the demo runs on any Linux box with
python3 and no installs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

WORLD_W = 960
WORLD_H = 600


@dataclass
class SceneObject:
    oid: int
    label: str
    keywords: set[str]
    category: str            # "vehicle" | "person"
    color: str               # fill color used by the viewer
    size: tuple[float, float]  # (length, width) in px along heading
    waypoints: list[tuple[float, float]]
    speed: float             # px/s
    start_delay: float = 0.0   # seconds before it starts moving
    stops: list[tuple[float, float]] = field(default_factory=list)

    def simulate(self, num_frames: int, fps: float) -> list[dict]:
        """Per-frame ground-truth state: x, y (center), heading (rad), visible."""
        dt = 1.0 / fps
        x, y = self.waypoints[0]
        seg = 0
        heading = 0.0
        states = []
        t = 0.0
        for _ in range(num_frames):
            moving = t >= self.start_delay and seg < len(self.waypoints) - 1
            if moving and any(a <= t < b for a, b in self.stops):
                moving = False
            if moving:
                tx, ty = self.waypoints[seg + 1]
                dx, dy = tx - x, ty - y
                dist = math.hypot(dx, dy)
                step = self.speed * dt
                if dist <= step:
                    x, y = tx, ty
                    seg += 1
                else:
                    x += dx / dist * step
                    y += dy / dist * step
                heading = math.atan2(dy, dx)
            visible = -30 < x < WORLD_W + 30 and -30 < y < WORLD_H + 30
            states.append({"x": x, "y": y, "heading": heading, "visible": visible})
            t += dt
        return states

    def bbox(self, state: dict) -> list[float]:
        """Axis-aligned box around the (oriented) footprint at a state."""
        length, width = self.size
        c, s = abs(math.cos(state["heading"])), abs(math.sin(state["heading"]))
        half_w = (length * c + width * s) / 2
        half_h = (length * s + width * c) / 2
        return [state["x"] - half_w, state["y"] - half_h,
                state["x"] + half_w, state["y"] + half_h]


def build_scene() -> dict:
    """The static world plus its actors."""
    objects = [
        SceneObject(
            oid=1, label="white pickup truck",
            keywords={"white", "pickup", "truck"}, category="vehicle",
            color="#E8E4D8", size=(34, 16), speed=52,
            waypoints=[(-30, 342), (400, 330), (620, 322), (1000, 310)],
            stops=[(8.0, 12.5)],
        ),
        SceneObject(
            oid=2, label="dark sedan",
            keywords={"dark", "sedan", "car", "black"}, category="vehicle",
            color="#23272C", size=(28, 14), speed=95,
            waypoints=[(614, -30), (606, 300), (600, 640)],
        ),
        SceneObject(
            oid=3, label="box truck",
            keywords={"box", "truck", "delivery"}, category="vehicle",
            color="#8A5A33", size=(42, 18), speed=68, start_delay=6.0,
            waypoints=[(-50, 118), (500, 128), (1010, 140)],
        ),
        SceneObject(
            oid=4, label="person walking",
            keywords={"person", "pedestrian", "walking"}, category="person",
            color="#B0483C", size=(7, 7), speed=13,
            waypoints=[(468, 392), (515, 305), (462, 252), (430, 200)],
        ),
    ]
    static = {
        "dirt_road": [(0, 345), (400, 332), (620, 322), (960, 310)],
        "paved_road": [(618, 0), (606, 300), (598, 600)],
        "second_road": [(0, 122), (500, 130), (960, 142)],
        "buildings": [
            [430, 352, 496, 402], [508, 360, 560, 412], [452, 412, 500, 448],
            [740, 180, 812, 236], [760, 250, 806, 288],
        ],
        "trees": [(120, 80), (180, 210), (90, 430), (260, 470), (330, 60),
                  (700, 60), (880, 220), (840, 420), (720, 500), (170, 540),
                  (300, 250), (520, 520), (900, 60)],
        "field": [40, 380, 360, 560],
    }
    return {"objects": objects, "static": static}


def match_query(query: str, objects: list[SceneObject]) -> list[SceneObject]:
    """Keyword stand-in for PLM open-vocabulary grounding."""
    q = {w.strip(".,").lower() for w in query.split()}
    if q & {"all", "vehicles", "everything", "objects"}:
        return [o for o in objects if o.category == "vehicle"]
    scored = [(len(q & o.keywords), o) for o in objects]
    best = max((s for s, _ in scored), default=0)
    return [o for s, o in scored if s == best and s > 0]


def _compass(dx: float, dy: float) -> str:
    dirs = ["east", "southeast", "south", "southwest",
            "west", "northwest", "north", "northeast"]
    return dirs[int((math.degrees(math.atan2(dy, dx)) + 22.5) % 360 // 45)]


def analyze_track(label: str, centers: list[tuple[int, float, float]],
                  fps: float) -> str:
    """Stand-in for PLM RDCap: segment the track into moving/stopped phases."""
    if len(centers) < 2:
        return f"{label}: too short to analyze."
    phases = []
    window = max(2, int(fps))
    i = 0
    while i < len(centers) - 1:
        j = min(i + window, len(centers) - 1)
        f0, x0, y0 = centers[i]
        f1, x1, y1 = centers[j]
        speed = math.hypot(x1 - x0, y1 - y0) / max((f1 - f0) / fps, 1e-6)
        state = ("stationary", "") if speed < 4 else \
                ("moving", _compass(x1 - x0, y1 - y0))
        if phases and phases[-1][0] == state:
            phases[-1] = (state, phases[-1][1], f1)
        else:
            phases.append((state, f0, f1))
        i = j
    parts = []
    for (kind, direction), f0, f1 in [(p[0], p[1], p[2]) for p in phases]:
        span = f"{f0 / fps:.1f}-{f1 / fps:.1f}s"
        parts.append(f"stationary ({span})" if kind == "stationary"
                     else f"moves {direction} ({span})")
    return f"{label}: " + "; ".join(parts) + "."


def run_mock_pipeline(query: str, duration_s: float = 20.0, fps: float = 10.0,
                      reacquire_every_s: float = 3.0, seed: int = 7) -> dict:
    """Full mock pipeline -> tracks.json payload (same schema as real mode)."""
    rng = random.Random(seed)
    num_frames = int(duration_s * fps)
    scene = build_scene()
    objects: list[SceneObject] = scene["objects"]
    gt = {o.oid: o.simulate(num_frames, fps) for o in objects}

    tracks: dict[int, dict] = {}   # oid -> track record
    reacquire_every = max(1, int(reacquire_every_s * fps))
    overlay_palette = ["#37B58C", "#D98E3B", "#5B8DD9", "#C25B5B", "#8E6BC2"]

    for f in range(num_frames):
        # "PLM grounding" runs on frame 0 and then periodically (re-acquisition)
        if f % reacquire_every == 0:
            for obj in match_query(query, objects):
                if obj.oid not in tracks and gt[obj.oid][f]["visible"]:
                    tracks[obj.oid] = {
                        "id": len(tracks) + 1, "label": obj.label,
                        "color": overlay_palette[len(tracks) % len(overlay_palette)],
                        "acquired_frame": f,
                        "boxes": {}, "centers": [],
                    }
        # "SAM 2 propagation": follow each acquired object with slight noise
        for oid, tr in tracks.items():
            state = gt[oid][f]
            if not state["visible"]:
                continue
            x1, y1, x2, y2 = objects_by_id(objects)[oid].bbox(state)
            jitter = lambda: rng.uniform(-1.6, 1.6)
            box = [round(x1 + jitter(), 1), round(y1 + jitter(), 1),
                   round(x2 + jitter(), 1), round(y2 + jitter(), 1)]
            tr["boxes"][str(f)] = box
            tr["centers"].append((f, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

    for tr in tracks.values():
        tr["analysis"] = analyze_track(tr["label"], tr["centers"], fps)
        del tr["centers"]

    return {
        "meta": {"mode": "mock", "query": query, "fps": fps,
                 "num_frames": num_frames, "width": WORLD_W, "height": WORLD_H},
        "scene": {
            "static": scene["static"],
            "actors": [
                {"oid": o.oid, "label": o.label, "color": o.color,
                 "category": o.category, "size": list(o.size),
                 "states": [[round(s["x"], 1), round(s["y"], 1),
                             round(s["heading"], 3), int(s["visible"])]
                            for s in gt[o.oid]]}
                for o in objects
            ],
        },
        "tracks": sorted(tracks.values(), key=lambda t: t["id"]),
    }


def objects_by_id(objects: list[SceneObject]) -> dict[int, SceneObject]:
    return {o.oid: o for o in objects}
