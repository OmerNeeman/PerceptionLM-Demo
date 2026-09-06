"""Write the standalone HTML viewer with the tracking result embedded."""

from __future__ import annotations

import json
import os

_TEMPLATE = os.path.join(os.path.dirname(__file__), "viewer_template.html")


def write_viewer(payload: dict, out_path: str) -> str:
    with open(_TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__TRACKS_JSON__", blob)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
