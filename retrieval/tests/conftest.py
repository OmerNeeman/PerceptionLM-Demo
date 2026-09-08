import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# This dev machine's data root (spec N-8: src/ itself carries no such
# default -- see src/config.py). setdefault so a test explicitly overriding
# AERIAL_DATA_ROOT (e.g. test_config_fails_loudly_without_data_root, which
# deletes it via monkeypatch for the duration of one test) is never clobbered.
# A fresh machine has no conftest.py doing this and must set the variable
# itself, per INSTRUCTIONS.md.
os.environ.setdefault(
    "AERIAL_DATA_ROOT", "/home/omer/PycharmProjects/Dynamic-Terrain/data"
)
