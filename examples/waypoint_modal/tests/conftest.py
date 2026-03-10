from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LUCID_SRC = ROOT / "packages" / "lucid" / "src"
EXAMPLE_SRC = ROOT / "examples" / "waypoint_modal" / "src"

for path in (EXAMPLE_SRC, LUCID_SRC):
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

os.environ.setdefault("WM_MODEL_MODULE", "waypoint_modal_example.model")
