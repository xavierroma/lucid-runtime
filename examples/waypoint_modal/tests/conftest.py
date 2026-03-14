from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LUCID_PKG = ROOT / "packages" / "lucid"
EXAMPLE_SRC = ROOT / "examples" / "waypoint_modal" / "src"

for path in (EXAMPLE_SRC, LUCID_PKG):
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
