from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LUCID_SRC = ROOT / "packages" / "lucid" / "src"
EXAMPLE_SRC = ROOT / "examples" / "helios_modal" / "src"

for path in (EXAMPLE_SRC, LUCID_SRC):
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
