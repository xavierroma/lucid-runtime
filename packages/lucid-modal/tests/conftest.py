from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LUCID_SRC = ROOT / "packages" / "lucid" / "src"
MODAL_SRC = ROOT / "packages" / "lucid-modal" / "src"

for path in (LUCID_SRC, MODAL_SRC):
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
