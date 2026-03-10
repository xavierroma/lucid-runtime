from __future__ import annotations

import os
import sys
from pathlib import Path

EXAMPLE_SRC = Path(__file__).resolve().parents[3] / "examples" / "yume_modal" / "src"
if str(EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SRC))
os.environ.setdefault("WM_MODEL_MODULE", "yume_modal_example.model")
