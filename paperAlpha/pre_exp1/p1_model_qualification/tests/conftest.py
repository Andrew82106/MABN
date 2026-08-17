from __future__ import annotations

import sys
from pathlib import Path

P1_ROOT = Path(__file__).resolve().parents[1]
PRE_EXP_ROOT = P1_ROOT.parents[0]
for source in (
    P1_ROOT / "src",
    PRE_EXP_ROOT / "p1_benign" / "src",
    PRE_EXP_ROOT / "src",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
