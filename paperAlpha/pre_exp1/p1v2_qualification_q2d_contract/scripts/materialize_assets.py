"""Materialize the frozen Q2-D contract assets from the independent snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from q2d_contract.contract import CARDS, PROFILE, _PUBLISHER_CASES, contract_snapshot


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8", newline="\n")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    assets = ROOT / "assets"
    _write(assets / "contract_snapshot.json", contract_snapshot())
    _write(assets / "configs" / "profile.json", PROFILE)
    _write(assets / "fixtures" / "publisher_cases.json", _PUBLISHER_CASES)
    _write(assets / "fixtures" / "cards.json", [{key: card[key] for key in ("card_id", "phase", "position", "role", "fixture", "expected_action", "expected_action_hash", "schema_hash", "allowed_tool", "runtime_canary", "coverage_cases")} for card in CARDS])
    for card in CARDS:
        _write(assets / "schemas" / f"{card['card_id']}.schema.json", card["schema"])
        _write(assets / "instructions" / f"{card['card_id']}.txt", card["prompt"] + "\n")
    print(json.dumps({"cards": len(CARDS), "schemas": len(list((assets / "schemas").glob("*.json"))), "instructions": len(list((assets / "instructions").glob("*.txt")))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

