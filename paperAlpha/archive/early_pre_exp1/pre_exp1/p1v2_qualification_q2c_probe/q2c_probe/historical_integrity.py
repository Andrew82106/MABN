"""Read-only aggregate integrity checks for the explicitly named Q2-B trees."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def _tree_digest(root: Path) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("historical_tree_unavailable")
    digest = sha256()
    count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("historical_tree_contains_symlink")
        if path.is_file():
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            content_digest = sha256(path.read_bytes()).digest()
            digest.update(content_digest)
            count += 1
    return {"file_count": count, "tree_sha256": digest.hexdigest()}


def historical_tree_snapshot() -> dict[str, dict[str, object]]:
    paper_alpha = Path(__file__).resolve().parents[3]
    targets = {
        "q2b_code": paper_alpha / "pre_exp1" / "p1v2_qualification_q2b",
        "q2b_data": paper_alpha / "data" / "pre_exp1" / "p1v2_qualification_q2b",
        "q2b_repair_code": paper_alpha / "pre_exp1" / "p1v2_qualification_q2b_repair",
        "q2b_repair_data": paper_alpha / "data" / "pre_exp1" / "p1v2_qualification_q2b_repair",
    }
    return {label: _tree_digest(path) for label, path in targets.items()}
