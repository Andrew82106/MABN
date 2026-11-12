"""Historical-path lookup without aliases or changes to frozen records."""
from functools import lru_cache
import json
from pathlib import Path


def project_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "code/workspace_paths.json").is_file():
        raise FileNotFoundError("historical path lookup requires the full source checkout")
    return root


@lru_cache(maxsize=1)
def _moves():
    data = json.loads((project_root() / "code/workspace_paths.json").read_text(encoding="utf-8"))
    return sorted(data["moves"], key=lambda item: len(item["source"]), reverse=True)


def resolve_historical_path(value: str | Path) -> Path:
    root = project_root()
    text = str(value).replace("\\", "/")
    prefix = root.as_posix() + "/"
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix):]
    elif text.startswith("paperAlpha/"):
        text = text[len("paperAlpha/"):]
    if Path(text).is_absolute() or ".." in Path(text).parts:
        raise ValueError("only paths inside this project can be resolved")
    for entry in _moves():
        old = entry["source"]
        if text == old or text.startswith(old + "/"):
            return root / (entry["destination"] + text[len(old):])
    candidate = root / text
    if not candidate.exists():
        raise FileNotFoundError(f"no recorded location for {text!r}")
    return candidate
