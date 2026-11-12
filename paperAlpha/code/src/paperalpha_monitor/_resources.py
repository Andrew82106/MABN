"""Read the single knowledge tree in a checkout or an installed wheel."""
from pathlib import Path, PurePosixPath


def static_resource(relative: str) -> Path:
    """Resolve a knowledge JSON; never fall back to the current directory."""
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("a relative POSIX knowledge-resource name is required")
    name = PurePosixPath(relative)
    if (name.is_absolute() or ".." in name.parts or len(name.parts) != 3
            or name.parts[0] != "knowledge"
            or name.parts[1] not in {"risks", "templates", "schemas"}
            or name.suffix != ".json"):
        raise ValueError("only JSON files in the knowledge tree are resources")
    package = Path(__file__).resolve().parent
    installed = package / "_resources" / name
    if installed.is_file():
        return installed
    root = package.parents[2]
    if package == root / "code/src/paperalpha_monitor":
        source = root / name
        if source.is_file():
            return source
    raise FileNotFoundError(f"static resource {relative!r} is absent from this package")
