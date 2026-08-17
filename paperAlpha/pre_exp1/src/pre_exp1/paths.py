"""Project path resolution with a single data-output boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PRE_EXP1_ROOT = PACKAGE_DIR.parents[1]
PAPER_ALPHA_ROOT = PACKAGE_DIR.parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    paper_alpha_root: Path
    pre_exp1_root: Path
    config_dir: Path
    data_root: Path
    shared_data_dir: Path
    pre_exp1_data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    manifests_dir: Path
    reports_dir: Path

    @classmethod
    def from_root(cls, paper_alpha_root: Path) -> "ProjectPaths":
        root = paper_alpha_root.resolve()
        pre_exp1 = root / "pre_exp1"
        data = root / "data"
        experiment_data = data / "pre_exp1"
        return cls(
            paper_alpha_root=root,
            pre_exp1_root=pre_exp1,
            config_dir=pre_exp1 / "configs",
            data_root=data,
            shared_data_dir=data / "shared",
            pre_exp1_data_dir=experiment_data,
            raw_dir=experiment_data / "raw",
            interim_dir=experiment_data / "interim",
            processed_dir=experiment_data / "processed",
            manifests_dir=experiment_data / "manifests",
            reports_dir=experiment_data / "reports",
        )


DEFAULT_PATHS = ProjectPaths.from_root(PAPER_ALPHA_ROOT)
CONFIG_DIR = DEFAULT_PATHS.config_dir
DATA_ROOT = DEFAULT_PATHS.data_root
SHARED_DATA_DIR = DEFAULT_PATHS.shared_data_dir
PRE_EXP1_DATA_DIR = DEFAULT_PATHS.pre_exp1_data_dir
RAW_DIR = DEFAULT_PATHS.raw_dir
INTERIM_DIR = DEFAULT_PATHS.interim_dir
PROCESSED_DIR = DEFAULT_PATHS.processed_dir
MANIFESTS_DIR = DEFAULT_PATHS.manifests_dir
REPORTS_DIR = DEFAULT_PATHS.reports_dir


def ensure_data_dirs(paths: ProjectPaths = DEFAULT_PATHS) -> None:
    """Create only the approved generated-data directories."""

    for directory in (
        paths.raw_dir,
        paths.interim_dir,
        paths.processed_dir,
        paths.manifests_dir,
        paths.reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def require_data_path(path: Path, data_root: Path = DATA_ROOT) -> Path:
    """Reject any generated-output path outside paperAlpha/data."""

    resolved = path.resolve()
    resolved_root = data_root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(
            f"Generated output must be under {resolved_root}: {resolved}"
        )
    return resolved


def relative_to_paper_alpha(
    path: Path,
    paper_alpha_root: Path = PAPER_ALPHA_ROOT,
) -> str:
    return path.resolve().relative_to(paper_alpha_root.resolve()).as_posix()
