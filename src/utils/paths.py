from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return repository root assuming this file is under src/utils/."""
    return Path(__file__).resolve().parents[2]


def resolve_from_root(path_like: str | Path) -> Path:
    """Resolve a potentially-relative path from the repository root."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root() / path


def ensure_dir(path_like: str | Path) -> Path:
    """Create directory if needed and return the resolved path."""
    path = resolve_from_root(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path
