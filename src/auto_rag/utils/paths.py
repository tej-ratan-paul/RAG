"""Filesystem path helpers.

The project root is located by walking up from this package until a
``pyproject.toml`` is found. This keeps paths stable whether the package is
run from source or installed in editable mode.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PACKAGE_FILE = Path(__file__).resolve()


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``pyproject.toml``.

    Args:
        start: Directory to begin searching from (defaults to this file's dir).

    Returns:
        The closest ancestor directory containing a ``pyproject.toml``.

    Raises:
        FileNotFoundError: If no ``pyproject.toml`` is found in the ancestor chain.
    """
    current = (start or _PACKAGE_FILE.parent).resolve()
    for ancestor in (current, *current.parents):
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    raise FileNotFoundError(
        f"Could not locate project root (no pyproject.toml) from {current}"
    )


def project_root() -> Path:
    """Cached lookup of the repository root."""
    root = find_project_root()
    logger.debug("Resolved project root: %s", root)
    return root


def ensure_directory(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; returns the path.

    Args:
        path: Directory to ensure exists.

    Returns:
        The same path, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured directory exists: %s", path)
    return path
