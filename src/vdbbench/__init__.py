"""Reproducible vector database benchmark harness."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vdbbench")
except PackageNotFoundError:  # pragma: no cover - editable install before metadata exists
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
