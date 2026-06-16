from importlib import metadata

__all__ = ["__version__"]

try:
    __version__ = metadata.version("robust-rl-observation-noise-benchmark")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
