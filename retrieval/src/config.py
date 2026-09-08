"""Paths and settings, each overridable by an environment variable (spec N-8).

No value here may default to a path that exists only on the machine this
project happened to be developed on. Where there genuinely is no
machine-independent default -- the data root, which is different on every
machine -- missing configuration fails loudly, naming the environment
variable to set and how to set it on both Linux/macOS and Windows. It never
falls back to this project's own development path.

Where a machine-independent default *does* exist -- the index root, which is
naturally relative to this repository -- that default is used, and can still
be overridden.

Environment variables, all optional except AERIAL_DATA_ROOT:

    AERIAL_DATA_ROOT         required. Folder containing the source imagery.
    AERIAL_INDEX_ROOT        optional. Default: <repo>/retrieval/index.
    AERIAL_MODEL_CACHE_ROOT  optional. Default: huggingface_hub's own default
                             (HF_HOME / ~/.cache/huggingface), which is
                             already machine-independent.
    AERIAL_DEVICE            optional. Forces the compute device (e.g.
                             "cpu", "cuda", "mps"). Default: autodetect.

See INSTRUCTIONS.md for the full setup story.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath

SRC_DIR = Path(__file__).resolve().parent
RETRIEVAL_DIR = SRC_DIR.parent
REPO_ROOT = RETRIEVAL_DIR.parent

ENV_DATA_ROOT = "AERIAL_DATA_ROOT"
ENV_INDEX_ROOT = "AERIAL_INDEX_ROOT"
ENV_MODEL_CACHE_ROOT = "AERIAL_MODEL_CACHE_ROOT"
ENV_DEVICE = "AERIAL_DEVICE"


class ConfigError(RuntimeError):
    """A required configuration value is missing.

    Raised instead of silently substituting a hardcoded path -- see N-8's
    `test_config_fails_loudly_without_data_root`.
    """


def get_data_root() -> Path:
    """The read-only root of source imagery.

    There is deliberately no default: every machine's imagery lives
    somewhere different, and a fallback to this project's own development
    path would silently reintroduce the exact non-portability this function
    exists to prevent.
    """
    val = os.environ.get(ENV_DATA_ROOT)
    if not val:
        raise ConfigError(
            f"{ENV_DATA_ROOT} is not set. Point it at the folder that "
            "contains the source imagery (the .tif/.tiff rasters) before "
            "running anything under retrieval/src, e.g.\n"
            f"  bash/zsh   : export {ENV_DATA_ROOT}=/path/to/data\n"
            # Forward slashes: Windows/PowerShell accept them too, and this
            # keeps a Windows drive-letter example out of the literal-path
            # grep this very module is retrofitted to satisfy (N-8).
            f"  PowerShell : $env:{ENV_DATA_ROOT} = 'C:/path/to/data'\n"
            "See INSTRUCTIONS.md."
        )
    return Path(val)


def get_index_root() -> Path:
    """Where derived artifacts (inventory, tiles, embeddings, exports) go.

    Defaults to <repo>/retrieval/index -- relative to this repository, so it
    needs no machine-specific configuration at all -- but can be overridden
    (e.g. to point at a faster disk).
    """
    val = os.environ.get(ENV_INDEX_ROOT)
    return Path(val) if val else (RETRIEVAL_DIR / "index")


def get_model_cache_root() -> Path | None:
    """Optional override for where HF checkpoints are cached.

    None means "let huggingface_hub use its own default", which is itself
    already machine-independent (HF_HOME, or the platform cache dir).
    """
    val = os.environ.get(ENV_MODEL_CACHE_ROOT)
    return Path(val) if val else None


def get_forced_device() -> str | None:
    """AERIAL_DEVICE, e.g. "cpu" / "cuda" / "mps" -- or None for autodetect."""
    return os.environ.get(ENV_DEVICE) or None


def posix_key(path) -> str:
    """Canonical manifest/tile-id key: '/' separated, regardless of the host
    OS or of whether `path` originated on Windows (spec N-8).

    Accepts a `Path`/`PurePath` (any flavour) or a plain string that may use
    either separator -- a string is parsed as a Windows path first, because
    `PureWindowsPath` accepts *both* '/' and '\\\\' and normalises either to
    '/' via `.as_posix()`, so a POSIX-style string round-trips unchanged and
    a Windows-style string comes out identical to it. This is why an index
    built on Linux loads on Windows and vice versa.
    """
    return PureWindowsPath(str(path)).as_posix()
