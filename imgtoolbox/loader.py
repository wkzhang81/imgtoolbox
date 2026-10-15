"""
Load each tool's script in place from the working directory, avoiding copying
large blocks of code into the package.

Original script locations (relative to this project's root ``integration/``)::

    integration/
        standalone/cells/cell_to_feature.py
        standalone/cells/cell_to_feature_resnet50.py
        standalone/fig_discrimination/group_similar_images.py
        standalone/fig_stitch/stitch_sift.py
        standalone/find_dup/find_dup.py
        standalone/ihc/ihc.py
        standalone/img_normalization/universal_stain_normalize.py
        standalone/lum_combine/lum_combine.py

Scripts are imported as modules, so their ``if __name__ == "__main__"`` blocks
do not execute.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Project root = the parent of imgtoolbox/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Relative paths of the original scripts (relative to the project root integration/)
_SCRIPT_MAP: dict[str, str] = {
    "cells_morph": "standalone/cells/cell_to_feature.py",
    "cells_resnet": "standalone/cells/cell_to_feature_resnet50.py",
    "fig_discrimination": "standalone/fig_discrimination/group_similar_images.py",
    "fig_stitch": "standalone/fig_stitch/stitch_sift.py",
    "find_dup": "standalone/find_dup/find_dup.py",
    "ihc": "standalone/ihc/ihc.py",
    "stain_norm": "standalone/img_normalization/universal_stain_normalize.py",
    "lum_combine": "standalone/lum_combine/lum_combine.py",
}

_CACHE: dict[str, object] = {}


def load_script(module_key: str):
    """Load a script module in place by key, with caching."""
    if module_key in _CACHE:
        return _CACHE[module_key]

    rel = _SCRIPT_MAP.get(module_key)
    if rel is None:
        raise KeyError(f"Unknown script key: {module_key}")

    path = (PROJECT_ROOT / rel).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Original script not found: {path}")

    # Add the script's directory to sys.path so imports from the same directory work
    sys.path.insert(0, str(path.parent))

    spec = importlib.util.spec_from_file_location(
        f"imgtoolbox_src_{module_key}",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    # Must register into sys.modules first, otherwise dataclass +
    # from __future__ import annotations, when resolving string annotations, looks
    # the module up in sys.modules[cls.__module__]; getting None raises
    # 'NoneType' has no attribute '__dict__'.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise ImportError(f"Failed to execute {path}: {exc}") from exc

    _CACHE[module_key] = module
    return module


def project_root() -> Path:
    return PROJECT_ROOT
