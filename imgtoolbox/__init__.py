"""
imgtoolbox — a scikit-image based toolkit for scientific image processing.

Wraps several standalone scripts from the working directory (cell analysis,
image stitching, similar-image grouping, IHC quantification, stain
normalization, chemiluminescence overlay, etc.) into a single unified package,
and lets you pick a tool, fill in parameters and run it through a tkinter GUI.
"""

__version__ = "0.1.0"

from .registry import REGISTRY, list_tools  # noqa: F401
