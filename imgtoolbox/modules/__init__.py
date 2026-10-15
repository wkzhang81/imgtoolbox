"""
Import each tool module to trigger its register() call.
"""

from . import (  # noqa: F401
    cells_morph,
    cells_resnet,
    fig_discrimination,
    fig_stitch,
    find_dup,
    ihc,
    stain_norm,
    lum_combine,
)
