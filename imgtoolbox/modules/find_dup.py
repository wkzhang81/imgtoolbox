"""
Detect duplicate regions within a single image (SIFT split in half + match_descriptors).

Original script: find_dup/find_dup.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from ..loader import load_script, project_root
from ..registry import Param, Tool, register


def run(params: dict) -> str:
    m = load_script("find_dup")

    image_path = params["image_path"]
    if not image_path:
        raise ValueError("Please select an input image.")
    max_ratio = float(params["max_distance_ratio"])
    min_dist = float(params["min_dist"])
    output_dir = params["output_dir"] or str(project_root() / "outputs" / "find_dup")

    out_path, n_pairs = m.detect_duplicate_with_match_descriptors(
        image_path,
        max_distance_ratio=max_ratio,
        min_dist=min_dist,
        output_dir=output_dir,
    )

    if out_path is None:
        return "Too few keypoints; no result image generated."
    return (
        f"Input: {image_path}\n"
        f"Duplicate match pairs: {n_pairs}\n"
        f"Result image: {out_path}"
    )


register(Tool(
    key="find_dup",
    label="Duplicate-region detection within a single image",
    description=(
        "Split the SIFT keypoints in half and match the first half against the second, "
        "to detect tiles that appear twice in the same image (repeated collage, watermarks, copy-paste regions)."
    ),
    params=[
        Param(
            key="image_path",
            label="Input image",
            type="file",
            file_mode="open",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")],
        ),
        Param(
            key="max_distance_ratio",
            label="Lowe ratio",
            type="float",
            default=0.5,
            min=0.1,
            max=1.0,
            help="Higher (e.g. 0.8) matches more; lower is stricter.",
        ),
        Param(
            key="min_dist",
            label="Minimum spatial distance (px)",
            type="float",
            default=20,
            min=0,
            max=2000,
            help="Match pairs closer than this distance are treated as adjacent texture and discarded.",
        ),
        Param(
            key="output_dir",
            label="Output directory",
            type="dir",
            default="",
            help="If empty, uses outputs/find_dup.",
            optional=True,
        ),
    ],
    run=run,
))
