"""
SIFT panoramic image stitching.

Original script: fig_stitch/stitch_sift.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from ..loader import load_script
from ..registry import Param, Tool, register


def run(params: dict) -> str:
    m = load_script("fig_stitch")

    # files is a semicolon-separated string of absolute paths
    image_paths = [p.strip() for p in params["image_files"].split(";") if p.strip()]
    if len(image_paths) < 2:
        raise ValueError("Please select at least 2 images to stitch.")

    output_path = params["output_path"]
    if not output_path:
        raise ValueError("Please specify an output path.")

    panorama = m.stitch_images(
        image_paths=image_paths,
        output_path=output_path,
        max_ratio=float(params["max_ratio"]),
        min_matches=int(params["min_matches"]),
        min_inliers=int(params["min_inliers"]),
        min_inlier_ratio=float(params["min_inlier_ratio"]),
        residual_threshold=float(params["residual_threshold"]),
        transform_model=params["transform_model"],
    )

    h, w = panorama.shape[:2]
    return (
        f"Input images: {len(image_paths)}\n"
        f"Panorama size: {w} x {h}\n"
        f"Output: {output_path}"
    )


register(Tool(
    key="fig_stitch",
    label="SIFT panoramic stitching",
    description=(
        "Run SIFT feature matching + RANSAC on multiple overlapping microscope/photo images, "
        "and output a single panorama with feather-weighted blending."
    ),
    params=[
        Param(
            key="image_files",
            label="Select multiple images",
            type="files",
            file_mode="open",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")],
            help="Hold Ctrl to select multiple.",
        ),
        Param(
            key="output_path",
            label="Output panorama path",
            type="file",
            file_mode="save",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif")],
            default="",
        ),
        Param(
            key="transform_model",
            label="Transform model",
            type="choice",
            default="translation",
            choices=["translation", "similarity", "projective"],
            help="translation = shift only (good for microscope scans); similarity = + rotation/scale; projective = full perspective.",
        ),
        Param(key="max_ratio", label="Lowe ratio", type="float", default=0.75, min=0.1, max=1.0),
        Param(key="min_matches", label="Minimum matches", type="int", default=12, min=4, max=1000),
        Param(key="min_inliers", label="Minimum inliers", type="int", default=10, min=4, max=1000),
        Param(key="min_inlier_ratio", label="Inlier ratio lower bound", type="float", default=0.25, min=0, max=1),
        Param(key="residual_threshold", label="RANSAC residual (px)", type="float", default=3.0, min=0.1, max=100),
    ],
    run=run,
))
