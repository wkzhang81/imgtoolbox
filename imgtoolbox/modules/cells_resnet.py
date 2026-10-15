"""
Phase-contrast cell segmentation + ResNet50 deep feature extraction (a 1024-dim vector per cell).

Original script: cells/cell_to_feature_resnet50.py
Requires torch / torchvision.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from ..loader import load_script, project_root
from ..registry import Param, Tool, register


def run(params: dict) -> str:
    m = load_script("cells_resnet")

    image_path = params["image_path"]
    output_dir = params["output_dir"] or str(project_root() / "outputs" / "cells_resnet")
    batch_size = int(params["batch_size"])
    crop_padding = int(params["crop_padding"])
    device = params["device"]

    result = m.analyze_phase_contrast_cells(
        image_path=image_path,
        output_dir=output_dir,
        batch_size=batch_size,
        crop_padding=crop_padding,
        device=device,
    )

    feats = result["features"]
    return (
        f"Cells: {len(feats)}\n"
        f"Feature dimensions: {feats.shape[1] - 3} (1024)\n"
        f"Output directory: {output_dir}"
    )


register(Tool(
    key="cells_resnet",
    label="Cell segmentation + ResNet50 features",
    description=(
        "Same watershed segmentation as the morphology version, but each cropped cell is passed through "
        "ResNet50 layer3, outputting a CSV of 1024-dim deep features. Requires torch / torchvision."
    ),
    params=[
        Param(
            key="image_path",
            label="Input cell image",
            type="file",
            file_mode="open",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")],
        ),
        Param(
            key="output_dir",
            label="Output directory",
            type="dir",
            default="",
            help="If empty, uses outputs/cells_resnet.",
            optional=True,
        ),
        Param(
            key="batch_size",
            label="batch_size",
            type="int",
            default=32,
            min=1,
            max=256,
        ),
        Param(
            key="crop_padding",
            label="Crop padding (pixels)",
            type="int",
            default=8,
            min=0,
            max=64,
        ),
        Param(
            key="device",
            label="Compute device",
            type="choice",
            default="auto",
            choices=["auto", "cpu", "cuda", "xpu"],
            help="auto = use the GPU if available, otherwise CPU.",
        ),
    ],
    run=run,
))
