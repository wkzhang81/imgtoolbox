"""
Phase-contrast cell segmentation + morphological-feature KMeans clustering (pure skimage/sklearn version).

Original script: cells/cell_to_feature.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from ..loader import load_script, project_root
from ..registry import Param, Tool, register


def run(params: dict) -> str:
    m = load_script("cells_morph")

    image_path = params["image_path"]
    output_dir = params["output_dir"] or str(project_root() / "outputs" / "cells_morph")
    n_clusters = int(params["n_clusters"])

    result = m.analyze_phase_contrast_cells(
        image_path=image_path,
        output_dir=output_dir,
        n_clusters=n_clusters,
    )

    labels = result["segmentation"]["labels"]
    return (
        f"Cells: {labels.max()}\n"
        f"Silhouette score: {result['clustering']['silhouette']:.3f}\n"
        f"Output directory: {output_dir}"
    )


register(Tool(
    key="cells_morph",
    label="Phase-contrast cell segmentation + morphology clustering",
    description=(
        "Apply background correction, confluent-area estimation and marker-controlled watershed to dense "
        "phase-contrast/brightfield cell images, extract per-cell morphology and grayscale features, and "
        "cluster with RobustScaler+PCA+KMeans."
    ),
    params=[
        Param(
            key="image_path",
            label="Input cell image",
            type="file",
            file_mode="open",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")],
            help="Phase-contrast or brightfield grayscale cell image.",
        ),
        Param(
            key="output_dir",
            label="Output directory",
            type="dir",
            default="",
            help="If empty, uses outputs/cells_morph.",
            optional=True,
        ),
        Param(
            key="n_clusters",
            label="Number of clusters n_clusters",
            type="int",
            default=5,
            min=2,
            max=20,
        ),
    ],
    run=run,
))
