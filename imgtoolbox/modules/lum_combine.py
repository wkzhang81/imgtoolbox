"""
Brightfield background + chemiluminescence TIF three-in-one overlay image.

Original script: lum_combine/lum_combine.py (logic lives in combine_luminescence).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from ..loader import load_script, project_root
from ..registry import Param, Tool, register


def run(params: dict) -> str:
    m = load_script("lum_combine")

    bg_path = params["bg_path"]
    lum_path = params["lum_path"]
    out_path = params["out_path"] or str(
        project_root() / "outputs" / "lum_combine" / "lum_combined.png"
    )

    if not bg_path or not lum_path:
        raise ValueError("Please select the brightfield background TIF and the chemiluminescence TIF.")

    # empty = None, letting the script estimate vmin/vmax automatically
    vmin = float(params["vmin"]) if params["vmin"] not in ("", None) else None
    vmax = float(params["vmax"]) if params["vmax"] not in ("", None) else None

    summary = m.combine_luminescence(
        bg_path=bg_path,
        lum_path=lum_path,
        out_path=out_path,
        vmin=vmin,
        vmax=vmax,
        noise_k=float(params["noise_k"]),
        alpha_floor=float(params["alpha_floor"]),
        alpha_gain=float(params["alpha_gain"]),
        bg_method=params["bg_method"],
        bg_sigma=float(params["bg_sigma"]),
        bg_radius=int(params["bg_radius"]),
        bg_percentile=float(params["bg_percentile"]),
    )

    return (
        f"Background: {bg_path}\n"
        f"Luminescence: {lum_path}{' (resized)' if summary['resized'] else ''}\n"
        f"Background subtraction: {summary['bg_desc']}\n"
        f"Colorbar range: [{summary['vmin']:.1f}, {summary['vmax']:.1f}]\n"
        f"Output: {summary['out_path']}"
    )


register(Tool(
    key="lum_combine",
    label="Brightfield + chemiluminescence three-in-one overlay",
    description=(
        "Align the brightfield TIF (anatomy) with the chemiluminescence TIF (signal spots) "
        "and output three panels: turbo overlay / brightfield grayscale / luminescence turbo with colorbar."
    ),
    params=[
        Param(
            key="bg_path",
            label="Brightfield background TIF",
            type="file",
            file_mode="open",
            filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")],
        ),
        Param(
            key="lum_path",
            label="Chemiluminescence TIF",
            type="file",
            file_mode="open",
            filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")],
        ),
        Param(
            key="out_path",
            label="Output PNG",
            type="file",
            file_mode="save",
            filetypes=[("PNG", "*.png")],
            default="",
            help="If empty, outputs to outputs/lum_combine/lum_combined.png.",
            optional=True,
        ),
        Param(
            key="bg_method",
            label="Background subtraction method",
            type="choice",
            default="gaussian",
            choices=["gaussian", "tophat", "global", "none"],
        ),
        Param(key="bg_sigma", label="(gaussian) sigma", type="float", default=25.0, min=1, max=500),
        Param(key="bg_radius", label="(tophat) radius (px)", type="int", default=50, min=1, max=500),
        Param(key="bg_percentile", label="(global) percentile", type="float", default=2.0, min=0, max=50),
        Param(key="vmin", label="vmin (empty = auto)", type="float", default="", optional=True),
        Param(key="vmax", label="vmax (empty = max value)", type="float", default="", optional=True),
        Param(key="noise_k", label="Auto vmin = median + k*sigma", type="float", default=2.5, min=0, max=20),
        Param(key="alpha_floor", label="alpha lower bound (0..1)", type="float", default=0.0, min=0, max=1),
        Param(key="alpha_gain", label="alpha exponent", type="float", default=0.5, min=0.1, max=5.0),
    ],
    run=run,
))
