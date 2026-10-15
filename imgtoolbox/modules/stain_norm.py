"""
General stain / fluorescence image normalization (H&E / IHC / Masson / IF).

Original script: img_normalization/universal_stain_normalize.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from ..loader import load_script
from ..registry import Param, Tool, register


def run(params: dict) -> str:
    m = load_script("stain_norm")

    input_dir = Path(params["input_dir"])
    output_dir = Path(params["output_dir"])
    mode = params["mode"]
    ref_file = params.get("reference_file") or ""
    ref_dir = params.get("reference_dir") or ""

    images = m.collect_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found under {input_dir}.")

    if ref_file and ref_dir:
        raise ValueError("Choose either a reference image or a reference directory, not both.")

    if ref_file:
        ref_paths = [Path(ref_file)]
    elif ref_dir:
        ref_paths = m.collect_images(Path(ref_dir))
        if not ref_paths:
            raise RuntimeError("No images in the reference directory.")
    else:
        ref_paths = images  # use this batch's median

    # decide the mode in auto mode
    if mode == "auto":
        modes = []
        for p in ref_paths[: min(len(ref_paths), 20)]:
            try:
                modes.append(m.detect_mode(m.read_image(p)))
            except Exception:
                continue
        if not modes:
            mode = "brightfield"
        else:
            mode = max(set(modes), key=modes.count)

    ill = float(params["illumination_strength"])
    chroma = float(params["chroma_strength"])
    light = float(params["lightness_strength"])
    gamma = float(params["if_gamma_strength"])
    save_float = bool(params.get("save_float_tiff", False))

    if mode == "brightfield":
        ref_med, ref_scale = m.compute_brightfield_reference_stats(ref_paths, ill)
    else:
        ref_stats = m.compute_fluorescence_reference_stats(ref_paths)

    ok, fail = 0, 0
    for p in images:
        rel = p.relative_to(input_dir)
        out_path = output_dir / rel
        try:
            img = m.read_image(p)
            if mode == "brightfield":
                if img.shape[-1] < 3:
                    raise ValueError("Brightfield normalization requires RGB.")
                out = m.normalize_brightfield(
                    img[..., :3],
                    ref_med=ref_med,
                    ref_scale=ref_scale,
                    illumination_strength=ill,
                    chroma_strength=chroma,
                    lightness_strength=light,
                )
                m.save_image(out_path, out)
            else:
                out = m.normalize_fluorescence(img, ref_stats=ref_stats, gamma_strength=gamma)
                m.save_image(out_path, out)
                if save_float:
                    fp = output_dir / "float_tiff" / rel.with_suffix(".tif")
                    m.save_float_tiff(fp, out)
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"[failed] {p.name}: {exc}")

    return (
        f"Input: {input_dir}\n"
        f"Output: {output_dir}\n"
        f"Mode: {mode}\n"
        f"Reference: {'self-reference (batch median)' if not (ref_file or ref_dir) else (ref_file or ref_dir)}\n"
        f"Success: {ok} / failed: {fail} / total {len(images)}"
    )


register(Tool(
    key="stain_norm",
    label="Stain/fluorescence normalization",
    description=(
        "Brightfield (H&E/IHC/Masson) uses robust Reinhard-style matching in Lab space; "
        "fluorescence (IF) uses per-channel percentile normalization + light gamma alignment. "
        "Supports single-image or batch reference."
    ),
    params=[
        Param(key="input_dir", label="Input image directory", type="dir"),
        Param(key="output_dir", label="Output directory", type="dir"),
        Param(
            key="mode",
            label="Mode",
            type="choice",
            default="auto",
            choices=["auto", "brightfield", "fluorescence"],
        ),
        Param(
            key="reference_file",
            label="Reference image (single, optional)",
            type="file",
            file_mode="open",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")],
            default="",
            optional=True,
        ),
        Param(
            key="reference_dir",
            label="Reference directory (batch, optional)",
            type="dir",
            default="",
            optional=True,
        ),
        Param(key="illumination_strength", label="Brightfield illumination correction strength", type="float", default=0.35, min=0, max=1),
        Param(key="chroma_strength", label="Brightfield a/b color strength", type="float", default=0.85, min=0, max=1),
        Param(key="lightness_strength", label="Brightfield L strength", type="float", default=0.75, min=0, max=1),
        Param(key="if_gamma_strength", label="Fluorescence gamma strength", type="float", default=0.35, min=0, max=1),
        Param(key="save_float_tiff", label="Fluorescence: also save 32-bit float TIFF", type="bool", default=False),
    ],
    run=run,
))
