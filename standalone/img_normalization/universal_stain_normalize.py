#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal stain / fluorescence normalization for H&E / IHC / Masson / IF.

Core idea:
1) Robust foreground/background masking
2) Optional illumination correction
3) Lab-space robust location/scale matching for brightfield images
4) Percentile-based intensity normalization for IF images
5) Batch processing with a reference image or reference folder

Dependencies:
    pip install numpy scipy scikit-image imageio

Examples:
    python universal_stain_normalize.py \
        --input ./input_images \
        --output ./normalized \
        --mode auto \
        --reference ./reference.png

    python universal_stain_normalize.py \
        --input ./input_images \
        --output ./normalized \
        --mode brightfield \
        --reference-dir ./reference_batch

    python universal_stain_normalize.py \
        --input ./if_images \
        --output ./normalized_if \
        --mode fluorescence \
        --reference-dir ./if_reference_batch
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import color, exposure, io, img_as_float, img_as_ubyte
from skimage.filters import threshold_otsu


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_image(path):
    img = io.imread(str(path))

    # Normalize common shapes to H x W x C
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        # Some TIFFs are C x H x W
        img = np.moveaxis(img, 0, -1)

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.shape[-1] > 4:
        # Multi-channel IF is allowed; keep channels as-is.
        pass
    elif img.shape[-1] == 4:
        # Drop alpha for RGB processing.
        img = img[..., :3]

    return img_as_float(img)


def robust_percentiles(x, low=1.0, high=99.0):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def detect_mode(img):
    """
    Heuristic:
    - IF usually has a dark background and bright sparse structures.
    - Brightfield H&E/IHC/Masson usually has a bright/white background.
    """
    rgb = img[..., :3] if img.shape[-1] >= 3 else img
    gray = color.rgb2gray(rgb[..., :3])
    mean_g = float(np.mean(gray))
    median_g = float(np.median(gray))
    bright_fraction = float(np.mean(gray > 0.8))
    dark_fraction = float(np.mean(gray < 0.2))

    if median_g < 0.35 and dark_fraction > bright_fraction:
        return "fluorescence"
    return "brightfield"


def brightfield_foreground_mask(rgb):
    """
    Tissue/positive-content mask that excludes most white background.

    Combines:
    - optical-density-like darkness
    - saturation
    - adaptive Otsu fallback
    """
    rgb = np.clip(rgb[..., :3], 1e-6, 1.0)
    hsv = color.rgb2hsv(rgb)
    gray = color.rgb2gray(rgb)

    sat = hsv[..., 1]
    darkness = 1.0 - gray

    # Conservative initial mask
    mask = (gray < 0.92) | (sat > 0.08)

    # Refine using Otsu on darkness if possible
    finite = darkness[np.isfinite(darkness)]
    if finite.size > 100:
        try:
            t = threshold_otsu(finite)
            otsu_mask = darkness > max(t * 0.65, 0.03)
            mask = mask & (otsu_mask | (sat > 0.10))
        except Exception:
            pass

    # If mask is too small, relax it
    frac = np.mean(mask)
    if frac < 0.01:
        mask = gray < 0.97

    return mask


def fluorescence_foreground_mask(img):
    """
    Multi-channel IF foreground mask based on total robust intensity.
    """
    x = np.clip(img, 0, None)
    if x.shape[-1] > 1:
        total = np.max(x, axis=-1)
    else:
        total = x[..., 0]

    lo, hi = robust_percentiles(total, 5, 99.5)
    scaled = np.clip((total - lo) / (hi - lo + 1e-8), 0, 1)

    try:
        t = threshold_otsu(scaled)
        mask = scaled > max(t, 0.03)
    except Exception:
        mask = scaled > 0.05

    if np.mean(mask) < 0.001:
        mask = scaled > 0.01
    return mask


def estimate_illumination(rgb, sigma_fraction=0.03):
    """
    Smooth illumination field estimated per channel.
    Useful for uneven scanner/light-source illumination.
    """
    h, w = rgb.shape[:2]
    sigma = max(8.0, sigma_fraction * max(h, w))
    field = np.empty_like(rgb, dtype=np.float64)
    for c in range(rgb.shape[-1]):
        field[..., c] = gaussian_filter(rgb[..., c], sigma=sigma)
    return np.clip(field, 1e-3, None)


def flatfield_correct_brightfield(rgb, strength=0.5):
    """
    Mild multiplicative illumination correction.
    strength=0 disables; strength=1 applies full correction.
    """
    if strength <= 0:
        return np.clip(rgb, 0, 1)

    field = estimate_illumination(rgb)
    target = np.median(field.reshape(-1, field.shape[-1]), axis=0)
    corrected = rgb * (target[None, None, :] / field)
    corrected = (1 - strength) * rgb + strength * corrected
    return np.clip(corrected, 0, 1)


def robust_lab_stats(rgb, mask):
    lab = color.rgb2lab(np.clip(rgb[..., :3], 0, 1))
    vals = lab[mask]
    if vals.shape[0] < 100:
        vals = lab.reshape(-1, 3)

    med = np.median(vals, axis=0)
    q25 = np.percentile(vals, 25, axis=0)
    q75 = np.percentile(vals, 75, axis=0)

    # Robust sigma estimate from IQR
    scale = (q75 - q25) / 1.349
    scale = np.maximum(scale, np.array([2.0, 1.0, 1.0]))
    return med, scale


def normalize_brightfield(
    rgb,
    ref_med,
    ref_scale,
    illumination_strength=0.35,
    chroma_strength=0.85,
    lightness_strength=0.75,
    preserve_background=True,
):
    """
    Robust Reinhard-like normalization in Lab space.

    L channel is intentionally adjusted less aggressively than a/b channels,
    reducing the risk of erasing true morphology or stain-density differences.
    """
    rgb = flatfield_correct_brightfield(
        np.clip(rgb[..., :3], 0, 1),
        strength=illumination_strength,
    )
    mask = brightfield_foreground_mask(rgb)

    src_med, src_scale = robust_lab_stats(rgb, mask)
    lab = color.rgb2lab(rgb)

    target = (lab - src_med) / src_scale * ref_scale + ref_med

    out_lab = lab.copy()
    out_lab[..., 0] = (
        (1 - lightness_strength) * lab[..., 0]
        + lightness_strength * target[..., 0]
    )
    out_lab[..., 1:] = (
        (1 - chroma_strength) * lab[..., 1:]
        + chroma_strength * target[..., 1:]
    )

    out = np.clip(color.lab2rgb(out_lab), 0, 1)

    if preserve_background:
        # Blend normalization only in foreground, with a soft colorfulness-based weight.
        hsv = color.rgb2hsv(rgb)
        gray = color.rgb2gray(rgb)
        weight = np.maximum(
            np.clip((0.96 - gray) / 0.20, 0, 1),
            np.clip(hsv[..., 1] / 0.18, 0, 1),
        )
        weight = gaussian_filter(weight.astype(np.float64), sigma=1.0)
        weight = np.clip(weight, 0, 1)[..., None]
        out = weight * out + (1 - weight) * rgb

    return np.clip(out, 0, 1)


def fluorescence_channel_stats(img, mask):
    """
    Per-channel robust percentiles on signal pixels.
    """
    stats = []
    for c in range(img.shape[-1]):
        vals = img[..., c][mask]
        if vals.size < 100:
            vals = img[..., c].ravel()

        # Include a low percentile for black-level offset and high percentile for scale
        p1, p50, p99 = np.percentile(vals, [1, 50, 99.5])
        if p99 <= p1:
            p99 = p1 + 1e-6
        stats.append((float(p1), float(p50), float(p99)))
    return np.asarray(stats, dtype=float)


def normalize_fluorescence(
    img,
    ref_stats,
    gamma_strength=0.35,
    clip_high=0.999,
):
    """
    IF normalization:
    - subtract robust channel-wise background
    - match channel-wise dynamic range
    - mild median alignment using gamma

    This preserves channel identity and works for RGB composites and
    arbitrary multi-channel fluorescence images.
    """
    x = np.clip(img.astype(np.float64), 0, None)
    mask = fluorescence_foreground_mask(x)
    src = fluorescence_channel_stats(x, mask)

    out = np.empty_like(x, dtype=float)

    n_channels = x.shape[-1]
    if ref_stats.shape[0] != n_channels:
        raise ValueError(
            f"Reference has {ref_stats.shape[0]} channels, "
            f"but target has {n_channels} channels."
        )

    for c in range(n_channels):
        s_lo, s_med, s_hi = src[c]
        r_lo, r_med, r_hi = ref_stats[c]

        y = (x[..., c] - s_lo) / (s_hi - s_lo + 1e-8)
        y = np.clip(y, 0, None)

        # Reference dynamic range normalization
        ref_med_norm = (r_med - r_lo) / (r_hi - r_lo + 1e-8)
        src_med_norm = (s_med - s_lo) / (s_hi - s_lo + 1e-8)

        # Estimate a mild gamma needed to align medians.
        if 1e-4 < src_med_norm < 0.999 and 1e-4 < ref_med_norm < 0.999:
            gamma = np.log(ref_med_norm) / np.log(src_med_norm)
            gamma = float(np.clip(gamma, 0.5, 2.0))
            gamma = 1.0 + gamma_strength * (gamma - 1.0)
            y = np.power(np.clip(y, 0, None), gamma)

        out[..., c] = y

    # Robust global clipping only to suppress extreme outliers.
    hi = np.quantile(out, clip_high)
    if hi > 0:
        out = out / hi

    return np.clip(out, 0, 1)


def collect_images(folder):
    folder = Path(folder)
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def compute_brightfield_reference_stats(paths, illumination_strength=0.35):
    meds = []
    scales = []

    for p in paths:
        img = read_image(p)
        if img.shape[-1] < 3:
            continue
        rgb = img[..., :3]
        rgb = flatfield_correct_brightfield(rgb, strength=illumination_strength)
        mask = brightfield_foreground_mask(rgb)
        med, scale = robust_lab_stats(rgb, mask)
        meds.append(med)
        scales.append(scale)

    if not meds:
        raise RuntimeError("No valid brightfield reference images found.")

    return np.median(np.stack(meds), axis=0), np.median(np.stack(scales), axis=0)


def compute_fluorescence_reference_stats(paths):
    all_stats = []
    channel_count = None

    for p in paths:
        img = read_image(p)
        if channel_count is None:
            channel_count = img.shape[-1]
        elif img.shape[-1] != channel_count:
            warnings.warn(
                f"Skipping {p.name}: channel count differs from reference batch."
            )
            continue

        mask = fluorescence_foreground_mask(img)
        all_stats.append(fluorescence_channel_stats(img, mask))

    if not all_stats:
        raise RuntimeError("No valid fluorescence reference images found.")

    return np.median(np.stack(all_stats), axis=0)


def save_image(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save normalized output as 8-bit for broad compatibility.
    # For quantitative IF workflows, use --save-float-tiff to preserve float data.
    io.imsave(str(path), img_as_ubyte(np.clip(arr, 0, 1)), check_contrast=False)


def save_float_tiff(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(path), np.asarray(arr, dtype=np.float32), check_contrast=False)


def main():
    parser = argparse.ArgumentParser(
        description="Universal normalization for H&E, IHC, Masson and IF images."
    )
    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument(
        "--mode",
        choices=["auto", "brightfield", "fluorescence"],
        default="auto",
        help="Image mode. H&E/IHC/Masson -> brightfield; IF -> fluorescence.",
    )
    parser.add_argument("--reference", help="Single reference image")
    parser.add_argument("--reference-dir", help="Folder of reference images")
    parser.add_argument(
        "--illumination-strength",
        type=float,
        default=0.35,
        help="Brightfield illumination correction strength [0..1]. Default 0.35",
    )
    parser.add_argument(
        "--chroma-strength",
        type=float,
        default=0.85,
        help="Brightfield a/b color normalization strength [0..1]. Default 0.85",
    )
    parser.add_argument(
        "--lightness-strength",
        type=float,
        default=0.75,
        help="Brightfield L normalization strength [0..1]. Default 0.75",
    )
    parser.add_argument(
        "--if-gamma-strength",
        type=float,
        default=0.35,
        help="IF median/gamma normalization strength [0..1]. Default 0.35",
    )
    parser.add_argument(
        "--save-float-tiff",
        action="store_true",
        help="For IF, additionally save 32-bit float TIFFs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    images = collect_images(input_dir)

    if not images:
        raise RuntimeError(f"No supported images found in {input_dir}")

    if args.reference and args.reference_dir:
        raise ValueError("Use either --reference or --reference-dir, not both.")

    if args.reference:
        ref_paths = [Path(args.reference)]
    elif args.reference_dir:
        ref_paths = collect_images(args.reference_dir)
        if not ref_paths:
            raise RuntimeError("No reference images found.")
    else:
        # Fallback: use all input images to define a batch-wide median target.
        ref_paths = images
        print(
            "[INFO] No explicit reference supplied. "
            "Using the median statistics of the input batch as target."
        )

    # If mode=auto, decide mode based on the reference set's majority.
    mode = args.mode
    if mode == "auto":
        modes = []
        for p in ref_paths[: min(len(ref_paths), 20)]:
            modes.append(detect_mode(read_image(p)))
        mode = max(set(modes), key=modes.count)
        print(f"[INFO] Auto-detected mode: {mode}")

    if mode == "brightfield":
        ref_med, ref_scale = compute_brightfield_reference_stats(
            ref_paths,
            illumination_strength=args.illumination_strength,
        )
        print("[INFO] Brightfield reference Lab median:", ref_med)
        print("[INFO] Brightfield reference Lab robust scale:", ref_scale)

    else:
        ref_stats = compute_fluorescence_reference_stats(ref_paths)
        print("[INFO] Fluorescence reference stats (p1, p50, p99.5/channel):")
        print(ref_stats)

    for i, p in enumerate(images, start=1):
        rel = p.relative_to(input_dir)
        out_path = output_dir / rel

        try:
            img = read_image(p)

            if mode == "brightfield":
                if img.shape[-1] < 3:
                    raise ValueError("Brightfield normalization requires RGB images.")

                normalized = normalize_brightfield(
                    img[..., :3],
                    ref_med=ref_med,
                    ref_scale=ref_scale,
                    illumination_strength=args.illumination_strength,
                    chroma_strength=args.chroma_strength,
                    lightness_strength=args.lightness_strength,
                )
                save_image(out_path, normalized)

            else:
                normalized = normalize_fluorescence(
                    img,
                    ref_stats=ref_stats,
                    gamma_strength=args.if_gamma_strength,
                )

                save_image(out_path, normalized)

                if args.save_float_tiff:
                    float_path = output_dir / "float_tiff" / rel.with_suffix(".tif")
                    save_float_tiff(float_path, normalized)

            print(f"[{i}/{len(images)}] OK: {p.name}")

        except Exception as exc:
            print(f"[{i}/{len(images)}] FAILED: {p.name} -> {exc}")

    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
