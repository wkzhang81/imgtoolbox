# -*- coding: utf-8 -*-
"""
Combine a bright-field background TIF with a chemiluminescence TIF.

Output (one figure, three aligned panels, left -> right):
  1. Overlay : turbo chemiluminescence composited on top of the grayscale
               background (alpha driven by luminescence intensity, so dim
               areas show the mouse and bright spots show through in color).
  2. Background (bright field), grayscale.
  3. Chemiluminescence rendered with the "turbo" colormap, with a colorbar.

If the two TIFs differ in size, the luminescence image is resized to match
the background using scikit-image.

Usage:
    python lum_combine.py --bg D:\\bg.tif --lum D:\\lum.tif --out D:\\out.png

Can also be imported by imgtoolbox to call combine_luminescence().
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from skimage import io, transform
from skimage.filters import gaussian
from skimage.morphology import disk, white_tophat


def mad_sigma(x):
    """Robust Gaussian-noise sigma estimate from median absolute deviation.
    Insensitive to a small number of bright outliers (the signal spot)."""
    med = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - med)))


def load_gray(path):
    """Read an image as a 2-D float array. uint16 TIF -> float."""
    img = io.imread(path)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img.astype(np.float32)


def align_to(lum, ref):
    """Resize lum to the shape of ref if they differ. Returns (lum, was_resized)."""
    if lum.shape == ref.shape:
        return lum, False
    lum_a = transform.resize(
        lum, ref.shape, order=1, preserve_range=True, anti_aliasing=True
    )
    return lum_a, True


def subtract_background(lum, method="gaussian", sigma=25.0, radius=50, pct=2.0):
    """Estimate and remove background from the luminescence image.

    Returns (corrected_float_array, background_estimate, method_used_str).
    Negative values after subtraction are clipped to 0.
    """
    if method == "none":
        return lum.copy(), np.zeros_like(lum), "no background subtraction"

    if method == "global":
        # Single scalar offset: low percentile of the whole frame.
        bg_val = float(np.percentile(lum, pct))
        corr = lum - bg_val
        return np.clip(corr, 0, None), np.full_like(lum, bg_val), f"global {pct:g}% = {bg_val:.1f}"

    if method == "tophat":
        # Morphological white top-hat: original - opening(original). Removes
        # broad, slowly varying background while keeping small bright peaks.
        selem = disk(radius)
        corr = white_tophat(lum, footprint=selem)
        bg_est = lum - corr
        return corr.astype(np.float32), bg_est.astype(np.float32), \
            f"white_tophat disk r={radius}"

    # default: gaussian rolling-ball estimate of the background field
    bg_est = gaussian(lum, sigma=sigma, preserve_range=True)
    corr = lum - bg_est
    return np.clip(corr, 0.0, None).astype(np.float32), bg_est.astype(np.float32), \
        f"gaussian rolling-ball sigma={sigma:g}"


def percentile_norm(img, low=2.0, high=99.9):
    """Stretch display contrast via percentile clipping. Returns float in [0,1]."""
    lo, hi = np.percentile(img, (low, high))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def combine_luminescence(
    bg_path,
    lum_path,
    out_path,
    vmin=None,
    vmax=None,
    noise_k=2.5,
    alpha_floor=0.0,
    alpha_gain=0.5,
    bg_method="gaussian",
    bg_sigma=25.0,
    bg_radius=50,
    bg_percentile=2.0,
):
    """Build the three-panel overlay figure and save it to out_path.

    vmin / vmax: None -> auto (noise floor / image max).
    Returns a summary dict.
    """
    bg = load_gray(bg_path)
    lum_raw = load_gray(lum_path)
    lum_raw = lum_raw.max() - lum_raw  # chemiluminescence is inverted first
    lum_raw, resized = align_to(lum_raw, bg)

    # --- Background subtraction on the luminescence image (before any color) ---
    lum, bg_est, bg_desc = subtract_background(
        lum_raw, method=bg_method, sigma=bg_sigma,
        radius=bg_radius, pct=bg_percentile,
    )
    print(f"background subtraction: {bg_desc}")
    print(f"lum raw: mean={lum_raw.mean():.1f} max={lum_raw.max():.0f} | "
          f"after subtract: mean={lum.mean():.1f} max={lum.max():.0f}")

    # Display scaling:
    #   - bright-field background: gentle percentile stretch so the mouse is visible
    #   - chemiluminescence: linear scale with a noise floor. Pixels below vmin
    #     are mapped to pure black (cmap.set_under), so true background reads as
    #     black instead of turbo's dark purple low-end.
    bg_disp = percentile_norm(bg, low=2.0, high=99.5)

    # Upper end
    vmax = float(vmax) if vmax is not None else float(lum.max())

    # Lower end / noise floor. Use a robust MAD-based sigma estimate on the
    # unclipped background-subtracted frame; insensitive to the bright spot.
    raw_diff = lum_raw - bg_est
    if vmin is not None:
        vmin = float(vmin)
        print(f"vmin (user-specified) = {vmin:.1f} raw counts")
    else:
        noise_sd = mad_sigma(raw_diff)
        # offset: median of the clipped background, robust to the hot spot
        bg_pixel_median = float(np.median(lum))
        vmin = bg_pixel_median + noise_k * noise_sd
        print(f"noise floor estimate: MAD sigma={noise_sd:.1f}, "
              f"bg median={bg_pixel_median:.1f}, vmin=median+{noise_k:g}*sigma={vmin:.1f}")
    if vmax <= vmin:
        vmax = vmin + 1.0

    # Turbo with pure black under the threshold
    turbo_black = plt.get_cmap("turbo").copy()
    turbo_black.set_under("black")
    lum_norm = Normalize(vmin=vmin, vmax=vmax)
    # Normalized lum for alpha: map [vmin, vmax] -> [0, 1], clip below to 0
    lum_n = np.clip((lum - vmin) / (vmax - vmin), 0.0, 1.0)

    # Per-pixel alpha for the overlay: dim -> transparent, bright -> opaque
    a_floor = float(np.clip(alpha_floor, 0.0, 1.0))
    alpha = np.clip((lum_n - a_floor) / (1.0 - a_floor + 1e-9), 0.0, 1.0)
    if float(alpha_gain) != 1.0:
        alpha = alpha ** float(alpha_gain)

    # Three panels, strictly aligned
    fig, (ax_o, ax_b, ax_l) = plt.subplots(
        1, 3, figsize=(16, 5.2), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.04}, constrained_layout=True,
    )

    # 1) Overlay: grayscale background under turbo luminescence with per-pixel alpha
    ax_o.imshow(bg_disp, cmap="gray", interpolation="nearest")
    ax_o.imshow(lum, cmap=turbo_black, norm=lum_norm,
                alpha=alpha, interpolation="nearest")
    ax_o.set_title("Overlay (turbo on bright field)", fontsize=12)
    ax_o.set_xlabel("pixels")
    ax_o.set_ylabel("pixels")

    # 2) Background only
    ax_b.imshow(bg_disp, cmap="gray", interpolation="nearest")
    ax_b.set_title("Background (bright field)", fontsize=12)
    ax_b.set_xlabel("pixels")

    # 3) Luminescence only, with colorbar
    im_l = ax_l.imshow(lum, cmap=turbo_black, norm=lum_norm, interpolation="nearest")
    ax_l.set_title(f"Chemiluminescence (turbo, {bg_desc})", fontsize=12)
    ax_l.set_xlabel("pixels")
    cbar = fig.colorbar(im_l, ax=ax_l, fraction=0.046, pad=0.04,
                        extend="min")
    cbar.set_label("intensity (bg-subtracted, raw counts)", fontsize=10)

    if resized:
        fig.suptitle(
            f"Aligned: luminescence resized from original to {bg.shape}",
            fontsize=10, y=1.02,
        )

    out_parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_parent, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"background shape={bg.shape}, lum shape={lum.shape}, resized={resized}")
    print(f"lum color scale = linear [{vmin:.1f}, {vmax:.1f}] raw counts; "
          f"below {vmin:.1f} = pure black (no percentile stretch)")
    print(f"overlay alpha floor={a_floor}, gain={alpha_gain}")

    return {
        "out_path": out_path,
        "vmin": vmin,
        "vmax": vmax,
        "bg_desc": bg_desc,
        "resized": resized,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bg", required=True, help="background TIF path")
    p.add_argument("--lum", required=True, help="luminescence TIF path")
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--vmin", type=float, default=None,
                   help="lower end of the turbo color scale, in raw counts after "
                        "background subtraction. Pixels below this are rendered "
                        "pure black (colormap under-color). Default: auto-estimated "
                        "from the background noise floor (see --noise-k).")
    p.add_argument("--vmax", type=float, default=None,
                   help="upper end of the turbo color scale, in raw counts after "
                        "background subtraction. Defaults to the image maximum. "
                        "Set this (instead of auto-max) to keep the colorbar "
                        "consistent across multiple images.")
    p.add_argument("--noise-k", type=float, default=2.5,
                   help="auto vmin = (median of background) + k * sigma_noise. "
                        "Higher k pushes the black threshold higher (darker "
                        "background, may clip very faint signal).")
    p.add_argument("--alpha-floor", type=float, default=0.0,
                   help="fraction of [vmin,vmax] below which overlay is fully "
                        "transparent (0..1). vmin already suppresses noise, so 0 "
                        "lets faint signal show through.")
    p.add_argument("--alpha-gain", type=float, default=0.5,
                   help="alpha curve exponent on normalized lum. <1 makes faint "
                        "signal more opaque (more prominent overlay); >1 makes "
                        "only the hottest spot opaque.")
    p.add_argument("--bg-method", choices=["gaussian", "tophat", "global", "none"],
                   default="gaussian",
                   help="background subtraction method applied to the lum image "
                        "before pseudocolor/overlay")
    p.add_argument("--bg-sigma", type=float, default=25.0,
                   help="(gaussian) sigma of the background estimate, in pixels; "
                        "should be larger than the signal spot")
    p.add_argument("--bg-radius", type=int, default=50,
                   help="(tophat) radius of the disk structuring element")
    p.add_argument("--bg-percentile", type=float, default=2.0,
                   help="(global) percentile used as a single background offset")
    args = p.parse_args()

    combine_luminescence(
        bg_path=args.bg,
        lum_path=args.lum,
        out_path=args.out,
        vmin=args.vmin,
        vmax=args.vmax,
        noise_k=args.noise_k,
        alpha_floor=args.alpha_floor,
        alpha_gain=args.alpha_gain,
        bg_method=args.bg_method,
        bg_sigma=args.bg_sigma,
        bg_radius=args.bg_radius,
        bg_percentile=args.bg_percentile,
    )


if __name__ == "__main__":
    main()
