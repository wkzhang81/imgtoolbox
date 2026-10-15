"""HED color separation + DAB quantification for IHC images.

Can be run as a script:
    python ihc.py --input <directory> --output <directory>
Can also be imported by imgtoolbox to call process_ihc().
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import skimage.io as io
from skimage.color import rgb2hed, hed2rgb

PATTERNS = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp')


def collect_images(input_dir):
    """List the supported images in the first level of a directory (case-insensitive)."""
    files = set()
    for pattern in PATTERNS:
        files.update(glob.glob(os.path.join(input_dir, pattern)))
        files.update(glob.glob(os.path.join(input_dir, pattern.upper())))
    return sorted(files)


def to_rgb(img):
    """Grayscale / RGBA -> RGB; rgb2hed only accepts 3 channels."""
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    return img[..., :3]


def process_ihc(input_dir, output_dir):
    """Separate the three HED channels and compute DAB intensity; returns the result list."""
    os.makedirs(output_dir, exist_ok=True)

    all_images = collect_images(input_dir)
    if not all_images:
        raise ValueError(f"No supported images found under {input_dir}.")

    quant = []
    for img_path in all_images:
        fname = os.path.basename(img_path)
        raw = io.imread(img_path)
        ihc_rgb = to_rgb(raw)
        if raw.dtype.kind in 'iu':
            ihc_rgb = ihc_rgb.astype('float64') / 255.0
        ihc_hed = rgb2hed(ihc_rgb)

        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
        ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
        ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

        io.imsave(os.path.join(output_dir, 'hematoxylin_' + fname),
                  np.clip(ihc_h * 255, 0, 255).astype('uint8'),
                  check_contrast=False)
        io.imsave(os.path.join(output_dir, 'eosin_' + fname),
                  np.clip(ihc_e * 255, 0, 255).astype('uint8'),
                  check_contrast=False)
        io.imsave(os.path.join(output_dir, 'dab_' + fname),
                  np.clip(ihc_d * 255, 0, 255).astype('uint8'),
                  check_contrast=False)

        quant.append({
            'filename': fname,
            'DAB_intensity': float(ihc_hed[:, :, 2].sum()),
        })
        print(f"[{len(quant)}/{len(all_images)}] {fname} "
              f"DAB={quant[-1]['DAB_intensity']:.1f}")

    csv_path = os.path.join(output_dir, 'quantify_result.csv')
    pd.DataFrame(quant).to_csv(csv_path, index=False)
    return quant, csv_path


def main():
    parser = argparse.ArgumentParser(description='Color separation and DAB quantification for IHC images')
    parser.add_argument('--input', required=True, help='Directory containing the input images')
    parser.add_argument('--output', required=True, help='Output directory for the results')
    args = parser.parse_args()

    quant, csv_path = process_ihc(args.input, args.output)
    print(f"Processed {len(quant)} images in total; quantification table: {csv_path}")


if __name__ == '__main__':
    main()
