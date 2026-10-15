# imgtoolbox

A scientific image processing toolbox based on **scikit-image**. Eight commonly used image-analysis scripts are packaged into one Python package, with a tkinter GUI on top: pick a tool → fill in parameters → run → watch the log.

---

## Overview

### What it does

| Category | Tool | In one sentence |
|---|---|---|
| Cell analysis | `cells_morph` | Phase-contrast cell segmentation + KMeans clustering on morphological features |
| Cell analysis | `cells_resnet` | Same segmentation, but outputs 1024-dim ResNet50 features per cell |
| Image stitching | `fig_stitch` | Multiple overlapping photos → panorama |
| Image dedup | `fig_discrimination` | Auto-groups similar images (resized / rotated / cropped / recompressed) within a directory |
| Image dedup | `find_dup` | Detects repeated regions inside a single image |
| Pathology staining | `ihc` | IHC slide HED three-channel separation + DAB quantification |
| Pathology staining | `stain_norm` | Stain normalization for H&E / IHC / fluorescence images |
| Animal imaging | `lum_combine` | Brightfield + chemiluminescence TIF three-in-one overlay image |

### Directory structure

```
imgtoolbox/
├── imgtoolbox/        # the package itself (do not edit directly)
├── standalone/        # the 8 original scripts (loaded in place by the package; do not delete)
├── samples/           # 39 sample input images (subdirectories per tool)
├── outputs/           # run outputs (created automatically)
├── run_app.py         # GUI entry point
└── requirements.txt
```

### Quick start (managed with uv)

The project ships with a `pyproject.toml`; install and run it with uv:

```powershell
cd imgtoolbox

# One-time: create the venv (inherits large packages such as torch xpu already installed in the system Python 3.12) + editable install
# Replace --python with your local Python 3.12 path if uv cannot find it.
uv venv --python 3.12 --system-site-packages .venv
uv pip install -e .

# Starting the GUI afterwards (any one of these)
imgtoolbox.exe                       # .venv\Scripts\ is already in PATH
uv run imgtoolbox                    # equivalent
.venv\Scripts\imgtoolbox.exe         # full path
```

> **About torch**: this project's torch is `2.14.0+xpu` (Intel XPU), which the default PyPI install cannot get. The venv uses `--system-site-packages` to inherit the version already installed in the system Python 3.12 — **do not** reinstall torch inside the venv. The core dependencies (numpy/scipy/pandas/skimage/matplotlib/imageio) are resolved automatically by uv.

GUI workflow: pick a tool from the dropdown at the top → fill in the parameters in the middle (click "Browse…" for files/directories) → click "Run" → the log at the bottom scrolls in real time. Processing runs on a background thread, so the UI never freezes. All sample input images live under `samples/`.

---

## 1. cells_morph — phase-contrast cell segmentation + morphological clustering

### Principle

Targets high-density phase-contrast / brightfield cell images. The full pipeline:

1. **Preprocessing**: large-scale Gaussian to estimate non-uniform illumination → high-pass background correction → small-scale Gaussian denoising → CLAHE to enhance local contrast.
2. **Coverage-area estimation**: weighted score from local grayscale std (texture) + Scharr gradient density (edges), binarized with an Otsu threshold, morphological open/close to remove the empty gaps of the culture dish.
3. **Marker-controlled watershed**: Scharr gradient → Gaussian smoothing → h-minima to find seeds → `watershed` segmentation within the coverage area. An area filter removes blobs that are too small / too large.
4. **Feature extraction**: `regionprops` for morphological features such as area, perimeter, circularity, aspect ratio, eccentricity, solidity + mean/std grayscale.
5. **Clustering**: log1p on skewed features → RobustScaler → PCA → KMeans, evaluated with the silhouette score.

Outputs: segmentation overlay, clustering overlay, feature CSV, coverage mask.

### GUI Parameters

| Parameter | Description |
|---|---|
| Input cell image | Phase-contrast / brightfield grayscale image (jpg/png/tiff) |
| Output directory | Leave empty to use `outputs/cells_morph` |
| Number of clusters | 2–20, default 5 |

---

## 2. cells_resnet — cell segmentation + ResNet50 features

### Principle

The first half (preprocessing / coverage area / watershed) is identical to `cells_morph`. The difference is in feature extraction:

- For each cell, take its bbox, use the mask to fill the background with that cell's median value, pad to a square, replicate into 3 channels.
- Pass through the global average pooling layer after **ResNet50 layer3**, producing a **1024-dim** deep feature vector.
- Output CSV: label, centroid, 1024-dim features.

Suited for downstream cell-subtype classification / retrieval with deep features instead of handcrafted morphology.

### GUI Parameters

| Parameter | Description |
|---|---|
| Input cell image | Same as above |
| Output directory | Leave empty to use `outputs/cells_resnet` |
| batch_size | Inference batch size, default 32 |
| crop_padding | Pixels to expand the cell bbox outward, default 8 |
| device | auto / cpu / cuda / xpu |

---

## 3. fig_discrimination — similar-image grouping

### Principle

Recognizes "the various variants of one image": rescaling, JPEG recompression, brightness/contrast changes, 90° rotations, arbitrary-angle rotations, cropping, added borders, slight perspective, local watermarks.

Two-phase approach:

1. **pHash pre-filter**: for each image, take 0/90/180/270 rotations × 8 crops (full/center/four edges), computing a DCT pHash (256 bits) for each. For a pair of images take the minimum Hamming distance over all combinations; only pairs below the threshold proceed to the next step — this avoids O(n²) all-pairs SIFT.
2. **SIFT precise matching**: SIFT features → cross-check + Lowe ratio matching → RANSAC tries Similarity / Affine / Projective in turn → verdict from inlier count, inlier ratio, normalized inliers, feature-point bounding-box coverage, and spatial spread.
3. **Union-find** links similar images into groups, moves/copies them into `similar_group_001/...`, and writes a CSV report.

### GUI Parameters (common)

| Parameter | Default | Description |
|---|---|---|
| Directory to process | — | Images at the first level (non-recursive) |
| Preview only | ✓ | Dry-run first to check the grouping, then uncheck to run for real |
| Copy | ✗ | Otherwise the files are moved |
| Also detect horizontal mirror | ✗ | Doubles the computation |
| pHash distance threshold | 100 | Smaller = stricter |
| Lowe ratio | 0.80 | Matching looseness |
| Min inliers | 10 | RANSAC inlier lower bound |

The remaining advanced thresholds (min_coverage / residual_threshold etc.) have defaults and generally don't need to be touched.

---

## 4. fig_stitch — SIFT panorama stitching

### Principle

Stitches multiple overlapping microscope images / photos into one panorama:

1. Extract SIFT features, pairwise `match_descriptors`.
2. Pick one of three transform models:
   - `translation` (default): estimates only the median displacement, the most stable, suited to microscope scan images;
   - `similarity`: adds rotation and uniform scaling;
   - `projective`: full homography, suited to real perspective photos.
3. Take the largest connected component and pick the reference image by cumulative match score.
4. Dijkstra finds the optimal path from each image to the reference, accumulating homographies along the way — avoiding direct-pair error accumulation.
5. Compute the panorama canvas size, blend with **feather weights** (distance transform: high weight in the center, low at the edges), then crop the empty borders.

### GUI Parameters

| Parameter | Default | Description |
|---|---|---|
| Multiple images | — | Ctrl-click for multi-select |
| Output path | — | .png recommended |
| Transform model | translation | Pick translation for microscope scans |
| Lowe ratio | 0.75 | |
| Min inliers | 10 | |
| RANSAC residual (px) | 3.0 | |

---

## 5. find_dup — repeated-region detection within a single image

### Principle

Detects patches that appear twice within the same image (repeated collage tiles, copy-paste, watermarks, recurring layout elements).

The trick: split the SIFT keypoints **in half** into groups A/B and have A match against B — breaking the "matching yourself" deadlock. Then drop pairs whose spatial distance is too small (consecutive points along the same texture edge); what remains are genuine cross-region repeats.

Output: a side-by-side two-panel comparison image (left: original; right: original + red lines for every repeated pair).

### GUI Parameters

| Parameter | Default | Description |
|---|---|---|
| Input image | — | |
| Lowe ratio | 0.5 | Raising it to 0.8 detects more |
| Min spatial distance (px) | 20 | Pairs closer than this are treated as adjacent texture and discarded |

---

## 6. ihc — IHC HED separation + DAB quantification

### Principle

An IHC slide is RGB, but DAB (brown positive signal), Hematoxylin (blue-purple nuclei) and Eosin (pink cytoplasm) are mixed together in RGB space.

1. `rgb2hed` converts to the HED color space (a deconvolution transform tailored to stains).
2. Rebuild each channel's monochrome image separately with `hed2rgb`:
   - `hematoxylin_*.jpg` — nuclear stain distribution
   - `eosin_*.jpg` — cytoplasm distribution
   - `dab_*.jpg` — positive signal
3. Sum all pixel values of the DAB channel as the per-image quantitative metric, written to `quantify_result.csv`.

### GUI Parameters

| Parameter | Description |
|---|---|
| Input directory | A batch of IHC jpg/png/tiff |
| Output directory | Three separated channel images + CSV |

---

## 7. stain_norm — stain / fluorescence normalization

### Principle

H&E, IHC and fluorescence images captured by different scanner/microscope batches differ a lot in color; analyzing them directly introduces batch effects.

- **Automatic mode detection**: dark background + sparse bright structures → fluorescence; bright background → brightfield.
- **Brightfield**:
  1. flatfield Gaussian illumination correction;
  2. foreground mask (tissue / positive regions, excluding the white background);
  3. robust Reinhard normalization in **Lab space**: estimate the scale from the median + IQR of foreground pixels, strongly match the a/b channels to the reference image, and match the L channel more weakly (protecting morphology and density differences);
  4. blend only within the foreground; the background keeps its original colors.
- **Fluorescence**:
  1. per-channel robust percentiles (p1 / p50 / p99.5);
  2. background subtraction + dynamic-range matching;
  3. light gamma to align channel medians.
- **Reference image**: a single image, a whole batch (its median is taken), or leave it empty to use this batch's median as the target.

### GUI Parameters

| Parameter | Default | Description |
|---|---|---|
| Input/output directories | — | |
| Mode | auto | Can also force brightfield / fluorescence |
| Reference image | empty | Single reference |
| Reference directory | empty | Batch reference (mutually exclusive with the reference image) |
| Illumination correction strength | 0.35 | Brightfield |
| a/b color strength | 0.85 | Brightfield |
| L strength | 0.75 | Brightfield |
| Fluorescence gamma strength | 0.35 | Fluorescence |

---

## 8. lum_combine — brightfield + chemiluminescence three-in-one overlay

### Principle

Chemiluminescence imaging of animals / tissues: the brightfield TIF provides the anatomical structure, the luminescence TIF provides the signal hotspots.

1. Read both TIFs as grayscale; **invert** the luminescence image (chemiluminescence is usually a bright signal on a dark background; invert it before processing).
2. If the sizes differ, resize to align them.
3. **Background subtraction** (luminescence image), pick one of four:
   - `gaussian` (default): Gaussian rolling-ball;
   - `tophat`: white tophat morphology, preserves small bright dots;
   - `global`: global low percentile;
   - `none`: no subtraction.
4. **Robust noise estimation**: MAD (median absolute deviation) estimates the noise sigma, `vmin = background median + k·sigma`; pixels below this are mapped to pure black (the under color of the turbo colormap).
5. alpha is driven by luminescence intensity: weak signals transparent, strong signals opaque.
6. Output three panels: turbo overlaid on the brightfield grayscale / brightfield grayscale / turbo alone + colorbar.

### GUI Parameters

| Parameter | Default | Description |
|---|---|---|
| Brightfield background TIF | — | |
| Chemiluminescence TIF | — | |
| Output PNG | — | |
| Background subtraction method | gaussian | |
| vmin / vmax | auto | Leave empty to estimate automatically; to compare colors across images, fix vmax manually |
| noise_k | 2.5 | vmin = median + k·sigma |
| alpha_floor / alpha_gain | 0.0 / 0.5 | Controls weak-signal transparency |

---

## Using as a library

```python
from imgtoolbox import list_tools
from imgtoolbox.registry import REGISTRY

list_tools()                       # list all 8 tools
tool = REGISTRY["ihc"]             # get by key
summary = tool.run({
    "input_dir": r"samples\ihc",
    "output_dir": r"outputs\ihc",
})
print(summary)
```

Each tool's parameter schema lives in `tool.params` (the `Param` dataclass), type ∈ `file / files / dir / int / float / str / bool / choice`.

## Dependencies

See `requirements.txt`. The local Python 3.12 already has `scikit-image 0.25.2 / torch 2.14.0+xpu` installed. `cells_resnet` requires torch/torchvision; the other 7 tools run on plain skimage.
