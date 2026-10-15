# Image Similarity Grouper

A high-performance image similarity grouper built on scikit-image, able to recognize similar images after a variety of transforms including rotation, cropping, compression and watermarks.

## Features

Supports recognizing the following types of similar images:

- Scaled versions of the same image
- JPEG-recompressed versions
- Brightness, contrast and slight color changes
- 90°, 180°, 270° rotations
- Arbitrary-angle rotations
- Local crop, center crop, edge crop
- Images with added borders or padding
- Slight perspective transformation
- Local watermark or text overlay
- Horizontal mirror (opt-in)

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

or install manually:

```bash
pip install numpy scipy scikit-image pillow
```

### Usage

**Preview mode (recommended to run first)**:
```bash
python group_similar_images.py "path/to/photos" --dry-run
```

**Move for real**:
```bash
python group_similar_images.py "path/to/photos"
```

**Copy instead of move**:
```bash
python group_similar_images.py "path/to/photos" --copy
```

**Enable mirror detection**:
```bash
python group_similar_images.py "path/to/photos" --include-flips
```

## Output

After running, the following are generated in the target directory:

- `similar_group_001/`, `similar_group_002/`, ... — similar-image group directories
- `similarity_report.csv` — detailed similarity detection report
- Ungrouped images are left in the original directory

## Algorithm

```
1. Scan the first-level images in the target directory
2. Compute multi-orientation, multi-crop perceptual hashes (pHash) to quickly filter candidate images
3. Use SIFT to extract rotation- and scale-invariant feature points
4. Use descriptor matching (Lowe ratio test + cross-check)
5. Sequentially apply RANSAC with similarity, affine and homography transforms
6. Judge based on inlier count, inlier ratio, spatial distribution and coverage
7. Cluster into groups using Union-Find
8. Move/copy similar images into group subdirectories
9. Output a detailed CSV report
```

### Two-stage filtering strategy

1. **Prefilter stage**: multi-orientation + multi-crop pHash, quickly ruling out clearly dissimilar image pairs
2. **Fine-matching stage**: SIFT features + RANSAC geometric verification, ensuring high accuracy

## Parameters

### Basic parameters

| Parameter | Default | Description |
|------|--------|------|
| `directory` | - | Directory of images to process (required) |
| `--dry-run` | False | Preview only, do not move files |
| `--copy` | False | Copy images instead of moving them |
| `--group-prefix` | similar_group | Group directory prefix |
| `--report-name` | similarity_report.csv | Report file name |

### Performance parameters

| Parameter | Default | Description |
|------|--------|------|
| `--max-size` | 1600 | Max pixels for the long side when processing with SIFT |
| `--phash-size` | 16 | Perceptual hash side length (256 bits) |
| `--phash-distance` | 100 | Minimum-distance threshold for multi-orientation hashes |
| `--no-phash-filter` | False | Disable perceptual-hash prefiltering |
| `--include-flips` | False | Additionally detect horizontally mirrored images |

### Matching thresholds

| Parameter | Default | Description |
|------|--------|------|
| `--max-ratio` | 0.80 | Lowe ratio threshold |
| `--min-raw-matches` | 9 | Minimum number of matches required to enter RANSAC |
| `--min-inliers` | 10 | Minimum number of inliers required to judge similar |
| `--min-inlier-ratio` | 0.25 | Minimum inlier ratio |
| `--min-normalized-inliers` | 0.018 | Minimum ratio of inliers to the smaller feature-point count |
| `--min-coverage` | 0.035 | Minimum area coverage of the inlier bounding box |
| `--min-spatial-spread` | 0.08 | Minimum ratio of the smaller horizontal/vertical span of matched points |

### RANSAC parameters

| Parameter | Default | Description |
|------|--------|------|
| `--residual-threshold` | 5.0 | RANSAC pixel residual threshold |
| `--max-trials` | 2000 | Maximum number of RANSAC iterations |

## Supported image formats

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tif`, `.tiff`
- `.webp`

## Examples

The project includes sample images and run results:

- `examples/` — 9 test images (containing various transforms)
- `examples_result/` — example run result (4 similar groups)

Run the example:

```bash
python group_similar_images.py examples --dry-run
```

## CSV report fields

| Field | Description |
|------|------|
| image1 / image2 | The two image file names being compared |
| min pHash distance | The minimum Hamming distance among the multi-orientation hashes |
| best hash combo | The hash-variant combination that achieved the minimum distance |
| SIFT matches | The number of raw feature-point matches |
| RANSAC inliers | The number of geometric-model inliers |
| inlier ratio | The ratio of inliers to the total match count |
| normalized inlier ratio | The ratio of inliers to the smaller feature-point count |
| image1 coverage / image2 coverage | The ratio of the inlier bounding box to the image area |
| image1 spatial spread / image2 spatial spread | The spatial-distribution span of matched points |
| best geometry model | similarity / affine / projective |
| estimated rotation angle | The estimated image rotation angle |
| similar | The final decision |
| description | The detailed decision basis |

## License

MIT License
