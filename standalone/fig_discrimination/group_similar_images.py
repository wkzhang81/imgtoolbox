#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image similarity grouper based on scikit-image
===================================

Supports recognizing:
- Scaled versions of the same image
- JPEG-recompressed versions
- Brightness, contrast, and slight color changes
- 90, 180, 270 degree rotations
- Arbitrary-angle rotations
- Local crop, center crop, edge crop
- Images with added borders or padding
- Slight perspective transformation
- Local watermark or text overlay

Processing logic:
1. Scan the first-level images in the target directory;
2. Compute multi-orientation, multi-crop perceptual hashes to quickly filter candidate images;
3. Use skimage.feature.SIFT to extract rotation- and scale-invariant features;
4. Use descriptor matching;
5. Sequentially apply RANSAC with similarity, affine and homography transforms;
6. Judge similarity based on inlier count, inlier ratio, spatial distribution and coverage;
7. Cluster using Union-Find;
8. Move similar images into subdirectories such as similar_group_001;
9. Files with no similar images stay in the original directory;
10. Output a CSV report.

Install dependencies:
    pip install numpy scipy scikit-image pillow

Preview first:
    python group_similar_images_final.py "D:/photos" --dry-run

Move for real:
    python group_similar_images_final.py "D:/photos"

Copy instead of moving:
    python group_similar_images_final.py "D:/photos" --copy
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.fft import dctn
from skimage import color, exposure, io, transform
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import (
    AffineTransform,
    ProjectiveTransform,
    SimilarityTransform,
)


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


@dataclass
class HashSignature:
    """One hash signature of an image."""
    name: str
    bits: np.ndarray


@dataclass
class ImageInfo:
    path: Path
    hash_signatures: list[HashSignature]
    keypoints: Optional[np.ndarray]
    descriptors: Optional[np.ndarray]
    processed_shape: Optional[tuple[int, int]]
    original_shape: Optional[tuple[int, int]]
    error: Optional[str] = None


@dataclass
class GeometryResult:
    model_name: str
    inlier_mask: Optional[np.ndarray]
    inliers: int
    median_error: float


@dataclass
class CompareResult:
    first: Path
    second: Path
    hash_distance: int
    hash_pair: str
    raw_matches: int
    inliers: int
    inlier_ratio: float
    normalized_inliers: float
    coverage_first: float
    coverage_second: float
    spatial_spread_first: float
    spatial_spread_second: float
    model_name: str
    estimated_rotation: Optional[float]
    similar: bool
    reason: str


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if self.rank[root_first] < self.rank[root_second]:
            self.parent[root_first] = root_second
        elif self.rank[root_first] > self.rank[root_second]:
            self.parent[root_second] = root_first
        else:
            self.parent[root_second] = root_first
            self.rank[root_first] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify rotation/crop-similar images using multi-orientation hashing, SIFT and RANSAC."
    )

    parser.add_argument("directory", type=Path, help="Directory of images to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not move files")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of moving them")
    parser.add_argument("--group-prefix", default="similar_group")
    parser.add_argument("--report-name", default="similarity_report.csv")

    parser.add_argument(
        "--max-size",
        type=int,
        default=1600,
        help="Max pixels for the long side when processing with SIFT, default 1600"
    )
    parser.add_argument(
        "--phash-size",
        type=int,
        default=16,
        help="Perceptual hash side length, default 16, i.e. 256 bits"
    )
    parser.add_argument(
        "--phash-distance",
        type=int,
        default=100,
        help="Minimum-distance threshold for multi-orientation/multi-crop hashes, default 100"
    )
    parser.add_argument(
        "--no-phash-filter",
        action="store_true",
        help="Disable perceptual-hash prefiltering; run SIFT on all image pairs"
    )
    parser.add_argument(
        "--include-flips",
        action="store_true",
        help="Additionally detect horizontally mirrored images, which increases computation"
    )

    parser.add_argument(
        "--max-ratio",
        type=float,
        default=0.80,
        help="Lowe ratio threshold, default 0.80"
    )
    parser.add_argument(
        "--min-raw-matches",
        type=int,
        default=9,
        help="Minimum number of matches required to enter RANSAC, default 9"
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=10,
        help="Minimum number of inliers required to judge similar, default 10"
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.25,
        help="Minimum inlier ratio, default 0.25"
    )
    parser.add_argument(
        "--min-normalized-inliers",
        type=float,
        default=0.018,
        help="Minimum ratio of inliers to the smaller feature-point count, default 0.018"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.035,
        help="Minimum area coverage of the inlier bounding box in at least one image, default 0.035"
    )
    parser.add_argument(
        "--min-spatial-spread",
        type=float,
        default=0.08,
        help="Minimum ratio of the smaller horizontal/vertical span of matched points in at least one image, default 0.08"
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=5.0,
        help="RANSAC pixel residual threshold, default 5.0"
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=2000,
        help="Maximum number of RANSAC iterations, default 2000"
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.directory.exists():
        raise ValueError(f"Directory does not exist: {args.directory}")
    if not args.directory.is_dir():
        raise ValueError(f"Path is not a directory: {args.directory}")
    if args.max_size < 128:
        raise ValueError("--max-size cannot be less than 128")
    if args.phash_size < 4:
        raise ValueError("--phash-size cannot be less than 4")
    max_bits = args.phash_size ** 2
    if not 0 <= args.phash_distance <= max_bits:
        raise ValueError(f"--phash-distance must be between 0 and {max_bits}")
    if not 0 < args.max_ratio <= 1:
        raise ValueError("--max-ratio must be between 0 and 1")
    if args.min_raw_matches < 4:
        raise ValueError("--min-raw-matches cannot be less than 4")
    if args.min_inliers < 4:
        raise ValueError("--min-inliers cannot be less than 4")
    for name in (
        "min_inlier_ratio",
        "min_normalized_inliers",
        "min_coverage",
        "min_spatial_spread",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.residual_threshold <= 0:
        raise ValueError("--residual-threshold must be greater than 0")
    if args.max_trials <= 0:
        raise ValueError("--max-trials must be greater than 0")


def collect_images(directory: Path) -> list[Path]:
    return sorted(
        (
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower()
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 4:
            image = color.rgba2rgb(image)
            gray = color.rgb2gray(image)
        elif channels == 3:
            gray = color.rgb2gray(image)
        elif channels == 1:
            gray = image[..., 0]
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

    gray = np.asarray(gray, dtype=np.float32)
    gray = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = exposure.rescale_intensity(
        gray,
        in_range="image",
        out_range=(0.0, 1.0)
    ).astype(np.float32)
    return gray


def read_image_gray(path: Path) -> np.ndarray:
    return to_grayscale(io.imread(path))


def resize_for_sift(gray: np.ndarray, max_size: int) -> np.ndarray:
    height, width = gray.shape[:2]
    longest = max(height, width)

    if longest <= max_size:
        return gray.astype(np.float32, copy=False)

    scale = max_size / longest
    target = (
        max(64, int(round(height * scale))),
        max(64, int(round(width * scale)))
    )
    return transform.resize(
        gray,
        target,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)


def safe_crop(
    image: np.ndarray,
    top: float,
    bottom: float,
    left: float,
    right: float
) -> np.ndarray:
    height, width = image.shape
    y1 = max(0, min(height - 1, int(round(height * top))))
    y2 = max(y1 + 1, min(height, int(round(height * bottom))))
    x1 = max(0, min(width - 1, int(round(width * left))))
    x2 = max(x1 + 1, min(width, int(round(width * right))))
    return image[y1:y2, x1:x2]


def compute_phash(gray: np.ndarray, hash_size: int) -> np.ndarray:
    dct_size = hash_size * 4
    small = transform.resize(
        gray,
        (dct_size, dct_size),
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)

    dct_result = dctn(small, type=2, norm="ortho")
    low = dct_result[:hash_size, :hash_size].copy()
    flat = low.ravel()
    median = np.median(flat[1:]) if flat.size > 1 else np.median(flat)
    return low > median


def build_hash_signatures(
    gray: np.ndarray,
    hash_size: int,
    include_flips: bool
) -> list[HashSignature]:
    """
    Generate multi-crop, multi-rotation hashes for the original image.

    The crop templates prevent locally cropped images from being wrongly excluded
    by the pHash prefilter;
    the rotation templates recognize 90/180/270 degree rotations.
    """
    crop_specs = [
        ("full", 0.00, 1.00, 0.00, 1.00),
        ("center90", 0.05, 0.95, 0.05, 0.95),
        ("center80", 0.10, 0.90, 0.10, 0.90),
        ("center70", 0.15, 0.85, 0.15, 0.85),
        ("top85", 0.00, 0.85, 0.00, 1.00),
        ("bottom85", 0.15, 1.00, 0.00, 1.00),
        ("left85", 0.00, 1.00, 0.00, 0.85),
        ("right85", 0.00, 1.00, 0.15, 1.00),
    ]

    signatures: list[HashSignature] = []

    base_variants = [("normal", gray)]
    if include_flips:
        base_variants.append(("hflip", np.fliplr(gray)))

    for base_name, base_image in base_variants:
        for rotation_k, angle in enumerate((0, 90, 180, 270)):
            rotated = np.rot90(base_image, k=rotation_k)
            for crop_name, top, bottom, left, right in crop_specs:
                cropped = safe_crop(rotated, top, bottom, left, right)
                signatures.append(
                    HashSignature(
                        name=f"{base_name}_r{angle}_{crop_name}",
                        bits=compute_phash(cropped, hash_size)
                    )
                )

    return signatures


def best_hash_distance(
    first: ImageInfo,
    second: ImageInfo
) -> tuple[int, str]:
    best_distance = math.inf
    best_pair = ""

    for signature_a in first.hash_signatures:
        for signature_b in second.hash_signatures:
            distance = int(
                np.count_nonzero(signature_a.bits != signature_b.bits)
            )
            if distance < best_distance:
                best_distance = distance
                best_pair = f"{signature_a.name}<->{signature_b.name}"

    if math.isinf(best_distance):
        return 10**9, ""

    return int(best_distance), best_pair


def extract_image_info(
    path: Path,
    max_size: int,
    phash_size: int,
    include_flips: bool
) -> ImageInfo:
    try:
        gray = read_image_gray(path)
        original_shape = gray.shape[:2]
        hash_signatures = build_hash_signatures(
            gray,
            phash_size,
            include_flips
        )

        sift_image = resize_for_sift(gray, max_size)
        sift = SIFT()
        sift.detect_and_extract(sift_image)

        if (
            sift.keypoints is None
            or sift.descriptors is None
            or len(sift.keypoints) < 4
        ):
            return ImageInfo(
                path=path,
                hash_signatures=hash_signatures,
                keypoints=None,
                descriptors=None,
                processed_shape=sift_image.shape[:2],
                original_shape=original_shape,
                error="Fewer than 4 valid SIFT feature points"
            )

        return ImageInfo(
            path=path,
            hash_signatures=hash_signatures,
            keypoints=np.asarray(sift.keypoints, dtype=np.float64),
            descriptors=np.asarray(sift.descriptors),
            processed_shape=sift_image.shape[:2],
            original_shape=original_shape
        )

    except Exception as exc:
        return ImageInfo(
            path=path,
            hash_signatures=[],
            keypoints=None,
            descriptors=None,
            processed_shape=None,
            original_shape=None,
            error=str(exc)
        )


def run_ransac_model(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    model_class,
    model_name: str,
    residual_threshold: float,
    max_trials: int
) -> GeometryResult:
    min_samples = 4 if model_class is ProjectiveTransform else 3
    if model_class is SimilarityTransform:
        min_samples = 2

    if len(source_points) < min_samples:
        return GeometryResult(model_name, None, 0, math.inf)

    kwargs = dict(
        data=(source_points, destination_points),
        model_class=model_class,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials
    )

    try:
        model, inlier_mask = ransac(
            **kwargs,
            rng=np.random.default_rng(42)
        )
    except TypeError:
        model, inlier_mask = ransac(**kwargs)
    except Exception:
        return GeometryResult(model_name, None, 0, math.inf)

    if inlier_mask is None or model is None:
        return GeometryResult(model_name, None, 0, math.inf)

    inlier_count = int(np.sum(inlier_mask))
    if inlier_count == 0:
        return GeometryResult(model_name, inlier_mask, 0, math.inf)

    try:
        predicted = model(source_points[inlier_mask])
        errors = np.linalg.norm(
            predicted - destination_points[inlier_mask],
            axis=1
        )
        median_error = float(np.median(errors))
    except Exception:
        median_error = math.inf

    return GeometryResult(
        model_name=model_name,
        inlier_mask=inlier_mask,
        inliers=inlier_count,
        median_error=median_error
    )


def choose_best_geometry(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    residual_threshold: float,
    max_trials: int
) -> GeometryResult:
    """
    The similarity transform suits rotation + scaling;
    the affine transform suits rotation + scaling + shearing;
    the homography suits perspective and more complex cropping relations.
    """
    candidates = [
        run_ransac_model(
            source_points, destination_points,
            SimilarityTransform, "similarity",
            residual_threshold, max_trials
        ),
        run_ransac_model(
            source_points, destination_points,
            AffineTransform, "affine",
            residual_threshold, max_trials
        ),
        run_ransac_model(
            source_points, destination_points,
            ProjectiveTransform, "projective",
            residual_threshold, max_trials
        ),
    ]

    return max(
        candidates,
        key=lambda item: (
            item.inliers,
            -item.median_error
        )
    )


def bounding_box_coverage(
    points_xy: np.ndarray,
    image_shape: tuple[int, int]
) -> float:
    if len(points_xy) < 2:
        return 0.0

    height, width = image_shape
    x_span = float(np.ptp(points_xy[:, 0]))
    y_span = float(np.ptp(points_xy[:, 1]))
    return (x_span * y_span) / max(float(width * height), 1.0)


def spatial_spread(
    points_xy: np.ndarray,
    image_shape: tuple[int, int]
) -> float:
    if len(points_xy) < 2:
        return 0.0

    height, width = image_shape
    x_ratio = float(np.ptp(points_xy[:, 0])) / max(width, 1)
    y_ratio = float(np.ptp(points_xy[:, 1])) / max(height, 1)
    return min(x_ratio, y_ratio)


def estimate_rotation_degrees(
    source_points: np.ndarray,
    destination_points: np.ndarray
) -> Optional[float]:
    if len(source_points) < 2:
        return None

    try:
        model = SimilarityTransform()
        success = model.estimate(source_points, destination_points)
        if not success:
            return None

        angle = math.degrees(model.rotation)
        angle = ((angle + 180.0) % 360.0) - 180.0
        return float(angle)
    except Exception:
        return None


def compare_images(
    first: ImageInfo,
    second: ImageInfo,
    hash_distance: int,
    hash_pair: str,
    args: argparse.Namespace
) -> CompareResult:
    empty = dict(
        first=first.path,
        second=second.path,
        hash_distance=hash_distance,
        hash_pair=hash_pair,
        raw_matches=0,
        inliers=0,
        inlier_ratio=0.0,
        normalized_inliers=0.0,
        coverage_first=0.0,
        coverage_second=0.0,
        spatial_spread_first=0.0,
        spatial_spread_second=0.0,
        model_name="",
        estimated_rotation=None,
        similar=False
    )

    if (
        first.keypoints is None
        or first.descriptors is None
        or second.keypoints is None
        or second.descriptors is None
        or first.processed_shape is None
        or second.processed_shape is None
    ):
        return CompareResult(
            **empty,
            reason="At least one image lacks enough SIFT features"
        )

    try:
        matches = match_descriptors(
            first.descriptors,
            second.descriptors,
            metric="euclidean",
            cross_check=True,
            max_ratio=args.max_ratio
        )
    except Exception as exc:
        return CompareResult(
            **empty,
            reason=f"Descriptor matching failed: {exc}"
        )

    raw_matches = int(len(matches))
    if raw_matches < args.min_raw_matches:
        return CompareResult(
            **{**empty, "raw_matches": raw_matches},
            reason=(
                f"Insufficient raw match count: "
                f"{raw_matches} < {args.min_raw_matches}"
            )
        )

    source_points = first.keypoints[matches[:, 0]][:, ::-1]
    destination_points = second.keypoints[matches[:, 1]][:, ::-1]

    geometry = choose_best_geometry(
        source_points,
        destination_points,
        args.residual_threshold,
        args.max_trials
    )

    if geometry.inlier_mask is None:
        return CompareResult(
            **{**empty, "raw_matches": raw_matches},
            reason="All RANSAC geometry models failed"
        )

    inlier_mask = geometry.inlier_mask
    inliers = geometry.inliers
    inlier_ratio = inliers / max(raw_matches, 1)
    normalized_inliers = inliers / max(
        min(len(first.descriptors), len(second.descriptors)),
        1
    )

    source_inliers = source_points[inlier_mask]
    destination_inliers = destination_points[inlier_mask]

    coverage_first = bounding_box_coverage(
        source_inliers,
        first.processed_shape
    )
    coverage_second = bounding_box_coverage(
        destination_inliers,
        second.processed_shape
    )
    spread_first = spatial_spread(
        source_inliers,
        first.processed_shape
    )
    spread_second = spatial_spread(
        destination_inliers,
        second.processed_shape
    )

    rotation = estimate_rotation_degrees(
        source_inliers,
        destination_inliers
    )

    # Cropped images often show: small coverage in the original but large coverage in the crop.
    # Therefore coverage and spatial spread only require at least one image to reach the threshold.
    coverage_ok = max(coverage_first, coverage_second) >= args.min_coverage
    spread_ok = max(spread_first, spread_second) >= args.min_spatial_spread

    similar = (
        inliers >= args.min_inliers
        and inlier_ratio >= args.min_inlier_ratio
        and normalized_inliers >= args.min_normalized_inliers
        and coverage_ok
        and spread_ok
    )

    rotation_text = (
        "unknown" if rotation is None else f"{rotation:.1f}°"
    )

    reason = (
        f"{'Pass' if similar else 'Fail'}: "
        f"model={geometry.model_name}, "
        f"inliers={inliers}/{args.min_inliers}, "
        f"inlier ratio={inlier_ratio:.3f}/{args.min_inlier_ratio:.3f}, "
        f"normalized inliers={normalized_inliers:.3f}/"
        f"{args.min_normalized_inliers:.3f}, "
        f"coverage={coverage_first:.3f},{coverage_second:.3f}, "
        f"spread={spread_first:.3f},{spread_second:.3f}, "
        f"estimated rotation={rotation_text}"
    )

    return CompareResult(
        first=first.path,
        second=second.path,
        hash_distance=hash_distance,
        hash_pair=hash_pair,
        raw_matches=raw_matches,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        normalized_inliers=normalized_inliers,
        coverage_first=coverage_first,
        coverage_second=coverage_second,
        spatial_spread_first=spread_first,
        spatial_spread_second=spread_second,
        model_name=geometry.model_name,
        estimated_rotation=rotation,
        similar=similar,
        reason=reason
    )


def generate_candidate_pairs(
    image_infos: list[ImageInfo],
    args: argparse.Namespace
) -> list[tuple[int, int, int, str]]:
    candidates: list[tuple[int, int, int, str]] = []
    total_pairs = len(image_infos) * (len(image_infos) - 1) // 2
    checked = 0

    print("\nStarting multi-orientation, multi-crop pHash candidate filtering:")

    for first_index in range(len(image_infos)):
        for second_index in range(first_index + 1, len(image_infos)):
            checked += 1
            first = image_infos[first_index]
            second = image_infos[second_index]

            if args.no_phash_filter:
                candidates.append(
                    (first_index, second_index, -1, "filter_disabled")
                )
                continue

            if not first.hash_signatures or not second.hash_signatures:
                continue

            distance, pair_name = best_hash_distance(first, second)

            if distance <= args.phash_distance:
                candidates.append(
                    (first_index, second_index, distance, pair_name)
                )

            if checked % 100 == 0 or checked == total_pairs:
                print(
                    f"\r  Filtered {checked}/{total_pairs} pairs, "
                    f"{len(candidates)} candidate pairs",
                    end="",
                    flush=True
                )

    if total_pairs:
        print()

    return candidates


def build_groups(
    image_infos: list[ImageInfo],
    candidates: list[tuple[int, int, int, str]],
    args: argparse.Namespace
) -> tuple[list[list[int]], list[CompareResult]]:
    union_find = UnionFind(len(image_infos))
    results: list[CompareResult] = []

    print(f"\nEntering SIFT fine matching: {len(candidates)} pairs")

    for number, (i, j, distance, hash_pair) in enumerate(
        candidates,
        start=1
    ):
        result = compare_images(
            image_infos[i],
            image_infos[j],
            distance,
            hash_pair,
            args
        )
        results.append(result)

        rotation_text = (
            "unknown"
            if result.estimated_rotation is None
            else f"{result.estimated_rotation:.1f}°"
        )
        print(
            f"[{number:>5}/{len(candidates)}] "
            f"{result.first.name} <-> {result.second.name}\n"
            f"    pHash={result.hash_distance}, "
            f"matches={result.raw_matches}, inliers={result.inliers}, "
            f"model={result.model_name or '-'}, "
            f"rotation={rotation_text}, "
            f"result={'similar' if result.similar else 'not similar'}"
        )

        if result.similar:
            union_find.union(i, j)

    groups_by_root: dict[int, list[int]] = {}
    for index in range(len(image_infos)):
        root = union_find.find(index)
        groups_by_root.setdefault(root, []).append(index)

    groups = [
        group for group in groups_by_root.values()
        if len(group) >= 2
    ]
    groups.sort(
        key=lambda group: (
            -len(group),
            image_infos[group[0]].path.name.lower()
        )
    )
    return groups, results


def make_unique_path(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination

    path = Path(filename)
    counter = 1
    while True:
        candidate = directory / f"{path.stem}_{counter}{path.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_or_copy_groups(
    image_infos: list[ImageInfo],
    groups: list[list[int]],
    root: Path,
    args: argparse.Namespace
) -> None:
    if not groups:
        print("\nNo similar image groups found.")
        return

    action = "copy" if args.copy else "move"
    print(f"\nFound {len(groups)} similar image groups:")

    for group_number, group in enumerate(groups, start=1):
        group_name = f"{args.group_prefix}_{group_number:03d}"
        group_directory = root / group_name

        print(f"\n{group_name}: {len(group)} images")
        for index in group:
            print(f"  - {image_infos[index].path.name}")

        if args.dry_run:
            print(f"  [preview mode] not performing {action}")
            continue

        group_directory.mkdir(parents=True, exist_ok=True)

        for index in group:
            source = image_infos[index].path
            if not source.exists():
                print(f"  Skipping nonexistent file: {source}")
                continue

            destination = make_unique_path(group_directory, source.name)
            try:
                if args.copy:
                    shutil.copy2(source, destination)
                else:
                    shutil.move(str(source), str(destination))
                print(
                    f"  {action}ed: {source.name} "
                    f"-> {group_name}/{destination.name}"
                )
            except Exception as exc:
                print(f"  {action} failed: {source.name}, reason: {exc}")


def write_report(path: Path, results: list[CompareResult]) -> None:
    try:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "image1", "image2",
                "min pHash distance", "best hash combo",
                "SIFT matches", "RANSAC inliers",
                "inlier ratio", "normalized inlier ratio",
                "image1 coverage", "image2 coverage",
                "image1 spatial spread", "image2 spatial spread",
                "best geometry model", "estimated rotation angle",
                "similar", "description"
            ])

            for result in results:
                writer.writerow([
                    result.first.name,
                    result.second.name,
                    result.hash_distance,
                    result.hash_pair,
                    result.raw_matches,
                    result.inliers,
                    f"{result.inlier_ratio:.6f}",
                    f"{result.normalized_inliers:.6f}",
                    f"{result.coverage_first:.6f}",
                    f"{result.coverage_second:.6f}",
                    f"{result.spatial_spread_first:.6f}",
                    f"{result.spatial_spread_second:.6f}",
                    result.model_name,
                    (
                        ""
                        if result.estimated_rotation is None
                        else f"{result.estimated_rotation:.6f}"
                    ),
                    "yes" if result.similar else "no",
                    result.reason
                ])

        print(f"\nDetection report saved: {path}")
    except Exception as exc:
        print(f"\nWarning: failed to save CSV report: {exc}")


def process(args: argparse.Namespace) -> dict:
    """Run the entire grouping pipeline (assumes args has passed validate_args).

    Returns a result-summary dict for reuse by main() and the imgtoolbox adapter layer.
    """
    start = time.perf_counter()

    root = args.directory.resolve()
    image_paths = collect_images(root)

    print("=" * 72)
    print("Automatic grouping tool for rotation/crop-similar images")
    print("=" * 72)
    print(f"Directory: {root}")
    print(f"Image count: {len(image_paths)}")
    print(
        f"Mode: "
        f"{'preview' if args.dry_run else ('copy' if args.copy else 'move')}"
    )
    print(f"Mirror detection: {'on' if args.include_flips else 'off'}")

    if len(image_paths) < 2:
        print("Fewer than 2 images, nothing to process.")
        return {"directory": root, "n_images": len(image_paths),
                "n_groups": 0, "n_grouped": 0, "report_path": None,
                "elapsed": 0.0}

    image_infos: list[ImageInfo] = []

    print("\nStarting multi-orientation hash and SIFT feature extraction:")
    for index, image_path in enumerate(image_paths, start=1):
        info = extract_image_info(
            image_path,
            args.max_size,
            args.phash_size,
            args.include_flips
        )
        image_infos.append(info)

        if info.error is None:
            print(
                f"[{index:>4}/{len(image_paths)}] "
                f"{image_path.name}: "
                f"{len(info.keypoints)} SIFT features, "
                f"{len(info.hash_signatures)} hash signatures"
            )
        else:
            print(
                f"[{index:>4}/{len(image_paths)}] "
                f"{image_path.name}: {info.error}"
            )

    total_pairs = len(image_infos) * (len(image_infos) - 1) // 2
    candidates = generate_candidate_pairs(image_infos, args)

    print("\nCandidate filtering result:")
    print(f"  Total image pairs: {total_pairs}")
    print(f"  Entering SIFT verification: {len(candidates)}")
    print(f"  Skipped: {total_pairs - len(candidates)}")

    groups, results = build_groups(image_infos, candidates, args)

    report_path = root / args.report_name
    write_report(report_path, results)

    grouped_indices = {
        index for group in groups for index in group
    }

    print("\nGrouping summary:")
    print(f"  Similar group count: {len(groups)}")
    print(f"  Similar image count: {len(grouped_indices)}")
    print(f"  Images with no similar match: {len(image_infos) - len(grouped_indices)}")

    move_or_copy_groups(image_infos, groups, root, args)

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 72)
    print(f"Processing complete, total time: {elapsed:.2f}s")

    if args.dry_run:
        print("Currently in preview mode; no files were moved or copied.")
        print("After confirming the result, remove the --dry-run argument and run again.")
    elif args.copy:
        print("Similar images have been copied into group directories; the originals are kept.")
    else:
        print("Similar images have been moved into group directories; images with no similar match stay in the original directory.")

    print("=" * 72)

    return {
        "directory": root,
        "n_images": len(image_paths),
        "n_groups": len(groups),
        "n_grouped": len(grouped_indices),
        "report_path": report_path,
        "elapsed": elapsed,
    }


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 1

    process(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
