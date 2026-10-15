from __future__ import annotations

import argparse
import glob
import heapq
import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import color, exposure, img_as_float
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp


@dataclass
class ImageFeatures:
    """image and SIFT features"""

    path: str
    image: np.ndarray
    gray: np.ndarray
    keypoints_rc: np.ndarray
    descriptors: np.ndarray


@dataclass
class PairwiseMatch:
    """
    fitness between two images
    transform_j_to_i:
        coord in image j -> coord in image i
    """

    i: int
    j: int
    transform_j_to_i: ProjectiveTransform | SimilarityTransform
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    median_error: float
    score: float


def read_rgb_image(path: str) -> np.ndarray:
    """Read the image and transform into RGB float numpy array in the range of [0, 1]"""

    image = iio.imread(path)

    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        elif image.shape[2] > 3:
            image = image[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}, file: {path}")

    return img_as_float(image).astype(np.float32)


def extract_sift_features(
    path: str,
    upsampling: int = 1,
) -> ImageFeatures:
    """Read the image and subtract SIFT features"""

    image = read_rgb_image(path)
    gray = color.rgb2gray(image)

    # Contrast stretching may improve keypoint detection in low-contrast images.
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2:
        gray_for_sift = exposure.rescale_intensity(
            gray,
            in_range=(p2, p98),
        )
    else:
        gray_for_sift = gray

    detector = SIFT(upsampling=upsampling)
    detector.detect_and_extract(gray_for_sift)

    return ImageFeatures(
        path=path,
        image=image,
        gray=gray,
        keypoints_rc=detector.keypoints.copy(),
        descriptors=detector.descriptors.copy(),
    )


def rc_to_xy(points_rc: np.ndarray) -> np.ndarray:
    """
    transform the skimage keypoint coord from (row, col) into (x, y).
    """

    return points_rc[:, ::-1].astype(np.float64)


def estimate_pairwise_transform(
    feature_i: ImageFeatures,
    feature_j: ImageFeatures,
    i: int,
    j: int,
    max_ratio: float = 0.75,
    min_matches: int = 12,
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.25,
    residual_threshold: float = 3.0,
    max_trials: int = 3000,
    rng_seed: int = 42,
    transform_model: str = "translation",
) -> PairwiseMatch | None:
    """Estimate the transformation from image j to image i.

    transform_model:
        translation: Translation only, without rotation, shear, or perspective distortion (default)
        similarity: Translation + uniform scaling + rotation
        projective: Full projective transformation; suitable for perspective changes but may introduce image tilt.
    """

    if (
        feature_i.descriptors is None
        or feature_j.descriptors is None
        or len(feature_i.descriptors) < min_matches
        or len(feature_j.descriptors) < min_matches
    ):
        return None

    matches = match_descriptors(
        feature_i.descriptors,
        feature_j.descriptors,
        metric="euclidean",
        cross_check=True,
        max_ratio=max_ratio,
    )

    if len(matches) < min_matches:
        return None

    keypoints_i_xy = rc_to_xy(feature_i.keypoints_rc[matches[:, 0]])
    keypoints_j_xy = rc_to_xy(feature_j.keypoints_rc[matches[:, 1]])

    try:
        if transform_model == "translation":
            # Compute the displacement from image j to image i for each matched keypoint, and use the median to suppress mismatches.
            displacements = keypoints_i_xy - keypoints_j_xy
            translation = np.median(displacements, axis=0)
            errors_all = np.linalg.norm(displacements - translation, axis=1)
            inliers = errors_all <= residual_threshold

            # Re-estimate the translation using the inliers from the first pass.
            if np.any(inliers):
                translation = np.median(displacements[inliers], axis=0)
                errors_all = np.linalg.norm(displacements - translation, axis=1)
                inliers = errors_all <= residual_threshold

            matrix = np.array(
                [
                    [1.0, 0.0, translation[0]],
                    [0.0, 1.0, translation[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            model = ProjectiveTransform(matrix=matrix)
        else:
            model_class = {
                "similarity": SimilarityTransform,
                "projective": ProjectiveTransform,
            }.get(transform_model)
            if model_class is None:
                raise ValueError(f"Unknown transformational model: {transform_model}")

            min_samples = 2 if transform_model == "similarity" else 4
            model, inliers = ransac(
                (keypoints_j_xy, keypoints_i_xy),
                model_class,
                min_samples=min_samples,
                residual_threshold=residual_threshold,
                max_trials=max_trials,
                stop_probability=0.999,
                rng=np.random.default_rng(rng_seed),
            )
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None

    if model is None or inliers is None:
        return None

    num_inliers = int(np.sum(inliers))
    inlier_ratio = num_inliers / len(matches)

    if num_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return None

    src_inliers = keypoints_j_xy[inliers]
    dst_inliers = keypoints_i_xy[inliers]
    projected = model(src_inliers)
    errors = np.linalg.norm(projected - dst_inliers, axis=1)
    median_error = float(np.median(errors))

    if not np.isfinite(median_error):
        return None

    score = num_inliers * inlier_ratio / (1.0 + median_error)

    return PairwiseMatch(
        i=i,
        j=j,
        transform_j_to_i=model,
        num_matches=len(matches),
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        median_error=median_error,
        score=score,
    )


def build_pairwise_matches(
    features: list[ImageFeatures],
    max_ratio: float = 0.75,
    min_matches: int = 12,
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.25,
    residual_threshold: float = 3.0,
    transform_model: str = "translation",
) -> list[PairwiseMatch]:
    """Pair all the images"""

    pairwise_matches: list[PairwiseMatch] = []
    n = len(features)

    for i in range(n):
        for j in range(i + 1, n):
            result = estimate_pairwise_transform(
                feature_i=features[i],
                feature_j=features[j],
                i=i,
                j=j,
                max_ratio=max_ratio,
                min_matches=min_matches,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
                residual_threshold=residual_threshold,
                rng_seed=42 + i * n + j,
                transform_model=transform_model,
            )

            name_i = os.path.basename(features[i].path)
            name_j = os.path.basename(features[j].path)

            if result is None:
                print(f"[unconnected] {name_i} <-> {name_j}")
                continue

            pairwise_matches.append(result)

            print(
                f"[connections] {name_i} <-> {name_j} | "
                f"paired={result.num_matches}, "
                f"inliers ={result.num_inliers}, "
                f"ratio of inliers ={result.inlier_ratio:.2%}, "
                f"medium error={result.median_error:.2f}px"
            )

    return pairwise_matches


def largest_connected_component(
    n_images: int,
    matches: list[PairwiseMatch],
) -> list[int]:
    """Find out the biggest connections between paired images"""

    adjacency = [[] for _ in range(n_images)]

    for match in matches:
        adjacency[match.i].append(match.j)
        adjacency[match.j].append(match.i)

    visited = set()
    components = []

    for start in range(n_images):
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        component = []

        while stack:
            node = stack.pop()
            component.append(node)

            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return max(components, key=len)


def choose_reference_image(
    component: list[int],
    matches: list[PairwiseMatch],
) -> int:
    """
    Select the reference image based on the accumulated matching scores with all other images. 
    In general, the image with the most connections and the most reliable matches is chosen as the reference (center) image.
    """

    scores = {node: 0.0 for node in component}

    for match in matches:
        if match.i in scores and match.j in scores:
            scores[match.i] += match.score
            scores[match.j] += match.score

    return max(scores, key=scores.get)


def invert_matrix(matrix: np.ndarray) -> np.ndarray:
    """Safely compute the inverse of the homography matrix."""

    inverse = np.linalg.inv(matrix)
    inverse /= inverse[2, 2]
    return inverse


def build_directed_graph(
    component: list[int],
    matches: list[PairwiseMatch],
):
    """
    Build a directed transformation graph.

    Each edge in graph[src] stores:
        destination
        the homography from src to destination
        matching score
    """

    component_set = set(component)
    graph = {node: [] for node in component}

    for match in matches:
        if match.i not in component_set or match.j not in component_set:
            continue

        h_j_to_i = match.transform_j_to_i.params
        h_i_to_j = invert_matrix(h_j_to_i)

        graph[match.j].append((match.i, h_j_to_i, match.score))
        graph[match.i].append((match.j, h_i_to_j, match.score))

    return graph


def compute_global_transforms(
    component: list[int],
    matches: list[PairwiseMatch],
    reference_index: int,
) -> dict[int, np.ndarray]:
    """
    Compute the accumulated homography that maps each image into the
    coordinate system of the reference image.
    
    To avoid using unreliable edges, define the path cost as:
        -log(normalized matching quality)
    """

    graph = build_directed_graph(component, matches)

    all_scores = [
        edge[2]
        for node in graph
        for edge in graph[node]
        if edge[2] > 0
    ]

    max_score = max(all_scores) if all_scores else 1.0

    distances = {node: np.inf for node in component}
    global_h = {}

    distances[reference_index] = 0.0
    global_h[reference_index] = np.eye(3, dtype=np.float64)

    queue = [(0.0, reference_index)]

    while queue:
        current_distance, src = heapq.heappop(queue)

        if current_distance > distances[src]:
            continue

        for dst, h_src_to_dst, score in graph[src]:
            quality = np.clip(score / max_score, 1e-6, 1.0)
            edge_cost = -np.log(quality) + 1e-6
            candidate_distance = current_distance + edge_cost

            if candidate_distance >= distances[dst]:
                continue

            # known src -> reference.
            # edge src -> dst.
            # therefore,  dst -> reference =
            # src -> reference @ dst -> src
            h_dst_to_src = invert_matrix(h_src_to_dst)
            h_dst_to_reference = global_h[src] @ h_dst_to_src
            h_dst_to_reference /= h_dst_to_reference[2, 2]

            distances[dst] = candidate_distance
            global_h[dst] = h_dst_to_reference
            heapq.heappush(queue, (candidate_distance, dst))

    missing = set(component) - set(global_h)
    if missing:
        raise RuntimeError(f"following images can not be included: {sorted(missing)}")

    return global_h


def image_corners_xy(image: np.ndarray) -> np.ndarray:
    """return (x, y) coords in the corners"""

    height, width = image.shape[:2]

    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float64,
    )


def compute_panorama_canvas(
    features: list[ImageFeatures],
    component: list[int],
    global_h: dict[int, np.ndarray],
):
    """Compute the panorama canvas size and the translation matrix."""

    transformed_corners = []

    for index in component:
        corners = image_corners_xy(features[index].image)
        transform = ProjectiveTransform(global_h[index])
        transformed_corners.append(transform(corners))

    all_corners = np.vstack(transformed_corners)

    min_xy = np.floor(np.min(all_corners, axis=0))
    max_xy = np.ceil(np.max(all_corners, axis=0))

    min_x, min_y = min_xy
    max_x, max_y = max_xy

    output_width = int(max_x - min_x + 1)
    output_height = int(max_y - min_y + 1)

    # Translate the entire reference coordinate system into the positive canvas coordinate region.
    translation = np.array(
        [
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if output_width <= 0 or output_height <= 0:
        raise RuntimeError("Invalid size of the calculated canvas.")

    return (output_height, output_width), translation


def create_feather_weight(height: int, width: int) -> np.ndarray:
    """
    Generate a feathering weight map for a single image.
    
    Pixels near the image center receive higher weights, while those near the borders receive lower weights, helping reduce visible seams in overlapping regions.
    """

    mask = np.ones((height, width), dtype=np.uint8)

    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0

    weight = distance_transform_edt(mask).astype(np.float32)

    maximum = weight.max()
    if maximum > 0:
        weight /= maximum
    else:
        weight.fill(1.0)

    weight = np.maximum(weight, 0.02)

    return weight


def warp_and_blend(
    features: list[ImageFeatures],
    component: list[int],
    global_h: dict[int, np.ndarray],
    output_shape: tuple[int, int],
    translation: np.ndarray,
) -> np.ndarray:
    """Project all images onto a common canvas and blend them using weighted averaging."""

    output_height, output_width = output_shape

    color_sum = np.zeros(
        (output_height, output_width, 3),
        dtype=np.float64,
    )
    weight_sum = np.zeros(
        (output_height, output_width),
        dtype=np.float64,
    )

    for index in component:
        image = features[index].image
        height, width = image.shape[:2]

        h_image_to_canvas = translation @ global_h[index]
        h_image_to_canvas /= h_image_to_canvas[2, 2]

        transform = ProjectiveTransform(h_image_to_canvas)

        warped_image = warp(
            image,
            inverse_map=transform.inverse,
            output_shape=output_shape,
            preserve_range=True,
            mode="constant",
            cval=0,
        )

        source_mask = np.ones((height, width), dtype=np.float32)
        warped_mask = warp(
            source_mask,
            inverse_map=transform.inverse,
            output_shape=output_shape,
            preserve_range=True,
            mode="constant",
            cval=0,
            order=0,
        )

        source_weight = create_feather_weight(height, width)
        warped_weight = warp(
            source_weight,
            inverse_map=transform.inverse,
            output_shape=output_shape,
            preserve_range=True,
            mode="constant",
            cval=0,
        )

        warped_weight *= warped_mask > 0.5

        color_sum += warped_image * warped_weight[..., np.newaxis]
        weight_sum += warped_weight

        print(f"[Completed!] {os.path.basename(features[index].path)}")

    panorama = color_sum / np.maximum(
        weight_sum[..., np.newaxis],
        1e-8,
    )

    panorama = np.clip(panorama, 0.0, 1.0)

    # set all uncovered area black
    panorama[weight_sum <= 1e-8] = 0

    return panorama.astype(np.float32)


def crop_empty_border(
    panorama: np.ndarray,
    threshold: float = 1e-6,
) -> np.ndarray:
    """Crop away the completely empty black borders surrounding the panorama."""

    valid = np.any(panorama > threshold, axis=2)

    rows = np.where(np.any(valid, axis=1))[0]
    cols = np.where(np.any(valid, axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return panorama

    return panorama[
        rows[0] : rows[-1] + 1,
        cols[0] : cols[-1] + 1,
    ]


def stitch_images(
    image_paths: list[str],
    output_path: str,
    max_ratio: float = 0.75,
    min_matches: int = 12,
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.25,
    residual_threshold: float = 3.0,
    max_output_pixels: int = 150_000_000,
    transform_model: str = "translation",
) -> np.ndarray:
    """Automatically stitch a set of overlapping images into a panorama."""

    if len(image_paths) < 2:
        raise ValueError("At least two images required.")

    print("Extracting SIFT features...")

    features = []

    for path in image_paths:
        try:
            feature = extract_sift_features(path)
        except RuntimeError as exc:
            print(f"[skip] unable to extract features: {path}, Cause: {exc}")
            continue

        features.append(feature)

        print(
            f"[SIFT] {os.path.basename(path)}: "
            f"{len(feature.keypoints_rc)} keypoints"
        )

    if len(features) < 2:
        raise RuntimeError("images with features less than 2.")

    print("\nPairing...")

    pairwise_matches = build_pairwise_matches(
        features,
        max_ratio=max_ratio,
        min_matches=min_matches,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        residual_threshold=residual_threshold,
        transform_model=transform_model,
    )

    if not pairwise_matches:
        raise RuntimeError(
            "No reliable overlap found"
            "Try to increase max_ratio, reduce min_inliers,"
            "or check overlap areas of the images."
        )

    component = largest_connected_component(
        len(features),
        pairwise_matches,
    )

    if len(component) < 2:
        raise RuntimeError("maximum connected images less than 2.")

    if len(component) < len(features):
        omitted = sorted(set(range(len(features))) - set(component))

        print("\nWarning: The following images could not be reliably connected to the main image group and will be ignored: ")
        for index in omitted:
            print(f"  - {features[index].path}")

    reference_index = choose_reference_image(
        component,
        pairwise_matches,
    )

    print(
        "\nReference image: "
        f"{os.path.basename(features[reference_index].path)}"
    )

    global_h = compute_global_transforms(
        component,
        pairwise_matches,
        reference_index,
    )

    output_shape, translation = compute_panorama_canvas(
        features,
        component,
        global_h,
    )

    output_height, output_width = output_shape
    total_pixels = output_height * output_width

    print(
        f"Full canvas size: {output_width} × {output_height} "
        f"({total_pixels / 1_000_000:.1f} MP)"
    )

    if total_pixels > max_output_pixels:
        raise RuntimeError(
            f"Too large a canvas: {total_pixels:,} pixels"
            "This usually indicates incorrect feature matches or excessive perspective differences between the images."
        )

    panorama = warp_and_blend(
        features,
        component,
        global_h,
        output_shape,
        translation,
    )

    panorama = crop_empty_border(panorama)

    output_parent = Path(output_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(
        output_path,
        np.round(panorama * 255).astype(np.uint8),
    )

    print(f"\nResults save into: {output_path}")

    return panorama


def collect_image_paths(input_pattern: str) -> list[str]:
    """collect images and exclude directory"""

    paths = [
        path
        for path in glob.glob(input_pattern)
        if os.path.isfile(path)
    ]

    paths.sort()

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Automatic image stitcher based on skimage SIFT and RANSAC"
    )

    parser.add_argument(
        "input",
        help='input images, e.g. "images/*.jpg"',
    )
    parser.add_argument(
        "-o",
        "--output",
        default="panorama.png",
        help="output directory, default panorama.png",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=0.75,
        help="describe the nearest-neighbor ratio threshold, default 0.75",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=12,
        help="least pairing points, default 12",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=10,
        help="least inliers required for RANSAC, default 10",
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.25,
        help="least inliers ratio, default 0.25",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=3.0,
        help="residual threshold for RANSAC, default 3.0",
    )
    parser.add_argument(
        "--transform-model",
        choices=("translation", "similarity", "projective"),
        default="translation",
        help=(
            "Transformation models: translation = translation only; no rotation or perspective distortion (default); "
            "similarity = allows rotation and uniform scaling; "
            "projective = allows full projective transformation"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show results after stitching",
    )

    args = parser.parse_args()

    image_paths = collect_image_paths(args.input)

    if not image_paths:
        raise FileNotFoundError(
            f"No images found for the follow model: {args.input}"
        )

    print(f"found {len(image_paths)} images: ")
    for path in image_paths:
        print(f"  - {path}")

    panorama = stitch_images(
        image_paths=image_paths,
        output_path=args.output,
        max_ratio=args.max_ratio,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        residual_threshold=args.residual_threshold,
        transform_model=args.transform_model,
    )

    if args.show:
        plt.figure(figsize=(16, 9))
        plt.imshow(panorama)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()