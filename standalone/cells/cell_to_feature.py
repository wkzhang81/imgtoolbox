from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import (
    io,
    color,
    util,
    exposure,
    filters,
    morphology,
    measure,
    segmentation,
)
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. Reading and preprocessing
# ============================================================

def read_gray_image(image_path):
    """Read an image and convert it to a 0~1 float grayscale image."""
    image = io.imread(image_path)

    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)

    return util.img_as_float32(image)


def preprocess_phase_contrast(
    image,
    background_sigma=25,
    denoise_sigma=1.2,
    clahe_clip=0.015,
):
    """
    Preprocess a phase-contrast/bright-field image.

    background_sigma:
        Use a large-scale Gaussian filter to estimate the non-uniform background.
        Should be clearly larger than a typical cell radius.

    clahe_clip:
        Strength of local contrast enhancement.
    """

    image = exposure.rescale_intensity(image, out_range=(0, 1))

    # Estimate the slowly varying illumination background
    background = filters.gaussian(
        image,
        sigma=background_sigma,
        preserve_range=True,
    )

    # High-pass style background correction
    corrected = image - background

    # Remap to 0~1
    corrected = exposure.rescale_intensity(
        corrected,
        in_range=(
            np.percentile(corrected, 1),
            np.percentile(corrected, 99),
        ),
        out_range=(0, 1),
    )

    # Remove high-frequency noise while preserving boundaries
    denoised = filters.gaussian(
        corrected,
        sigma=denoise_sigma,
        preserve_range=True,
    )

    # Enhance local boundaries
    enhanced = exposure.equalize_adapthist(
        denoised,
        clip_limit=clahe_clip,
    )

    return {
        "original": image,
        "background": background,
        "corrected": corrected,
        "enhanced": enhanced,
    }



# ============================================================
# 2. First distinguish cell-covered areas from dish gaps
# ============================================================

def estimate_cell_coverage_mask(
    original,
    enhanced,
    texture_sigma=5.0,
    gradient_sigma=2.0,
    threshold_scale=0.85,
    closing_radius=7,
    opening_radius=2,
    min_coverage_area=300,
    max_hole_area=1200,
):
    """
    Estimate the "cell-covered regions" based on local texture and edge density.

    In phase-contrast images:
    - Cell regions usually have higher local grayscale std and edge density;
    - Dish gaps are usually smoother, with lower local texture and gradient.

    Returns
    -------
    coverage_mask : bool ndarray
        True marks the regions where cell instance segmentation is allowed;
        False marks dish gaps / smooth background.

    score : float ndarray
        Heuristic score of cell-coverage probability, useful for tuning and visualization.
    """

    original = exposure.rescale_intensity(original, out_range=(0, 1))
    enhanced = exposure.rescale_intensity(enhanced, out_range=(0, 1))

    # Local standard deviation: E[x^2] - E[x]^2
    local_mean = filters.gaussian(
        original,
        sigma=texture_sigma,
        preserve_range=True,
    )
    local_mean_sq = filters.gaussian(
        original ** 2,
        sigma=texture_sigma,
        preserve_range=True,
    )
    local_variance = np.maximum(
        local_mean_sq - local_mean ** 2,
        0,
    )
    local_std = np.sqrt(local_variance)

    # Edge density: cell edges and phase-contrast halos produce strong gradients
    gradient = filters.scharr(enhanced)
    gradient_density = filters.gaussian(
        gradient,
        sigma=gradient_sigma,
        preserve_range=True,
    )

    # Robust normalization separately, to avoid extreme bright points dominating the score
    def robust_normalize(arr):
        low, high = np.percentile(arr, (2, 98))
        if high <= low:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - low) / (high - low), 0, 1)

    texture_norm = robust_normalize(local_std)
    gradient_norm = robust_normalize(gradient_density)

    # For this kind of phase-contrast image, texture is usually more stable than single-point gradient
    score = 0.60 * texture_norm + 0.40 * gradient_norm
    score = filters.gaussian(
        score,
        sigma=1.5,
        preserve_range=True,
    )

    # Otsu automatic threshold, allowing recall to be tuned with threshold_scale
    otsu_threshold = filters.threshold_otsu(score)
    threshold = otsu_threshold * threshold_scale
    coverage_mask = score > threshold

    # Connect cell-covered regions broken by halos or weak texture
    if closing_radius > 0:
        coverage_mask = morphology.binary_closing(
            coverage_mask,
            morphology.disk(closing_radius),
        )

    # Remove isolated noise
    if opening_radius > 0:
        coverage_mask = morphology.binary_opening(
            coverage_mask,
            morphology.disk(opening_radius),
        )

    coverage_mask = morphology.remove_small_objects(
        coverage_mask,
        min_size=min_coverage_area,
    )

    # Fill small holes inside cells, but keep the real large dish gaps
    coverage_mask = morphology.remove_small_holes(
        coverage_mask,
        area_threshold=max_hole_area,
    )

    return coverage_mask, score


# ============================================================
# 3. Gradient-based marker-controlled Watershed
# ============================================================

def segment_confluent_cells(
    enhanced,
    coverage_mask=None,
    gradient_sigma=4.0,
    minima_h=0.015,
    compactness=0.0005,
    min_area=80,
    max_area=5000,
    remove_border_cells=True,
):
    """
    Gradient Watershed for high-density phase-contrast cells.

    Unlike binary thresholding, this does not assume a distinct background.
    The whole image is divided into cell regions based on edge ridges.

    gradient_sigma:
        Gradient smoothing scale.
        Small -> more seeds, easy to over-segment
        Large -> fewer seeds, easy to miss small cells

    minima_h:
        h-minima depth threshold.
        Small -> keep more minima, cell count increases
        Large -> keep only stable minima, cell count decreases
    """

    # Scharr is quite sensitive to the bright/dark halos in phase-contrast images
    gradient_raw = filters.scharr(enhanced)

    # Suppress small textures and noisy minima inside a single cell
    gradient = filters.gaussian(
        gradient_raw,
        sigma=gradient_sigma,
        preserve_range=True,
    )

    # If no coverage area is provided, fall back to the whole image
    if coverage_mask is None:
        coverage_mask = np.ones_like(enhanced, dtype=bool)
    else:
        coverage_mask = coverage_mask.astype(bool)

    # Cell interiors usually correspond to low-value basins in the smoothed gradient map
    minima = morphology.h_minima(
        gradient,
        h=minima_h,
    )

    # Dish gaps must not become watershed seeds
    minima &= coverage_mask

    # Remove very small minima regions
    minima = morphology.remove_small_objects(
        minima,
        min_size=4,
    )

    markers = measure.label(minima)

    print("Initial watershed seed count:", markers.max())

    if markers.max() == 0:
        raise RuntimeError(
            "No watershed seeds detected. Please reduce minima_h."
        )

    # Only partition within the cell-covered area; dish gaps remain as background 0
    labels = segmentation.watershed(
        gradient,
        markers=markers,
        mask=coverage_mask,
        compactness=compactness,
        watershed_line=True,
    )

    if remove_border_cells:
        labels = segmentation.clear_border(labels)

    # Area filtering, but must not use measure.label(labels > 0),
    # otherwise adjacent cells could merge back into one region
    filtered = np.zeros_like(labels, dtype=np.int32)
    next_label = 1

    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue

        if max_area is not None and region.area > max_area:
            continue

        filtered[labels == region.label] = next_label
        next_label += 1

    return {
        "gradient_raw": gradient_raw,
        "gradient": gradient,
        "minima": minima,
        "markers": markers,
        "coverage_mask": coverage_mask,
        "labels": filtered,
    }


# ============================================================
# 4. Extract per-cell morphology and grayscale features
# ============================================================

def intensity_std(region_mask, intensity_image):
    """Custom region grayscale-standard-deviation feature for regionprops_table."""
    values = intensity_image[region_mask]

    if values.size == 0:
        return np.nan

    return float(np.std(values))


def intensity_p10(region_mask, intensity_image):
    values = intensity_image[region_mask]
    return float(np.percentile(values, 10)) if values.size else np.nan


def intensity_p90(region_mask, intensity_image):
    values = intensity_image[region_mask]
    return float(np.percentile(values, 90)) if values.size else np.nan


def extract_cell_features(image, labels):
    """Extract morphology and grayscale features from each watershed region."""

    properties = [
        "label",
        "area",
        "perimeter",
        "perimeter_crofton",
        "convex_area",
        "filled_area",
        "equivalent_diameter_area",
        "major_axis_length",
        "minor_axis_length",
        "eccentricity",
        "solidity",
        "extent",
        "orientation",
        "centroid",
        "mean_intensity",
        "min_intensity",
        "max_intensity",
    ]

    table = measure.regionprops_table(
        labels,
        intensity_image=image,
        properties=properties,
        extra_properties=[
            intensity_std,
            intensity_p10,
            intensity_p90,
        ],
    )

    df = pd.DataFrame(table)

    eps = 1e-8

    # Circularity: closer to 1 means closer to a circle
    df["circularity"] = (
        4 * np.pi * df["area"]
        / (df["perimeter_crofton"] ** 2 + eps)
    )

    # Aspect ratio
    df["aspect_ratio"] = (
        df["major_axis_length"]
        / (df["minor_axis_length"] + eps)
    )

    # Area to convex-hull area ratio
    df["convex_area_ratio"] = (
        df["area"]
        / (df["convex_area"] + eps)
    )

    # Degree of holes inside the region
    df["filled_ratio"] = (
        df["area"]
        / (df["filled_area"] + eps)
    )

    # Grayscale dynamic range
    df["intensity_range"] = (
        df["intensity_p90"]
        - df["intensity_p10"]
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# ============================================================
# 5. Morphology clustering
# ============================================================

def cluster_cell_morphology(
    feature_df,
    n_clusters=3,
):
    """
    RobustScaler + PCA + KMeans.

    The clustering result represents computationally defined morphology groups,
    and must not be called biological cell types without validation.
    """

    feature_columns = [
        "area",
        "perimeter_crofton",
        "equivalent_diameter_area",
        "major_axis_length",
        "minor_axis_length",
        "eccentricity",
        "solidity",
        "extent",
        "circularity",
        "aspect_ratio",
        "convex_area_ratio",
        "filled_ratio",
        "mean_intensity",
        "intensity_std",
        "intensity_range",
    ]

    df = feature_df.copy()

    X = df[feature_columns].copy()
    X = X.fillna(X.median())

    # Apply log1p to skewed features such as area, to avoid large cells dominating the clustering
    log_columns = [
        "area",
        "perimeter_crofton",
        "equivalent_diameter_area",
        "major_axis_length",
        "minor_axis_length",
        "aspect_ratio",
    ]

    for column in log_columns:
        X[column] = np.log1p(X[column].clip(lower=0))

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(
        8,
        X_scaled.shape[1],
        X_scaled.shape[0] - 1,
    )

    if n_components < 2:
        raise ValueError("Too few valid cells to perform PCA clustering.")

    pca = PCA(
        n_components=n_components,
        random_state=42,
    )

    X_pca = pca.fit_transform(X_scaled)

    model = KMeans(
        n_clusters=n_clusters,
        n_init=30,
        random_state=42,
    )

    clusters = model.fit_predict(X_pca)

    df["cluster"] = clusters
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    score = silhouette_score(X_pca, clusters)

    return {
        "features": df,
        "feature_columns": feature_columns,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "silhouette": score,
    }


def map_clusters_to_image(labels, clustered_df):
    """Map each cell's cluster number back onto the original image."""

    cluster_image = np.zeros_like(labels, dtype=np.int32)

    mapping = dict(
        zip(
            clustered_df["label"].astype(int),
            clustered_df["cluster"].astype(int) + 1,
        )
    )

    for cell_label, cluster_label in mapping.items():
        cluster_image[labels == cell_label] = cluster_label

    return cluster_image


# ============================================================
# 6. Visualization
# ============================================================

def visualize_segmentation(
    original,
    preprocessing,
    segmentation_result,
    output_dir,
):
    labels = segmentation_result["labels"]

    overlay = label2rgb(
        labels,
        image=original,
        bg_label=0,
        alpha=0.30,
    )

    boundaries = find_boundaries(
        labels,
        mode="outer",
    )

    boundary_overlay = color.gray2rgb(original)
    boundary_overlay[boundaries] = [1, 0, 0]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 11),
    )

    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("original")

    axes[0, 1].imshow(
        preprocessing["coverage_score"],
        cmap="magma",
    )
    axes[0, 1].set_title("cell coverage score")

    axes[0, 2].imshow(
        preprocessing["coverage_mask"],
        cmap="gray",
    )
    axes[0, 2].set_title("cell coverage mask")

    axes[1, 0].imshow(
        segmentation_result["gradient"],
        cmap="magma",
    )
    axes[1, 0].set_title("gradient smoothed")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(
        f"segment into: {labels.max()} areas"
    )

    axes[1, 2].imshow(boundary_overlay)
    axes[1, 2].set_title("bondary_check")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "result1.png", dpi = 300)


def visualize_clusters(
    original,
    labels,
    clustering_result,
    output_dir,
):
    df = clustering_result["features"]

    cluster_image = map_clusters_to_image(
        labels,
        df,
    )

    cluster_overlay = label2rgb(
        cluster_image,
        image=original,
        bg_label=0,
        alpha=0.42,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    axes[0].imshow(cluster_overlay)
    axes[0].set_title(
        "Cluster_color_overlay_original_figure"
    )
    axes[0].axis("off")

    scatter = axes[1].scatter(
        df["PC1"],
        df["PC2"],
        c=df["cluster"],
        cmap="tab10",
        s=22,
        alpha=0.8,
    )

    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title(
        "PC analysis\n"
        f"silhouette={clustering_result['silhouette']:.3f}"
    )

    axes[1].legend(
        *scatter.legend_elements(),
        title="clusters",
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "result2.png", dpi = 300)

    return cluster_image


# ============================================================
# 7. Main function
# ============================================================

def analyze_phase_contrast_cells(
    image_path,
    output_dir="cell_analysis_output",
    n_clusters=5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = read_gray_image(image_path)

    prep = preprocess_phase_contrast(
        image,
        background_sigma=25,
        denoise_sigma=1.2,
        clahe_clip=0.015,
    )

    # First distinguish cell-covered areas from dish gaps
    coverage_mask, coverage_score = estimate_cell_coverage_mask(
        image,
        prep["enhanced"],
        texture_sigma=5.0,
        gradient_sigma=2.0,
        threshold_scale=0.85,
        closing_radius=7,
        opening_radius=2,
        min_coverage_area=300,
        max_hole_area=1200,
    )

    prep["coverage_mask"] = coverage_mask
    prep["coverage_score"] = coverage_score

    print(
        "Estimated cell coverage rate:",
        f"{coverage_mask.mean() * 100:.1f}%"
    )

    seg = segment_confluent_cells(
        prep["enhanced"],
        coverage_mask=coverage_mask,

        # For the uploaded 900x675 image, start from the following parameters
        gradient_sigma=4.0,
        minima_h=0.015,
        compactness=0.0005,

        min_area=80,
        max_area=5000,
        remove_border_cells=True,
    )

    labels = seg["labels"]

    print("Cell region count after area filtering:", labels.max())

    if labels.max() < n_clusters:
        raise RuntimeError(
            f"Only got {labels.max()} regions, "
            f"cannot cluster into {n_clusters} classes."
        )

    features = extract_cell_features(
        prep["enhanced"],
        labels,
    )

    clustering = cluster_cell_morphology(
        features,
        n_clusters=n_clusters,
    )

    visualize_segmentation(
        image,
        prep,
        seg,
        output_dir,
    )

    cluster_image = visualize_clusters(
        image,
        labels,
        clustering,
        output_dir,
    )

    # Save the feature table
    feature_path = output_dir / "cell_features_clusters.csv"
    clustering["features"].to_csv(
        feature_path,
        index=False,
    )

    # Save the initially obtained cell coverage mask
    coverage_mask_path = output_dir / "cell_coverage_mask.png"
    io.imsave(
        coverage_mask_path,
        (prep["coverage_mask"].astype(np.uint8) * 255),
        check_contrast=False,
    )

    # Save the coverage score map, useful for judging whether the threshold is reasonable
    coverage_score_path = output_dir / "cell_coverage_score.png"
    io.imsave(
        coverage_score_path,
        util.img_as_ubyte(
            exposure.rescale_intensity(
                prep["coverage_score"],
                out_range=(0, 1),
            )
        ),
        check_contrast=False,
    )

    # Optional: save a color mask containing only cell instances, without the original background, for manual inspection
    instance_mask_rgb = label2rgb(
        labels,
        bg_label=0,
        bg_color=(0, 0, 0),
    )
    instance_mask_path = output_dir / "instance_mask_rgb.png"
    io.imsave(
        instance_mask_path,
        util.img_as_ubyte(instance_mask_rgb),
        check_contrast=False,
    )

    # Save the color segmentation map
    instance_rgb = label2rgb(
        labels,
        image=image,
        bg_label=0,
        alpha=0.35,
    )

    io.imsave(
        output_dir / "instance_overlay.png",
        util.img_as_ubyte(instance_rgb),
    )

    cluster_rgb = label2rgb(
        cluster_image,
        image=image,
        bg_label=0,
        alpha=0.42,
    )

    io.imsave(
        output_dir / "cluster_overlay.png",
        util.img_as_ubyte(cluster_rgb),
    )

    print("\nResults saved to:", output_dir.resolve())
    print("Feature file:", feature_path)
    print("Cell coverage mask:", coverage_mask_path)
    print("Coverage score map:", coverage_score_path)
    print("Instance color mask:", instance_mask_path)
    print(
        "Clustering silhouette score:",
        round(clustering["silhouette"], 3),
    )

    summary_columns = [
        "area",
        "circularity",
        "aspect_ratio",
        "eccentricity",
        "solidity",
        "mean_intensity",
        "intensity_std",
    ]

    summary = (
        clustering["features"]
        .groupby("cluster")[summary_columns]
        .median()
        .round(3)
    )

    print("\nFeature medians of different morphology clusters:")
    print(summary)

    return {
        "image": image,
        "preprocessing": prep,
        "segmentation": seg,
        "features": clustering["features"],
        "clustering": clustering,
        "cluster_image": cluster_image,
        "summary": summary,
    }


if __name__ == "__main__":
    result = analyze_phase_contrast_cells(
        image_path="cells.jpg",
        output_dir="cell_analysis_output",
        n_clusters=5,
    )
