from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import color, exposure, filters, io, measure, morphology, segmentation, util
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


# ============================================================
# 1. Image loading and preprocessing
# ============================================================


def read_gray_image(image_path):
    """Read an image and convert it to a float32 grayscale image in [0, 1]."""
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
    """Preprocess a phase-contrast or bright-field image."""
    image = exposure.rescale_intensity(image, out_range=(0, 1))

    background = filters.gaussian(
        image,
        sigma=background_sigma,
        preserve_range=True,
    )

    corrected = image - background
    corrected = exposure.rescale_intensity(
        corrected,
        in_range=(
            np.percentile(corrected, 1),
            np.percentile(corrected, 99),
        ),
        out_range=(0, 1),
    )

    denoised = filters.gaussian(
        corrected,
        sigma=denoise_sigma,
        preserve_range=True,
    )

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
# 2. Cell-coverage estimation
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
    """Estimate image regions that are covered by cells."""
    original = exposure.rescale_intensity(original, out_range=(0, 1))
    enhanced = exposure.rescale_intensity(enhanced, out_range=(0, 1))

    local_mean = filters.gaussian(
        original,
        sigma=texture_sigma,
        preserve_range=True,
    )
    local_mean_sq = filters.gaussian(
        original**2,
        sigma=texture_sigma,
        preserve_range=True,
    )
    local_variance = np.maximum(local_mean_sq - local_mean**2, 0)
    local_std = np.sqrt(local_variance)

    gradient = filters.scharr(enhanced)
    gradient_density = filters.gaussian(
        gradient,
        sigma=gradient_sigma,
        preserve_range=True,
    )

    def robust_normalize(array):
        low, high = np.percentile(array, (2, 98))
        if high <= low:
            return np.zeros_like(array, dtype=np.float32)
        return np.clip((array - low) / (high - low), 0, 1)

    texture_norm = robust_normalize(local_std)
    gradient_norm = robust_normalize(gradient_density)

    score = 0.60 * texture_norm + 0.40 * gradient_norm
    score = filters.gaussian(score, sigma=1.5, preserve_range=True)

    threshold = filters.threshold_otsu(score) * threshold_scale
    coverage_mask = score > threshold

    if closing_radius > 0:
        coverage_mask = morphology.binary_closing(
            coverage_mask,
            morphology.disk(closing_radius),
        )

    if opening_radius > 0:
        coverage_mask = morphology.binary_opening(
            coverage_mask,
            morphology.disk(opening_radius),
        )

    coverage_mask = morphology.remove_small_objects(
        coverage_mask,
        min_size=min_coverage_area,
    )
    coverage_mask = morphology.remove_small_holes(
        coverage_mask,
        area_threshold=max_hole_area,
    )

    return coverage_mask, score


# ============================================================
# 3. Marker-controlled watershed segmentation
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
    """Segment confluent cells using marker-controlled watershed."""
    gradient_raw = filters.scharr(enhanced)
    gradient = filters.gaussian(
        gradient_raw,
        sigma=gradient_sigma,
        preserve_range=True,
    )

    if coverage_mask is None:
        coverage_mask = np.ones_like(enhanced, dtype=bool)
    else:
        coverage_mask = coverage_mask.astype(bool)

    minima = morphology.h_minima(gradient, h=minima_h)
    minima &= coverage_mask
    minima = morphology.remove_small_objects(minima, min_size=4)
    markers = measure.label(minima)

    print("Initial watershed marker count:", markers.max())

    if markers.max() == 0:
        raise RuntimeError(
            "No watershed markers were detected. Reduce minima_h and try again."
        )

    labels = segmentation.watershed(
        gradient,
        markers=markers,
        mask=coverage_mask,
        compactness=compactness,
        watershed_line=True,
    )

    if remove_border_cells:
        labels = segmentation.clear_border(labels)

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
# 4. Per-cell 1024-dimensional ResNet50 feature extraction
# ============================================================


class CellCropDataset(Dataset):
    """Create masked, square cell crops for ResNet50 inference."""

    def __init__(self, image, labels, padding=8):
        self.image = exposure.rescale_intensity(image, out_range=(0, 1))
        self.labels = labels
        self.regions = list(measure.regionprops(labels))
        self.padding = padding
        self.transform = Compose(
            [
                ToTensor(),
                Resize((224, 224), antialias=True),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, index):
        region = self.regions[index]
        min_row, min_col, max_row, max_col = region.bbox

        min_row = max(0, min_row - self.padding)
        min_col = max(0, min_col - self.padding)
        max_row = min(self.image.shape[0], max_row + self.padding)
        max_col = min(self.image.shape[1], max_col + self.padding)

        crop = self.image[min_row:max_row, min_col:max_col]
        crop_mask = self.labels[min_row:max_row, min_col:max_col] == region.label

        background_value = float(np.median(crop))
        masked_crop = np.where(crop_mask, crop, background_value)

        height, width = masked_crop.shape
        side = max(height, width)
        square_crop = np.full((side, side), background_value, dtype=np.float32)
        row_offset = (side - height) // 2
        col_offset = (side - width) // 2
        square_crop[
            row_offset : row_offset + height,
            col_offset : col_offset + width,
        ] = masked_crop

        rgb_crop = np.repeat(square_crop[..., None], 3, axis=-1).astype(np.float32)
        tensor = self.transform(rgb_crop)

        return {
            "image": tensor,
            "label": int(region.label),
            "centroid_row": float(region.centroid[0]),
            "centroid_col": float(region.centroid[1]),
        }


class ResNet50Layer3Encoder(nn.Module):
    """Return the globally pooled 1024-channel output of ResNet50 layer3."""

    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        features = self.features(images)
        features = self.pool(features)
        return torch.flatten(features, 1)


def extract_resnet50_cell_features(
    image,
    labels,
    batch_size=32,
    padding=8,
    device=None,
):
    """Extract one 1024-dimensional ResNet50 feature vector per cell."""
    if labels.max() == 0:
        raise ValueError("No segmented cells are available for feature extraction.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CellCropDataset(image, labels, padding=padding)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    model = ResNet50Layer3Encoder().to(device)
    model.eval()

    all_features = []
    all_labels = []
    all_centroid_rows = []
    all_centroid_cols = []

    with torch.inference_mode():
        for batch in loader:
            batch_images = batch["image"].to(device, non_blocking=True)
            batch_features = model(batch_images).cpu().numpy()

            all_features.append(batch_features)
            all_labels.extend(batch["label"].numpy().tolist())
            all_centroid_rows.extend(batch["centroid_row"].numpy().tolist())
            all_centroid_cols.extend(batch["centroid_col"].numpy().tolist())

    feature_matrix = np.concatenate(all_features, axis=0)

    if feature_matrix.shape[1] != 1024:
        raise RuntimeError(
            f"Expected 1024 features per cell, but received {feature_matrix.shape[1]}."
        )

    feature_columns = [f"feature_{index:04d}" for index in range(1024)]
    feature_df = pd.DataFrame(feature_matrix, columns=feature_columns)
    feature_df.insert(0, "centroid_col", all_centroid_cols)
    feature_df.insert(0, "centroid_row", all_centroid_rows)
    feature_df.insert(0, "label", all_labels)

    return feature_df


# ============================================================
# 5. Segmentation visualization
# ============================================================


def visualize_segmentation(original, preprocessing, segmentation_result, output_dir):
    """Save the segmentation quality-control figure as result1.png."""
    labels = segmentation_result["labels"]

    overlay = label2rgb(labels, image=original, bg_label=0, alpha=0.30)
    boundaries = find_boundaries(labels, mode="outer")
    boundary_overlay = color.gray2rgb(original)
    boundary_overlay[boundaries] = [1, 0, 0]

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original image")

    axes[0, 1].imshow(preprocessing["coverage_score"], cmap="magma")
    axes[0, 1].set_title("Cell-coverage score")

    axes[0, 2].imshow(preprocessing["coverage_mask"], cmap="gray")
    axes[0, 2].set_title("Cell-coverage mask")

    axes[1, 0].imshow(segmentation_result["gradient"], cmap="magma")
    axes[1, 0].set_title("Smoothed gradient")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"Segmented cells: {labels.max()}")

    axes[1, 2].imshow(boundary_overlay)
    axes[1, 2].set_title("Boundary check")

    for axis in axes.flat:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "result1.png", dpi=300)
    plt.close(fig)


# ============================================================
# 6. Main analysis function
# ============================================================


def analyze_phase_contrast_cells(
    image_path,
    output_dir="cell_analysis_output",
    batch_size=32,
    crop_padding=8,
    device=None,
):
    """Segment cells and extract a 1024-dimensional ResNet50 vector per cell."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_gray_image(image_path)

    preprocessing = preprocess_phase_contrast(
        image,
        background_sigma=25,
        denoise_sigma=1.2,
        clahe_clip=0.015,
    )

    coverage_mask, coverage_score = estimate_cell_coverage_mask(
        image,
        preprocessing["enhanced"],
        texture_sigma=5.0,
        gradient_sigma=2.0,
        threshold_scale=0.85,
        closing_radius=7,
        opening_radius=2,
        min_coverage_area=300,
        max_hole_area=1200,
    )

    preprocessing["coverage_mask"] = coverage_mask
    preprocessing["coverage_score"] = coverage_score

    print(f"Estimated cell coverage: {coverage_mask.mean() * 100:.1f}%")

    segmentation_result = segment_confluent_cells(
        preprocessing["enhanced"],
        coverage_mask=coverage_mask,
        gradient_sigma=4.0,
        minima_h=0.015,
        compactness=0.0005,
        min_area=80,
        max_area=5000,
        remove_border_cells=True,
    )

    labels = segmentation_result["labels"]
    print("Cell regions after area filtering:", labels.max())

    features = extract_resnet50_cell_features(
        preprocessing["enhanced"],
        labels,
        batch_size=batch_size,
        padding=crop_padding,
        device=device,
    )

    visualize_segmentation(
        image,
        preprocessing,
        segmentation_result,
        output_dir,
    )

    feature_path = output_dir / "cell_resnet50_features_1024.csv"
    features.to_csv(feature_path, index=False)

    coverage_mask_path = output_dir / "cell_coverage_mask.png"
    io.imsave(
        coverage_mask_path,
        preprocessing["coverage_mask"].astype(np.uint8) * 255,
        check_contrast=False,
    )

    coverage_score_path = output_dir / "cell_coverage_score.png"
    io.imsave(
        coverage_score_path,
        util.img_as_ubyte(
            exposure.rescale_intensity(
                preprocessing["coverage_score"],
                out_range=(0, 1),
            )
        ),
        check_contrast=False,
    )

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

    instance_overlay = label2rgb(
        labels,
        image=image,
        bg_label=0,
        alpha=0.35,
    )
    io.imsave(
        output_dir / "instance_overlay.png",
        util.img_as_ubyte(instance_overlay),
    )

    print("\nResults saved to:", output_dir.resolve())
    print("ResNet50 feature file:", feature_path)
    print("Cell-coverage mask:", coverage_mask_path)
    print("Cell-coverage score:", coverage_score_path)
    print("Instance mask:", instance_mask_path)
    print("Feature matrix shape:", features.shape)

    return {
        "image": image,
        "preprocessing": preprocessing,
        "segmentation": segmentation_result,
        "features": features,
    }


if __name__ == "__main__":
    result = analyze_phase_contrast_cells(
        image_path="cells.jpg",
        output_dir="cell_analysis_output",
        batch_size=32,
        crop_padding=8,
    )
