"""Detection of repeated regions within a single image (SIFT split in half + match_descriptors).

Can be run as a script:
    python find_dup.py <image path> [--ratio 0.5] [--min-dist 20]
Can also be imported by imgtoolbox to call detect_duplicate_with_match_descriptors().
"""

import os


def detect_duplicate_with_match_descriptors(image_path, max_distance_ratio=0.5,
                                            min_dist=20, output_dir=None):
    """
    Detect repeated regions within a single image by cleverly splitting the
    feature set and using the match_descriptors function.
    Returns (result image path, number of duplicate pairs).
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import io, color
    from skimage.feature import SIFT, match_descriptors

    # 1. Read the image and convert it to grayscale
    img = io.imread(image_path)
    if img.ndim == 3:
        img_gray = color.rgb2gray(img[..., :3])
    else:
        img_gray = img
    img_disp = img[..., :3] if img.ndim == 3 else img

    # 2. Extract SIFT features
    detector = SIFT()
    detector.detect_and_extract(img_gray)

    keypoints = detector.keypoints
    descriptors = detector.descriptors

    print(f"Total SIFT keypoints found: {len(keypoints)}")
    if len(keypoints) < 4:
        print("Too few keypoints to perform block matching.")
        return None, 0

    # 3. Core trick: split the keypoint set in half to break the deadlock of
    # "matching a set against itself"
    # We let the first half of the keypoints (set A) match the second half (set B)
    mid = len(keypoints) // 2

    kp_A, desc_A = keypoints[:mid], descriptors[:mid]
    kp_B, desc_B = keypoints[mid:], descriptors[mid:]

    # 4. Strictly call the match_descriptors function
    # Since A and B are different sets, match_descriptors can now work properly!
    matches = match_descriptors(desc_A, desc_B,
                                metric='euclidean',
                                max_ratio=max_distance_ratio)

    print(f"Preliminary number of matched pairs found by match_descriptors: {len(matches)}")

    # 5. Filter out false matches whose spatial distance is too small
    # (e.g. two consecutive points along the same texture edge)
    valid_matches = []
    for m in matches:
        idx_A, idx_B = m[0], m[1]

        kp1 = kp_A[idx_A]
        kp2 = kp_B[idx_B]  # Note: this indexes into set B

        # Compute the actual physical distance between the two points in the image
        dist = np.linalg.norm(kp1 - kp2)

        if dist > min_dist:
            valid_matches.append((kp1, kp2))

    print(f"After removing near-neighbor false positives, number of red line pairs finally drawn: {len(valid_matches)}")

    # 6. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img_disp)
    ax[0].set_title('original figure')

    ax[1].imshow(img_disp)
    # Draw all SIFT keypoints
    ax[1].scatter(keypoints[:, 1], keypoints[:, 0], s=10,
                  facecolors='none', edgecolors='blue', alpha=0.5,
                  label='SIFT Keypoints')

    # Draw the matched red lines
    if len(valid_matches) > 0:
        for kp1, kp2 in valid_matches:
            ax[1].plot([kp1[1], kp2[1]], [kp1[0], kp2[0]], color='red',
                       linewidth=1.5, alpha=0.8)
            ax[1].scatter([kp1[1], kp2[1]], [kp1[0], kp2[0]], s=30, color='red')
        # Fake a legend entry
        ax[1].plot([], [], color='red', linewidth=1.5, label='Detected Duplicates')
    else:
        print("No cross-region duplicate tiles detected. Try raising max_distance_ratio (e.g. 0.8)")

    ax[1].set_title(f"Duplicate Detection via match_descriptors (Pairs: {len(valid_matches)})")

    for a in ax.ravel():
        a.axis('off')
    plt.legend(loc='upper right')

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(image_path)) or '.'
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f'dup_detect_{stem}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path, len(valid_matches)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect repeated regions within a single image")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--ratio", type=float, default=0.5, help="Lowe ratio")
    parser.add_argument("--min-dist", type=float, default=20, help="Minimum spatial distance (px)")
    args = parser.parse_args()

    detect_duplicate_with_match_descriptors(
        args.image, max_distance_ratio=args.ratio, min_dist=args.min_dist
    )
