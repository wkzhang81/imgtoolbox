"""
Similar-image grouping (multi-orientation pHash + SIFT + RANSAC + union-find).

Original script: fig_discrimination/group_similar_images.py
The script only has an argparse main; here we build a Namespace and call its process().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from ..loader import load_script
from ..registry import Param, Tool, register


def run(params: dict) -> str:
    m = load_script("fig_discrimination")

    directory = Path(params["directory"]).resolve()
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    args = argparse.Namespace(
        directory=directory,
        dry_run=bool(params["dry_run"]),
        copy=bool(params["copy"]),
        group_prefix="similar_group",
        report_name="similarity_report.csv",
        max_size=int(params["max_size"]),
        phash_size=int(params["phash_size"]),
        phash_distance=int(params["phash_distance"]),
        no_phash_filter=bool(params["no_phash_filter"]),
        include_flips=bool(params["include_flips"]),
        max_ratio=float(params["max_ratio"]),
        min_raw_matches=int(params["min_raw_matches"]),
        min_inliers=int(params["min_inliers"]),
        min_inlier_ratio=float(params["min_inlier_ratio"]),
        min_normalized_inliers=float(params["min_normalized_inliers"]),
        min_coverage=float(params["min_coverage"]),
        min_spatial_spread=float(params["min_spatial_spread"]),
        residual_threshold=float(params["residual_threshold"]),
        max_trials=int(params["max_trials"]),
    )

    m.validate_args(args)
    summary = m.process(args)

    return (
        f"Directory: {summary['directory']}\n"
        f"Images: {summary['n_images']}\n"
        f"Similar groups: {summary['n_groups']}\n"
        f"Similar images: {summary['n_grouped']}\n"
        f"Report: {summary['report_path']}\n"
        f"Mode: {'preview (no move)' if args.dry_run else ('copy' if args.copy else 'move')}\n"
        f"Elapsed: {summary['elapsed']:.1f}s"
    )


register(Tool(
    key="fig_discrimination",
    label="Similar-image grouping",
    description=(
        "Use multi-orientation/multi-crop pHash prefiltering + SIFT/RANSAC precise matching to detect "
        "near-duplicates from rescaling, recompression, rotation, cropping, borders, perspective, watermarks, "
        "etc., and file them into subdirectories by group."
    ),
    params=[
        Param(
            key="directory",
            label="Image directory to process",
            type="dir",
            help="Images in the top level of this directory are scanned (non-recursive).",
        ),
        Param(key="dry_run", label="Preview only (no move/copy)", type="bool", default=True),
        Param(key="copy", label="Copy (otherwise move)", type="bool", default=False),
        Param(key="include_flips", label="Also detect horizontal flips", type="bool", default=False),
        Param(key="no_phash_filter", label="Disable pHash prefilter (all-pairs SIFT)", type="bool", default=False),

        Param(key="max_size", label="SIFT max long-edge pixels", type="int", default=1600, min=128, max=8000),
        Param(key="phash_size", label="pHash edge length", type="int", default=16, min=4, max=32),
        Param(key="phash_distance", label="pHash distance threshold", type="int", default=100, min=0, max=1024),

        Param(key="max_ratio", label="Lowe ratio threshold", type="float", default=0.80, min=0.1, max=1.0),
        Param(key="min_raw_matches", label="Minimum raw matches", type="int", default=9, min=4, max=1000),
        Param(key="min_inliers", label="Minimum RANSAC inliers", type="int", default=10, min=4, max=1000),
        Param(key="min_inlier_ratio", label="Minimum inlier ratio", type="float", default=0.25, min=0, max=1),
        Param(key="min_normalized_inliers", label="Normalized inlier lower bound", type="float", default=0.018, min=0, max=1),
        Param(key="min_coverage", label="Minimum coverage", type="float", default=0.035, min=0, max=1),
        Param(key="min_spatial_spread", label="Minimum spatial spread", type="float", default=0.08, min=0, max=1),
        Param(key="residual_threshold", label="RANSAC residual threshold (px)", type="float", default=5.0, min=0.1, max=100),
        Param(key="max_trials", label="RANSAC max iterations", type="int", default=2000, min=10, max=50000),
    ],
    run=run,
))
