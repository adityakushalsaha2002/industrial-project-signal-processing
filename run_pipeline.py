#!/usr/bin/env python3
"""
Main pipeline orchestration script.
"""
import sys

# Status lines use emoji; keep them from crashing on Windows consoles/pipes
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

from modules.preprocessing import run_batch_preprocessing
from modules.region_detection import run_batch_region_extraction
from modules.peak_detection import run_batch_peak_detection
from modules.tumor_volume_estimation import run_batch_volume_estimation
from modules.visualization import run_full_batch_visualization


def main():
    print("=" * 60)
    print("SIGNAL PROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Preprocessing (parallelized, skips existing)
    print("\n[STEP 1] PREPROCESSING")
    print("-" * 40)
    preprocessing_results = run_batch_preprocessing()

    # Step 2: Region Detection (parallelized, skips existing)
    print("\n[STEP 2] REGION DETECTION")
    print("-" * 40)
    region_results = run_batch_region_extraction()

    # Step 3: Peak Detection (sequential, can be parallelized if needed)
    print("\n[STEP 3] PEAK DETECTION")
    print("-" * 40)
    peak_results = run_batch_peak_detection()

    # Step 4: Tumor Volume Estimation (parallelized)
    print("\n[STEP 4] TUMOR VOLUME ESTIMATION")
    print("-" * 40)
    volume_results = run_batch_volume_estimation()

    # Step 5: Visualization (parallelized)
    print("\n[STEP 5] VISUALIZATION")
    print("-" * 40)
    plot_paths = run_full_batch_visualization()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()