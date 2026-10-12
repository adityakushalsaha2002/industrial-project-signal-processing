#!/usr/bin/env python3
"""
Main pipeline orchestration script.
"""
from modules.preprocessing import run_batch_preprocessing
from modules.region_detection import run_batch_region_extraction
from modules.peak_detection import run_batch_peak_detection
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

    # Step 4: Visualization (sequential)
    print("\n[STEP 4] VISUALIZATION")
    print("-" * 40)
    plot_paths = run_full_batch_visualization()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()