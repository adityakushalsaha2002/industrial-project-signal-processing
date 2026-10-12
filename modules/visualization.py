"""
Visualization module: plot full signals with detected peaks.
FIXED: Correctly locates peak metadata files in Region/ directory.
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing as mp
from modules.preprocessing import CFG
from modules.region_detection import REGION_CONFIG
from modules.peak_detection import PEAK_CONFIG


VISUALIZATION_CONFIG = {
    'results_dir': Path('./Test signals from previous years/Suction data/Results')
}
PREPROCESSING_DIR = CFG['output_dir']
REGION_DIR = REGION_CONFIG['output_dir']


def plot_full_signal_with_peaks(bin_path):
    """Plot complete signal with peaks + active/inactive region shading."""
    VISUALIZATION_CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    meta_path = bin_path.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text())
    raw_data = np.memmap(str(bin_path), dtype='uint16', mode='r')
    centered = (raw_data.astype(np.int32) - meta['value_offset'])
    time_axis = np.arange(len(centered)) / meta['target_fs']

    fig, ax = plt.subplots(figsize=(16, 4))

    # =============================
    # STEP 1: Load REGION metadata (NOT peak metadata)
    # =============================
    region_meta_name = f"{bin_path.stem}_regions.meta.json"
    region_meta_path = REGION_DIR / region_meta_name  # This is KEY!
    
    if region_meta_path.exists():
        region_meta = json.loads(region_meta_path.read_text())
        stored_regions = region_meta.get('stored_regions', [])
        
        # Shade ALL active regions in light red (draw FIRST so signal overlays)
        for region in stored_regions:
            start_t = region['global_start'] / meta['target_fs']
            end_t = region['global_end'] / meta['target_fs']
            ax.axvspan(start_t, end_t, color='#ffcccc', alpha=0.4, zorder=0, linewidth=0)
        
        print(f"  ℹ️ Found {len(stored_regions)} active regions to shade")
    else:
        print(f"  ⚠️ Region meta not found: {region_meta_path}")

    # =============================
    # STEP 2: Plot signal (on top of shaded regions)
    # =============================
    ax.plot(time_axis, centered, 'b-', linewidth=0.3, alpha=0.6, label='Signal', zorder=1)

    # =============================
    # STEP 3: Load and mark PEAKS
    # =============================
    peak_meta_name = f"{bin_path.stem}_regions_peaks.meta.json"  # Check your actual save location
    peak_meta_path = PEAK_CONFIG['peaks_dir'] / peak_meta_name
    all_peak_times = []
    all_peak_values = []
    
    if peak_meta_path.exists():
        peak_meta = json.loads(peak_meta_path.read_text())
        detected_peaks = peak_meta.get('detected_peaks', [])
        
        for region_peak in detected_peaks:
            region_start = region_peak['region_start_sec']
            for t, v in zip(region_peak['local_times_sec'], region_peak['peak_values']):
                all_peak_times.append(region_start + t)
                all_peak_values.append(v)
        
        if len(all_peak_times) > 0:
            ax.scatter(all_peak_times, all_peak_values,
                      c='red', s=8, zorder=3, marker='^',
                      edgecolors='darkred', linewidths=0.3,
                      label=f'Detected peaks (n={len(all_peak_times)})')
        print(f"  ✅ Found peaks: {len(all_peak_times)}")
    else:
        print(f"  ⚠️ Peak metadata not found: {peak_meta_path}")

    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Amplitude (centered)', fontsize=10)
    ax.set_title(f"{bin_path.name}\nDuration: {meta['duration_sec']:.1f}s | {meta['samples']:,} samples | Peaks: {len(all_peak_times) if len(all_peak_times) > 0 else 0}",
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    output_path = VISUALIZATION_CONFIG['results_dir'] / (bin_path.stem + '_full_signal.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def run_full_batch_visualization():
    """Batch process all files for visualization."""
    bin_files = sorted(PREPROCESSING_DIR.glob('*.bin'))

    print(f"Preprocessed data in: {PREPROCESSING_DIR}")
    print(f"Peak files in: {PEAK_CONFIG['peaks_dir']}")
    print(f"Images will be saved to: {VISUALIZATION_CONFIG['results_dir']}")
    print()

    plot_paths = []
    for bin_path in bin_files:
        try:
            plot_path = plot_full_signal_with_peaks(bin_path)
            plot_paths.append(plot_path)
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone! Generated {len(plot_paths)} plots")
    return plot_paths


if __name__ == '__main__':
    run_full_batch_visualization()