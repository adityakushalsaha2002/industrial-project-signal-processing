"""
Visualization module: plot full signals with detected peaks.
FIXED: Correctly locates peak metadata files in Region/ directory.
PARALLELIZED: Uses ProcessPoolExecutor for batch processing.
UPDATED: Displays calculated tumor volume in legend.
"""
import time
import pandas as pd
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from modules.preprocessing import CFG
from modules.region_detection import REGION_CONFIG
from modules.peak_detection import PEAK_CONFIG
from modules.tumor_volume_estimation import VOLUME_CONFIG

VISUALIZATION_CONFIG = {
    'results_dir': Path('./Test signals from previous years/Suction data/Results')
}
PREPROCESSING_DIR = CFG['output_dir']
REGION_DIR = REGION_CONFIG['output_dir']
# Convert string path to Path for file operations
VOLUME_DIR = Path(VOLUME_CONFIG['output_dir'])

def plot_full_signal_with_peaks(bin_path_str):
    """Plot complete signal with peaks + active/inactive region shading.
    
    Note: Takes string path for multiprocessing compatibility.
    """
    bin_path = Path(bin_path_str)  # Convert string back to Path
    VISUALIZATION_CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    output_path = VISUALIZATION_CONFIG['results_dir'] / (bin_path.stem + '_full_signal.png')
    if output_path.exists():
        return {'status': 'skipped', 'file': bin_path.name}

    # Load preprocessed data
    meta_path = bin_path.with_suffix('.meta.json')
    if not meta_path.exists():
        return {'status': 'error', 'file': bin_path.name, 'reason': 'meta.json not found'}
    
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
    
    regions_found = 0
    if region_meta_path.exists():
        region_meta = json.loads(region_meta_path.read_text())
        stored_regions = region_meta.get('stored_regions', [])
        regions_found = len(stored_regions)
        
        # Shade ALL active regions in light red (draw FIRST so signal overlays)
        for region in stored_regions:
            start_t = region['global_start'] / meta['target_fs']
            end_t = region['global_end'] / meta['target_fs']
            ax.axvspan(start_t, end_t, color='#ffcccc', alpha=0.4, zorder=0, linewidth=0)
    else:
        pass  # Silent fail for missing region meta

    # =============================
    # STEP 2: Plot signal (on top of shaded regions)
    # =============================
    ax.plot(time_axis, centered, 'b-', linewidth=0.3, alpha=0.6, label='Signal', zorder=1)

    # =============================
    # STEP 3: Load and mark PEAKS
    # =============================
    peak_meta_name = f"{bin_path.stem}_regions_peaks.meta.json"
    peak_meta_path = PEAK_CONFIG['peaks_dir'] / peak_meta_name
    
    num_peaks = 0
    if peak_meta_path.exists():
        peak_meta = json.loads(peak_meta_path.read_text())
        detected_peaks = peak_meta.get('detected_peaks', [])
        
        all_peak_times = []
        all_peak_values = []
        
        for region_peak in detected_peaks:
            region_start = region_peak['region_start_sec']
            for t, v in zip(region_peak['local_times_sec'], region_peak['peak_values']):
                all_peak_times.append(region_start + t)
                all_peak_values.append(v)
        
        num_peaks = len(all_peak_times)
        
        if num_peaks > 0:
            ax.scatter(all_peak_times, all_peak_values,
                      c='red', s=8, zorder=3, marker='^',
                      edgecolors='darkred', linewidths=0.3,
                      label=f'Detected peaks (n={num_peaks})')

    # =============================
    # STEP 4: Load VOLUME data (n_eff SE band, dimensionless units)
    # =============================
    volume_meta_name = f"{bin_path.stem}_regions_volume.meta.json"
    volume_meta_path = VOLUME_DIR / volume_meta_name

    volume_display = ""
    bounds_annotation = ""
    total_units = None
    units_range = None
    volume_meta = None
    if volume_meta_path.exists():
        volume_meta = json.loads(volume_meta_path.read_text())
        total_units = volume_meta.get('total_units', 0)
        lower = volume_meta.get('total_units_lower', total_units)
        upper = volume_meta.get('total_units_upper', total_units)
        units_range = f"{lower:.1f}-{upper:.1f}"
        num_regions_with_peaks = volume_meta.get('num_regions_with_peaks', 0)
        total_peaks = volume_meta.get('total_peak_count', 0)

        volume_display = (f"Estimate: {units_range} units (point {total_units:.1f}) | "
                          f"Regions w/ peaks: {num_regions_with_peaks}")

        # Optional annotations — present only if the volume module logs them
        region_details = volume_meta.get('region_details', [])
        ks = [r.get('correlation_factor_k') for r in region_details]
        ks = [k for k in ks if k is not None]
        if ks:
            bounds_annotation = (f"Band: {volume_meta.get('bounds_method', 'n_eff SE')} | "
                                 f"median k = {np.median(ks):.2f}")
    
    # =============================
    # Labels and title
    # =============================
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Amplitude (centered)', fontsize=10)
    
    # Title assembly
    title_parts = [f"{bin_path.name}",
                   f"Duration: {meta['duration_sec']:.1f}s | {meta['samples']:,} samples"]
    if num_peaks > 0:
        title_parts.append(f"Peaks: {num_peaks}")
    if volume_display:
        title_parts.append(volume_display)
    ax.set_title("\n".join(title_parts), fontsize=11, fontweight='bold')

    if bounds_annotation:
        ax.text(0.01, 0.98, bounds_annotation, transform=ax.transAxes,
                fontsize=8, va='top', alpha=0.7, style='italic')
    
    ax.set_title("\n".join(title_parts), fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    output_path = VISUALIZATION_CONFIG['results_dir'] / (bin_path.stem + '_full_signal.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Explicitly close figure

    return {
        'status': 'completed',
        'file': bin_path.name,
        'output_file': str(output_path),
        'num_regions': regions_found,
        'num_peaks': num_peaks,
        'total_units': total_units,
        'units_range': units_range,
    }

def run_full_batch_visualization(num_workers=None):
    """Batch process all files for visualization in parallel."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    bin_files = list(PREPROCESSING_DIR.glob('*.bin'))

    results = []
    total_time_start = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(plot_full_signal_with_peaks, str(f)): f for f in bin_files}
        
        for future in as_completed(futures):
            result = future.result()
            status = "✅" if result.get('status') == 'completed' else "❌"
            vol_info = (f"Units: {result.get('units_range', 'n/a')}" if result.get('units_range') is not None else "No estimate")
            print(f"{status}: {result['file']} | Regions: {result.get('num_regions', 0)} | Peaks: {result.get('num_peaks', 0)} | {vol_info}")
            results.append(result)

    elapsed_time = time.time() - total_time_start
    completed = sum(1 for r in results if r.get('status') == 'completed')

    # Save summary CSV
    df = pd.DataFrame(results)
    VISUALIZATION_CONFIG['results_dir'].parent.mkdir(exist_ok=True)
    df.to_csv(VISUALIZATION_CONFIG['results_dir'] / 'visualization_summary.csv', index=False)
    
    return results