"""
Peak detection module: find peaks within detected regions.
"""
import numpy as np
import json
from pathlib import Path
from scipy.signal import find_peaks
import pandas as pd
from modules.region_detection import REGION_CONFIG

PEAK_CONFIG = {
    'peaks_dir': Path('./Test signals from previous years/Suction data/Peaks'),

    # Peak detection parameters
    'min_peak_height': 25,          # Minimum height above baseline (in centered units)
    'min_peak_percentage': 5,       # Scales with file amplitude
    'min_distance_samples': 10,     # min_distance_samples/Hz = seconds between peaks, where Hz is the downsampled Hz
    'prominence': 0,                # Peak prominence threshold
    'prominence_percentage': 0,     # Scales with file amplitude
}
PEAK_CONFIG['peaks_dir'].mkdir(parents=True, exist_ok=True)


def load_region_data(region_bin_path):
    """Load region-based data (plain .bin files only)."""
    meta_path = region_bin_path.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text())

    regions_data = np.fromfile(region_bin_path, dtype=np.uint16)
    centered = regions_data.astype(np.int32) - meta['value_offset']

    regions = []
    idx = 0
    for region_info in meta['stored_regions']:
        region_size = region_info['global_end'] - region_info['global_start']
        region_data = centered[idx:idx + region_size]
        regions.append({
            'timestamp_sec': region_info['global_start'] / meta['target_fs'],
            'data': region_data,
            'duration_sec': region_info['duration_sec'],
        })
        idx += region_size

    return {
        'regions': regions,
        'num_regions': len(regions),
        'meta': meta,
    }


def detect_peaks_in_regions(region_data_result):
    """Detect peaks in each loaded region using scipy.signal.find_peaks."""
    all_peaks = []
    regions = region_data_result['regions']
    fs = region_data_result['meta']['target_fs']

    for reg_idx, region in enumerate(regions):
        data = region['data']
        time_axis = np.arange(len(data)) / fs

        signal_amplitude = np.max(np.abs(data)) - np.min(np.abs(data))
        adaptive_height = max(PEAK_CONFIG['min_peak_height'], signal_amplitude * PEAK_CONFIG['min_peak_percentage'] / 100)
        adaptive_prominence = max(PEAK_CONFIG['prominence'], signal_amplitude * PEAK_CONFIG['prominence_percentage'] / 100)

        peaks, properties = find_peaks(
            data,
            height=adaptive_height,
            distance=PEAK_CONFIG['min_distance_samples'],
            prominence=adaptive_prominence
        )

        region_peaks = {
            'region_index': reg_idx,
            'region_start_sec': region['timestamp_sec'],
            'num_peaks': len(peaks),
            'peak_times_sec': (time_axis[peaks] + region['timestamp_sec']).tolist(),
            'peak_values': data[peaks].tolist(),
            'local_times_sec': time_axis[peaks].tolist(),
        }
        all_peaks.append(region_peaks)

    return all_peaks


def run_batch_peak_detection():
    """Process all region files and save peak detection results."""
    region_files = sorted(REGION_CONFIG['output_dir'].glob('*_regions.bin'))

    all_results = []

    for region_bin_path in region_files:
        try:
            print(f"Processing: {region_bin_path.name}")
            region_result = load_region_data(region_bin_path)
            peaks = detect_peaks_in_regions(region_result)

            file_total_peaks = sum(p['num_peaks'] for p in peaks)
            file_meta = region_result['meta']

            all_results.append({
                'file': region_bin_path.name,
                'num_regions': region_result['num_regions'],
                'total_peaks': file_total_peaks,
                'original_duration_sec': file_meta['original_duration_sec'],
                'region_efficiency': file_meta['region_efficiency'],
            })

            # Save individual peak metadata
            peak_meta_path = PEAK_CONFIG['peaks_dir'] / (region_bin_path.stem + '_peaks.meta.json')
            meta_output = {
                **file_meta,
                'detected_peaks': peaks,
                'total_peaks': file_total_peaks,
            }
            peak_meta_path.write_text(json.dumps(meta_output, indent=2))

        except Exception as e:
            print(f"{region_bin_path.name} | ERROR: {e}")
            all_results.append({'file': region_bin_path.name, 'error': str(e)})

    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(PEAK_CONFIG['peaks_dir'] / 'peak_detection_summary.csv', index=False)

    print(f"\nBatch complete! Processed {len(region_files)} files.")
    print(df_summary[['file', 'num_regions', 'total_peaks']].to_string())

    return all_results