"""
Tumor volume estimation module: integrate signal values within regions containing peaks.
Only regions with detected peaks contribute to the total tumor collection estimate.
"""
import numpy as np
import json
from pathlib import Path
import pandas as pd
from modules.region_detection import REGION_CONFIG
from modules.peak_detection import PEAK_CONFIG

VOLUME_CONFIG = {
    'output_dir': './Test signals from previous years/Suction data/Volume',  # Changed from Path to string
    
    # Negative value handling: 'ignore', 'absolute', or 'signed'
    'negative_handling': 'ignore',  # Set to 'absolute' to count absolute values, 'signed' for raw integration
    
    # Integration parameters
    'baseline_correction': True,     # Subtract region baseline before integration
    'min_signal_value': 0,           # Floor for integration (prevents negative contributions if 'ignore')
}
Path(VOLUME_CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

def load_region_and_peak_data(region_bin_path):
    """Load region data and corresponding peak metadata."""
    # Load region data
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

    # Load peak metadata
    peak_meta_name = f"{region_bin_path.stem}_peaks.meta.json"
    peak_meta_path = PEAK_CONFIG['peaks_dir'] / peak_meta_name
    
    if not peak_meta_path.exists():
        return None
    
    peak_meta = json.loads(peak_meta_path.read_text())
    
    return {
        'regions': regions,
        'num_regions': len(regions),
        'meta': meta,
        'peak_meta': peak_meta,
        'detected_peaks': peak_meta.get('detected_peaks', []),
    }

def calculate_region_volume(region_data_result):
    """Calculate integrated tumor volume for regions containing peaks."""
    regions = region_data_result['regions']
    detected_peaks = region_data_result['detected_peaks']
    config = region_data_result['config']
    fs = region_data_result['meta']['target_fs']
    
    region_volumes = []
    total_volume = 0
    total_positive_volume = 0
    total_negative_volume = 0
    peak_count = 0
    
    for reg_idx, region in enumerate(regions):
        # Check if this region has peaks
        region_has_peaks = False
        for peak_record in detected_peaks:
            if peak_record['region_index'] == reg_idx:
                region_has_peaks = True
                peak_count += peak_record['num_peaks']
                break
        
        # Skip regions without peaks
        if not region_has_peaks:
            continue
        
        data = region['data'].copy()
        
        # Apply baseline correction if enabled
        if config.get('baseline_correction', False):
            baseline = np.median(data)
            data = data - baseline
        
        # Handle negative values based on configuration
        if config['negative_handling'] == 'ignore':
            # Set negative values to 0 (conservative estimate)
            data = np.maximum(data, 0)
        elif config['negative_handling'] == 'absolute':
            # Take absolute value of all samples
            data = np.abs(data)
        # 'signed' leaves data unchanged
        
        # Apply minimum signal floor
        min_val = config.get('min_signal_value', 0)
        data = np.maximum(data, min_val)
        
        # Integrate (sum * sampling interval for physical units)
        region_integral = np.sum(data) / fs  # Convert samples to seconds
        
        region_volumes.append({
            'region_index': reg_idx,
            'region_start_sec': region['timestamp_sec'],
            'duration_sec': region['duration_sec'],
            'peak_count_in_region': sum(1 for p in detected_peaks if p['region_index'] == reg_idx),
            'integrated_volume': round(region_integral, 4),
            'max_amplitude': int(np.max(data)),
            'mean_amplitude': round(np.mean(data), 2),
        })
        
        total_volume += region_integral
        
        # Track positive/negative separately for reporting
        positive_sum = np.sum(np.maximum(data, 0)) / fs
        negative_sum = np.sum(np.minimum(data, 0)) / fs
        total_positive_volume += positive_sum
        total_negative_volume += negative_sum
    
    return {
        'total_volume_units': round(total_volume, 4),
        'total_positive_volume': round(total_positive_volume, 4),
        'total_negative_volume': round(total_negative_volume, 4),
        'total_peak_count': peak_count,
        'num_regions_with_peaks': len(region_volumes),
        'total_regions': len(regions),
        'region_details': region_volumes,
    }

def run_batch_volume_estimation(num_workers=None):
    """Process all region files and calculate tumor volume estimates."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    # Convert string back to Path for file operations
    output_dir = Path(VOLUME_CONFIG['output_dir'])
    region_files = sorted(REGION_CONFIG['output_dir'].glob('*_regions.bin'))
    
    print(f"Processing {len(region_files)} region files for volume estimation...")
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        for region_bin_path in region_files:
            try:
                # Prepare data for worker
                region_data_result = load_region_and_peak_data(region_bin_path)
                
                if region_data_result is None:
                    print(f"⏭️ SKIPPED: {region_bin_path.name} (no peaks detected)")
                    all_results.append({
                        'file': region_bin_path.name,
                        'status': 'skipped',
                        'reason': 'no peaks',
                    })
                    continue
                
                region_data_result['config'] = VOLUME_CONFIG
                futures[executor.submit(calculate_region_volume, region_data_result)] = region_bin_path
                
            except Exception as e:
                print(f"⚠️ ERROR loading {region_bin_path.name}: {e}")
                all_results.append({
                    'file': region_bin_path.name,
                    'status': 'error',
                    'reason': str(e),
                })
        
        for future in as_completed(futures):
            result = future.result()
            region_bin_path = futures[future]
            
            if result['num_regions_with_peaks'] == 0:
                print(f"⏭️ SKIPPED: {region_bin_path.name} (no peaks in any region)")
                all_results.append({
                    'file': region_bin_path.name,
                    'status': 'skipped',
                    'reason': 'no peaks in regions',
                })
            else:
                print(f"✅ {region_bin_path.name}: {result['total_peak_count']} peaks | "
                      f"{result['total_volume_units']:.2f} units | "
                      f"{result['num_regions_with_peaks']}/{result['total_regions']} regions")
                
                # Save individual file volume metadata (use string version of path for JSON)
                volume_meta_path = output_dir / (region_bin_path.stem + '_volume.meta.json')
                
                # Create a serializable version of the config (convert Paths to strings)
                serializable_config = {k: str(v) if isinstance(v, Path) else v 
                                      for k, v in VOLUME_CONFIG.items()}
                
                volume_meta = {
                    'source_file': region_bin_path.name,
                    'volume_config': serializable_config,
                    **result,
                }
                volume_meta_path.write_text(json.dumps(volume_meta, indent=2))
                
                all_results.append({
                    'file': region_bin_path.name,
                    'status': 'completed',
                    'total_volume_units': result['total_volume_units'],
                    'total_positive_volume': result['total_positive_volume'],
                    'total_negative_volume': result['total_negative_volume'],
                    'total_peak_count': result['total_peak_count'],
                    'num_regions_with_peaks': result['num_regions_with_peaks'],
                    'total_regions': result['total_regions'],
                })
    
    # Save summary CSV
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(output_dir / 'volume_estimation_summary.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE! Processed {len([r for r in all_results if r.get('status') == 'completed'])} files")
    print(f"{'='*60}")
    
    # Show summary statistics
    completed_results = [r for r in all_results if r.get('status') == 'completed']
    if completed_results:
        total_all_files = sum(r['total_volume_units'] for r in completed_results)
        total_all_peaks = sum(r['total_peak_count'] for r in completed_results)
        avg_per_file = total_all_files / len(completed_results)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total files processed: {len(completed_results)}")
        print(f"   Total tumor volume: {total_all_files:.2f} units")
        print(f"   Total peaks detected: {total_all_peaks}")
        print(f"   Average per file: {avg_per_file:.2f} units")
    
    return all_results