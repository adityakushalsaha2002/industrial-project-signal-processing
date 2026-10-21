"""
Region detection module: activity-based region extraction.
ACTIVITY IS CALCULATED ON RAW PREPROCESSED DATA BEFORE ANY BASELINE REMOVAL.
"""
import numpy as np
import pandas as pd
import json, gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from modules.preprocessing import CFG

REGION_CONFIG = {
    'data_dir': Path('./Test signals from previous years/Suction data/Preprocessed'),
    'output_dir': Path('./Test signals from previous years/Suction data/Region'),

    # Activity detection thresholds (calculated on RAW preprocessed data)
    'activity_window_samples': 10,      # activity_window_samples*1000/Hz = ms of sliding window, where Hz is the downsampled Hz
    'activity_multiplier': 0.2,         # >activity_multiplier*baseline = active 
    'min_region_duration_sec': 0.01,    # Ignore regions shorter than this (noise)
    'padding_sec': 0,                   # Add buffer before/after each region, in seconds
}
REGION_CONFIG['target_fs'] = CFG['target_fs']
REGION_CONFIG['padding_samples'] = int(REGION_CONFIG['padding_sec'] * REGION_CONFIG['target_fs'])
REGION_CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

# wip: why are there more regions with peaks than peaks sometimes?
def detect_regions_and_extract(bin_path, cfg):
    """
    Detect contiguous active regions → store as blocks → save
    CRITICAL: Activity calculation happens BEFORE subtracting value_offset
    (using raw uint16 data as-is to capture true signal variance)
    """
    meta_path = bin_path.with_suffix('.meta.json')
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found for {bin_path}")

    meta = json.loads(meta_path.read_text())

    # Memory-map ORIGINAL data (uint16, NOT centered yet for activity calc)
    raw_data = np.memmap(str(bin_path), dtype='uint16', mode='r')

    # ACTIVITY CALCULATION: Use raw uint16 data directly
    # Rolling std captures local variance in the raw signal
    activity_std = pd.Series(raw_data).rolling(cfg['activity_window_samples'], center=True, min_periods=1).std().to_numpy()

    # PER-FILE BASELINE: Overall std of the ENTIRE file
    # wip: should be changed to rolling?
    baseline_std = np.std(raw_data.astype(np.float64))
    activity_threshold = cfg['activity_multiplier'] * baseline_std

    # Mark active samples
    active_mask = activity_std > activity_threshold

    # Find contiguous active regions
    regions = []
    in_region = False
    region_start = 0

    for i, is_active in enumerate(active_mask):
        if is_active and not in_region:
            region_start = i
            in_region = True
        elif not is_active and in_region:
            region_end = i

            padded_start = max(0, region_start - cfg['padding_samples'])
            padded_end = min(len(raw_data), region_end + cfg['padding_samples'])

            duration_sec = (padded_end - padded_start) / cfg['target_fs']
            if duration_sec >= cfg['min_region_duration_sec']:
                regions.append({
                    'global_start': int(padded_start),
                    'global_end': int(padded_end),
                    'duration_sec': round(duration_sec, 4),
                })

            in_region = False

    if in_region:
        padded_start = max(0, region_start - cfg['padding_samples'])
        padded_end = len(raw_data)
        duration_sec = (padded_end - padded_start) / cfg['target_fs']

        if duration_sec >= cfg['min_region_duration_sec']:
            regions.append({
                'global_start': int(padded_start),
                'global_end': int(padded_end),
                'duration_sec': round(duration_sec, 4),
            })

    # Extract and convert regions to centered int32 for storage
    all_region_data = []
    for region in regions:
        segment = (raw_data[region['global_start']:region['global_end']].astype(np.int32)
                   - meta['value_offset'])
        all_region_data.append(segment)

    if len(all_region_data) > 0:
        # Store back as uint16 with offset for compatibility
        compressed_regions = np.concatenate(all_region_data).astype(np.uint16) + meta['value_offset']
    else:
        compressed_regions = np.array([], dtype=np.uint16)

    # Stats
    total_stored_samples = len(compressed_regions)
    total_original_samples = meta['samples']
    compression_ratio = total_original_samples / total_stored_samples if total_stored_samples > 0 else 1.0
    total_region_duration = sum(r['duration_sec'] for r in regions)

    output_meta = {
        'source_file': bin_path.name,
        'original_duration_sec': meta['duration_sec'],
        'original_samples': total_original_samples,
        'num_regions': len(regions),
        'stored_regions': regions,
        'total_stored_samples': total_stored_samples,
        'compression_ratio': round(compression_ratio, 2),
        'original_size_mb': meta.get('original_size_mb', 0),
        'stored_size_mb': round(total_stored_samples * 2 / 1024**2, 3),
        'total_region_duration_sec': round(total_region_duration, 2),
        'region_efficiency': round(total_region_duration / meta['duration_sec'], 4),
        'baseline_std': round(baseline_std, 2),
        'activity_threshold': round(activity_threshold, 2),
        'value_offset': meta['value_offset'],
        'target_fs': cfg['target_fs'],
        'zstd_compressed': False,
    }

    output_path = cfg['output_dir'] / (bin_path.stem + '_regions.bin')
    meta_path_out = cfg['output_dir'] / (bin_path.stem + '_regions.meta.json')

    # SKIP if already exists
    # wip: these types of checks should happen sooner
    if output_path.exists():
        print(f"⏭️ SKIPPED: {bin_path.name} (regions already extracted)")
        return {'status': 'skipped', 'file': bin_path.name, **output_meta}

    if len(compressed_regions) > 0:
        compressed_regions.tofile(output_path)
    meta_path_out.write_text(json.dumps(output_meta, indent=2))

    print(f"✅ {output_meta['original_size_mb']:.2f}MB → {output_meta['stored_size_mb']:.2f}MB "
          f"({output_meta['region_efficiency']*100:.1f}% stored) | {bin_path.name}")

    return {'status': 'completed', 'file': bin_path.name, **output_meta}


def run_batch_region_extraction(num_workers=None):
    """Parallel batch region extraction."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    bin_files = sorted(REGION_CONFIG['data_dir'].glob('*.bin'))
    print(f"{len(bin_files)} files to process")

    results = []
    total_original_mb = 0
    total_stored_mb = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(detect_regions_and_extract, f, REGION_CONFIG): f for f in bin_files}
        for future in as_completed(futures):
            result = future.result()

            if result.get('status') != 'skipped':
                total_original_mb += result.get('original_size_mb', 0)
                total_stored_mb += result.get('stored_size_mb', 0)

            results.append({
                'file': result['file'],
                **result,
            })

    total_ratio = total_original_mb / total_stored_mb if total_stored_mb > 0 else 1
    if total_original_mb > 0:
        pct_stored = (total_stored_mb / total_original_mb * 100)
    else:
        pct_stored = 0.0
    print(f"\n{total_original_mb:.2f}MB → {total_stored_mb:.2f}MB "
        f"({pct_stored:.1f}% stored)")

    df = pd.DataFrame(results)
    df.to_csv(REGION_CONFIG['output_dir'] / 'region_extraction_results.csv', index=False)

    return results