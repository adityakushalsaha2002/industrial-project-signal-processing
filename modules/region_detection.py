"""
Region detection module: activity-based region extraction.

Activity is calculated on CENTERED data (value_offset removed), so the
threshold anchor no longer conflates slow baseline wander with signal
variance.

Threshold anchor: the file-wide std used previously was inflated by active
bursts AND baseline drift. Instead we use the MEDIAN of the rolling-std
activity signal itself as the noise-floor estimate. The median is robust to
the minority of samples belonging to bursts, so activity cannot raise its
own threshold (no self-suppression, unlike a rolling threshold).
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from modules.preprocessing import CFG

REGION_CONFIG = {
    'data_dir': Path('./Test signals from previous years/Suction data/Preprocessed'),
    'output_dir': Path('./Test signals from previous years/Suction data/Region'),

    # Activity detection thresholds (calculated on centered preprocessed data)
    'activity_window_samples': 10,      # 0.1 s sliding window at 100 Hz
    'noise_floor_multiplier': 4.0,      # active when rolling std > 4x median rolling std
    'min_region_duration_sec': 0.05,     # >= 3 activity windows AND >= 30 samples at 100 Hz
    'padding_sec': 0,                   # buffer before/after each region, in seconds
    'bridge_gap_samples': 10
}
REGION_CONFIG['target_fs'] = CFG['target_fs']
REGION_CONFIG['padding_samples'] = int(REGION_CONFIG['padding_sec'] * REGION_CONFIG['target_fs'])
REGION_CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)


def output_paths_for(bin_path, cfg):
    """Canonical output paths for a given preprocessed .bin file."""
    out_bin = cfg['output_dir'] / (bin_path.stem + '_regions.bin')
    out_meta = cfg['output_dir'] / (bin_path.stem + '_regions.meta.json')
    return out_bin, out_meta


def find_contiguous_regions(active_mask, cfg):
    """Walk the boolean mask and collect padded regions meeting min duration."""
    regions = []
    in_region = False
    region_start = 0
    n = len(active_mask)

    for i, is_active in enumerate(active_mask):
        if is_active and not in_region:
            region_start = i
            in_region = True
        elif not is_active and in_region:
            region_end = i
            padded_start = max(0, region_start - cfg['padding_samples'])
            padded_end = min(n, region_end + cfg['padding_samples'])
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
        padded_end = n
        duration_sec = (padded_end - padded_start) / cfg['target_fs']
        if duration_sec >= cfg['min_region_duration_sec']:
            regions.append({
                'global_start': int(padded_start),
                'global_end': int(padded_end),
                'duration_sec': round(duration_sec, 4),
            })

    return regions


def detect_regions_and_extract(bin_path, cfg):
    """Detect contiguous active regions -> store as blocks -> save."""
    out_bin, out_meta_path = output_paths_for(bin_path, cfg)

    # SKIP FIRST: before any memmap, filtering, or region walking
    if out_bin.exists():
        return {'status': 'skipped', 'file': bin_path.name}

    meta_path = bin_path.with_suffix('.meta.json')
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found for {bin_path}")

    meta = json.loads(meta_path.read_text())

    raw_data = np.memmap(str(bin_path), dtype='uint16', mode='r')

    # Center FIRST, then compute activity. A 10-sample rolling std is nearly
    # identical on raw vs centered data, but the THRESHOLD anchor differs:
    # centered data removes slow baseline drift from the noise estimate.
    centered = raw_data.astype(np.int32) - meta['value_offset']

    activity_std = pd.Series(centered).rolling(
        cfg['activity_window_samples'], center=True, min_periods=1
    ).std().to_numpy()

    # Robust noise floor: median of the activity signal itself.
    noise_floor = float(np.median(activity_std))
    activity_threshold = cfg['noise_floor_multiplier'] * noise_floor

    active_mask = activity_std > activity_threshold

    # Bridge silent gaps shorter than bridge_gap: a brief dip below threshold
    # mid-event splits one event into fragments; glue them back together.
    bridge_gap = cfg.get('bridge_gap_samples', 10)   # 0.1 s at 100 Hz
    silent = np.where(~active_mask)[0]
    if len(silent) > 1:
        runs = np.split(silent, np.where(np.diff(silent) > 1)[0] + 1)
        for run in runs:
            if len(run) <= bridge_gap and 0 < run[0] and run[-1] < len(active_mask) - 1:
                active_mask[run[0]:run[-1] + 1] = True

    regions = find_contiguous_regions(active_mask, cfg)

    # Extract and store regions (centered + offset -> uint16 for compatibility)
    all_region_data = []
    for region in regions:
        all_region_data.append(centered[region['global_start']:region['global_end']])

    if all_region_data:
        stored = np.concatenate(all_region_data).astype(np.uint16) + meta['value_offset']
    else:
        stored = np.array([], dtype=np.uint16)

    total_stored_samples = len(stored)
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
        'noise_floor_std': round(noise_floor, 2),
        'noise_floor_multiplier': cfg['noise_floor_multiplier'],
        'activity_threshold': round(activity_threshold, 2),
        'value_offset': meta['value_offset'],
        'target_fs': cfg['target_fs'],
    }

    if total_stored_samples > 0:
        stored.tofile(out_bin)
    out_meta_path.write_text(json.dumps(output_meta, indent=2))

    print(f"✅ {output_meta['original_size_mb']:.2f}MB → {output_meta['stored_size_mb']:.2f}MB "
          f"({output_meta['region_efficiency']*100:.1f}% stored) | {bin_path.name}")

    return {'status': 'completed', 'file': bin_path.name, **output_meta}


def run_batch_region_extraction(num_workers=None):
    """Parallel batch region extraction. Skipped files never enter the pool."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    results = []
    todo = []
    for bin_path in sorted(REGION_CONFIG['data_dir'].glob('*.bin')):
        out_bin, _ = output_paths_for(bin_path, REGION_CONFIG)
        if out_bin.exists():
            results.append({'file': bin_path.name, 'status': 'skipped'})
            print(f"⏭️ SKIPPED: {bin_path.name} (regions already extracted)")
        else:
            todo.append(bin_path)

    print(f"{len(todo)} files to process ({len(results)} already skipped)")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(detect_regions_and_extract, f, REGION_CONFIG): f for f in todo}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    total_original_mb = sum(r.get('original_size_mb', 0) for r in results if r.get('status') != 'skipped')
    total_stored_mb = sum(r.get('stored_size_mb', 0) for r in results if r.get('status') != 'skipped')
    if total_original_mb > 0:
        pct_stored = total_stored_mb / total_original_mb * 100
    else:
        pct_stored = 0.0
    print(f"\n{total_original_mb:.2f}MB → {total_stored_mb:.2f}MB ({pct_stored:.1f}% stored)")

    df = pd.DataFrame(results)
    df.to_csv(REGION_CONFIG['output_dir'] / 'region_extraction_results.csv', index=False)

    return results