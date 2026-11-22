"""
Tumor volume estimation module: integrate signal values within regions containing peaks.

Estimates are reported as bounded ranges in dimensionless "units" (never
physical volumes such as ml). This follows the report's convention of
expressing uncertain quantities as lower-upper ranges instead of single
values (e.g. a detection threshold "between 0.6 and 1.8 umol/L").

Bounds rationale:
- Baseline noise makes a positive, systematic contribution to a rectified
  integral: pure noise with std sigma has an expected rectified offset of
  sigma * sqrt(2/pi) per sample.
- The point estimate subtracts exactly this expected offset (multiplier 1.0).
- LOWER bound: assumes MORE of the signal is noise (subtracts 1.5x the
  expected offset) -> conservative, high-confidence tumor content.
- UPPER bound: assumes LESS of the signal is noise (subtracts 0.5x).
For 'signed' integrals noise is zero-mean, so the band is instead the
standard error of the noise sum (sigma * sqrt(n) / fs).

Millilitre estimate:
- The unit range is converted to ml with one fixed scalar (ml_per_unit).
  It is blind to concentration: it was fitted (least squares through the
  origin) on the labelled 'high_1gr/2gr/3gr' recordings, assuming tissue
  density ~1 g/ml so that 1 gr ~ 1 ml.
- Each resuction sends the same tissue through the detector again, so the
  ml estimate is divided by the number of resuctions named in the file name
  (e.g. '3 resuctions', 'resuc x2', '2resuc'; a bare 'resuc' counts as 1).
- A negative lower bound is clipped to 0 ml.
"""
import re
import numpy as np
import json
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from modules.region_detection import REGION_CONFIG
from modules.peak_detection import PEAK_CONFIG

VOLUME_CONFIG = {
    'output_dir': Path('./Test signals from previous years/Suction data/Volume'),

    # Negative value handling: 'ignore', 'absolute', 'signed', or 'noise_corrected'
    'negative_handling': 'noise_corrected',

    'baseline_correction': True,
    'min_signal_value': 0,

    # Bounded estimation: multipliers on the expected noise offset used for
    # the lower / upper bounds (point estimate always uses 1.0).
    'bound_offset_low': 1.5,    # subtract more noise -> conservative lower bound
    'bound_offset_high': 0.5,   # subtract less noise -> optimistic upper bound

    # Units -> ml scalar, calibrated on the labelled 1/2/3 gr recordings
    # (point estimates of 10 files, 1 g ~ 1 ml). Concentration-independent.
    'ml_per_unit': 0.0023,
}
VOLUME_CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

_NUM_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
              'single': 1, 'double': 2, 'twice': 2, 'triple': 3, 'thrice': 3}
_NUM = r'(\d+|' + '|'.join(_NUM_WORDS) + r')'
_RESUC = r're-?suc\w*'


def _base(name):
    """Lower-case file name without a known extension (raw files may have none,
    and '1.5ml' must not lose its decimal part to Path.stem)."""
    s = Path(str(name)).name.lower()
    return re.sub(r'(_regions)?\.(csv|txt|tsv|bin|json)$', '', s)


def _to_int(token):
    token = token.lower()
    return int(token) if token.isdigit() else _NUM_WORDS[token]


def parse_resuctions(name):
    """Number of resuctions named in a file name (0 if none).

    Recognised: '3 resuctions', '3x resuction', '2resuc', 'resuc2',
    'resuc x2', 'resuction (3)', 'double resuction'. A bare 'resuc'/'re-suction'
    counts as 1. A number joined by '_' before 'resuc' (as in
    'high_1gr_2_resuc') is a trial index, not a count, so it is ignored.
    """
    s = _base(name)
    if not re.search(_RESUC, s):
        return 0
    patterns = (
        _NUM + r'(?:\s*x)?\s*' + _RESUC,                    # '3 resuctions', '3x resuc', '2resuc'
        _RESUC + r'\s*(?:x\s*|\(\s*)?(\d+)',                # 'resuc2', 'resuc x2', 'resuction (3)'
    )
    for pat in patterns:
        m = re.search(r'(?<![a-z0-9.])' + pat if pat.startswith(_NUM) else pat, s)
        if m:
            n = _to_int(m.group(1))
            if n > 0:
                return n
    return 1


def parse_labelled_ml(name):
    """Labelled tissue amount in the file name as ml (1 g ~ 1 ml), or None.

    Only used to report the label next to the estimate; the estimate itself
    never reads it.
    """
    s = _base(name)
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(ml|gr|g)(?![a-z])', s)
    return float(m.group(1).replace(',', '.')) if m else None


def add_ml_estimate(result, name, config=VOLUME_CONFIG):
    """Add ml fields (point, lower, upper) to a calculate_region_volume result."""
    resuctions = parse_resuctions(name)
    divisor = max(resuctions, 1)
    k = config.get('ml_per_unit', VOLUME_CONFIG['ml_per_unit'])

    def ml(units):
        return round(max(units * k / divisor, 0.0), 3)

    result.update({
        'ml_per_unit': k,
        'resuctions': resuctions,
        'ml_divisor': divisor,
        'total_ml': ml(result['total_units']),
        'total_ml_lower': ml(result['total_units_lower']),
        'total_ml_upper': ml(result['total_units_upper']),
        'labelled_ml': parse_labelled_ml(name),
    })
    for r in result.get('region_details', []):
        r['integrated_ml'] = round(r['integrated_units'] * k / divisor, 4)
    return result


def load_region_and_peak_data(region_bin_path):
    """Load region data and corresponding peak metadata."""
    meta_path = region_bin_path.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text())

    regions_data = np.fromfile(region_bin_path, dtype=np.uint16)
    centered = regions_data.astype(np.int32) - meta['value_offset']

    regions = []
    idx = 0
    for region_info in meta['stored_regions']:
        region_size = region_info['global_end'] - region_info['global_start']
        regions.append({
            'timestamp_sec': region_info['global_start'] / meta['target_fs'],
            'data': centered[idx:idx + region_size],
            'duration_sec': region_info['duration_sec'],
        })
        idx += region_size

    peak_meta_path = PEAK_CONFIG['peaks_dir'] / f"{region_bin_path.stem}_peaks.meta.json"
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


def region_peak_counts(detected_peaks):
    """Map region_index -> number of peaks, omitting regions with zero peaks.

    This is the single source of truth for the 'region has peaks' gate, so
    zero-peak records written by peak_detection can never mark a region
    as active again.
    """
    return {
        pr['region_index']: pr['num_peaks']
        for pr in detected_peaks
        if pr.get('num_peaks', 0) > 0
    }


def noise_parameters(data):
    """Baseline noise std, expected rectified offset, and correlation factor k.

    k = 1 + 2*sum_lag rho_l accounts for autocorrelation: the variance of a
    sum of n correlated samples is sigma^2 * n * k, i.e. n_eff = n / k
    effectively independent samples. Assumes the baseline segment is
    representative of the noise throughout the region.
    """
    n_baseline = max(int(len(data) * 0.2), 10)
    x = data[:n_baseline] - np.median(data[:n_baseline])
    noise_std = float(np.std(x))
    noise_expectation = noise_std * np.sqrt(2 / np.pi)

    n = len(x)
    max_lag = min(n // 4, 200)
    x0 = x - x.mean()
    denom = float(np.dot(x0, x0))
    k = 1.0
    if denom > 0:
        for lag in range(1, max_lag + 1):
            rho = float(np.dot(x0[:-lag], x0[lag:]) / denom)
            if rho <= 0:
                break
            k += 2.0 * rho
    return noise_std, noise_expectation, k


def calculate_region_volume(region_data_result):
    """Calculate bounded integrated tumor estimates for regions containing peaks."""
    regions = region_data_result['regions']
    config = region_data_result['config']
    fs = region_data_result['meta']['target_fs']

    peak_counts = region_peak_counts(region_data_result['detected_peaks'])

    region_records = []
    k_values = []
    total_units = 0.0
    total_units_lower = 0.0
    total_units_upper = 0.0
    total_positive_units = 0.0
    total_negative_units = 0.0

    mode = config['negative_handling']

    Z = config.get('confidence_z', 2.0)   # band half-width in units of SE

    for reg_idx, region in enumerate(regions):
        if reg_idx not in peak_counts:
            continue

        data = region['data'].copy().astype(np.float64)

        if config.get('baseline_correction', False):
            data = data - np.median(data)

        noise_std, noise_expectation, k = noise_parameters(data)
        k_values.append(k)
        pos = neg = None

        if mode in ('noise_corrected', 'ignore', 'absolute'):
            base = np.abs(data) if mode == 'absolute' else data
            if mode == 'ignore':
                base = np.maximum(base, config.get('min_signal_value', 0))
            if mode == 'noise_corrected':
                base = data
                base = np.maximum(base - noise_expectation, 0)
            point = np.sum(base) / fs
            se = (noise_std * np.sqrt(1 - 2 / np.pi)
                * np.sqrt(len(data) * k) / fs)
            lower, upper = point - Z * se, point + Z * se
        else:  # signed
            point = np.sum(data) / fs
            se = noise_std * np.sqrt(len(data) * k) / fs
            lower, upper = point - Z * se, point + Z * se

        region_records.append({
            'region_index': reg_idx,
            'region_start_sec': region['timestamp_sec'],
            'duration_sec': region['duration_sec'],
            'peak_count_in_region': peak_counts[reg_idx],   # actual peaks, not record count
            'integrated_units': round(point, 4),
            'integrated_units_lower': round(lower, 4),
            'integrated_units_upper': round(upper, 4),
            'max_amplitude': round(float(np.max(data)), 4),
            'mean_amplitude': round(float(np.mean(data)), 2),
            'noise_std': round(noise_std, 2),
            'bounds_method': f'n_eff SE band, z={Z} '
                 f'(median k over regions: {np.median(k_values):.2f})'
        })

        total_units += point
        total_units_lower += lower
        total_units_upper += upper
        if pos is not None:
            total_positive_units += pos
            total_negative_units += neg

    return {
        'unit': 'units',
        'bounds_method': (f'n_eff SE band, z={Z} '
                          f'(median k over regions: {np.median(k_values):.2f})'),
        'total_units': round(total_units, 4),
        'total_units_lower': round(total_units_lower, 4),
        'total_units_upper': round(total_units_upper, 4),
        'total_positive_units': round(total_positive_units, 4),
        'total_negative_units': round(total_negative_units, 4),
        'total_peak_count': sum(peak_counts.values()),
        'num_regions_with_peaks': len(region_records),
        'total_regions': len(regions),
        'region_details': region_records,
    }


def process_volume_file(region_bin_path_str):
    """Worker entry point: load data INSIDE the worker so disk I/O parallelizes."""
    region_bin_path = Path(region_bin_path_str)

    region_data_result = load_region_and_peak_data(region_bin_path)
    if region_data_result is None:
        return {'file': region_bin_path.name, 'status': 'skipped', 'reason': 'no peaks detected'}

    region_data_result['config'] = VOLUME_CONFIG
    result = calculate_region_volume(region_data_result)
    add_ml_estimate(result, region_bin_path.name)

    if result['num_regions_with_peaks'] == 0:
        return {'file': region_bin_path.name, 'status': 'skipped', 'reason': 'no peaks in regions'}

    serializable_config = {k: str(v) if isinstance(v, Path) else v
                           for k, v in VOLUME_CONFIG.items()}
    volume_meta_path = VOLUME_CONFIG['output_dir'] / (region_bin_path.stem + '_volume.meta.json')
    volume_meta_path.write_text(json.dumps({
        'source_file': region_bin_path.name,
        'volume_config': serializable_config,
        **result,
    }, indent=2))

    return {'file': region_bin_path.name, 'status': 'completed', **result}


def run_batch_volume_estimation(num_workers=None):
    """Process all region files and calculate bounded tumor estimates."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    region_files = sorted(REGION_CONFIG['output_dir'].glob('*_regions.bin'))
    print(f"Processing {len(region_files)} region files for bounded estimation...")

    results = []
    todo = []
    for region_bin_path in region_files:
        volume_meta_path = VOLUME_CONFIG['output_dir'] / (region_bin_path.stem + '_volume.meta.json')
        peaks_exist = (PEAK_CONFIG['peaks_dir'] / f"{region_bin_path.stem}_peaks.meta.json").exists()

        if volume_meta_path.exists():
            results.append({'file': region_bin_path.name, 'status': 'skipped', 'reason': 'already exists'})
            print(f"⏭️ SKIPPED: {region_bin_path.name} (already estimated)")
        elif not peaks_exist:
            results.append({'file': region_bin_path.name, 'status': 'skipped', 'reason': 'no peaks file'})
        else:
            todo.append(region_bin_path)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_volume_file, str(p)): p for p in todo}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result['status'] == 'completed':
                print(f"✅ {result['file']}: {result['total_peak_count']} peaks | "
                      f"{result['total_units_lower']:.2f}-{result['total_units_upper']:.2f} units "
                      f"(point {result['total_units']:.2f}) | "
                      f"{result['total_ml_lower']:.2f}-{result['total_ml_upper']:.2f} ml "
                      f"(point {result['total_ml']:.2f}"
                      f"{', ÷%d resuctions' % result['ml_divisor'] if result['ml_divisor'] > 1 else ''}) | "
                      f"{result['num_regions_with_peaks']}/{result['total_regions']} regions")
            else:
                print(f"⏭️ SKIPPED: {result['file']} ({result['reason']})")

    df_summary = pd.DataFrame(results)
    df_summary.to_csv(VOLUME_CONFIG['output_dir'] / 'volume_estimation_summary.csv', index=False)

    completed_results = [r for r in results if r.get('status') == 'completed']
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE! Processed {len(completed_results)} files")
    if completed_results:
        grand_lower = sum(r['total_units_lower'] for r in completed_results)
        grand_upper = sum(r['total_units_upper'] for r in completed_results)
        grand_point = sum(r['total_units'] for r in completed_results)
        total_all_peaks = sum(r['total_peak_count'] for r in completed_results)
        print(f"   Total files processed: {len(completed_results)}")
        print(f"   Total estimate: {grand_lower:.2f}-{grand_upper:.2f} units "
              f"(point {grand_point:.2f})")
        print(f"   Total peaks detected: {total_all_peaks}")
        print(f"   Total estimate: {sum(r['total_ml_lower'] for r in completed_results):.2f}-"
              f"{sum(r['total_ml_upper'] for r in completed_results):.2f} ml "
              f"(point {sum(r['total_ml'] for r in completed_results):.2f})")

    return results