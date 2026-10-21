"""
Preprocessing module: downsampling and per-file baseline normalization.
"""
import numpy as np
import pandas as pd
import json, time, gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

CFG = {
    'input_dir': Path('./Test signals from previous years/Suction data/Original'),
    'output_dir': Path('./Test signals from previous years/Suction data/Preprocessed'),
    'original_fs': 50_000,      # Hz
    'target_fs': 100,           # Hz
    'adc_column': 'adc2',
    'adc_col_index': None,
    'chunk_rows': 100_000_000,
    'window_size_sec': 60,
    'uint16_offset': 32_768,
    'overwrite': False,
}
CFG['window_size_samples'] = CFG['window_size_sec'] * CFG['target_fs']
CFG['pool_block'] = CFG['original_fs'] // CFG['target_fs']
CFG['output_dir'].mkdir(parents=True, exist_ok=True)


def sniff_csv(path):
    """Return (delimiter, has_header, n_columns) from the first line."""
    with open(path, 'r') as f:
        first = f.readline()
    delim = '\t' if first.count('\t') >= first.count(',') else ','
    parts = [p.strip() for p in first.rstrip('\r\n').split(delim) if p != '']
    is_numeric = all(p.lstrip('-').replace('.', '', 1).isdigit() for p in parts)
    return delim, (not is_numeric), len(parts)


def load_adc_chunks(path, cfg):
    """Yield the ADC column as int32 arrays, one chunk at a time."""
    delim, header, ncols = sniff_csv(path)
    if header:
        cols = pd.read_csv(path, sep=delim, nrows=0).columns.tolist()
        usecols = cfg['adc_column'] if cfg['adc_column'] in cols else cols[cfg['adc_col_index']]
    else:
        usecols = cfg['adc_col_index']

    for chunk in pd.read_csv(path, sep=delim, usecols=[usecols],
                             header=0 if header else None,
                             chunksize=cfg['chunk_rows'],
                             dtype=np.int32, engine='c'):
        yield chunk.iloc[:, 0].to_numpy(dtype=np.int32)
        del chunk
        gc.collect()


def pool_downsample(chunks, block):
    """Mean-pool every `block` raw samples → 1 output sample."""
    parts, carry = [], np.empty(0, dtype=np.int32)
    total_raw = 0
    for arr in chunks:
        total_raw += len(arr)
        arr = np.concatenate([carry, arr]) if len(carry) else arr
        n_blocks = len(arr) // block
        if n_blocks:
            usable = n_blocks * block
            parts.append(arr[:usable].reshape(n_blocks, block).mean(axis=1, dtype=np.int32))
            carry = arr[usable:].copy()
        else:
            carry = arr
        del arr
    if len(carry):
        parts.append(np.array([carry.mean()], dtype=np.int32))
    return np.concatenate(parts), total_raw


def preprocess_file(csv_path, cfg):
    """Single file preprocessing: decimate → rolling baseline → uint16 .bin + sidecar."""
    bin_path = cfg['output_dir'] / (csv_path.stem + '.bin')
    meta_path = cfg['output_dir'] / (csv_path.stem + '.meta.json')

    if bin_path.exists() and not cfg['overwrite']:
        return {'status': 'skipped', 'file': csv_path.name, 'reason': 'already exists'}

    t0 = time.time()
    raw_mb = csv_path.stat().st_size / 1024**2

    # 1. Read + downsample
    pooled, n_raw = pool_downsample(load_adc_chunks(csv_path, cfg), cfg['pool_block'])
    expected = n_raw / cfg['pool_block']
    assert abs(len(pooled) - expected) <= 1, f"samples lost: {len(pooled)} vs {expected:.1f}"

    baseline = pd.Series(pooled).rolling(window=cfg['window_size_samples'], center=True, min_periods=1).median().values
    centered = pooled - baseline

    # 3. Convert to uint16
    u16 = np.clip(centered + cfg['uint16_offset'], 0, 65_535).astype(np.uint16)
    u16.tofile(bin_path)

    # Sidecar info
    meta = {
        'source_file': csv_path.name,
        'original_size_mb': round(raw_mb, 2),
        'raw_samples': n_raw,
        'raw_fs': cfg['original_fs'],
        'target_fs': cfg['target_fs'],
        'samples': int(len(u16)),
        'duration_sec': round(len(u16) / cfg['target_fs'], 2),
        'dtype': 'uint16',
        'value_offset': cfg['uint16_offset'],
        'baseline_window_sec': cfg['window_size_sec'],
        'preprocess_time_sec': round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return {'status': 'completed', 'file': csv_path.name, **meta}


def run_batch_preprocessing(num_workers=None):
    """Parallel batch preprocessing."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    files = sorted(CFG['input_dir'].glob('*.csv'), key=lambda p: p.stat().st_size)
    print(f"{len(files)} files to process")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(preprocess_file, f, CFG): f for f in files}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "⏭️ SKIPPED" if result.get('status') == 'skipped' else "✅ DONE"
            print(f"{status}: {result['file']}")

    # Save summary
    df = pd.DataFrame(results)
    CFG['output_dir'].parent.mkdir(exist_ok=True)
    df.to_csv(CFG['output_dir'] / 'preprocessing_summary.csv', index=False)
    return results