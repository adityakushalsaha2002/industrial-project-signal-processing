# Industrial Project – Signal Processing (Lucent)

Detects tumor tissue events in suction-sensor recordings (`adc2` column, 50 kHz) and estimates
how much tumor passed the detector. There are two ways to run it:

- **Lucent** (`gui_app.py`) – a desktop app that analyses one recording at a time.
- **`run_pipeline.py`** – a batch command that analyses every recording in a folder and saves plots.

Both use the same processing code in `modules/`.

## Setup

Python 3.10+ is needed.

```bash
python -m venv venv
# Windows:   venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` holds the packages for both the app and the batch pipeline
(numpy, scipy, pandas, matplotlib, PySide6, pyqtgraph).

## Input files

A recording is a comma- or tab-separated text file with an `adc2` column, for example:

```
adc1,adc2
508,653
500,656
```

Files without a header row are not supported unless `adc_col_index` is set in
`modules/preprocessing.py`. Raw files without a file extension (like `high_1gr_1`) work in
Lucent; for `run_pipeline.py`, give them a `.csv` extension.

The file name is also read for two things (see [Volume estimate in ml](#volume-estimate-in-ml)):
the number of **resuctions** (e.g. `3 resuctions`, `resuc x2`) and, for comparison only, a
**labelled amount** (e.g. `2gr`, `1.5ml`).

## Using Lucent (the app)

Start it from the repository folder:

```bash
python gui_app.py                 # opens the app
python gui_app.py path/to/file.csv  # opens the app with a recording already loaded
```

1. Click **Open recording…** and choose a `.csv`/`.txt` recording (or *All files* for files
   without an extension).
2. Click **Analyse**. The left panel shows each processing step as it runs: check the file,
   downsample 50 kHz → 100 Hz, remove baseline drift, store compactly, find active regions,
   detect peaks, estimate tumor volume. Click **Stop** to cancel.
3. Read the results:
   - **Signal plot** – the cleaned signal; red triangles are detected peaks, shaded bands are
     active regions. Mouse wheel zooms, dragging pans, `−`/`+`/**Reset zoom** control the view.
     Clicking a row in the table zooms to that peak.
   - **Tumor events table** – every peak with its time, height, region and the region's volume
     (in units).
   - **Right-hand cards** – number of tumor events (peaks per minute), **tumor volume in ml**
     (point estimate, with the ml range, the raw unit estimate, the resuction divisor if any, and
     the label from the file name if any), number of active regions, and recording length.
4. **Peaks vs volume…** shows whether the peak count goes with the estimated volume: one dot per
   region in this recording, and one dot per recording for every file analysed so far (the
   history can be cleared from that window).
5. **Export CSV** saves the tumor events table (`<recording>_tumor_events.csv`, to a location
   you choose).
6. **Dark mode / Light mode** switches the theme; **Built by** lists the team.

Self-test without opening a window (writes a JSON summary, exit code 0 on success):

```bash
python gui_app.py --selftest path/to/file.csv summary.json
```

### Where Lucent writes files

| What | Running `python gui_app.py` | Packaged app (frozen executable) |
| --- | --- | --- |
| Intermediate files of the current analysis (cleared at the start of every run) | `gui_cache/` next to `gui_app.py` | `%LOCALAPPDATA%\Lucent\gui_cache\` |
| Run logs (`run_YYYYMMDD_HHMMSS.log`) | `gui_logs/` | `%LOCALAPPDATA%\Lucent\gui_logs\` |
| Crash report | `crash.log` | `%LOCALAPPDATA%\Lucent\crash.log` |
| Empty pipeline output folders (created when the modules are imported) | `Test signals from previous years/Suction data/…` in the repository | `%LOCALAPPDATA%\Lucent\work\Test signals from previous years\…` |
| Exported events table | wherever you save it | wherever you save it |
| Settings (last folder, theme, peaks-vs-volume history) | Qt settings for "Lucent" (Windows registry `HKCU\Software\Lucent\Lucent`) | same |

Lucent does not write plot images; use `run_pipeline.py` for those.

## Using `run_pipeline.py` (batch)

1. Put the recordings (with a `.csv` extension) in:

   ```
   Test signals from previous years/Suction data/Original/
   ```

2. From the repository folder run:

   ```bash
   python run_pipeline.py
   ```

It runs five steps over every file (in parallel, using all CPU cores but one): preprocessing,
region detection, peak detection, volume estimation and visualization. Steps 1, 2 and 4 skip
files whose output already exists, so delete the output folders (or the files for one
recording) to force a re-run, e.g. after changing a parameter.

The console prints, per file, the peak count, the estimate in units and ml, and the number of
regions with peaks, followed by totals.

### Where `run_pipeline.py` writes files

All paths are relative to the folder you run it from, under
`Test signals from previous years/Suction data/`:

| Folder | Contents |
| --- | --- |
| `Original/` | **Input**: your recordings (`*.csv`) |
| `Preprocessed/` | `<name>.bin` (100 Hz 16-bit signal), `<name>.meta.json`, `preprocessing_summary.csv` |
| `Region/` | `<name>_regions.bin`, `<name>_regions.meta.json` (active regions), `region_extraction_results.csv` |
| `Peaks/` | `<name>_regions_peaks.meta.json` (peak times and heights per region), `peak_detection_summary.csv` |
| `Volume/` | `<name>_regions_volume.meta.json` (unit and ml estimates, per-region details), `volume_estimation_summary.csv` |
| `Results/` | **Plot images** `<name>_full_signal.png` (signal, peaks, active regions; the title shows the unit and ml estimate), `visualization_summary.csv` |

## Volume estimate in ml

The pipeline integrates the signal inside every active region that contains at least one peak,
after subtracting the expected noise contribution. This gives a point estimate and a range in
dimensionless **units** (the range is a ±2 standard-error noise band).

The unit numbers are converted to **ml** with one fixed scalar,
`VOLUME_CONFIG['ml_per_unit'] = 0.0023` in `modules/tumor_volume_estimation.py`:

```
ml = units × 0.0023 ÷ (number of resuctions in the file name, or 1 if none)
```

- **Calibration.** The scalar is a least-squares fit (through the origin) of the point
  estimates of the ten labelled recordings `high_1gr_1` … `high_3gr_resuc` against their
  labelled amount, taking 1 g of tissue ≈ 1 ml. It does not know the concentration, so the same
  scalar is used for every file. With it every labelled file falls inside its ml range:

  | File | Label | Estimate (point, range) |
  | --- | --- | --- |
  | high_1gr_1 / _2 / _3 | 1 ml | 1.15, 1.41, 0.80 (ranges 0.0–2.5) |
  | high_1gr_1/2/3_resuc | 1 ml | 0.68, 1.16, 1.42 (ranges 0.1–2.3) |
  | high_2gr / high_2gr_resuc | 2 ml | 2.00 (0.00–4.50), 1.91 (0.57–3.25) |
  | high_3gr / high_3gr_resuc | 3 ml | 3.28 (0.52–6.05), 2.12 (0.00–4.49) |

- **Resuctions.** A resuction sends the same tissue past the detector again, so the estimate is
  divided by the number of resuctions named in the file name. Recognised forms: `3 resuctions`,
  `3x resuction`, `2resuc`, `resuc2`, `resuc x2`, `resuction (3)`, `double resuction`; a bare
  `resuc` / `re-suction` counts as 1. A number joined with `_` before `resuc` (as in
  `high_1gr_2_resuc`) is treated as a trial index, not a count.
- A negative lower bound is shown as 0 ml.
- The labelled amount in the file name (`2gr`, `1.5ml`) is only displayed next to the estimate
  for comparison; it is never used to compute it.

## Repository layout

```
gui_app.py                       Lucent desktop app
run_pipeline.py                  batch pipeline
modules/preprocessing.py         anti-aliased 50 kHz -> 100 Hz downsampling, baseline removal
modules/region_detection.py      active-region detection
modules/peak_detection.py        peak detection inside regions
modules/tumor_volume_estimation.py  unit and ml volume estimates
modules/visualization.py         full-signal plots for run_pipeline.py
meeting-notes/                   meeting notes
old-code/                        earlier notebooks and app versions (not used)
```

Processing parameters live in the `*_CONFIG` / `CFG` dictionaries at the top of each module.
