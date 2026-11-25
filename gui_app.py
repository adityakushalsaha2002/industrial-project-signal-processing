#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

APP = "Lucent"
TEAM = [("lichenaut", "https://codeberg.org/lichenaut"),
        ("Arthiya Shiva", ""),
        ("Aditya K", "")]
FROZEN = getattr(sys, "frozen", False)                
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP if FROZEN else HERE
CACHE_DIR = DATA_DIR / "gui_cache"
LOG_DIR = DATA_DIR / "gui_logs"
log = logging.getLogger("lucent")

LIGHT = dict(
    ground="#F5F5F7", card="#FFFFFF", text="#1D1D1F", dim="#6E6E73", hair="#E3E3E8", border="rgba(0,0,0,0.14)",
    off_text="#AEAEB2", off_bg="#FAFAFC", track="#E5E5EA", select="rgba(0,113,227,0.12)",
    blue="#0071E3", blue_off="#9CC8F4", tumor="#D3194A", green="#1F7A3A", red_text="#C0143E",
    plot_bg="#1C1C1E", plot_fg="#AEAEB2", trace="#7CC4FF", tri="#FF453A",
)
DARK = dict(
    ground="#111214", card="#1C1D20", text="#F2F2F5", dim="#9A9AA2", hair="#2E2F34", border="rgba(255,255,255,0.14)",
    off_text="#5C5D63", off_bg="#18191C", track="#2E2F34", select="rgba(10,132,255,0.22)",
    blue="#0A84FF", blue_off="#24456B", tumor="#FF375F", green="#30D158", red_text="#FF6B81",
    plot_bg="#0B0C0E", plot_fg="#9A9AA2", trace="#7CC4FF", tri="#FF453A",
)
C = dict(LIGHT)                                       


def build_qss():
    return f"""
* {{ font-family: "Segoe UI", "Helvetica Neue", "Noto Sans", sans-serif; font-size: 13px; color: {C['text']}; }}
QMainWindow, QDialog, QStackedWidget#bg {{ background: {C['ground']}; }}
QToolBar {{ background: {C['card']}; border: 0; border-bottom: 1px solid {C['hair']}; padding: 8px 12px; spacing: 8px; }}
QPushButton, QToolButton {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 9px;
                            padding: 8px 16px; font-weight: 500; }}
QPushButton:hover, QToolButton:hover {{ background: {C['ground']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {C['off_text']}; background: {C['off_bg']}; }}
QToolButton[primary="true"], QPushButton[primary="true"] {{ background: {C['blue']}; color: white; border-color: {C['blue']}; }}
QToolButton[primary="true"]:disabled {{ background: {C['blue_off']}; border-color: {C['blue_off']}; color: white; }}
QToolButton[stop="true"] {{ color: {C['red_text']}; border-color: {C['tumor']}; }}
QPushButton[small="true"] {{ padding: 4px 10px; min-width: 18px; }}
QPushButton[seg="true"] {{ padding: 4px 12px; border-radius: 7px; }}
QPushButton[seg="true"]:checked {{ background: {C['blue']}; color: white; border-color: {C['blue']}; }}
QFrame[card="true"] {{ background: {C['card']}; border-radius: 14px; }}
QLabel[dim="true"] {{ color: {C['dim']}; }}
QLabel[caps="true"] {{ color: {C['dim']}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QTableWidget {{ background: {C['card']}; border: 0; gridline-color: transparent;
               selection-background-color: {C['select']}; selection-color: {C['text']}; }}
QHeaderView::section {{ background: {C['card']}; border: 0; border-bottom: 1px solid {C['hair']}; padding: 6px;
                       color: {C['dim']}; font-size: 11px; font-weight: 600; }}
QProgressBar {{ background: {C['track']}; border: 0; border-radius: 3px; max-height: 6px; color: transparent; }}
QProgressBar::chunk {{ background: {C['blue']}; border-radius: 3px; }}
QStatusBar {{ background: {C['card']}; color: {C['dim']}; border-top: 1px solid {C['hair']}; }}
QSplitter::handle {{ background: transparent; }}
"""


def apply_theme(app, dark):
    """Switch every colour in the app between light and dark."""
    C.clear()
    C.update(DARK if dark else LIGHT)
    pal = QtGui.QPalette()                            # native parts (scrollbars, file dialogs) follow too
    for role, key in ((QtGui.QPalette.Window, "ground"), (QtGui.QPalette.Base, "card"),
                      (QtGui.QPalette.AlternateBase, "ground"), (QtGui.QPalette.Button, "card"),
                      (QtGui.QPalette.WindowText, "text"), (QtGui.QPalette.Text, "text"),
                      (QtGui.QPalette.ButtonText, "text"), (QtGui.QPalette.ToolTipBase, "card"),
                      (QtGui.QPalette.ToolTipText, "text"), (QtGui.QPalette.Highlight, "blue"),
                      (QtGui.QPalette.Mid, "hair"), (QtGui.QPalette.PlaceholderText, "dim")):
        pal.setColor(role, QtGui.QColor(C[key]))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("white"))
    app.setPalette(pal)
    app.setStyleSheet(build_qss())

def load_modules():
    sys.path.insert(0, str(HERE))
    cwd = os.getcwd()
    work = DATA_DIR / "work" if FROZEN else HERE       
    work.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(work)
        from modules import peak_detection, preprocessing, region_detection, tumor_volume_estimation
    finally:
        os.chdir(cwd)
    return preprocessing, region_detection, peak_detection, tumor_volume_estimation


def secs(t):
    return f"{t:.2f}s" if t < 10 else f"{t:.1f}s"


def step_list(pre, rd, pk, tv):
    fs, cfg, rcfg, pcfg = pre.CFG["target_fs"], pre.CFG, rd.REGION_CONFIG, pk.PEAK_CONFIG
    return [
        ("prepare", "Check the file", f"Find the {cfg['adc_column']} column; comma or tab separated"),
        ("prepare", f"Downsample {cfg['original_fs'] // 1000}kHz → {fs}Hz",
         f"Average every {cfg['pool_block']} samples"),
        ("prepare", "Remove baseline drift", f"Subtract a {cfg['window_size_sec']}s rolling median"),
        ("prepare", "Store compactly", "16-bit binary file for the next steps"),
        ("analyse", "Find active regions",
         f"Spread over {rcfg['activity_window_samples'] * 1000 // fs}ms above "
         f"{rcfg['noise_floor_multiplier']:g} × its typical (median) level; gaps under "
         f"{rcfg.get('bridge_gap_samples', 0) / fs * 1000:g}ms are joined and short blips discarded"),
        ("analyse", "Detect peaks",
         f"At least {pcfg['min_peak_height']} tall, or {pcfg.get('min_peak_sn', 0):g}× the noise floor "
         f"(≈{pcfg.get('min_peak_sn', 0) * 1.5:.0f} units), and ≥ {pcfg['min_distance_samples'] / fs:g}s apart"),
        ("analyse", "Estimate tumor volume",
         f"Add up the signal inside regions with peaks; × {tv.VOLUME_CONFIG['ml_per_unit']:g} ml per unit, "
         f"÷ resuctions named in the file"),
    ]


class Stopped(Exception):
    pass


class _PrintToLog(io.TextIOBase):
    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                logging.getLogger("marginum").info(line.rstrip())
        return len(s)


class PipelineWorker(QObject):
    step = Signal(int, str, str)
    progress = Signal(float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, mods, path: Path):
        super().__init__()
        self.mods, self.path = mods, path
        self._stop = False
        self._current = 0

    def stop(self):
        self._stop = True

    def _done(self, i, text):
        self.step.emit(i, "done", f"{text} · {secs(time.time() - self._t)}")

    def _start(self, i, text=""):
        self._t = time.time()
        if self._stop:
            raise Stopped()
        self._current = i
        self.step.emit(i, "running", text)

    @Slot()
    def run(self):
        try:
            with contextlib.redirect_stdout(_PrintToLog()):
                result = self._run()
            self.finished.emit(result)
        except Stopped:
            self.step.emit(self._current, "failed", "stopped")
            self.failed.emit("Stopped.")
        except Exception as exc:
            log.exception("analysis failed")
            self.step.emit(self._current, "failed", str(exc) or type(exc).__name__)
            self.failed.emit(str(exc) or type(exc).__name__)

    def _run(self):
        pre, rd, pk, tv = self.mods
        path, fs = self.path, pre.CFG["target_fs"]
        size = path.stat().st_size
        t_all = time.time()

        self._start(0)
        delim, has_header, ncols = pre.sniff_csv(path)
        adc = pre.CFG["adc_column"]
        if has_header:
            cols = pd.read_csv(path, sep=delim, nrows=0).columns.tolist()
            if adc not in cols and pre.CFG["adc_col_index"] is None:
                raise ValueError(f"No '{adc}' column in this file (columns: {', '.join(map(str, cols))})")
        elif pre.CFG["adc_col_index"] is None:
            raise ValueError(f"The file has no header row, so the '{adc}' column cannot be found")
        sep = "tab" if delim == "\t" else "comma"
        self._done(0, f"{sep} separated · {ncols} columns · {size / 1e6:.0f}MB")

        st = path.stat()
        key = f"{path.resolve()}|{st.st_size}|{st.st_mtime}|{json.dumps(pre.CFG, default=str, sort_keys=True)}"
        work = CACHE_DIR / hashlib.md5(key.encode()).hexdigest()[:12]
        shutil.rmtree(CACHE_DIR, ignore_errors=True)   # always a fresh run: clear everything from earlier runs
        work.mkdir(parents=True, exist_ok=True)
        bin_path = work / (path.stem + ".bin")
        if True:
            self._start(1)
            cfg = dict(pre.CFG, output_dir=work, overwrite=False,
                       chunk_rows=min(pre.CFG["chunk_rows"], 5_000_000))   # smaller chunks -> live progress
            est, rows, t0, t_read = max(size / 9.5, 1.0), [0], time.time(), [None]
            original_reader = pre.load_adc_chunks

            def reader(p, c):
                for arr in original_reader(p, c):
                    if self._stop:
                        raise Stopped()
                    rows[0] += len(arr)
                    self.progress.emit(0.6 * min(rows[0] / est, 0.99))
                    self.step.emit(1, "running", f"{rows[0] / 1e6:.0f} M samples read")
                    yield arr
                t_read[0] = time.time()
                self.step.emit(1, "done", f"{rows[0]:,} → {rows[0] // pre.CFG['pool_block']:,} samples · "
                                          f"{secs(t_read[0] - t0)}")
                self._current = 2
                self.step.emit(2, "running", "")

            pre.load_adc_chunks = reader
            try:
                res = pre.preprocess_file(path, cfg)
            finally:
                pre.load_adc_chunks = original_reader
            t_pre = secs(time.time() - (t_read[0] or t0))   # the backend does steps 3 and 4 in one call
            self.step.emit(2, "done", f"drift removed · {t_pre} with step 4")
            self.step.emit(3, "done", f"{size / 1e6:.0f}MB → {bin_path.stat().st_size / 1e6:.2f}MB · {t_pre} with step 3")
            log.info("preprocessed %s: %s", path.name, res)
        meta = json.loads(bin_path.with_suffix(".meta.json").read_text())
        signal = (np.fromfile(bin_path, dtype=np.uint16).astype(np.int32) - meta["value_offset"]).astype(np.float32)
        self.progress.emit(0.7)

        self._start(4)
        rcfg = dict(rd.REGION_CONFIG, output_dir=work)
        rcfg["padding_samples"] = int(rcfg["padding_sec"] * rcfg["target_fs"])
        region_bin = work / (bin_path.stem + "_regions.bin")
        region_bin.unlink(missing_ok=True)
        region_bin.with_suffix(".meta.json").unlink(missing_ok=True)
        reg = rd.detect_regions_and_extract(bin_path, rcfg)
        self._done(4, f"{reg['num_regions']:,} regions · {reg['region_efficiency'] * 100:.0f}% of the signal")
        self.progress.emit(0.8)

        rows_out, starts, ends, volume, reg_peaks, reg_vol = [], [], [], 0.0, [], []
        lo = hi = 0.0
        reg_details = []
        ml = dict(total_ml=0.0, total_ml_lower=0.0, total_ml_upper=0.0, resuctions=tv.parse_resuctions(path.name),
                  ml_divisor=max(tv.parse_resuctions(path.name), 1), labelled_ml=tv.parse_labelled_ml(path.name))
        if reg["num_regions"] == 0:
            self._done(5, "no active regions, so no peaks")
            self._done(6, "0 units · 0 ml")
        else:
            self._start(5)
            loaded = pk.load_region_data(region_bin)
            peaks = pk.detect_peaks_in_regions(loaded)
            n_peaks = sum(p["num_peaks"] for p in peaks)
            self._done(5, f"{n_peaks:,} peaks")
            self.progress.emit(0.9)

            self._start(6)
            vres = tv.calculate_region_volume({"regions": loaded["regions"], "detected_peaks": peaks,
                                    "config": tv.VOLUME_CONFIG, "meta": loaded["meta"]})
            point = float(vres["total_units"])
            lo = float(vres["total_units_lower"])
            hi = float(vres["total_units_upper"])
            ml = tv.add_ml_estimate(vres, path.name)
            div = f" · ÷{ml['ml_divisor']} resuctions" if ml["ml_divisor"] > 1 else ""
            self._done(6, f"{ml['total_ml_lower']:.2f}–{ml['total_ml_upper']:.2f} ml (point {ml['total_ml']:.2f}){div} · "
                          f"{lo:,.1f}–{hi:,.1f} units (point {point:,.1f})")
            volume = point
            reg_details = vres["region_details"]
            vol_by_region = {d["region_index"]: d["integrated_units"]
                             for d in vres["region_details"]}

            for i, (info, pr) in enumerate(zip(loaded["meta"]["stored_regions"], peaks)):
                s, e = info["global_start"] / fs, info["global_end"] / fs
                starts.append(s)
                ends.append(e)
                if i in vol_by_region:
                    reg_peaks.append(int(pr["num_peaks"]))
                    reg_vol.append(float(vol_by_region[i]))
                for t, v in zip(pr["peak_times_sec"], pr["peak_values"]):
                    rows_out.append(dict(peak_time_s=round(float(t), 3), height=float(v), region_start_s=round(s, 3),
                                         region_end_s=round(e, 3), region_volume_units=float(vol_by_region.get(i, 0.0))))

        self.progress.emit(1.0)
        events = pd.DataFrame(rows_out, columns=["peak_time_s", "height", "region_start_s", "region_end_s",
                                                 "region_volume_units"]).sort_values("peak_time_s", ignore_index=True)
        log.info("%s: %d regions, %d peaks, volume %.2f units (%.2f ml), %.1f s", path.name, reg["num_regions"],
                 len(events), volume, ml["total_ml"], time.time() - t_all)
        return dict(name=path.name, fs=fs, duration=float(meta["duration_sec"]), signal=signal, events=events,
                    starts=np.array(starts), ends=np.array(ends), n_regions=int(reg["num_regions"]), volume=volume,
                    volume_lower=lo, volume_upper=hi, ml=float(ml["total_ml"]), ml_lower=float(ml["total_ml_lower"]),
                    ml_upper=float(ml["total_ml_upper"]), ml_divisor=int(ml["ml_divisor"]),
                    labelled_ml=ml["labelled_ml"], reg_peaks=np.array(reg_peaks), reg_vol=np.array(reg_vol),
                    reg_details=reg_details,
                    ml_scale=tv.VOLUME_CONFIG["ml_per_unit"] / max(int(ml["ml_divisor"]), 1))


def label(text="", dim=False, caps=False, size=None, bold=False, color=None, wrap=False):
    lb = QtWidgets.QLabel(text)
    if dim:
        lb.setProperty("dim", True)
    if caps:
        lb.setProperty("caps", True)
    style = [f"font-size:{size}px" if size else "", "font-weight:600" if bold else "", f"color:{color}" if color else ""]
    if any(style):
        lb.setStyleSheet(";".join(x for x in style if x))
    lb.setWordWrap(wrap)
    return lb


def card(layout, margins=(16, 14, 16, 14)):
    f = QtWidgets.QFrame()
    f.setProperty("card", True)
    layout.setContentsMargins(*margins)
    f.setLayout(layout)
    return f


class LogoWidget(QtWidgets.QWidget):
    def __init__(self, size=80):
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(self.width() / 512.0, self.width() / 512.0)
        g = QtGui.QLinearGradient(0, 0, 0, 512)
        g.setColorAt(0, QtGui.QColor("#FFFFFF"))
        g.setColorAt(1, QtGui.QColor("#E3E3E8"))
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 30), 6))
        p.setBrush(g)
        p.drawRoundedRect(QtCore.QRectF(3, 3, 506, 506), 116, 116)
        red = QtGui.QColor("#D3194A")
        glow = QtGui.QColor(red)
        glow.setAlphaF(0.14)
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(QtCore.QPointF(256, 256), 84, 84)
        p.setPen(QtGui.QPen(QtGui.QColor("#1D1D1F"), 40, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawArc(QtCore.QRectF(132, 132, 248, 248), 75 * 16, 300 * 16)
        p.setPen(QtGui.QPen(red, 22, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QtCore.QRectF(86, 86, 340, 340), 28 * 16, 34 * 16)
        p.setPen(Qt.NoPen)
        p.setBrush(red)
        p.drawEllipse(QtCore.QPointF(256, 256), 46, 46)


class Spinner(QtWidgets.QWidget):

    def __init__(self, size=64):
        super().__init__()
        self.setFixedSize(size, size)
        self.angle = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)

    def start(self):
        self.timer.start(16)

    def stop(self):
        self.timer.stop()

    def _tick(self):
        self.angle = (self.angle + 6) % 360
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = 6
        r = QtCore.QRectF(w, w, self.width() - 2 * w, self.height() - 2 * w)
        p.setPen(QtGui.QPen(QtGui.QColor(C["hair"]), w))
        p.drawEllipse(r)
        p.setPen(QtGui.QPen(QtGui.QColor(C["tumor"]), w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(r, -self.angle * 16, 100 * 16)


class StepRow(QtWidgets.QWidget):
    MARKS = {"pending": ("○", C["dim"]), "running": ("◐", C["blue"]), "done": ("✓", C["green"]),
             "failed": ("✕", C["red_text"])}
    SPIN = "◐◓◑◒"

    def __init__(self, number, title, detail):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(10)
        self.mark = label("", size=15, bold=True)
        self.mark.setFixedWidth(18)
        self.mark.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        lay.addWidget(self.mark)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(label(f"{number}. {title}", bold=True))
        text.addWidget(label(detail, dim=True, size=12, wrap=True))
        self.result = label("", size=12, wrap=True)
        text.addWidget(self.result)
        lay.addLayout(text, 1)
        self._spin_i = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._spin)
        self.set_state("pending")

    def _spin(self):
        self._spin_i = (self._spin_i + 1) % len(self.SPIN)
        self.mark.setText(self.SPIN[self._spin_i])

    def restyle(self):
        self.set_state(self._state, self._text)

    def set_state(self, state, result=""):
        self._state, self._text = state, result
        colors = {"pending": C["dim"], "running": C["blue"], "done": C["green"], "failed": C["red_text"]}
        mark, color = self.MARKS[state][0], colors[state]
        if state == "running":
            if not self._timer.isActive():
                self._timer.start(150)
        else:
            self._timer.stop()
        self.mark.setText(mark)
        self.mark.setStyleSheet(f"font-size:15px;font-weight:600;color:{color}")
        self.result.setText(result)
        self.result.setStyleSheet(f"font-size:12px;color:{color if state != 'running' else C['blue']}")
        self.result.setVisible(bool(result))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mods):
        super().__init__()
        self.mods = mods
        self.path: Path | None = None
        self.result: dict | None = None
        self.thread = self.worker = None
        self.setWindowTitle(APP)
        self.setMinimumSize(1200, 720)
        self._build_toolbar()
        self._build_body()
        self.status = QtWidgets.QLabel("Open a recording to begin")
        self.pbar = QtWidgets.QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setFixedWidth(200)
        self.pbar.hide()
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.pbar)
        self.set_theme(QtCore.QSettings(APP, APP).value("dark", False, bool))

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        def tbtn(text, slot, primary=False):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setCursor(Qt.PointingHandCursor)
            b.setProperty("primary", primary)
            b.clicked.connect(slot)
            tb.addWidget(b)
            return b

        self.b_open = tbtn("Open recording…", self.open_recording)
        self.b_run = tbtn("▶  Analyse", self.toggle_run, primary=True)
        self.b_export = tbtn("Export CSV", self.export_csv)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.b_theme = tbtn("", self.toggle_theme)
        self.b_theme.setToolTip("Switch between light and dark mode")
        self.b_built = tbtn("Built by", self.show_built_by)

    def toggle_theme(self):
        self.set_theme(not self.dark)

    def set_theme(self, dark):
        self.dark = dark
        apply_theme(QtWidgets.QApplication.instance(), dark)
        QtCore.QSettings(APP, APP).setValue("dark", dark)
        self.b_theme.setText("☀  Light mode" if dark else "☾  Dark mode")
        for row in self.steps:
            row.restyle()
        for w in (self.plot, self.line_plot, self.bubble_plot):
            w.setBackground(C["plot_bg"])
        if self.result is not None:
            self._draw_line(self.result)
            self._draw_bubbles(self.result)
        self.load_bar.setStyleSheet("QProgressBar{max-height:10px;min-height:10px;border-radius:5px}"
                                    f"QProgressBar::chunk{{background:{C['tumor']};border-radius:5px}}")
        if self.result is not None:
            self.c_peaks[1].setStyleSheet(f"font-size:34px;font-weight:600;color:{C['text']}")
        self.spinner.update()
        self._update_buttons()

    def show_built_by(self):
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Built by")
        d.setMinimumWidth(320)
        v = QtWidgets.QVBoxLayout(d)
        v.setContentsMargins(32, 26, 32, 22)
        v.setSpacing(10)
        v.addWidget(LogoWidget(56), 0, Qt.AlignHCenter)
        t = label(APP, size=22, bold=True)
        t.setAlignment(Qt.AlignHCenter)
        v.addWidget(t)
        v.addWidget(label("BUILT BY", caps=True), 0, Qt.AlignHCenter)
        for i, (name, url) in enumerate(TEAM, 1):
            text = f'{i}.  <a href="{url}" style="color:{C["tumor"]}">{name}</a>' if url else f"{i}.  {name}"
            n = label(text, size=15)
            n.setAlignment(Qt.AlignHCenter)
            if url:
                n.setOpenExternalLinks(True)
                n.setToolTip(url)
            v.addWidget(n)
        v.addSpacing(6)
        ok = QtWidgets.QPushButton("Close")
        ok.setProperty("primary", True)
        ok.clicked.connect(d.accept)
        v.addWidget(ok, 0, Qt.AlignHCenter)
        d.exec()

    def _build_body(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QHBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 10)
        outer.setSpacing(14)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        rec = QtWidgets.QVBoxLayout()
        rec.addWidget(label("RECORDING", caps=True))
        self.rec_name = label("No recording open", bold=True, wrap=True)
        self.rec_info = label("", dim=True, size=12)
        rec.addWidget(self.rec_name)
        rec.addWidget(self.rec_info)
        left.addWidget(card(rec))

        steps = QtWidgets.QVBoxLayout()
        steps.setSpacing(2)
        pre, rd, pk, tv = self.mods
        self.steps, self.step_titles = [], []
        group_titles = {"prepare": "BEFORE ANALYSIS · PREPARE THE DATA", "analyse": "ANALYSIS"}
        last_group = None
        for n, (group, title, detail) in enumerate(step_list(pre, rd, pk, tv), 1):
            if group != last_group:
                if last_group is not None:
                    steps.addSpacing(10)
                steps.addWidget(label(group_titles[group], caps=True))
                last_group = group
            row = StepRow(n, title, detail)
            self.steps.append(row)
            self.step_titles.append(title)
            steps.addWidget(row)
        steps.addStretch(1)
        left.addWidget(card(steps), 1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(330)
        outer.addWidget(lw)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setObjectName("bg")
        self.stack.addWidget(self._welcome_page())
        self.stack.addWidget(self._results_page())
        self.stack.addWidget(self._loading_page())
        outer.addWidget(self.stack, 1)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        self.c_peaks = self._stat("Tumor events (peaks)", big=True)
        self.c_volume = self._stat("Tumor volume")
        self.c_regions = self._stat("Active regions")
        self.c_length = self._stat("Recording length")
        for c in (self.c_peaks, self.c_volume, self.c_regions, self.c_length):
            right.addWidget(c[0])
        right.addStretch(1)
        self.b_corr = QtWidgets.QPushButton("Peaks vs volume…")
        self.b_corr.setToolTip("Is the number of peaks linearly related to the estimated volume?")
        self.b_corr.setCursor(Qt.PointingHandCursor)
        self.b_corr.clicked.connect(self.show_correlation)
        right.addWidget(self.b_corr)
        rw = QtWidgets.QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(240)
        outer.addWidget(rw)

    def _welcome_page(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setAlignment(Qt.AlignCenter)
        v.setSpacing(12)
        v.addWidget(LogoWidget(80), 0, Qt.AlignHCenter)
        v.addWidget(label(f"Welcome to {APP}", size=30, bold=True), 0, Qt.AlignHCenter)
        v.addWidget(label("Open a sensor recording, then press Analyse.\n"
                          "The steps on the left show what happens to the data.", dim=True, size=14),
                    0, Qt.AlignHCenter)
        b = QtWidgets.QPushButton("Open recording…")
        b.setProperty("primary", True)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self.open_recording)
        v.addWidget(b, 0, Qt.AlignHCenter)
        v.addWidget(label("Research prototype. An aid to the surgeon’s judgement, not for clinical use.", dim=True, size=12),
                    0, Qt.AlignHCenter)
        return w

    def _loading_page(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setAlignment(Qt.AlignCenter)
        v.setSpacing(14)
        self.spinner = Spinner(64)
        v.addWidget(self.spinner, 0, Qt.AlignHCenter)
        self.load_title = label("Analysing…", size=24, bold=True)
        self.load_title.setAlignment(Qt.AlignHCenter)
        v.addWidget(self.load_title, 0, Qt.AlignHCenter)
        self.load_step = label("", size=15)
        self.load_step.setAlignment(Qt.AlignHCenter)
        v.addWidget(self.load_step, 0, Qt.AlignHCenter)
        self.load_detail = label("", dim=True, size=13)
        self.load_detail.setAlignment(Qt.AlignHCenter)
        v.addWidget(self.load_detail, 0, Qt.AlignHCenter)
        row = QtWidgets.QHBoxLayout()
        self.load_bar = QtWidgets.QProgressBar()
        self.load_bar.setRange(0, 100)
        self.load_bar.setFixedWidth(420)
        self.load_bar.setStyleSheet("QProgressBar{max-height:10px;min-height:10px;border-radius:5px}"
                                    f"QProgressBar::chunk{{background:{C['tumor']};border-radius:5px}}")
        self.load_pct = label("0%", bold=True, size=14)
        self.load_pct.setFixedWidth(50)
        row.addStretch(1)
        row.addWidget(self.load_bar)
        row.addWidget(self.load_pct)
        row.addStretch(1)
        v.addLayout(row)
        self.load_hint = label("Every run starts fresh. Large recordings (GBs) take a few minutes to read.",
                               dim=True, size=12)
        self.load_hint.setAlignment(Qt.AlignHCenter)
        v.addWidget(self.load_hint, 0, Qt.AlignHCenter)
        return w

    def _results_page(self):
        page = QtWidgets.QWidget()
        split = QtWidgets.QSplitter(Qt.Vertical, page)
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        pc = QtWidgets.QVBoxLayout()
        head = QtWidgets.QHBoxLayout()
        self.graph_title = label("", caps=True)
        head.addWidget(self.graph_title)
        head.addSpacing(12)
        self.view_group = QtWidgets.QButtonGroup(self)
        self.view_group.setExclusive(True)
        for i, (text, tip) in enumerate(self.GRAPH_VIEWS):
            b = QtWidgets.QPushButton(text)
            b.setProperty("seg", True)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            self.view_group.addButton(b, i)
            head.addWidget(b)
        self.view_group.idClicked.connect(self.set_graph_view)
        head.addStretch(1)
        self.zoom_span = label("", dim=True, size=12)
        head.addWidget(self.zoom_span)
        head.addSpacing(8)
        for text, factor, tip in (("−", 2.0, "Zoom out (Ctrl -)"), (None, None, None), ("+", 0.5, "Zoom in (Ctrl +)")):
            if text is None:
                self.zoom_lbl = label("100%", bold=True)
                self.zoom_lbl.setFixedWidth(64)
                self.zoom_lbl.setAlignment(Qt.AlignCenter)
                head.addWidget(self.zoom_lbl)
                continue
            b = QtWidgets.QPushButton(text)
            b.setProperty("small", True)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, f=factor: self.zoom(f))
            head.addWidget(b)
        head.addSpacing(8)
        reset = QtWidgets.QPushButton("Reset zoom")
        reset.clicked.connect(self.reset_zoom)
        head.addWidget(reset)
        QtGui.QShortcut(QtGui.QKeySequence.ZoomIn, self, lambda: self.zoom(0.5))
        QtGui.QShortcut(QtGui.QKeySequence.ZoomOut, self, lambda: self.zoom(2.0))
        pc.addLayout(head)
        self.plot = pg.PlotWidget(background=C["plot_bg"])
        self.plot.showGrid(x=False, y=True, alpha=0.12)
        self.plot.setLabel("bottom", "seconds")
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.getPlotItem().vb.sigXRangeChanged.connect(self._update_zoom_label)   # wheel, drag, buttons
        self.regions_item = pg.BarGraphItem(x0=[], x1=[], y0=[], height=[], brush=pg.mkBrush(255, 69, 58, 40), pen=None)
        self.plot.addItem(self.regions_item)
        self.curve = self.plot.plot(pen=pg.mkPen(C["trace"], width=1.1))
        self.curve.setDownsampling(auto=True, method="peak")
        self.curve.setClipToView(True)
        self.peaks_item = pg.ScatterPlotItem(symbol="t", size=10, brush=pg.mkBrush(C["tri"]), pen=None)
        self.plot.addItem(self.peaks_item)
        self.band = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(255, 255, 255, 35), pen=pg.mkPen(None))
        self.band.hide()
        self.plot.addItem(self.band, ignoreBounds=True)

        # Line graph: running total of the tumor volume estimate (ml) along the recording
        self.line_plot = self._graph("cumulative tumor volume (ml)")
        self.line_band = None
        self.line_curve = self.line_plot.plot()
        self.line_marks = pg.ScatterPlotItem(size=6, pen=None)
        self.line_plot.addItem(self.line_marks)

        # Bubble chart: one bubble per active region with peaks; size = its volume
        self.bubble_plot = self._graph("peaks in the region")
        self.bubbles = None

        self.graphs = QtWidgets.QStackedWidget()
        for w in (self.plot, self.line_plot, self.bubble_plot):
            self.graphs.addWidget(w)
        pc.addWidget(self.graphs, 1)
        self.graph_note = label("", dim=True, size=12, wrap=True)
        pc.addWidget(self.graph_note)
        split.addWidget(card(pc, (14, 12, 14, 10)))
        view = QtCore.QSettings(APP, APP).value("graph_view", 0, int)
        self.set_graph_view(view if 0 <= view < len(self.GRAPH_VIEWS) else 0)

        tc = QtWidgets.QVBoxLayout()
        tc.addWidget(label("TUMOR EVENTS", caps=True))
        self.table = QtWidgets.QTableWidget(0, 5)
        for j, text in enumerate(["#", "PEAK TIME (s)", "HEIGHT (units)", "REGION (s)", "REGION VOLUME (units)"]):
            h = QtWidgets.QTableWidgetItem(text)
            h.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if j == 0 else Qt.AlignRight))
            self.table.setHorizontalHeaderItem(j, h)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setShowGrid(False)
        self.table.currentCellChanged.connect(lambda row, *_: self._zoom_to(row))
        tc.addWidget(self.table, 1)
        tc.addWidget(label("Click a row to zoom to that peak.", dim=True, size=12))
        split.addWidget(card(tc, (14, 12, 14, 10)))
        split.setSizes([520, 300])
        return page

    GRAPH_VIEWS = [("Signal", "The cleaned signal with detected peaks and active regions"),
                   ("Line", "Running total of the tumor volume estimate (ml) along the recording"),
                   ("Bubble", "One bubble per active region: time, peaks and volume (bubble size)")]
    GRAPH_TITLES = ["SIGNAL · ADC2, BASELINE REMOVED", "LINE · CUMULATIVE TUMOR VOLUME",
                    "BUBBLE · ACTIVE REGIONS WITH PEAKS"]
    GRAPH_NOTES = ["Red triangles = detected peaks · shaded = active regions · wheel zooms, drag pans",
                   "Line = point estimate added up region by region (red dot = end of a region) · shaded band = "
                   "lower–upper range · wheel zooms, drag pans",
                   "Each bubble is an active region with peaks: across = time, up = peaks in it, size = its volume "
                   "(ml) · hover a bubble for details · wheel zooms, drag pans"]

    def _graph(self, y_label):
        """A time-axis plot whose zoom and pan follow the signal plot."""
        w = pg.PlotWidget(background=C["plot_bg"])
        w.showGrid(x=False, y=True, alpha=0.12)
        w.setLabel("bottom", "seconds")
        w.setLabel("left", y_label)
        w.setMouseEnabled(x=True, y=False)
        w.setMenuEnabled(False)
        w.setXLink(self.plot)
        vb = w.getPlotItem().vb
        vb.setAutoVisible(y=True)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        return w

    def set_graph_view(self, i):
        self.graphs.setCurrentIndex(i)
        self.view_group.button(i).setChecked(True)
        self.graph_title.setText(self.GRAPH_TITLES[i])
        self.graph_note.setText(self.GRAPH_NOTES[i])
        QtCore.QSettings(APP, APP).setValue("graph_view", i)

    @staticmethod
    def _region_rows(r):
        """Per-region start, duration, peaks, ml point/lower/upper and max height, sorted by time."""
        k = r["ml_scale"]
        rows = sorted(((d["region_start_sec"], d["duration_sec"], d["peak_count_in_region"],
                        d["integrated_units"] * k, d["integrated_units_lower"] * k, d["integrated_units_upper"] * k,
                        d["max_amplitude"]) for d in r["reg_details"]), key=lambda t: t[0])
        return np.array(rows, dtype=float).reshape(-1, 7)

    def _draw_line(self, r):
        rows = self._region_rows(r)
        if self.line_band is not None:
            self.line_plot.removeItem(self.line_band)
            self.line_band = None
        if not len(rows):
            self.line_curve.setData([], [])
            self.line_marks.setData([], [])
            return
        t_end = rows[:, 0] + rows[:, 1]
        x = np.concatenate([[0.0], t_end, [r["duration"]]])
        point, lower, upper = (np.maximum(np.concatenate([[0.0], np.cumsum(rows[:, c])]), 0) for c in (3, 4, 5))
        point, lower, upper = (np.append(a, a[-1]) for a in (point, lower, upper))
        band = QtGui.QColor(C["trace"])
        band.setAlpha(45)
        self.line_band = pg.FillBetweenItem(pg.PlotDataItem(x, lower), pg.PlotDataItem(x, upper),
                                            brush=pg.mkBrush(band))
        self.line_plot.addItem(self.line_band)
        self.line_curve.setData(x, point, pen=pg.mkPen(C["trace"], width=2))
        self.line_curve.setZValue(2)
        self.line_marks.setData(t_end, point[1:-1], brush=pg.mkBrush(C["tri"]))
        self.line_marks.setZValue(3)

    def _draw_bubbles(self, r):
        rows = self._region_rows(r)
        if self.bubbles is not None:
            self.bubble_plot.removeItem(self.bubbles)
            self.bubbles = None
        if not len(rows):
            return
        mid = rows[:, 0] + rows[:, 1] / 2
        ml = np.maximum(rows[:, 3], 0)
        size = 8 + 42 * np.sqrt(ml / ml.max()) if ml.max() > 0 else np.full(len(ml), 8.0)
        fill = QtGui.QColor(C["trace"])
        fill.setAlpha(110)
        spots = [dict(pos=(mid[i], rows[i, 2]), size=float(size[i]),
                      data=(rows[i, 0], rows[i, 0] + rows[i, 1], int(rows[i, 2]), ml[i], rows[i, 6]))
                 for i in range(len(rows))]
        self.bubbles = pg.ScatterPlotItem(
            spots=spots, brush=pg.mkBrush(fill), pen=pg.mkPen(C["trace"], width=1),
            hoverable=True, hoverBrush=pg.mkBrush(C["tri"]),
            tip=lambda x, y, data: (f"{data[0]:.2f}–{data[1]:.2f} s\n{data[2]} peaks\n"
                                    f"{data[3]:.3f} ml\nmax height {data[4]:.0f}"))
        self.bubble_plot.addItem(self.bubbles)

    def _stat(self, title, big=False):
        v = QtWidgets.QVBoxLayout()
        v.setSpacing(2)
        v.addWidget(label(title, dim=True, size=12))
        val = label("–", size=34 if big else 22, bold=True)
        sub = label("", dim=True, size=12, wrap=True)
        v.addWidget(val)
        v.addWidget(sub)
        return card(v), val, sub

    def _running(self):
        return self.thread is not None

    def _update_buttons(self):
        run = self._running()
        self.b_open.setEnabled(not run)
        self.b_run.setEnabled(self.path is not None or run)
        self.b_run.setText("■  Stop" if run else "▶  Analyse")
        self.b_run.setProperty("primary", not run)
        self.b_run.setProperty("stop", run)
        self.b_run.style().unpolish(self.b_run)
        self.b_run.style().polish(self.b_run)
        self.b_export.setEnabled(self.result is not None and not run)
        if hasattr(self, "b_corr"):
            self.b_corr.setEnabled(self.result is not None and not run)

    def open_recording(self):
        last = QtCore.QSettings(APP, APP).value("last_folder", "")
        start = Path(last) if last and Path(last).is_dir() else HERE / "Test signals from previous years"
        if not start.is_dir():
            start = Path.home() / "Documents"
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open recording", str(start),
                                                     "Sensor recordings (*.csv *.txt);;All files (*)")
        if p:
            QtCore.QSettings(APP, APP).setValue("last_folder", str(Path(p).parent))
            self.load(Path(p))

    def load(self, path: Path):
        self.path, self.result = path, None
        self.rec_name.setText(path.name)
        self.rec_info.setText(f"{path.stat().st_size / 1e6:,.0f}MB · not analysed yet")
        for row in self.steps:
            row.set_state("pending")
        self._clear_results()
        self.stack.setCurrentIndex(0)
        self.status.setText("Press Analyse to run the steps on the left")
        log.info("opened %s", path)
        self._update_buttons()

    def toggle_run(self):
        if self._running():
            self.worker.stop()
            self.status.setText("Stopping…")
            return
        if self.path is None:
            return
        for row in self.steps:
            row.set_state("pending")
        self._clear_results()
        self.worker = PipelineWorker(self.mods, self.path)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.step.connect(self._on_step)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.pbar.setValue(0)
        self.pbar.show()
        self.status.setText(f"Analysing {self.path.name}…")
        self.load_title.setText(f"Analysing {self.path.name}")
        self.load_step.setText("Starting…")
        self.load_detail.setText("")
        self.load_bar.setValue(0)
        self.load_bar.show()
        self.load_pct.setText("0%")
        self.load_pct.show()
        self.load_hint.show()
        self.spinner.show()
        self.spinner.start()
        self.stack.setCurrentIndex(2)
        self.thread.start()
        self._update_buttons()

    def _end_thread(self):
        self.thread.quit()
        self.thread.wait(3000)
        self.thread = self.worker = None
        self.pbar.hide()
        self.spinner.stop()
        self._update_buttons()

    @Slot(int, str, str)
    def _on_step(self, i, state, text):
        self.steps[i].set_state(state, text)
        if state == "running":
            self.load_step.setText(f"Step {i + 1} of {len(self.steps)} · {self.step_titles[i]}")
            self.load_detail.setText(text)

    @Slot(float)
    def _on_progress(self, fraction):
        pct = int(fraction * 100)
        self.pbar.setValue(pct)
        self.load_bar.setValue(pct)
        self.load_pct.setText(f"{pct}%")

    @Slot(object)
    def _on_finished(self, res):
        self._end_thread()
        self.result = res
        self._show(res)
        self.status.setText(f"Done · {len(res['events'])} tumor events found in {res['name']}")
        self._update_buttons()

    @Slot(str)
    def _on_failed(self, msg):
        self._end_thread()
        self.status.setText(f"Not analysed: {msg}")
        self.spinner.hide()
        self.load_title.setText("Analysis stopped" if msg == "Stopped." else "Could not analyse this file")
        self.load_step.setText(msg)
        self.load_detail.setText("The step marked ✕ on the left shows where it stopped.")
        self.load_bar.hide()
        self.load_pct.hide()
        self.load_hint.hide()

    def _clear_results(self):
        self.curve.setData([], [])
        self.peaks_item.setData([], [])
        self.regions_item.setOpts(x0=[], x1=[], y0=[], height=[])
        self._draw_line(dict(reg_details=[], ml_scale=1.0))
        self._draw_bubbles(dict(reg_details=[], ml_scale=1.0))
        self.table.setRowCount(0)
        for c in (self.c_peaks, self.c_volume, self.c_regions, self.c_length):
            c[1].setText("–")
            c[2].setText("")

    def _show(self, r):
        self.stack.setCurrentIndex(1)
        sig, fs, ev = r["signal"], r["fs"], r["events"]
        self.curve.setData(np.arange(len(sig)) / fs, sig)
        self.peaks_item.setData(ev["peak_time_s"].to_numpy(), ev["height"].to_numpy())
        lo, hi = float(sig.min()), float(max(sig.max(), ev["height"].max() if len(ev) else 0))
        self.regions_item.setOpts(x0=r["starts"], x1=r["ends"], y0=np.full(len(r["starts"]), lo),
                                  height=np.full(len(r["starts"]), hi - lo))
        self._draw_line(r)
        self._draw_bubbles(r)
        self.table.setRowCount(len(ev))
        for i, e in ev.iterrows():
            cells = [str(i + 1), f"{e.peak_time_s:.2f}", f"{e.height:.0f}",
                     f"{e.region_start_s:.2f} – {e.region_end_s:.2f}", f"{e.region_volume_units:.2f}"]
            for j, text in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if j == 0 else Qt.AlignRight))
                self.table.setItem(i, j, it)
        n = len(ev)
        self.c_peaks[1].setText(str(n))
        self.c_peaks[1].setStyleSheet(f"font-size:34px;font-weight:600;color:{C['text']}")
        self.c_peaks[2].setText(f"{n / r['duration'] * 60:.1f} per minute" if n else "No tumor detected")
        self.c_volume[1].setText(f"{r['ml']:,.2f} ml")
        sub = [f"range {r['ml_lower']:,.2f}–{r['ml_upper']:,.2f} ml",
               f"{r['volume']:,.1f} units ({r['volume_lower']:,.1f}–{r['volume_upper']:,.1f})"]
        if r["ml_divisor"] > 1:
            sub.append(f"÷{r['ml_divisor']} for resuctions in the file name")
        if r.get("labelled_ml") is not None:
            sub.append(f"label in file name: {r['labelled_ml']:g} ml")
        self.c_volume[2].setText("\n".join(sub))
        self.c_regions[1].setText(f"{r['n_regions']:,}")
        self.c_regions[2].setText("parts of the signal with activity")
        self.c_length[1].setText(f"{int(r['duration'] // 60)}min {int(r['duration'] % 60)}s"
                                 if r["duration"] >= 60 else f"{r['duration']:.1f}s")
        self.rec_info.setText(f"{self.path.stat().st_size / 1e6:,.0f}MB · analysed")
        history = self._history()
        history[r["name"]] = [n, round(r["volume"], 2)]
        QtCore.QSettings(APP, APP).setValue("history", json.dumps(history))
        self.reset_zoom()

    @staticmethod
    def _history():
        try:
            return json.loads(QtCore.QSettings(APP, APP).value("history", "{}") or "{}")
        except (TypeError, ValueError):
            return {}

    def show_correlation(self):
        if self.result is None:
            return
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Peaks vs volume")
        d.resize(980, 460)
        v = QtWidgets.QVBoxLayout(d)
        row = QtWidgets.QHBoxLayout()
        history = self._history()
        names = list(history)
        here = names.index(self.result["name"]) if self.result["name"] in history else None
        charts = [("THIS RECORDING · ONE DOT PER ACTIVE REGION", self.result["reg_peaks"], self.result["reg_vol"],
                   "peaks in the region", "region volume (units)", None, False),
                  ("EVERY RECORDING ANALYSED · ONE DOT PER FILE", [history[k][0] for k in names],
                   [history[k][1] for k in names], "peaks in the recording", "volume (units)", here, True)]
        for title, x, y, xl, yl, hi, files in charts:
            x, y = np.asarray(x, float), np.asarray(y, float)
            col = QtWidgets.QVBoxLayout()
            col.addWidget(label(title, caps=True))
            w = pg.PlotWidget(background=C["plot_bg"])
            w.setMenuEnabled(False)
            w.showGrid(x=True, y=True, alpha=0.12)
            w.setLabel("bottom", xl)
            w.setLabel("left", yl)
            w.addItem(pg.ScatterPlotItem(x, y, size=8, brush=pg.mkBrush(C["trace"]), pen=None))
            if hi is not None:
                w.addItem(pg.ScatterPlotItem([x[hi]], [y[hi]], size=13, brush=pg.mkBrush(C["tri"]), pen=None))
            if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                slope, icpt = np.polyfit(x, y, 1)
                r = float(np.corrcoef(x, y)[0, 1])
                xs = np.array([x.min(), x.max()])
                w.plot(xs, slope * xs + icpt, pen=pg.mkPen(C["tri"], width=2, style=Qt.DashLine))
                note = f"r = {r:.2f} · r² = {r * r:.2f} · volume ≈ {slope:.2f} × peaks {icpt:+.1f} · n = {len(x)}"
            elif files:
                note = f"n = {len(x)} · analyse at least 3 recordings to fit a line"
            else:
                note = f"n = {len(x)} · not enough spread to fit a line"
            col.addWidget(w, 1)
            col.addWidget(label(note, dim=True, size=12, wrap=True))
            row.addWidget(card(col, (14, 12, 14, 10)), 1)
        v.addLayout(row, 1)
        foot = QtWidgets.QHBoxLayout()
        foot.addWidget(label("r near 1 means more peaks go with more volume. The right chart keeps every file you "
                             "analyse (latest result per file); the red dot is the current one.", dim=True, size=12,
                             wrap=True), 1)
        clear = QtWidgets.QPushButton("Clear history")
        clear.clicked.connect(lambda: (QtCore.QSettings(APP, APP).remove("history"), d.accept()))
        foot.addWidget(clear)
        ok = QtWidgets.QPushButton("Close")
        ok.setProperty("primary", True)
        ok.clicked.connect(d.accept)
        foot.addWidget(ok)
        v.addLayout(foot)
        d.exec()

    def reset_zoom(self):
        self.band.hide()
        r = self.result
        if not r:
            return
        sig, ev = r["signal"], r["events"]
        top = max(float(sig.max()), float(ev["height"].max()) * 1.1 if len(ev) else 0.0)
        bottom = float(sig.min())
        pad = 0.05 * ((top - bottom) or 1)
        self.plot.setXRange(0, len(sig) / r["fs"], padding=0.01)
        self.plot.setYRange(bottom - pad, top + pad, padding=0)

    def zoom(self, factor):
        if not self.result:
            return
        full = len(self.result["signal"]) / self.result["fs"]
        x0, x1 = self.plot.getPlotItem().vb.viewRange()[0]
        span = min(max((x1 - x0) * factor, 0.05), full)
        centre = min(max((x0 + x1) / 2, span / 2), full - span / 2)
        self.plot.setXRange(centre - span / 2, centre + span / 2, padding=0)

    def _update_zoom_label(self, *_):
        if not self.result:
            self.zoom_lbl.setText("100%")
            self.zoom_span.setText("")
            return
        full = len(self.result["signal"]) / self.result["fs"]
        x0, x1 = self.plot.getPlotItem().vb.viewRange()[0]
        span = max(x1 - x0, 1e-9)
        self.zoom_lbl.setText(f"{max(full / span, 1.0) * 100:,.0f}%")
        self.zoom_span.setText(f"showing {span:,.1f}s of {full:,.1f}s" if span < full * 0.98 else "whole recording")

    def _zoom_to(self, row):
        if self.result is None or row < 0 or row >= len(self.result["events"]):
            return
        e = self.result["events"].iloc[row]
        self.plot.setXRange(e.peak_time_s - 1.0, e.peak_time_s + 1.0, padding=0)
        self.band.setRegion((e.region_start_s, e.region_end_s))
        self.band.show()

    def export_csv(self):
        if self.result is None:
            return
        name = Path(self.result["name"]).stem + "_tumor_events.csv"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", name, "CSV (*.csv)")
        if p:
            self.result["events"].to_csv(p, index=False)
            self.status.setText(f"Saved {p}")

    def closeEvent(self, e):
        if self._running():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(5000)
        super().closeEvent(e)


def logo_icon():
    icon = QtGui.QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        pix = QtGui.QPixmap(size, size)
        pix.fill(Qt.transparent)
        LogoWidget(size).render(pix, QtCore.QPoint(0, 0), QtGui.QRegion(),
                                QtWidgets.QWidget.RenderFlag.DrawChildren)
        icon.addPixmap(pix)
    return icon


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s", "%H:%M:%S")
    handlers = [logging.FileHandler(LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    for name in ("lucent", "marginum"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        lg.propagate = False
        for h in handlers:
            h.setFormatter(fmt)
            lg.addHandler(h)


def selftest(csv_path, out_path):
    try:
        worker = PipelineWorker(load_modules(), Path(csv_path))
        res = worker._run()
        summary = {"ok": True, "file": res["name"], "peaks": len(res["events"]), "regions": res["n_regions"],
                   "volume_units": res["volume"], "volume_ml": res["ml"],
                   "volume_ml_range": [res["ml_lower"], res["ml_upper"]], "duration_s": res["duration"]}
    except Exception as exc:
        log.exception("selftest failed")
        summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    Path(out_path).write_text(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    setup_logging()
    if len(sys.argv) == 4 and sys.argv[1] == "--selftest":
        return selftest(sys.argv[2], sys.argv[3])
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("UEF.IDP2026.Lucent")
        except Exception:
            pass
    pg.setConfigOptions(antialias=True, foreground=C["plot_fg"])
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setStyle("Fusion")
    apply_theme(app, QtCore.QSettings(APP, APP).value("dark", False, bool))
    app.setWindowIcon(logo_icon())
    try:
        mods = load_modules()
    except Exception as exc:
        log.exception("could not import modules/")
        QtWidgets.QMessageBox.critical(None, APP, f"Could not load the pipeline in modules/.\n\n{exc}\n\n"
                                                  "Put gui_app.py in the marginum folder next to modules/.")
        return 1
    win = MainWindow(mods)
    win.resize(1500, 900)
    win.show()
    if len(sys.argv) > 1:
        win.load(Path(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "crash.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
