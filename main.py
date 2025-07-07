#!/usr/bin/env python3
"""Minimal non-blocking GUI for two antennas + recorder.

Architecture
------------
GUI thread               – Qt / user interaction.
TrackerThread("N")       – owns *one* `Tracker` instance (north antenna).
TrackerThread("S")       – owns *one* `Tracker` instance (south antenna).
RecorderThread           – owns *one* `Recorder` instance.

Communication is via `queue.Queue` instances:
GUI → worker : tuples like (command, kwargs)
worker → GUI : Qt signals (`ProgressInfo` + plain log strings).

The GUI never blocks; all potentially long calls are executed in the
three dedicated worker threads.
"""

from __future__ import annotations

import sys
import time
import queue
import threading
import logging
import signal
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QProgressBar,
)
from PyQt5.QtCore import QObject, pyqtSignal, Qt

from tracking import Tracker, Source
from tracking.utils.progress import ProgressInfo, ProgressCallback
from recording import Recorder

# -----------------------------------------------------------------------------
# Helper: monkey-patch `signal.signal` in a worker thread before constructing
# Tracker / Recorder (these libraries might try to set signal handlers).
# -----------------------------------------------------------------------------

def _construct_safely(factory: Callable):
    """Call *factory* with a temporary dummy `signal.signal` to avoid ValueError."""
    original_signal = signal.signal

    def dummy_signal(signum, handler):  # noqa: D401
        return None

    signal.signal = dummy_signal
    try:
        return factory()
    finally:
        signal.signal = original_signal


# -----------------------------------------------------------------------------
# Qt bridges – log & progress from worker → GUI
# -----------------------------------------------------------------------------

class UILogBridge(QObject):
    log_signal = pyqtSignal(str)

    def write(self, msg: str):  # logging.Handler expects write-like API
        self.log_signal.emit(msg)


class UIProgressBridge(ProgressCallback, QObject):
    progress_signal = pyqtSignal(object)  # ProgressInfo

    def __init__(self):
        super().__init__()

    def __call__(self, info: ProgressInfo):  # noqa: D401
        self.progress_signal.emit(info)


# Bridge for recorder parameter updates
class ParamBridge(QObject):
    param_signal = pyqtSignal(dict)  # {'fftshift': int, 'acclen': int, 'overflow': int|None, 'recording': bool}

    def emit_update(self, data: dict):
        self.param_signal.emit(data)


# -----------------------------------------------------------------------------
# Worker threads
# -----------------------------------------------------------------------------

class TrackerThread(threading.Thread):
    """Dedicated thread owning a single `Tracker` instance."""

    def __init__(self, ant: str, prog_cb: UIProgressBridge, log: logging.Logger):
        super().__init__(daemon=True)
        self.ant = ant
        self.prog_cb = prog_cb
        self.log = log.getChild(f"tracker_{ant}")
        self.cmd_q: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.running = True
        self.tracker: Tracker | None = None

    # Public API --------------------------------------------------------
    def submit(self, cmd: str, **kwargs):
        """Queue a command for the tracker thread."""
        self.cmd_q.put((cmd, kwargs))

    # Internal ----------------------------------------------------------
    def _init_tracker(self):
        try:
            self.tracker = _construct_safely(Tracker)
            self.log.info("Tracker initialised")
        except Exception as exc:
            self.log.error(f"Failed to init Tracker: {exc}")
            self.tracker = None

    def run(self):
        self._init_tracker()
        while self.running:
            try:
                cmd, kw = self.cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.tracker:
                self.log.warning("Tracker unavailable – ignoring command %s", cmd)
                continue

            try:
                if cmd == "slew":
                    self.log.info("Slew → az %(az)s el %(el)s", kw)
                    self.tracker.run_slew(ant=self.ant, progress_callback=self.prog_cb, **kw)
                elif cmd == "track":
                    self.log.info("Track → %(source)s", kw)
                    self.tracker.run_track(ant=self.ant, progress_callback=self.prog_cb, **kw)
                elif cmd == "park":
                    self.log.info("Park")
                    self.tracker.run_park(ant=self.ant, progress_callback=self.prog_cb)
                elif cmd == "stop":
                    self.log.info("Stop")
                    self.tracker._stop()
                else:
                    self.log.warning("Unknown command %s", cmd)
            except Exception as exc:
                self.log.error("%s command failed: %s", cmd, exc)


class RecorderThread(threading.Thread):
    """Dedicated thread owning a single `Recorder` instance."""

    def __init__(self, log: logging.Logger, param_bridge: ParamBridge):
        super().__init__(daemon=True)
        self.log = log.getChild("recorder")
        self.cmd_q: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.recorder: Recorder | None = None
        self.running = True
        self.rec_worker: threading.Thread | None = None  # internal worker for blocking start_recording
        self.param_bridge = param_bridge
        self._last_param_emit = 0.0

    # Public API --------------------------------------------------------
    def submit(self, cmd: str, **kwargs):
        self.cmd_q.put((cmd, kwargs))

    # Internal ----------------------------------------------------------
    def _init_recorder(self):
        try:
            self.recorder = _construct_safely(Recorder)
            self.log.info("Recorder initialised")
        except Exception as exc:
            self.log.error(f"Failed to init Recorder: {exc}")
            self.recorder = None

    def run(self):
        self._init_recorder()
        while self.running:
            try:
                cmd, kw = self.cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.recorder:
                self.log.warning("Recorder unavailable – ignoring command %s", cmd)
                continue

            try:
                if cmd == "start":
                    p = kw  # params dict
                    if self.rec_worker and self.rec_worker.is_alive():
                        self.log.warning("Recording already in progress")
                        continue

                    def _record_task():
                        try:
                            self.recorder.set_fftshift(p["fftshift"])
                            self.recorder.set_acclen(p["acclen"])
                            # Blocking call – will run until stop_recording() sets flag.
                            self.recorder.start_recording("observation", 3600)
                        except Exception as exc:
                            self.log.error("Recording task error: %s", exc)

                    self.log.info("Start recording fft=%s acc=%s", p["fftshift"], p["acclen"])
                    self.rec_worker = threading.Thread(target=_record_task, daemon=True)
                    self.rec_worker.start()
                elif cmd == "stop":
                    if self.recorder.is_recording:
                        self.log.info("Stop recording requested")
                        self.recorder.stop_recording()
                    else:
                        self.log.info("Recorder not active")
                elif cmd == "set_params":
                    p = kw
                    try:
                        if "fftshift" in p:
                            self.recorder.set_fftshift(p["fftshift"])
                        if "acclen" in p:
                            self.recorder.set_acclen(p["acclen"])
                        self.log.info("Params updated via GUI")
                    except Exception as exc:
                        self.log.error("Failed to set params: %s", exc)
                else:
                    self.log.warning("Unknown recorder cmd %s", cmd)
            except Exception as exc:
                self.log.error("Recorder %s failed: %s", cmd, exc)

            # Periodic parameter emit every 2 seconds
            now = time.time()
            if self.recorder and now - self._last_param_emit > 2:
                try:
                    fft0, _ = self.recorder.get_fftshift()
                    acclen = self.recorder.get_acclen()
                    overflow = None
                    try:
                        overflow = self.recorder._device.cross_corr.get_overflow_count()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self.param_bridge.emit_update(
                        {
                            "fftshift": fft0,
                            "acclen": acclen,
                            "overflow": overflow,
                            "recording": self.recorder.is_recording,
                        }
                    )
                except Exception as exc:
                    self.log.debug("Param emit error: %s", exc)
                self._last_param_emit = now


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

@dataclass
class AntennaUI:
    ra_spin: QDoubleSpinBox
    dec_spin: QDoubleSpinBox
    az_spin: QDoubleSpinBox
    el_spin: QDoubleSpinBox
    prog: QProgressBar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Antenna GUI")
        self.resize(800, 500)

        # ---- Logging setup (Qt-bridge handler) -------------------------
        self.log_bridge = UILogBridge()
        self.log_bridge.log_signal.connect(self._append_log)
        handler = logging.Handler()
        handler.emit = lambda record: self.log_bridge.write(logging.getLogger().handlers[0].format(record))  # type: ignore
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

        # ---- Progress bridges -----------------------------------------
        self.prog_n = UIProgressBridge()
        self.prog_s = UIProgressBridge()
        self.prog_n.progress_signal.connect(self._on_progress)
        self.prog_s.progress_signal.connect(self._on_progress)

        # ---- Worker threads ------------------------------------------
        self.trk_N = TrackerThread("N", self.prog_n, root)
        self.trk_S = TrackerThread("S", self.prog_s, root)
        self.param_bridge = ParamBridge()
        self.param_bridge.param_signal.connect(self._update_rec_params)
        self.rec_thread = RecorderThread(root, self.param_bridge)
        for t in (self.trk_N, self.trk_S, self.rec_thread):
            t.start()

        # ---- Build UI -------------------------------------------------
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)

        # Antenna panels
        self.ui_n = self._antenna_panel("North", "N")
        self.ui_s = self._antenna_panel("South", "S")
        v_left = QVBoxLayout()
        v_left.addWidget(self.ui_n.box)
        v_left.addWidget(self.ui_s.box)
        v_left.addStretch()
        h.addLayout(v_left, 1)

        # Recorder + logs
        v_right = QVBoxLayout()
        v_right.addWidget(self._recorder_panel())
        v_right.addWidget(self._log_panel())
        h.addLayout(v_right, 1)

    # ---------------- Antenna panel -----------------------------------
    def _antenna_panel(self, title: str, ant: str):
        box = QGroupBox(title)
        v = QVBoxLayout(box)

        # Track row
        track_row = QHBoxLayout()
        ra = QDoubleSpinBox(); ra.setRange(0, 24); ra.setDecimals(4)
        dec = QDoubleSpinBox(); dec.setRange(-90, 90); dec.setDecimals(4)
        btn_track = QPushButton("Track")
        track_row.addWidget(QLabel("RA")); track_row.addWidget(ra)
        track_row.addWidget(QLabel("Dec")); track_row.addWidget(dec)
        track_row.addWidget(btn_track)
        v.addLayout(track_row)

        # Slew row
        slew_row = QHBoxLayout()
        az = QDoubleSpinBox(); az.setRange(0, 360); az.setDecimals(2)
        el = QDoubleSpinBox(); el.setRange(0, 90); el.setDecimals(2)
        btn_slew = QPushButton("Slew")
        slew_row.addWidget(QLabel("Az")); slew_row.addWidget(az)
        slew_row.addWidget(QLabel("El")); slew_row.addWidget(el)
        slew_row.addWidget(btn_slew)
        v.addLayout(slew_row)

        # Park / stop row
        row = QHBoxLayout()
        btn_park = QPushButton("Park")
        btn_stop = QPushButton("Stop")
        row.addWidget(btn_park); row.addWidget(btn_stop)
        v.addLayout(row)

        # Progress bar
        prog = QProgressBar(); prog.setRange(0, 100)
        v.addWidget(prog)

        # Button wiring
        btn_track.clicked.connect(lambda: self._cmd_track(ant, ra.value(), dec.value()))
        btn_slew.clicked.connect(lambda: self._cmd_slew(ant, az.value(), el.value()))
        btn_park.clicked.connect(lambda: self._cmd_park(ant))
        btn_stop.clicked.connect(lambda: self._cmd_stop(ant))

        ui = AntennaUI(ra, dec, az, el, prog)
        ui.box = box  # type: ignore
        return ui

    # ---------------- Recorder panel ----------------------------------
    def _recorder_panel(self):
        box = QGroupBox("Recorder")
        v = QVBoxLayout(box)

        # Param row
        row = QHBoxLayout()
        self.sp_fft = QSpinBox(); self.sp_fft.setRange(0, 4095)
        self.sp_acc = QSpinBox(); self.sp_acc.setRange(1, 1_000_000); self.sp_acc.setValue(100)
        btn_apply = QPushButton("Apply")
        row.addWidget(QLabel("fftshift")); row.addWidget(self.sp_fft)
        row.addWidget(QLabel("acclen")); row.addWidget(self.sp_acc)
        row.addWidget(btn_apply)
        v.addLayout(row)

        # Status row
        status_row = QHBoxLayout()
        self.btn_rec = QPushButton("Start")
        self.lbl_rec = QLabel("Idle")
        self.lbl_overflow = QLabel("OF: N/A")
        status_row.addWidget(self.btn_rec)
        status_row.addWidget(self.lbl_rec)
        status_row.addWidget(self.lbl_overflow)
        v.addLayout(status_row)

        # Signals
        btn_apply.clicked.connect(self._apply_rec_params)
        self.btn_rec.clicked.connect(self._toggle_record)

        return box

    # ---------------- Log panel ---------------------------------------
    def _log_panel(self):
        box = QGroupBox("Logs")
        v = QVBoxLayout(box)
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        v.addWidget(self.log_view)
        return box

    # ------------------------------------------------------------------
    # Command helpers (GUI → worker threads)
    # ------------------------------------------------------------------
    def _target_thread(self, ant: str) -> TrackerThread:
        return self.trk_N if ant == "N" else self.trk_S

    def _cmd_track(self, ant: str, ra: float, dec: float):
        src = Source(ra, dec)
        self._target_thread(ant).submit("track", source=src, duration_hours=1.0)

    def _cmd_slew(self, ant: str, az: float, el: float):
        self._target_thread(ant).submit("slew", az=az, el=el)

    def _cmd_park(self, ant: str):
        self._target_thread(ant).submit("park")

    def _cmd_stop(self, ant: str):
        self._target_thread(ant).submit("stop")

    def _toggle_record(self):
        if self.btn_rec.text() == "Start":
            self.rec_thread.submit("start", fftshift=self.sp_fft.value(), acclen=self.sp_acc.value())
            self.btn_rec.setText("Stop")
            self.lbl_rec.setText("Recording")
        else:
            self.rec_thread.submit("stop")
            self.btn_rec.setText("Start")
            self.lbl_rec.setText("Idle")

    def _apply_rec_params(self):
        fft = self.sp_fft.value()
        acc = self.sp_acc.value()
        self.rec_thread.submit("set_params", fftshift=fft, acclen=acc)

    # ------------------------------------------------------------------
    # Slots (worker → GUI)
    # ------------------------------------------------------------------
    def _on_progress(self, info: ProgressInfo):
        ui = self.ui_n if info.antenna == "N" else self.ui_s
        ui.prog.setValue(int(info.percent_complete))

    def _append_log(self, text: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {text}")

    def _update_rec_params(self, data: dict):
        self.sp_fft.setValue(data.get("fftshift", self.sp_fft.value()))
        self.sp_acc.setValue(data.get("acclen", self.sp_acc.value()))
        of = data.get("overflow")
        self.lbl_overflow.setText(f"OF: {of if of is not None else 'N/A'}")
        self.lbl_rec.setText("Recording" if data.get("recording") else "Idle")

    # ------------------------------------------------------------------
    def closeEvent(self, event):  # pylint: disable=invalid-name
        # Graceful shutdown: signal threads to stop.
        for t in (self.trk_N, self.trk_S, self.rec_thread):
            t.running = False
        event.accept()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(); win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 