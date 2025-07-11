#!/usr/bin/env python3
"""Simple non-blocking GUI for controlling two antennas and a recorder.

Threads
~~~~~~~
GUI thread – Qt widgets & user interaction.
TrackerThread("N"), TrackerThread("S") – own their respective *Tracker* objects.
RecorderCmdThread – handles start/stop/parameter-set commands on a shared *Recorder*.
RecorderStatusThread – polls current fftshift (p0,p1), acclen, overflow (p0,p1) once/sec and emits a Qt signal so the GUI can update live status labels.

No long-running call blocks the GUI.
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
from typing import Dict, Tuple

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
    QCheckBox,
)
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from tracking import Tracker, Source
from tracking.utils.progress import ProgressCallback as TrkProgCB, ProgressInfo as TrkProgInfo
from recording import Recorder
from recording.utils.progress import ProgressCallback as RecProgCB, ProgressInfo as RecProgInfo, OperationType as RecOpType
from tracking.utils.coordinate_utils import get_targets_offsets, radec_to_azel, get_ovro_location_info
from utils.ftx import FTXController, Antenna, Polarization

# -----------------------------------------------------------------------------
# Helper – safe constructor for Tracker in worker threads (handles signal.signal)
# -----------------------------------------------------------------------------

def _construct_safely(factory):
    """Call *factory* replacing signal.signal with a dummy inside this thread."""
    original = signal.signal

    def dummy(_signum, _handler):
        return None

    signal.signal = dummy
    try:
        return factory()
    finally:
        signal.signal = original

# -----------------------------------------------------------------------------
# Qt bridges (log, progress, recorder status)
# -----------------------------------------------------------------------------

class UILogBridge(QObject):
    log_signal = pyqtSignal(str)

    def write(self, msg: str):  # logging.Handler-compatible
        try:
            self.log_signal.emit(msg)
        except RuntimeError:
            # Qt object has been deleted, ignore the signal emission
            pass

class UIProgressBridge(QObject):
    progress_signal = pyqtSignal(object)  # TrkProgInfo or RecProgInfo

    def __call__(self, info):  # noqa: D401
        try:
            self.progress_signal.emit(info)
        except RuntimeError:
            # Qt object has been deleted, ignore the signal emission
            pass

class RecorderStatusBridge(QObject):
    status_signal = pyqtSignal(tuple)  # (fft_p0, fft_p1, acclen, ovf_p0, ovf_p1, recording_bool)

    def emit_status(self, status: Tuple[int, int, int, int, int, bool]):
        try:
            self.status_signal.emit(status)
        except RuntimeError:
            # Qt object has been deleted, ignore the signal emission
            pass

class DataBufferBridge(QObject):
    data_signal = pyqtSignal(list)  # List of data arrays

    def emit_data(self, data: list):
        try:
            self.data_signal.emit(data)
        except RuntimeError:
            # Qt object has been deleted, ignore the signal emission
            pass

class AntennaPositionBridge(QObject):
    position_signal = pyqtSignal(str, float, float)  # (antenna, az, el)

    def emit_position(self, antenna: str, az: float, el: float):
        try:
            self.position_signal.emit(antenna, az, el)
        except RuntimeError:
            # Qt object has been deleted, ignore the signal emission
            pass

class TrackerCompletionBridge(QObject):
    completion_signal = pyqtSignal(str)  # (antenna)

    def emit_completion(self, antenna: str):
        try:
            self.completion_signal.emit(antenna)
        except RuntimeError:
            # Qt object has been deleted, ignore the signal emission
            pass

# -----------------------------------------------------------------------------
# Worker threads
# -----------------------------------------------------------------------------

class TrackerThread(threading.Thread):
    def __init__(self, ant: str, prog_cb: UIProgressBridge, completion_cb: TrackerCompletionBridge, logger_: logging.Logger):
        super().__init__(daemon=True)
        self.ant = ant
        self.log = logger_.getChild(f"tracker_{ant}")
        self.prog_cb = prog_cb
        self.completion_cb = completion_cb
        self.cmd_q: queue.Queue[Tuple[str, Dict]] = queue.Queue()
        self.running = True
        self.tracker: Tracker | None = None
        self._stop_requested = False  # Flag to request operation interruption
        self._shutting_down = False  # Flag to prevent logging during shutdown

    def submit(self, cmd: str, **kw):
        self.cmd_q.put((cmd, kw))

    def request_stop(self):
        """Request immediate stop of current operation."""
        self._stop_requested = True
        if not self._shutting_down:
            self.log.info("Stop requested for current operation")

    def cleanup_tracker(self):
        """Cleanup tracker resources including MQTT connection."""
        self._shutting_down = True  # Prevent logging during cleanup
        if self.tracker:
            try:
                self.tracker.cleanup()
                # Don't log during shutdown to avoid Qt object deletion errors
            except Exception as exc:
                # Don't log during shutdown to avoid Qt object deletion errors
                pass

    def _progress_callback_with_stop_check(self, progress_info):
        """Progress callback that checks for stop requests and interrupts if needed."""
        # Check if stop was requested
        if self._stop_requested:
            self._stop_requested = False  # Reset flag
            # Stop the tracker motion immediately
            if self.tracker:
                self.tracker.stop()
            # Raise exception to interrupt the current operation
            raise InterruptedError(f"Operation interrupted by user request")
        
        # Forward progress info to UI
        self.prog_cb(progress_info)

    # internal helpers
    def _init_tracker(self):
        try:
            self.tracker = _construct_safely(Tracker)
            self.log.info("Tracker initialised")
        except Exception as exc:
            self.log.error("Tracker init failed: %s", exc)
            self.tracker = None

    # run loop
    def run(self):
        self._init_tracker()
        while self.running:
            try:
                cmd, kw = self.cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.tracker:
                self.log.warning("Tracker unavailable – ignoring %s", cmd)
                continue

            # Reset stop flag at start of new operation
            self._stop_requested = False

            try:
                if cmd == "slew":
                    self.tracker.run_slew(ant=self.ant, progress_callback=self._progress_callback_with_stop_check, auto_cleanup=False, **kw)
                    self.completion_cb.emit_completion(self.ant)
                elif cmd == "track":
                    self.tracker.run_track(ant=self.ant, progress_callback=self._progress_callback_with_stop_check, auto_cleanup=False, **kw)
                    self.completion_cb.emit_completion(self.ant)
                elif cmd == "park":
                    self.tracker.run_park(ant=self.ant, progress_callback=self._progress_callback_with_stop_check, auto_cleanup=False)
                    self.completion_cb.emit_completion(self.ant)
                elif cmd == "rasta_scan":
                    # Extract parameters and create source
                    source = kw.get('source')
                    max_distance_deg = kw.get('max_dist_deg')
                    steps_deg = kw.get('step_deg')
                    position_angle_deg = kw.get('position_angle_deg')
                    duration_hours = kw.get('duration_hours')
                    slew = kw.get('slew', True)  # Default to True if not provided
                    park = kw.get('park', True)   # Default to True if not provided
                    
                    # Debug logging to see what parameters are being passed
                    self.log.info(f"RASTA scan parameters: max_dist_deg={max_distance_deg}, steps_deg={steps_deg}, position_angle_deg={position_angle_deg}, duration_hours={duration_hours}, slew={slew}, park={park}")
                    self.log.info(f"All kw parameters: {kw}")
                    
                    # Check that all required parameters are present
                    if source is None:
                        raise ValueError("source parameter is required")
                    if max_distance_deg is None:
                        raise ValueError("max_dist_deg parameter is required")
                    if steps_deg is None:
                        raise ValueError("step_deg parameter is required")
                    if position_angle_deg is None:
                        raise ValueError("position_angle_deg parameter is required")
                    if duration_hours is None:
                        raise ValueError("duration_hours parameter is required")
                    
                    # Additional debug logging
                    self.log.info(f"About to call run_rasta_scan with duration_hours={duration_hours}")
                    
                    self.tracker.run_rasta_scan(
                        ant=self.ant, 
                        source=source, 
                        max_distance_deg=max_distance_deg, 
                        steps_deg=steps_deg, 
                        position_angle_deg=position_angle_deg, 
                        duration_hours=duration_hours, 
                        slew=slew,
                        park=park,
                        progress_callback=self._progress_callback_with_stop_check, 
                        auto_cleanup=False
                    )
                    self.completion_cb.emit_completion(self.ant)
                elif cmd == "pointing_offsets":
                    source = kw.get('source')
                    closest_dist_deg = kw.get('closest_dist_deg')
                    number_of_points = kw.get('number_of_points')
                    duration_hours = kw.get('duration_hours')
                    slew = kw.get('slew', True)
                    park = kw.get('park', True)

                    if source is None or closest_dist_deg is None or number_of_points is None or duration_hours is None:
                        raise ValueError("Missing parameters for pointing_offsets command")

                    self.log.info("POINTING parameters: dist=%s, npts=%s, duration=%s", closest_dist_deg, number_of_points, duration_hours)

                    self.tracker.run_pointing_offsets(
                        ant=self.ant,
                        source=source,
                        closest_distance_deg=closest_dist_deg,
                        number_of_points=number_of_points,
                        duration_hours=duration_hours,
                        slew=slew,
                        park=park,
                        progress_callback=self._progress_callback_with_stop_check,
                        auto_cleanup=False,
                    )
                    self.completion_cb.emit_completion(self.ant)
                elif cmd == "stop":
                    # For immediate stop, use the request mechanism
                    self.request_stop()
                else:
                    self.log.warning("Unknown cmd %s", cmd)
            except InterruptedError as exc:
                self.log.info("Operation interrupted: %s", exc)
                self.completion_cb.emit_completion(self.ant)
            except Exception as exc:
                self.log.error("Tracker %s failed: %s", cmd, exc)
                self.completion_cb.emit_completion(self.ant)

class RecorderCmdThread(threading.Thread):
    """Thread handling start/stop/param-set commands for a shared Recorder."""

    def __init__(self, recorder: Recorder, prog_cb: UIProgressBridge, logger_: logging.Logger):
        super().__init__(daemon=True)
        self.recorder = recorder
        self.prog_cb = prog_cb
        self.log = logger_.getChild("rec_cmd")
        self.cmd_q: queue.Queue[Tuple[str, Dict]] = queue.Queue()
        self.running = True
        self.rec_worker: threading.Thread | None = None
        self._shutting_down = False  # Flag to prevent logging during shutdown

    def submit(self, cmd: str, **kw):
        self.cmd_q.put((cmd, kw))

    def _start_worker(self, fftshift: int, acclen: int):
        def _task():
            try:
                self.recorder.set_fftshift(fftshift)
                self.recorder.set_acclen(acclen)
                self.recorder.start_recording("observation", progress_callback=self.prog_cb)
            except Exception as exc:
                if not self._shutting_down:
                    self.log.error("Recording task error: %s", exc)
            finally:
                self.rec_worker = None
        self.rec_worker = threading.Thread(target=_task, daemon=True)
        self.rec_worker.start()

    def cleanup_recording(self):
        """Clean up recording resources and wait for worker thread to finish."""
        self._shutting_down = True
        
        # Stop recording if active
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        
        # Wait for worker thread to finish
        if self.rec_worker is not None:
            try:
                if self.rec_worker.is_alive():
                    self.rec_worker.join(timeout=5.0)
                    if self.rec_worker.is_alive():
                        if not self._shutting_down:
                            self.log.warning("Recording worker thread did not finish gracefully")
                    else:
                        if not self._shutting_down:
                            self.log.info("Recording worker thread finished gracefully")
                else:
                    if not self._shutting_down:
                        self.log.info("Recording worker thread already finished")
            except AttributeError:
                # Worker thread was set to None between our checks
                if not self._shutting_down:
                    self.log.info("Recording worker thread already finished")

    def run(self):
        """Main loop processing recorder commands."""
        while self.running:
            try:
                cmd, kw = self.cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if cmd == "start":
                    if self.recorder.is_recording:
                        if not self._shutting_down:
                            self.log.warning("Recorder already running")
                        continue
                    self._start_worker(kw["fftshift"], kw["acclen"])

                elif cmd == "stop":
                    if self.recorder.is_recording:
                        self.recorder.stop_recording()

                elif cmd == "set_params":
                    self.recorder.set_fftshift(kw["fftshift"])
                    self.recorder.set_acclen(kw["acclen"])

                else:
                    if not self._shutting_down:
                        self.log.warning("Unknown recorder cmd %s", cmd)

            except Exception as exc:
                if not self._shutting_down:
                    self.log.error("Recorder cmd %s failed: %s", cmd, exc)

class RecorderStatusThread(threading.Thread):
    """Periodically polls recorder for status and emits via bridge."""

    def __init__(self, recorder: Recorder, status_bridge: RecorderStatusBridge, data_bridge: DataBufferBridge, logger_: logging.Logger):
        super().__init__(daemon=True)
        self.recorder = recorder
        self.status_bridge = status_bridge
        self.data_bridge = data_bridge
        self.log = logger_.getChild("rec_status")
        self.running = True
        self._shutting_down = False  # Flag to prevent logging during shutdown
        self._data_requested = False  # Flag to indicate data buffer is requested

    def request_data_buffer(self):
        """Request the data buffer to be emitted immediately."""
        self._data_requested = True
        # Emit data immediately instead of waiting for next poll
        try:
            data_buffer = self.recorder.get_data_buffer()
            self.data_bridge.emit_data(data_buffer)
            self._data_requested = False
        except Exception as exc:
            if not self._shutting_down:
                self.log.error("Immediate data buffer request failed: %s", exc)

    def run(self):
        while self.running:
            try:
                fft_p0, fft_p1 = self.recorder.get_fftshift()
                acclen = self.recorder.get_acclen()
                ovf_p0, ovf_p1 = self.recorder.get_overflow_cnt()
                status = (fft_p0, fft_p1, acclen, ovf_p0, ovf_p1, self.recorder.is_recording)
                self.status_bridge.emit_status(status)
                
                # Emit data buffer if requested (fallback for any missed immediate requests)
                if self._data_requested:
                    data_buffer = self.recorder.get_data_buffer()
                    self.data_bridge.emit_data(data_buffer)
                    self._data_requested = False
                    
            except Exception as exc:
                if not self._shutting_down:
                    self.log.error("Status poll error: %s", exc)
            time.sleep(1.0)

class AntennaPositionThread(threading.Thread):
    """Monitors antenna positions via MQTT and emits via bridge."""

    def __init__(self, ant: str, position_bridge: AntennaPositionBridge, logger_: logging.Logger):
        super().__init__(daemon=True)
        self.ant = ant
        self.bridge = position_bridge
        self.log = logger_.getChild(f"pos_{ant}")
        self.running = True
        self.mqtt_client = None
        self.mqtt_connected = False
        self._shutting_down = False  # Flag to prevent logging during shutdown

    def _setup_mqtt(self):
        """Setup MQTT connection for position monitoring."""
        try:
            import paho.mqtt.client as mqtt
            from tracking.utils.config import config
            
            # Get broker IP based on antenna
            if self.ant == "N":
                broker_ip = config.mqtt.north_broker_ip
            elif self.ant == "S":
                broker_ip = config.mqtt.south_broker_ip
            else:
                raise ValueError(f"Invalid antenna '{self.ant}'. Must be 'N' or 'S'.")

            # Create MQTT client
            self.mqtt_client = mqtt.Client(client_id=f"pos_monitor_{self.ant}")
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message
            
            # Connect to broker
            self.mqtt_client.connect(broker_ip, config.mqtt.port, config.mqtt.connection_timeout)
            self.mqtt_client.loop_start()
            
            # Wait for connection
            timeout = config.mqtt.connection_wait_timeout
            start_time = time.time()
            while not self.mqtt_connected and (time.time() - start_time) < timeout:
                time.sleep(config.mqtt.poll_sleep_interval)
            
            if not self.mqtt_connected:
                raise ConnectionError(f"Failed to connect to MQTT broker {broker_ip}:{config.mqtt.port} within {timeout}s")
            
            # Subscribe to position topics
            self.mqtt_client.subscribe(config.mqtt.topic_az_status)
            self.mqtt_client.subscribe(config.mqtt.topic_el_status)
            
            if not self._shutting_down:
                self.log.info(f"MQTT position monitoring started for antenna {self.ant}")
            
        except Exception as exc:
            if not self._shutting_down:
                self.log.error(f"MQTT setup failed for antenna {self.ant}: {exc}")
            self.mqtt_client = None

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.mqtt_connected = True
            if not self._shutting_down:
                self.log.info(f"MQTT connected for antenna {self.ant}")
        else:
            if not self._shutting_down:
                self.log.error(f"MQTT connection failed for antenna {self.ant}: {rc}")

    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            import json
            from tracking.utils.config import config
            
            data = json.loads(msg.payload.decode())
            
            if not data or 'v' not in data:
                return
                
            v = data['v']
            
            # Extract position based on topic
            if msg.topic == config.mqtt.topic_az_status:
                if 'act_pos' in v:
                    az = float(v['act_pos'])
                    # Store azimuth and emit when we have both az and el
                    self._current_az = az
                    self._emit_position_if_ready()
            elif msg.topic == config.mqtt.topic_el_status:
                if 'act_pos' in v:
                    el = float(v['act_pos'])
                    # Store elevation and emit when we have both az and el
                    self._current_el = el
                    self._emit_position_if_ready()
                    
        except Exception as exc:
            if not self._shutting_down:
                self.log.error(f"Error processing MQTT message: {exc}")

    def _emit_position_if_ready(self):
        """Emit position if both azimuth and elevation are available."""
        if hasattr(self, '_current_az') and hasattr(self, '_current_el'):
            self.bridge.emit_position(self.ant, self._current_az, self._current_el)

    def cleanup_mqtt(self):
        """Cleanup MQTT connection."""
        self._shutting_down = True  # Prevent logging during cleanup
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                # Don't log during shutdown to avoid Qt object deletion errors
            except Exception as exc:
                # Don't log during shutdown to avoid Qt object deletion errors
                pass

    def run(self):
        """Main monitoring loop."""
        self._setup_mqtt()
        
        while self.running:
            time.sleep(1.0)  # Check for stop every second
            
        self.cleanup_mqtt()

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

@dataclass
class AntUI:
    ra: QDoubleSpinBox
    dec: QDoubleSpinBox
    az: QDoubleSpinBox
    el: QDoubleSpinBox
    duration: QDoubleSpinBox
    chk_no_slew: QCheckBox
    chk_no_park: QCheckBox
    status_label: QLabel
    prog: QProgressBar
    lbl_pos: QLabel
    box: QGroupBox
    # RASTA scan controls
    rasta_max_dist: QDoubleSpinBox
    rasta_step: QDoubleSpinBox
    rasta_position_angle: QDoubleSpinBox
    btn_rasta: QPushButton
    pointing_dist: QDoubleSpinBox
    pointing_npts: QSpinBox
    btn_point: QPushButton

class FTXPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controllers = {
            (Antenna.NORTH, 0): FTXController(Antenna.NORTH),
            (Antenna.NORTH, 1): FTXController(Antenna.NORTH),
            (Antenna.SOUTH, 0): FTXController(Antenna.SOUTH),
            (Antenna.SOUTH, 1): FTXController(Antenna.SOUTH),
        }
        self.init_ui()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_all)
        self.refresh_timer.start(3000)  # every 3 seconds
        self.refresh_all()

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("FTX Monitor & Control")
        title.setStyleSheet("font-weight: bold; font-size: 14pt;")
        layout.addWidget(title)
        self.blocks = {}
        for ant in [Antenna.NORTH, Antenna.SOUTH]:
            for pol in [0, 1]:
                block = self._make_block(ant, pol)
                layout.addWidget(block['group'])
                self.blocks[(ant, pol)] = block
        layout.addStretch()

    def _make_block(self, ant, pol):
        label_map = {
            (Antenna.NORTH, 0): "Y(N)",
            (Antenna.NORTH, 1): "X(N)",
            (Antenna.SOUTH, 0): "Y(S)",
            (Antenna.SOUTH, 1): "X(S)",
        }
        group_label = label_map.get((ant, pol), f"{ant.value} Pol {pol}")
        group = QGroupBox(group_label)
        v = QVBoxLayout(group)
        # Monitor labels
        labels = {}
        for key in ["attenuation", "rf_power", "laser_current"]:
            lbl = QLabel(f"{key.replace('_',' ').title()}: --")
            v.addWidget(lbl)
            labels[key] = lbl
        # Controls
        h = QHBoxLayout()
        atten = QDoubleSpinBox(); atten.setRange(0, 31.75); atten.setSingleStep(0.25); atten.setDecimals(2)
        laser = QDoubleSpinBox(); laser.setRange(0, 50); laser.setDecimals(2)
        lna = QCheckBox("LNA Enable"); lna.setChecked(True)
        btn = QPushButton("Set")
        h.addWidget(QLabel("Atten (dB)")); h.addWidget(atten)
        h.addWidget(QLabel("Laser (mA)")); h.addWidget(laser)
        h.addWidget(lna); h.addWidget(btn)
        v.addLayout(h)
        # Connect set button
        btn.clicked.connect(lambda _, a=ant, p=pol, at=atten, la=laser, ln=lna: self.set_ftx(a, p, at.value(), la.value(), ln.isChecked()))
        return {'group': group, 'labels': labels, 'atten': atten, 'laser': laser, 'lna': lna}

    def refresh_all(self):
        for (ant, pol), block in self.blocks.items():
            try:
                ctrl = self.controllers[(ant, pol)]
                data = ctrl.get_monitor_data(pol)
                block['labels']['attenuation'].setText(f"Attenuation: {data.attenuation_db:.2f} dB")
                block['labels']['rf_power'].setText(f"RF Power: {data.rf_power_dbm:.2f} dBm")
                block['labels']['laser_current'].setText(f"Laser Current: {data.ld_current_ma:.2f} mA")
                # Optionally prefill controls with current values
                block['atten'].setValue(data.attenuation_db)
                block['laser'].setValue(data.ld_current_ma)
                # block['lna'].setChecked(...) # If you can infer
            except Exception as e:
                for lbl in block['labels'].values():
                    lbl.setText(f"Error: {e}")

    def set_ftx(self, ant, pol, atten_db, laser_ma, lna_enabled):
        ctrl = self.controllers[(ant, pol)]
        try:
            ctrl.set_attenuation(pol, atten_db)
            ctrl.set_laser_current(pol, laser_ma)
            ctrl.set_lna_enabled(pol, lna_enabled)
        except Exception as e:
            # Optionally show error in UI
            pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antenna & Recorder Control")
        self.resize(900, 400)

        # Initialize components
        self._setup_logging()
        self._setup_recorder()
        self._setup_bridges()
        self._setup_threads()
        self._build_ui()

    def _setup_logging(self):
        """Setup logging bridges and handlers."""
        # Create log bridges
        self.bridge_trk_log = UILogBridge()
        self.bridge_rec_log = UILogBridge()
        self.bridge_trk_log.log_signal.connect(self._append_track_log)
        self.bridge_rec_log.log_signal.connect(self._append_rec_log)

        # Setup formatters and handlers
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        handler_trk = logging.StreamHandler()
        handler_trk.setFormatter(fmt)
        handler_trk.emit = lambda record: self.bridge_trk_log.write(handler_trk.format(record))

        handler_rec = logging.StreamHandler()
        handler_rec.setFormatter(fmt)
        handler_rec.emit = lambda record: self.bridge_rec_log.write(handler_rec.format(record))

        # Setup loggers
        logger_tracking = logging.getLogger("tracking")
        logger_tracking.setLevel(logging.INFO)
        logger_tracking.addHandler(handler_trk)
        logger_tracking.propagate = False

        logger_recording = logging.getLogger("recording")
        logger_recording.setLevel(logging.INFO)
        logger_recording.addHandler(handler_rec)
        logger_recording.propagate = False

        # App logger for misc messages
        self.root_log = logging.getLogger("app")
        self.root_log.setLevel(logging.INFO)
        self.root_log.addHandler(handler_trk)  # send to track pane

    def _setup_recorder(self):
        """Initialize recorder instance."""
        self.recorder = Recorder()

    def _setup_bridges(self):
        """Setup all Qt signal bridges."""
        # Progress bridges
        self.br_trk_n = UIProgressBridge()
        self.br_trk_s = UIProgressBridge()
        self.br_rec = UIProgressBridge()
        self.br_trk_n.progress_signal.connect(self._on_trk_progress)
        self.br_trk_s.progress_signal.connect(self._on_trk_progress)
        self.br_rec.progress_signal.connect(self._on_rec_progress)

        # Status bridges
        self.br_status = RecorderStatusBridge()
        self.br_status.status_signal.connect(self._update_rec_status)
        
        # Position bridges
        self.br_pos_n = AntennaPositionBridge()
        self.br_pos_s = AntennaPositionBridge()
        self.br_pos_n.position_signal.connect(self._update_antenna_position)
        self.br_pos_s.position_signal.connect(self._update_antenna_position)
        
        # Completion bridge
        self.br_completion = TrackerCompletionBridge()
        self.br_completion.completion_signal.connect(self._reset_tracker_status)
        
        # Data bridge
        self.br_data = DataBufferBridge()
        self.br_data.data_signal.connect(self._update_data_buffer)

    def _setup_threads(self):
        """Initialize and start all worker threads."""
        # Create threads
        self.thr_trk_n = TrackerThread("N", self.br_trk_n, self.br_completion, self.root_log)
        self.thr_trk_s = TrackerThread("S", self.br_trk_s, self.br_completion, self.root_log)
        self.thr_rec_cmd = RecorderCmdThread(self.recorder, self.br_rec, self.root_log)
        self.thr_rec_status = RecorderStatusThread(self.recorder, self.br_status, self.br_data, self.root_log)
        self.thr_pos_n = AntennaPositionThread("N", self.br_pos_n, self.root_log)
        self.thr_pos_s = AntennaPositionThread("S", self.br_pos_s, self.root_log)
        
        # Start all threads
        threads = [self.thr_trk_n, self.thr_trk_s, self.thr_rec_cmd, 
                  self.thr_rec_status, self.thr_pos_n, self.thr_pos_s]
        for thread in threads:
            thread.start()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)

        # Left column: Antenna panels (fixed width)
        v_left = QVBoxLayout()
        
        # Antenna panels
        self.ui_n = self._build_ant_panel("North", "N")
        self.ui_s = self._build_ant_panel("South", "S")
        v_left.addWidget(self.ui_n.box)
        v_left.addWidget(self.ui_s.box)
        v_left.addStretch()
        
        # Set left column to fixed size
        left_widget = QWidget()
        left_widget.setLayout(v_left)
        left_widget.setFixedWidth(450)  # Fixed width for left column
        h.addWidget(left_widget)

        # Middle column: Recording controls and coordinate calculations (fixed width)
        v_middle = QVBoxLayout()
        
        # Recording controls
        v_middle.addWidget(self._build_rec_panel())
        
        # Coordinate calculation controls
        v_middle.addWidget(self._build_coord_panel())
        v_middle.addStretch()
        
        # Set middle column to fixed size
        middle_widget = QWidget(); middle_widget.setLayout(v_middle); middle_widget.setFixedWidth(300)
        h.addWidget(middle_widget)

        # FTX column (new, fixed width)
        v_ftx = QVBoxLayout()
        self.ftx_panel = FTXPanel()
        v_ftx.addWidget(self.ftx_panel)
        ftx_widget = QWidget(); ftx_widget.setLayout(v_ftx); ftx_widget.setFixedWidth(350)
        h.addWidget(ftx_widget)

        # Right column: Plot and logs (expandable)
        v_right = QVBoxLayout()
        
        # Plot widget
        plot_box = QGroupBox("Data Plot")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self._create_plot_widget())
        v_right.addWidget(plot_box)
        
        # Log panels
        v_right.addLayout(self._build_log_panels())
        
        # Set right column to expand
        right_widget = QWidget()
        right_widget.setLayout(v_right)
        h.addWidget(right_widget, 1)  # Stretch factor of 1 for expansion

    # --------------- Antenna panel ------------------
    def _build_ant_panel(self, title: str, ant: str) -> AntUI:
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        
        # Status label row
        status_label = QLabel("Idle")
        status_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel(f"Status:"))
        status_row.addWidget(status_label)
        status_row.addStretch()
        v.addLayout(status_row)
        
        # Track/RASTA row (shared RA/Dec coordinates)
        ra = QDoubleSpinBox()
        ra.setRange(0, 24)
        ra.setDecimals(4)
        dec = QDoubleSpinBox()
        dec.setRange(-90, 90)
        dec.setDecimals(4)
        duration = QDoubleSpinBox()
        duration.setRange(0.0, 24.0)
        duration.setValue(1.0)
        duration.setDecimals(4)
        btn_track = QPushButton("TRACK")
        btn_track.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")  # Green button
        
        row = QHBoxLayout()
        row.addWidget(QLabel("RA(h)"))
        row.addWidget(ra)
        row.addWidget(QLabel("Dec(°)"))
        row.addWidget(dec)
        row.addWidget(QLabel("Length(h)"))
        row.addWidget(duration)
        row.addWidget(btn_track)
        v.addLayout(row)
        
        # RASTA scan controls - single row
        rasta_max_dist = QDoubleSpinBox()
        rasta_max_dist.setRange(0.1, 50.0)
        rasta_max_dist.setDecimals(2)
        rasta_max_dist.setValue(10.0)
        rasta_step = QDoubleSpinBox()
        rasta_step.setRange(0.1, 2.0)
        rasta_step.setDecimals(2)
        rasta_step.setValue(0.5)
        rasta_position_angle = QDoubleSpinBox()
        rasta_position_angle.setRange(0, 360)
        rasta_position_angle.setDecimals(1)
        rasta_position_angle.setValue(0.0)
        btn_rasta = QPushButton("RASTA")
        btn_rasta.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")  # Green button
        
        rasta_row = QHBoxLayout()
        rasta_row.addWidget(QLabel("Max(°)"))
        rasta_row.addWidget(rasta_max_dist)
        rasta_row.addWidget(QLabel("Step(°)"))
        rasta_row.addWidget(rasta_step)
        rasta_row.addWidget(QLabel("Angle(°)"))
        rasta_row.addWidget(rasta_position_angle)
        rasta_row.addWidget(btn_rasta)
        v.addLayout(rasta_row)

        # POINTING (offset pattern) controls – single row just below RASTA
        pointing_dist = QDoubleSpinBox(); pointing_dist.setRange(0.01, 20.0); pointing_dist.setDecimals(2); pointing_dist.setValue(0.5)
        pointing_npts = QSpinBox(); pointing_npts.setRange(5, 13); pointing_npts.setSingleStep(2); pointing_npts.setValue(5)
        btn_point = QPushButton("POINTING SCAN")
        btn_point.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")  # Green button


        pointing_row = QHBoxLayout()
        pointing_row.addWidget(QLabel("Dist(°)")); pointing_row.addWidget(pointing_dist)
        pointing_row.addWidget(QLabel("Npts")); pointing_row.addWidget(pointing_npts)
        pointing_row.addWidget(btn_point)
        v.addLayout(pointing_row)
        
        # Track/RASTA/POINTING options row (shared)
        chk_no_slew = QCheckBox("No Slew")
        chk_no_park = QCheckBox("No Park")
        
        row_options = QHBoxLayout()
        row_options.addWidget(chk_no_slew)
        row_options.addWidget(chk_no_park)
        row_options.addStretch()
        v.addLayout(row_options)
        
        # Slew row
        az = QDoubleSpinBox()
        az.setRange(0, 360)
        az.setDecimals(2)
        el = QDoubleSpinBox()
        el.setRange(0, 90)
        el.setDecimals(2)
        btn_slew = QPushButton("SLEW")
        btn_slew.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")  # Green button
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Az(°)"))
        row2.addWidget(az)
        row2.addWidget(QLabel("El(°)"))
        row2.addWidget(el)
        row2.addWidget(btn_slew)
        v.addLayout(row2)
        
        # Park/Stop
        btn_park = QPushButton("PARK")
        btn_park.setStyleSheet("QPushButton { background-color: orange; color: white; }")  # Orange button
        btn_stop = QPushButton("STOP")
        btn_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")  # Red button
        row3 = QHBoxLayout()
        row3.addWidget(btn_park)
        row3.addWidget(btn_stop)
        v.addLayout(row3)
        

        
        # Position display
        lbl_pos = QLabel("Position: --")
        lbl_pos.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 2px; border: 1px solid #ccc; }")
        v.addWidget(lbl_pos)
        
        # Progress
        prog = QProgressBar()
        prog.setRange(0, 100)
        v.addWidget(prog)
        
        # Connect signals
        btn_track.clicked.connect(lambda: self._cmd_track(ant, ra.value(), dec.value(), duration.value(), chk_no_slew.isChecked(), chk_no_park.isChecked()))
        btn_slew.clicked.connect(lambda: self._cmd_slew(ant, az.value(), el.value()))
        btn_park.clicked.connect(lambda: self._cmd_park(ant))
        btn_stop.clicked.connect(lambda: self._cmd_stop(ant))
        
        btn_rasta.clicked.connect(lambda: self._cmd_rasta_scan(ant, ra.value(), dec.value(), rasta_max_dist.value(), rasta_step.value(), rasta_position_angle.value(), duration.value(), chk_no_slew.isChecked(), chk_no_park.isChecked()))
        btn_point.clicked.connect(lambda: self._cmd_pointing_offsets(ant, ra.value(), dec.value(), pointing_dist.value(), pointing_npts.value(), duration.value(), chk_no_slew.isChecked(), chk_no_park.isChecked()))
        
        return AntUI(ra, dec, az, el, duration, chk_no_slew, chk_no_park, status_label, prog, lbl_pos, box, rasta_max_dist, rasta_step, rasta_position_angle, btn_rasta, pointing_dist, pointing_npts, btn_point)

    # --------------- Recorder panel ------------------
    def _build_rec_panel(self):
        box = QGroupBox("Recorder")
        h = QVBoxLayout(box)
        
        # Status label row
        self.recorder_status_label = QLabel("Idle")
        self.recorder_status_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        status_row.addWidget(self.recorder_status_label)
        status_row.addStretch()
        h.addLayout(status_row)
        
        # Top row controls
        row = QHBoxLayout()
        self.sp_fft = QSpinBox()
        self.sp_fft.setRange(0, 4095)
        self.sp_fft.setValue(1704)
        self.sp_acc = QSpinBox()
        self.sp_acc.setRange(1, 1_000_000)
        self.sp_acc.setValue(131072)
        
        row.addWidget(QLabel("fftshift"))
        row.addWidget(self.sp_fft)
        row.addWidget(QLabel("acclen"))
        row.addWidget(self.sp_acc)
        h.addLayout(row)
        
        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_set = QPushButton("Set Params")
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_set)
        h.addLayout(btn_row)
        
        # Status row
        self.lbl_fft = QLabel("FFT: -, -")
        self.lbl_acc = QLabel("AccLen: -")
        self.lbl_ovf = QLabel("Overflow: -, -")
        
        for lab in (self.lbl_fft, self.lbl_acc, self.lbl_ovf):
            h.addWidget(lab)
        
        # Connect signals
        self.btn_start.clicked.connect(self._toggle_record)
        self.btn_set.clicked.connect(self._apply_params)
        
        return box

    # --------------- Coordinate calculation panel ------------------
    def _build_coord_panel(self):
        box = QGroupBox("Coordinate Calculations")
        layout = QVBoxLayout(box)
        
        # Input row
        input_row = QHBoxLayout()
        self.coord_ra = QDoubleSpinBox()
        self.coord_ra.setRange(0, 24)
        self.coord_ra.setDecimals(4)
        self.coord_ra.setValue(12.0)
        self.coord_dec = QDoubleSpinBox()
        self.coord_dec.setRange(-90, 90)
        self.coord_dec.setDecimals(4)
        self.coord_dec.setValue(45.0)
        
        input_row.addWidget(QLabel("RA(h):"))
        input_row.addWidget(self.coord_ra)
        input_row.addWidget(QLabel("Dec(°):"))
        input_row.addWidget(self.coord_dec)
        layout.addLayout(input_row)
        
        # Buttons row
        button_row = QHBoxLayout()
        self.btn_get_offsets = QPushButton("Get Offsets")
        self.btn_get_azel = QPushButton("Get Az/El")
        self.btn_clear_results = QPushButton("Clear Results")
        
        button_row.addWidget(self.btn_get_offsets)
        button_row.addWidget(self.btn_get_azel)
        button_row.addWidget(self.btn_clear_results)
        layout.addLayout(button_row)
        
        # Offset parameters row
        offset_row = QHBoxLayout()
        self.offset_amplitude = QDoubleSpinBox()
        self.offset_amplitude.setRange(0.1, 10.0)
        self.offset_amplitude.setDecimals(2)
        self.offset_amplitude.setValue(0.5)
        self.offset_count = QSpinBox()
        self.offset_count.setRange(5, 13)
        self.offset_count.setValue(5)
        
        offset_row.addWidget(QLabel("Offset (deg):"))
        offset_row.addWidget(self.offset_amplitude)
        offset_row.addWidget(QLabel("Count:"))
        offset_row.addWidget(self.offset_count)
        layout.addLayout(offset_row)
        
        # Results display
        self.txt_coord_results = QTextEdit()
        self.txt_coord_results.setReadOnly(True)
        self.txt_coord_results.setMaximumHeight(150)
        layout.addWidget(self.txt_coord_results)
        
        # Connect signals
        self.btn_get_offsets.clicked.connect(self._calculate_offsets)
        self.btn_get_azel.clicked.connect(self._calculate_azel)
        self.btn_clear_results.clicked.connect(self._clear_coord_results)
        
        return box

    def _create_plot_widget(self):
        """Create the matplotlib plot widget with refresh button."""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Add refresh button
        self.btn_refresh = QPushButton("Refresh Plot")
        self.btn_refresh.clicked.connect(self._refresh_plot)
        layout.addWidget(self.btn_refresh)
        
        # Create matplotlib figure
        fig = Figure(figsize=(8, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title("Data Plot")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        
        # Store canvas reference for later use
        self.plot_widget = canvas
        layout.addWidget(canvas)
        
        return container

    # --------------- Log panels ------------------
    def _build_log_panels(self):
        layout = QVBoxLayout()

        # Tracking log
        box_trk = QGroupBox("Tracking Log")
        v1 = QVBoxLayout(box_trk)
        self.txt_log_trk = QTextEdit()
        self.txt_log_trk.setReadOnly(True)
        v1.addWidget(self.txt_log_trk)
        layout.addWidget(box_trk)

        # Recording log
        box_rec = QGroupBox("Recording Log")
        v2 = QVBoxLayout(box_rec)
        self.txt_log_rec = QTextEdit()
        self.txt_log_rec.setReadOnly(True)
        v2.addWidget(self.txt_log_rec)
        layout.addWidget(box_rec)

        return layout

    # ------------------------------------------------------------------
    # Command helpers
    def _cmd_track(self, ant: str, ra: float, dec: float, duration: float, no_slew: bool = False, no_park: bool = False):
        src = Source(ra_hrs=ra, dec_deg=dec)
        # Convert no_slew/no_park to slew/park (invert the logic)
        slew = not no_slew
        park = not no_park
        self._target_thr(ant).submit("track", source=src, duration_hours=duration, slew=slew, park=park)
        # Update status immediately
        ui = self.ui_n if ant == "N" else self.ui_s
        ui.status_label.setText(f"Tracking {ra:.2f}h, {dec:.2f}°")

    def _cmd_slew(self, ant: str, az: float, el: float):
        self._target_thr(ant).submit("slew", az=az, el=el)
        # Update status immediately
        ui = self.ui_n if ant == "N" else self.ui_s
        ui.status_label.setText(f"Slewing {az:.2f}°, {el:.2f}°")

    def _cmd_park(self, ant: str):
        self._target_thr(ant).submit("park")
        # Update status immediately
        ui = self.ui_n if ant == "N" else self.ui_s
        ui.status_label.setText("Parking")

    def _cmd_stop(self, ant: str):
        self._target_thr(ant).request_stop()
        # Update status immediately
        ui = self.ui_n if ant == "N" else self.ui_s
        ui.status_label.setText("Stopping...")

    def _cmd_rasta_scan(self, ant: str, ra: float, dec: float, max_dist: float, step: float, position_angle: float, duration: float, no_slew: bool = False, no_park: bool = False):
        src = Source(ra_hrs=ra, dec_deg=dec)
        
        # Convert no_slew/no_park to slew/park (invert the logic)
        slew = not no_slew
        park = not no_park
        
        # Debug logging to see what values are being passed
        print(f"DEBUG: RASTA scan called with duration={duration} hours")
        print(f"DEBUG: All parameters: ra={ra}, dec={dec}, max_dist={max_dist}, step={step}, position_angle={position_angle}, duration={duration}, slew={slew}, park={park}")
        
        self._target_thr(ant).submit("rasta_scan", source=src, max_dist_deg=max_dist, step_deg=step, position_angle_deg=position_angle, duration_hours=duration, slew=slew, park=park)
        # Update status immediately
        ui = self.ui_n if ant == "N" else self.ui_s
        ui.status_label.setText(f"RASTA Scanning {ra:.2f}h, {dec:.2f}°")

    def _cmd_pointing_offsets(self, ant: str, ra: float, dec: float, dist: float, npts: int, duration: float, no_slew: bool = False, no_park: bool = False):
        src = Source(ra_hrs=ra, dec_deg=dec)
        slew = not no_slew; park = not no_park
        self._target_thr(ant).submit("pointing_offsets", source=src, closest_dist_deg=dist, number_of_points=npts, duration_hours=duration, slew=slew, park=park)
        ui = self.ui_n if ant == "N" else self.ui_s
        ui.status_label.setText(f"POINTING {ra:.2f}h, {dec:.2f}°")

    def _target_thr(self, ant):
        return self.thr_trk_n if ant == "N" else self.thr_trk_s

    def _toggle_record(self):
        if self.btn_start.text() == "Start":
            self.thr_rec_cmd.submit("start", fftshift=self.sp_fft.value(), acclen=self.sp_acc.value())
            self.btn_start.setText("Stop")
            self.recorder_status_label.setText("Recording")
        else:
            self.thr_rec_cmd.submit("stop")
            self.btn_start.setText("Start")
            self.recorder_status_label.setText("Stopping...")

    def _apply_params(self):
        self.thr_rec_cmd.submit("set_params", fftshift=self.sp_fft.value(), acclen=self.sp_acc.value())
        self.recorder_status_label.setText("Setting parameters...")

    # ------------------------------------------------------------------
    # Slots
    def _on_trk_progress(self, info: TrkProgInfo):
        ui = self.ui_n if info.antenna == "N" else self.ui_s
        ui.prog.setValue(int(info.percent_complete))
        
        # Update status label based on operation type
        # Only set "Idle" when operation is actually complete (not just progress = 100%)
        if hasattr(info, 'operation_type'):
            if info.operation_type == 'track':
                ui.status_label.setText(f"Tracking... ({info.percent_complete:.1f}%)")
            elif info.operation_type == 'slew':
                ui.status_label.setText(f"Slewing... ({info.percent_complete:.1f}%)")
            elif info.operation_type == 'park':
                ui.status_label.setText("Parking...")
            elif info.operation_type == 'rasta_scan':
                ui.status_label.setText(f"RASTA Scanning... ({info.percent_complete:.1f}%)")
        else:
            # Fallback based on progress info
            if info.percent_complete < 100:
                ui.status_label.setText("Operating...")
            # Don't automatically set to "Idle" here - let the operation complete naturally
        
        # Check if tracking operation is complete and stop recording if active
        if info.is_complete and info.operation_type in ['track', 'rasta_scan']:
            if self.recorder.is_recording:
                self.root_log.info(f"Tracking operation completed for antenna {info.antenna}, stopping recording")
                self.thr_rec_cmd.submit("stop")
                # Update UI to reflect that recording is being stopped
                self.btn_start.setText("Start")
                self.recorder_status_label.setText("Stopping...")

    def _on_rec_progress(self, info: RecProgInfo):
        pct = int(info.percent_complete)
        self.lbl_status.setText(f"Recording ({pct}%)" if not info.is_complete else "Idle")
        
        # Update recorder status label
        if info.is_complete:
            self.recorder_status_label.setText("Idle")
        else:
            self.recorder_status_label.setText(f"Recording ({pct}%)")

    def _update_rec_status(self, status: tuple):
        fft0, fft1, acc, ovf0, ovf1, rec = status
        self.lbl_fft.setText(f"FFT: {fft0}, {fft1}")
        self.lbl_acc.setText(f"AccLen: {acc}")
        self.lbl_ovf.setText(f"Overflow: {ovf0}, {ovf1}")
        
        if not rec and self.btn_start.text() == "Stop":
            self.btn_start.setText("Start")
            # Also reset the recorder status label when recording stops
            self.recorder_status_label.setText("Idle")

    def _update_antenna_position(self, antenna: str, az: float, el: float):
        """Update antenna position display."""
        if antenna == "N":
            self.ui_n.lbl_pos.setText(f"Position: Az {az:.2f}°, El {el:.2f}°")
        elif antenna == "S":
            self.ui_s.lbl_pos.setText(f"Position: Az {az:.2f}°, El {el:.2f}°")

    def _update_data_buffer(self, data: list):
        """Update the data buffer display in the GUI."""
        # Store the data for plotting
        self._current_data_buffer = data
        print(f"Received data buffer with {len(data)} samples")
        
        # If we just requested data (refresh button was clicked), plot it immediately
        if hasattr(self, '_refresh_requested') and self._refresh_requested:
            print(f"Plotting data from refresh request with {len(data)} samples")
            self._plot_data(data)
            self._refresh_requested = False

    def _refresh_plot(self):
        """Refresh the plot with current data buffer."""
        print(f"Refresh button clicked - requesting immediate data buffer")
        # Always request fresh data buffer from recorder status thread
        self._refresh_requested = True  # Mark that we're expecting data
        self.thr_rec_status.request_data_buffer()

    def _plot_data(self, data_buffer: list):
        """
        Plot the data from the buffer.
        
        Args:
            data_buffer: List of data arrays to plot
        """
        # Clear the current plot
        ax = self.plot_widget.figure.axes[0]
        ax.clear()
        
        if not data_buffer:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            self.plot_widget.draw()
            return
        
        try:
            # Process data similar to the provided code

            specs = []
            tims = []
            nt = 0
            
            # Process each time sample in the buffer
            for data_array in data_buffer:
                d = data_array
                axi1 = d.shape[0]
                specs.append(d.reshape((int(axi1/10), 10, 8195))[:,:,3:])
                tims.append(d.reshape((int(axi1/10), 10, 8195))[:,:,0])
                nt += int(axi1/10)
            
            if nt == 0:
                ax.text(0.5, 0.5, 'No valid data samples', ha='center', va='center', transform=ax.transAxes)
                self.plot_widget.draw()
                return
            
            # Create arrays for spectra and times
            spectra = np.zeros((nt, 10, 8192), dtype=np.complex64)
            times = np.zeros(nt, dtype=np.float64)
            myt = 0
            
            for i in range(len(data_buffer)):
                d = specs[i]
                tt = d.shape[0]
                spectra[myt:myt+tt, :, :] = d
                times[myt:myt+tt] = np.real(tims[i][:, 0])
                myt += tt
            
            # Extract autocorrs and crosscorrs
            autocorrs = np.real(spectra[:, [0, 4, 7, 9], :]).astype(np.float64)  # XX_1, YY_1, XX_2, YY_2
            crosscorrs = spectra[:, [2, 3, 5, 6], :]  # XX_12, XY_12, YX_12, YY_12
            
            
            # Apply FFT shifts
            XX_12 = np.fft.fftshift(crosscorrs[:, 0, :], axes=1)
            XY_12 = np.fft.fftshift(crosscorrs[:, 1, :], axes=1)
            YX_12 = np.fft.fftshift(crosscorrs[:, 2, :], axes=1)
            YY_12 = np.fft.fftshift(crosscorrs[:, 3, :], axes=1)
            
            XX_1 = np.fft.fftshift(autocorrs[:, 0, :], axes=1)
            YY_1 = np.fft.fftshift(autocorrs[:, 1, :], axes=1)
            
            XX_2 = np.fft.fftshift(autocorrs[:, 2, :], axes=1)
            YY_2 = np.fft.fftshift(autocorrs[:, 3, :], axes=1)

            myf = np.linspace(0, 2400, 8192)
            
            # Plot autocorrs
            tmin = 0
            tmax = autocorrs.shape[0] - 1
            
            # Check if we have valid data to plot
            if tmax < tmin or autocorrs.shape[0] == 0:
                ax.text(0.5, 0.5, 'No valid autocorrelation data', ha='center', va='center', transform=ax.transAxes)
                self.plot_widget.draw()
                return
            
            labels = ["XX(S)", "YY(S)", "XX(N)", "YY(N)"]
            colors = ['blue', 'red', 'green', 'orange']
            
            plotted_lines = []  # Track if we actually plot anything
            
            for i in range(4):  # Plot XX2 and YY2
                # Check if the slice has data
                if tmax >= tmin and autocorrs[tmin:tmax, i, :].size > 0:
                    y = np.mean(autocorrs[tmin:tmax, i, :], axis=0)
                    # Handle zero values to avoid log10(0) warning
                    y = np.where(y > 0, y, 1e-10)  # Replace zeros with small positive value
                    y = 10. * np.log10(y)
                    line = ax.plot(myf, np.fft.fftshift(y), '-', label=labels[i], color=colors[i], linewidth=1, alpha = 0.5)
                    plotted_lines.extend(line)
            
            # Only add legend if we actually plotted something
            if plotted_lines:
                ax.legend(fontsize=10)
            
            # Add frequency markers
            freq_markers = [1420.4, 1612.0, 1665.0, 1667.0, 1720.0]
            for freq in freq_markers:
                ax.axvline(x=freq, color='black', linestyle='--', alpha=0.3)
            
            ax.set_xlabel("Frequency (MHz)", fontsize=12)
            ax.set_ylabel("Amplitude (arb. dB)", fontsize=12)
            # ax.set_xlim(1600, 1750)
            # ax.set_ylim(-60, -40)
            ax.set_title(f"Autocorrs - {len(data_buffer)} time integrations", fontsize=12)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            # If there's an error in processing, show a simple plot
            ax.text(0.5, 0.5, f'Error processing data: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        
        # Redraw the plot
        self.plot_widget.draw()

    def _append_track_log(self, txt: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log_trk.append(f"[{ts}] {txt}")

    def _append_rec_log(self, txt: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log_rec.append(f"[{ts}] {txt}")

    # ------------------------------------------------------------------
    # Coordinate calculation methods
    # ------------------------------------------------------------------
    def _calculate_offsets(self):
        """Calculate RA/Dec offsets and display results."""
        try:
            ra = self.coord_ra.value()
            dec = self.coord_dec.value()
            amp_offset = self.offset_amplitude.value()
            num_offsets = self.offset_count.value()
            
            ra_list, dec_list = get_targets_offsets(amp_offset, num_offsets, ra, dec)
            
            # Format results
            result_text = f"RA/Dec Offsets (amplitude: {amp_offset}°, count: {num_offsets}):\n"
            result_text += f"Center: RA={ra:.4f}h, Dec={dec:.4f}°\n\n"
            
            for i, (ra_val, dec_val) in enumerate(zip(ra_list, dec_list)):
                result_text += f"Point {i+1}: RA={ra_val:.4f}h, Dec={dec_val:.4f}°\n"
            
            self.txt_coord_results.setText(result_text)
            
        except Exception as e:
            self.txt_coord_results.setText(f"Error calculating offsets: {str(e)}")

    def _calculate_azel(self):
        """Calculate Az/El for the given RA/Dec and display results."""
        try:
            ra = self.coord_ra.value()
            dec = self.coord_dec.value()
            
            az, el = radec_to_azel(ra, dec)
            
            # Format results
            result_text = f"Azimuth/Elevation for RA={ra:.4f}h, Dec={dec:.4f}°:\n"
            result_text += f"Azimuth: {az:.2f}°\n"
            result_text += f"Elevation: {el:.2f}°\n\n"
            result_text += get_ovro_location_info()
            
            self.txt_coord_results.setText(result_text)
            
        except Exception as e:
            self.txt_coord_results.setText(f"Error calculating Az/El: {str(e)}")

    def _clear_coord_results(self):
        """Clear the coordinate results display."""
        self.txt_coord_results.clear()

    def _reset_tracker_status(self, ant: str):
        """Reset tracker status to Idle when operation completes and stop recording if active."""
        ui = self.ui_n if ant == "N" else self.ui_s
        ui.status_label.setText("Idle")
        
        # Stop recording if it's currently active
        if self.recorder.is_recording:
            self.root_log.info(f"Tracking operation completed for antenna {ant}, stopping recording")
            self.thr_rec_cmd.submit("stop")
            # Update UI to reflect that recording is being stopped
            self.btn_start.setText("Start")
            self.recorder_status_label.setText("Stopping...")

    # ------------------------------------------------------------------
    def closeEvent(self, event):  # pylint: disable=invalid-name
        """Cleanup resources when closing the application."""
        # Set shutdown flags to prevent logging during cleanup
        threads = [self.thr_trk_n, self.thr_trk_s, self.thr_rec_cmd, 
                  self.thr_rec_status, self.thr_pos_n, self.thr_pos_s]
        
        for thread in threads:
            if hasattr(thread, '_shutting_down'):
                thread._shutting_down = True
            thread.running = False
            
        # Give threads a moment to stop gracefully
        time.sleep(0.1)
        
        # Clean up recording resources and wait for worker thread to finish
        self.thr_rec_cmd.cleanup_recording()
        
        # Cleanup recorder (this will clear buffers and close device)
        try:
            self.recorder.cleanup()
        except Exception:
            # Don't log during shutdown to avoid Qt object deletion errors
            pass
        
        # Cleanup resources
        for thread in threads:
            if isinstance(thread, TrackerThread):
                thread.cleanup_tracker()
            elif isinstance(thread, AntennaPositionThread):
                thread.cleanup_mqtt()
        
        event.accept()

# -----------------------------------------------------------------------------
# Main application entry point
# -----------------------------------------------------------------------------

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 