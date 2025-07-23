from __future__ import annotations
import sys
import time
import logging
from dataclasses import dataclass
from datetime import datetime

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
    QSplitter,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from recording import Recorder
from tracking import Source
from tracking.utils.progress import ProgressInfo as TrkProgInfo
from recording.utils.progress import ProgressInfo as RecProgInfo
from tracking.utils.coordinate_utils import get_targets_offsets, radec_to_azel, get_ovro_location_info

from core.bridges import (
    UILogBridge,
    UIProgressBridge,
    RecorderStatusBridge,
    AntennaPositionBridge,
    TrackerCompletionBridge,
    DataBufferBridge,
    TrackingEventsBridge,
)
from core.threads import (
    TrackerThread,
    RecorderCmdThread,
    RecorderStatusThread,
    AntennaPositionThread,
)
from gui.observation_panel import ObservationPanel, Observation
from gui.ftx_panel import FTXPanel
from core.bridges import (
    TrackerThreadBridge, RecorderCmdThreadBridge, RecorderStatusThreadBridge, AntennaPositionThreadBridge
)
from core.threads import TrackerThread, RecorderCmdThread, RecorderStatusThread, AntennaPositionThread


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
    # RASTA and POINTING scan controls
    rasta_max_dist: QDoubleSpinBox
    rasta_step: QDoubleSpinBox
    rasta_position_angle: QDoubleSpinBox
    btn_rasta: QPushButton
    pointing_dist: QDoubleSpinBox
    pointing_npts: QSpinBox
    btn_point: QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antenna & Recorder Control")
        self.resize(900, 400)

        # Track if we started the recorder for the current observation
        self._recorder_started_for_obs = False
        self._ant_tracking = {"N": False, "S": False}  # Track tracking state of both antennas

        # Initialize components
        self._setup_logging()
        self._setup_recorder()
        self._setup_bridges()
        self._setup_threads()
        self._build_ui()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _setup_logging(self):
        """Setup logging bridges and handlers."""
        # Create log bridges
        self.bridge_trk_log = UILogBridge()
        self.bridge_rec_log = UILogBridge()
        # Do NOT connect signals here; connect after UI is built
        # self.bridge_trk_log.log_signal.connect(self._append_track_log)
        # self.bridge_rec_log.log_signal.connect(self._append_rec_log)

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
        """Setup all Qt signal bridges (unified)."""
        # Tracker bridges
        self.trk_n_bridge = TrackerThreadBridge()
        self.trk_s_bridge = TrackerThreadBridge()
        self.trk_n_bridge.progress_signal.connect(self._on_trk_progress)
        self.trk_s_bridge.progress_signal.connect(self._on_trk_progress)
        self.trk_n_bridge.completion_signal.connect(self._reset_tracker_status)
        self.trk_s_bridge.completion_signal.connect(self._reset_tracker_status)
        self.trk_n_bridge.tracking_event_signal.connect(self._on_tracking_event)
        self.trk_s_bridge.tracking_event_signal.connect(self._on_tracking_event)
        self.trk_n_bridge.log_signal.connect(self._append_track_log)
        self.trk_s_bridge.log_signal.connect(self._append_track_log)

        # Recorder command bridge
        self.rec_cmd_bridge = RecorderCmdThreadBridge()
        self.rec_cmd_bridge.progress_signal.connect(self._on_rec_progress)
        self.rec_cmd_bridge.log_signal.connect(self._append_rec_log)

        # Recorder status bridge
        self.rec_status_bridge = RecorderStatusThreadBridge()
        self.rec_status_bridge.status_signal.connect(self._update_rec_status)
        self.rec_status_bridge.data_signal.connect(self._update_data_buffer)
        self.rec_status_bridge.log_signal.connect(self._append_rec_log)

        # Antenna position bridges
        self.pos_n_bridge = AntennaPositionThreadBridge()
        self.pos_s_bridge = AntennaPositionThreadBridge()
        self.pos_n_bridge.position_and_target_signal.connect(self._update_antenna_position_and_target)
        self.pos_s_bridge.position_and_target_signal.connect(self._update_antenna_position_and_target)
        self.pos_n_bridge.log_signal.connect(self._append_track_log)
        self.pos_s_bridge.log_signal.connect(self._append_track_log)

    def _setup_threads(self):
        """Initialize and start all worker threads (unified bridges)."""
        
        self.thr_trk_n = TrackerThread("N", self.trk_n_bridge, self.root_log)
        self.thr_trk_s = TrackerThread("S", self.trk_s_bridge, self.root_log)
        self.thr_rec_cmd = RecorderCmdThread(self.recorder, self.rec_cmd_bridge, self.root_log)
        self.thr_rec_status = RecorderStatusThread(self.recorder, self.rec_status_bridge, self.root_log)
        self.thr_pos_n = AntennaPositionThread("N", self.pos_n_bridge, self.root_log)
        self.thr_pos_s = AntennaPositionThread("S", self.pos_s_bridge, self.root_log)
        threads = [self.thr_trk_n, self.thr_trk_s, self.thr_rec_cmd, 
                  self.thr_rec_status, self.thr_pos_n, self.thr_pos_s]
        for thread in threads:
            thread.start()

    # ------------------------------------------------------------------
    # UI building
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QWidget {
                background-color: #f0f0f0;
                color: #333;
                font-family: Arial, sans-serif;
            }
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 10pt;
            }
            QPushButton {
                font-size: 10pt;
                padding: 8px;
                border-radius: 4px;
                background-color: #007bff;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
            QRadioButton, QCheckBox {
                font-size: 10pt;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 9pt;
            }
            QDoubleSpinBox, QSpinBox {
                padding: 4px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QProgressBar {
                text-align: center;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #007bff;
            }
            QSplitter::handle {
                background-color: #ccc;
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)

        # Container for left and middle columns
        left_container = QWidget()
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left column: Observation panel (fixed width)
        self.observation_panel = ObservationPanel()
        self.observation_panel.setFixedWidth(400)  # Fixed width for observation panel
        left_layout.addWidget(self.observation_panel)
        
        # Middle column: Recording controls, antenna status, and coordinate calculations (fixed width)
        v_middle = QVBoxLayout()
        v_middle.addWidget(self._build_rec_panel())
        v_middle.addWidget(self._build_antenna_status_panel())
        v_middle.addWidget(self._build_coord_panel())
        v_middle.addStretch()
        middle_widget = QWidget()
        middle_widget.setLayout(v_middle)
        middle_widget.setFixedWidth(300)  # Fixed width for middle column
        left_layout.addWidget(middle_widget)

        # FTX column (fixed width)
        self.ftx_panel = FTXPanel()
        self.ftx_panel.setFixedWidth(200)  # Fixed width for FTX panel
        left_layout.addWidget(self.ftx_panel)
        
        # Set the entire left container to fixed width (sum of all columns)
        left_container.setFixedWidth(400 + 300 + 300)  # 1000 total
        
        splitter.addWidget(left_container)

        # Right column: Plot and logs
        right_widget = QWidget()
        v_right = QVBoxLayout(right_widget)
        plot_box = QGroupBox("Data Plot")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self._create_plot_widget())
        v_right.addWidget(plot_box)
        v_right.addLayout(self._build_log_panels())
        splitter.addWidget(right_widget)

        # Set initial splitter sizes and configure splitter behavior
        splitter.setSizes([900, 400])  # Left side is now 900 pixels (fixed width), right side 400 pixels
        
        # Prevent the left side from being resizable by setting stretch factors
        # This ensures only the right side can resize when the window is resized
        splitter.setStretchFactor(0, 0)  # Left side (index 0) has stretch factor 0 (no stretch)
        splitter.setStretchFactor(1, 1)  # Right side (index 1) has stretch factor 1 (can stretch)
        
        h.addWidget(splitter)
        
        # Connect observation panel signals to main window slots
        self.observation_panel.observation_requested.connect(self._on_observation_requested)
        self.observation_panel.command_requested.connect(self._on_command_requested)
        self.observation_panel.log_message_requested.connect(self._append_track_log)
        
        # Connect completion signal to queue logic after panel is created
        if hasattr(self, 'trk_n_bridge'): # Changed from br_completion to trk_n_bridge
            self.trk_n_bridge.completion_signal.connect(self.observation_panel._on_tracker_completion)
            print("DEBUG: Connected trk_n_bridge.completion_signal to observation_panel._on_tracker_completion")

        # Now connect log bridge signals, after log widgets exist
        self.bridge_trk_log.log_signal.connect(self._append_track_log)
        self.bridge_rec_log.log_signal.connect(self._append_rec_log)

    def _build_rec_panel(self):
        box = QGroupBox("Recorder")
        h = QVBoxLayout(box)
        
        # Status label row
        self.recorder_status_label = QLabel("Idle")
        self.recorder_status_label.setStyleSheet("color: #007bff; font-weight: bold;")
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        status_row.addWidget(self.recorder_status_label)
        status_row.addStretch()
        h.addLayout(status_row)
        
        # Top row controls
        row = QHBoxLayout()
        self.sp_fft = QSpinBox()
        self.sp_fft.setRange(0, 4095)
        self.sp_fft.setValue(1904)
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
        self.btn_start.setStyleSheet("background-color: #28a745;")
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
        self.btn_get_offsets.setStyleSheet("background-color: #6c757d;")
        self.btn_get_azel = QPushButton("Get Az/El")
        self.btn_get_azel.setStyleSheet("background-color: #6c757d;")
        self.btn_clear_results = QPushButton("Clear")
        self.btn_clear_results.setStyleSheet("background-color: #6c757d;")
        
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

    def _build_antenna_status_panel(self):
        box = QGroupBox("Antenna Status")
        layout = QVBoxLayout(box)
        
        # North antenna status
        north_status = QVBoxLayout()
        
        # Status line
        north_status_line = QHBoxLayout()
        self.lbl_north_status = QLabel("North: Idle")
        self.lbl_north_status.setStyleSheet("color: #007bff; font-weight: bold;")
        north_status_line.addWidget(self.lbl_north_status)
        north_status_line.addStretch()
        north_status.addLayout(north_status_line)
        
        # Position and Target stacked vertically
        self.lbl_north_pos = QLabel("Position: --")
        self.lbl_north_pos.setStyleSheet("background-color: #e9ecef; padding: 4px; border: 1px solid #ccc; border-radius: 3px;")
        north_status.addWidget(self.lbl_north_pos)
        
        self.lbl_north_target = QLabel("Target: --")
        self.lbl_north_target.setStyleSheet("background-color: #f0f9ff; padding: 4px; border: 1px solid #ccc; border-radius: 3px;")
        north_status.addWidget(self.lbl_north_target)
        
        layout.addLayout(north_status)
        
        # North progress bar
        self.prog_north = QProgressBar()
        self.prog_north.setRange(0, 100)
        self.prog_north.setValue(0)
        layout.addWidget(self.prog_north)
        
        # South antenna status
        south_status = QVBoxLayout()
        
        # Status line
        south_status_line = QHBoxLayout()
        self.lbl_south_status = QLabel("South: Idle")
        self.lbl_south_status.setStyleSheet("color: #007bff; font-weight: bold;")
        south_status_line.addWidget(self.lbl_south_status)
        south_status_line.addStretch()
        south_status.addLayout(south_status_line)
        
        # Position and Target stacked vertically
        self.lbl_south_pos = QLabel("Position: --")
        self.lbl_south_pos.setStyleSheet("background-color: #e9ecef; padding: 4px; border: 1px solid #ccc; border-radius: 3px;")
        south_status.addWidget(self.lbl_south_pos)
        
        self.lbl_south_target = QLabel("Target: --")
        self.lbl_south_target.setStyleSheet("background-color: #f0f9ff; padding: 4px; border: 1px solid #ccc; border-radius: 3px;")
        south_status.addWidget(self.lbl_south_target)
        
        layout.addLayout(south_status)
        
        # South progress bar
        self.prog_south = QProgressBar()
        self.prog_south.setRange(0, 100)
        self.prog_south.setValue(0)
        layout.addWidget(self.prog_south)
        
        return box

    def _create_plot_widget(self):
        """Create the matplotlib plot widget with refresh button."""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Add refresh button
        self.btn_refresh = QPushButton("Refresh Plot")
        self.btn_refresh.setStyleSheet("background-color: #17a2b8;")
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
    # Command handlers / slots
    # ------------------------------------------------------------------
    def _submit_tracker_command(self, ant: str, mode: str, params: dict):
        """Submit a command to the appropriate tracker thread."""
        # For slews, determine if it's RA/Dec or Az/El
        if mode == "slew":
            if "ra" in params and "dec" in params:
                src = Source(ra_hrs=params["ra"], dec_deg=params["dec"])
                self._target_thr(ant).submit("slew", source=src)
            else:
                self._target_thr(ant).submit("slew", az=params["az"], el=params["el"])
            return

        # For all other modes, a source object is expected
        src = Source(ra_hrs=params["ra"], dec_deg=params["dec"])
        
        if mode == "track":
            self._target_thr(ant).submit(
                "track", 
                source=src, 
                duration_hours=params["duration_hours"], 
                slew=params["slew"], 
                park=params["park"]
            )
        elif mode == "rasta":
            self._target_thr(ant).submit(
                "rasta_scan", 
                source=src, 
                max_dist_deg=params["max_dist_deg"], 
                step_deg=params["step_deg"], 
                position_angle_deg=params["position_angle_deg"], 
                duration_hours=params["duration_hours"], 
                slew=params["slew"], 
                park=params["park"]
            )
        elif mode == "pointing_scan":
            self._target_thr(ant).submit(
                "pointing_offsets", 
                source=src, 
                closest_dist_deg=params["closest_dist_deg"], 
                number_of_points=params["number_of_points"], 
                duration_hours=params["duration_hours"], 
                slew=params["slew"], 
                park=params["park"]
            )

    def _on_observation_requested(self, obs: Observation):
        ant = obs.ant
        mode = obs.mode
        params = obs.params

        # Handle split-track mode for RASTA and pointing scans
        is_split_mode = ant in ["N", "S"] and mode in ["rasta", "pointing_scan"] and params.get("split_track", False)
        if is_split_mode:
            scan_ant = ant
            track_ant = "N" if ant == "S" else "S"
            
            # Submit scan command to the designated antenna
            self._submit_tracker_command(scan_ant, mode, params)
            
            # Submit a long-duration track command to the other antenna
            # Calculate total duration needed for the scan
            if mode == "rasta":
                n_points = int((2 * params["max_dist_deg"]) / params["step_deg"]) + 1
            else: # pointing_scan
                n_points = params["number_of_points"]
            track_duration = params["duration_hours"] * n_points + 0.04  # Add buffer

            track_params = {
                "ra": params["ra"],
                "dec": params["dec"],
                "duration_hours": track_duration,
                "slew": True, # Always slew for tracking part of split obs
                "park": True, # Always park after tracking part of split obs
            }
            self._submit_tracker_command(track_ant, "track", track_params)
            return

        # Handle regular (non-split) observations
        ants_to_command = ["N", "S"] if ant == "both" else [ant]
        for a in ants_to_command:
            self._submit_tracker_command(a, mode, params)

    def _on_command_requested(self, ant: str, cmd: str):
        """Slot to handle simple commands like 'park' or 'stop'."""
        if ant == "both":
            ants = ["N", "S"]
        else:
            ants = [ant]

        for a in ants:
            if cmd == "park":
                self._target_thr(a).submit("park")
            elif cmd == "stop":
                self._target_thr(a).request_stop()

    def _target_thr(self, ant):
        if ant == "N":
            return self.thr_trk_n
        elif ant == "S":
            return self.thr_trk_s
        else:
            raise ValueError(f"Unknown antenna: {ant}")

    def _set_observation_name_from_metadata(self, extra_metadata: dict) -> None:
        """Set observation name based on metadata if available."""
        if extra_metadata and 'mode' in extra_metadata and 'antenna' in extra_metadata:
            mode = extra_metadata['mode']
            antenna = extra_metadata['antenna']
            if antenna == 'both':
                antenna = 'NS'
            obs_name = f"{mode}_{antenna}"
            self.recorder.set_observation_name(obs_name)
            self.root_log.info(f"Set observation name to: {obs_name}")

    def _gather_run_metadata(self, is_automatic_start: bool = False) -> dict:
        """
        Gather comprehensive observation/run info for metadata file.
        
        Args:
            is_automatic_start: True if recording was started automatically
        
        Returns:
            Dictionary of metadata
        """
        # Start with basic metadata
        metadata = {
            'recording_start_type': 'automatic' if is_automatic_start else 'manual',
        }
        
        # Get current observation if available
        obs = getattr(self.observation_panel, '_current_obs', None)
        
        if obs:
            # Use observation data
            metadata.update(self._extract_observation_metadata(obs))
        else:
            # Fallback to UI state
            metadata.update(self._extract_ui_metadata())
        
        return metadata

    def _extract_observation_metadata(self, obs: Observation) -> dict:
        """Extract metadata from current observation."""
        params = obs.params
        ant = obs.ant
        mode = obs.mode
        
        # Determine active antennas
        active_antennas = self._get_active_antennas(ant, mode, params)
        
        metadata = {
            'mode': mode,
            'antenna': ant,
            'active_antennas': active_antennas,
            'target_ra': params.get('ra'),
            'target_dec': params.get('dec'),
            'target_az': params.get('az'),
            'target_el': params.get('el'),
            'duration_hours': params.get('duration_hours'),
            'slew': params.get('slew'),
            'park': params.get('park'),
            'record': params.get('record'),
        }
        
        # Add scan-specific parameters
        if mode == 'rasta':
            metadata.update({
                'max_dist_deg': params.get('max_dist_deg'),
                'step_deg': params.get('step_deg'),
                'position_angle_deg': params.get('position_angle_deg'),
            })
        elif mode == 'pointing_scan':
            metadata.update({
                'closest_dist_deg': params.get('closest_dist_deg'),
                'number_of_points': params.get('number_of_points'),
            })
        
        # Handle split mode
        if params.get('split_track', False) and ant in ["N", "S"] and mode in ["rasta", "pointing_scan"]:
            track_ant = "N" if ant == "S" else "S"
            metadata.update({
                'split_track': True,
                'split_mode': True,
                'scan_antenna': ant,
                'track_antenna': track_ant,
                'scan_mode': mode,
                'track_mode': 'track',
            })
        else:
            metadata['split_track'] = params.get('split_track', False)
        
        return metadata

    def _extract_ui_metadata(self) -> dict:
        """Extract metadata from UI state when no observation is active."""
        try:
            # Determine antenna selection
            if self.observation_panel.rb_north.isChecked():
                ant = 'N'
            elif self.observation_panel.rb_south.isChecked():
                ant = 'S'
            else:
                ant = 'both'
            
            # Determine mode
            mode = None
            for rb, m in [
                (self.observation_panel.rb_track, 'track'),
                (self.observation_panel.rb_slew, 'slew'),
                (self.observation_panel.rb_rasta, 'rasta'),
                (self.observation_panel.rb_point, 'pointing_scan'),
            ]:
                if rb.isChecked():
                    mode = m
                    break
            
            # Get coordinates if available
            target_ra = getattr(self, 'coord_ra', None)
            target_dec = getattr(self, 'coord_dec', None)
            target_az = getattr(self, 'coord_az', None)
            target_el = getattr(self, 'coord_el', None)
            
            return {
                'mode': mode,
                'antenna': ant,
                'active_antennas': self._get_active_antennas(ant, mode, {}),
                'target_ra': target_ra.value() if target_ra else None,
                'target_dec': target_dec.value() if target_dec else None,
                'target_az': target_az.value() if target_az else None,
                'target_el': target_el.value() if target_el else None,
            }
        except Exception as e:
            self.root_log.warning(f"Error gathering metadata from UI: {e}")
            return {'error': 'Failed to gather complete metadata'}

    def _get_active_antennas(self, ant: str, mode: str, params: dict) -> list:
        """Determine which antennas are active for an observation."""
        if ant == "both":
            return ["N", "S"]
        elif ant in ["N", "S"]:
            # Check for split mode
            if params.get("split_track", False) and mode in ["rasta", "pointing_scan"]:
                track_ant = "N" if ant == "S" else "S"
                return [ant, track_ant]  # scan antenna + track antenna
            else:
                return [ant]
        else:
            return []

    def _toggle_record(self):
        extra_metadata = self._gather_run_metadata(is_automatic_start=False)
        if self.btn_start.text() == "Start":
            # Set observation name if we have metadata
            self._set_observation_name_from_metadata(extra_metadata)
            
            self.thr_rec_cmd.submit("start", fftshift=self.sp_fft.value(), acclen=self.sp_acc.value(), extra_metadata=extra_metadata)
            self.btn_start.setText("Stop")
            self.recorder_status_label.setText("Recording")
            # Manual start clears observation flag so auto-stop logic ignores
            self._recorder_started_for_obs = False
        else:
            self.thr_rec_cmd.submit("stop")
            self.btn_start.setText("Start")
            self.recorder_status_label.setText("Stopping...")
            # If user stops manually, ensure flag is cleared
            self._recorder_started_for_obs = False

    def _apply_params(self):
        self.thr_rec_cmd.submit("set_params", fftshift=self.sp_fft.value(), acclen=self.sp_acc.value())
        self.recorder_status_label.setText("Setting parameters...")

    # ------------------------------------------------------------------
    # Slots
    def _on_trk_progress(self, info: TrkProgInfo):
        # Update antenna status and progress in the observation panel
        if hasattr(self, 'observation_panel'):
            if hasattr(info, 'operation_type'):
                op = info.operation_type.value if hasattr(info.operation_type, 'value') else str(info.operation_type)
                msg = info.message.strip() if hasattr(info, 'message') and info.message else ''
                # Try to extract and format coordinates from the message
                def format_coords(msg):
                    # Try to find RA/Dec or AZ/EL in the message
                    import re
                    ra_dec = re.search(r'RA=([\d.\-]+)h,?\s*Dec=([\d.\-]+)', msg)
                    az_el = re.search(r'AZ=([\d.\-]+).?°,?\s*EL=([\d.\-]+)', msg)
                    if ra_dec:
                        ra = float(ra_dec.group(1))
                        dec = float(ra_dec.group(2))
                        return f"Ra {ra:.2f}, Dec {dec:.2f}"
                    elif az_el:
                        az = float(az_el.group(1))
                        el = float(az_el.group(2))
                        return f"Az {az:.2f}, El {el:.2f}"
                    return msg  # fallback: just return the message
                if op == 'track':
                    status = f"Tracking {format_coords(msg)}"
                elif op == 'slew':
                    status = f"Slewing {format_coords(msg)}"
                elif op == 'park':
                    status = "Parking..."
                elif op == 'rasta_scan':
                    status = f"Rasta {format_coords(msg)}"
                elif op == 'pointing_offsets':
                    status = f"Pointing {format_coords(msg)}"
                else:
                    status = f"Operating {format_coords(msg)}"
            else:
                if info.percent_complete < 100:
                    status = "Operating..."
                else:
                    status = "Idle"
            self.update_antenna_status(info.antenna, status)
            self.update_antenna_progress(info.antenna, int(info.percent_complete))
        
        # Check if tracking operation is complete and stop recording if active
        if info.is_complete and self._recorder_started_for_obs and self.recorder.is_recording:
                self.root_log.info(f"Tracking/scan operation completed for antenna {info.antenna}, stopping recording")
                self.thr_rec_cmd.submit("stop")
                # Update UI to reflect that recording is being stopped
                self.btn_start.setText("Start")
                self.recorder_status_label.setText("Stopping...")
                self._recorder_started_for_obs = False

    def _on_rec_progress(self, info: RecProgInfo):
        pct = int(info.percent_complete)
        self.recorder_status_label.setText(f"Recording ({pct}%)" if not info.is_complete else "Idle")
        
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

    def _on_tracking_event(self, ant: str, event_type: str, source=None, source_idx=None):
        """Handle tracking start/stop events to control the recorder."""
        self.root_log.info(f"Received tracking event: ant={ant}, event_type={event_type}, source_idx={source_idx}")
        
        obs = self.observation_panel._current_obs
        if not obs or not obs.params.get('record', False):
            self.root_log.info(f"Ignoring tracking event for {ant} because recording is not enabled.")
            return

        # Handle point start/completion events for scans
        if event_type == 'start_point':
            self._handle_point_start(ant, source, source_idx, obs)
            return
        if event_type == 'stop_point':
            self._handle_point_completion(ant, source, source_idx, obs)
            return

        # Handle overall tracking start/stop events
        if event_type in ['start', 'start_scan']:
            self._handle_tracking_start(ant, event_type, obs)
        elif event_type in ['stop', 'stop_scan']:
            self._handle_tracking_stop(ant, event_type, obs)

    def _handle_point_start(self, ant: str, source, source_idx: int, obs: Observation):
        """Handle point start for scan operations to notify the recorder. This does NOT affect recording state."""
        if self.recorder.is_recording and self._recorder_started_for_obs:
            self.root_log.info(f"Scan point {source_idx} started by antenna {ant}. Notifying recorder.")
            self.thr_rec_cmd.submit(
                "start_point_recording",
                source_idx=source_idx
            )

    def _handle_point_completion(self, ant: str, source, source_idx: int, obs: Observation):
        """Handle point completion for scan operations. This does NOT affect recording state."""
        mode = obs.mode
        is_split_mode = obs.params.get('split_track', False)
        
        # In split mode, only process events from the scan antenna
        if is_split_mode and mode in ['rasta', 'pointing_offsets']:
            scan_antenna = obs.ant
            if ant != scan_antenna:
                self.root_log.info(f"Split mode: ignoring point completion from tracking antenna {ant}")
                return
            else:
                self.root_log.info(f"Split mode: point {source_idx} completed by scan antenna {ant}")
        else:
            self.root_log.info(f"Scan point {source_idx} completed by antenna {ant}")
        
        # Log point completion if recording is active
        if self.recorder.is_recording and self._recorder_started_for_obs:
            self.thr_rec_cmd.submit(
                "log_point_completion",
                source_ra=source.ra_hrs,
                source_dec=source.dec_deg,
                source_idx=source_idx,
                antenna=ant
            )
        else:
            self.root_log.warning(f"Point {source_idx} completed, but recording is not active.")

    def _handle_tracking_start(self, ant: str, event_type: str, obs: Observation):
        """Handle tracking start events."""
        self.root_log.info(f"Received {event_type} event for antenna {ant}")
        self._ant_tracking[ant] = True
        
        # Determine if we should start recording
        should_start_recording = self._should_start_recording(ant, event_type, obs)
        
        if should_start_recording and not self.recorder.is_recording:
            self._start_automatic_recording(obs)
        elif self.recorder.is_recording:
            self.root_log.info(f"Recorder is already running, not starting again")

    def _handle_tracking_stop(self, ant: str, event_type: str, obs: Observation):
        """Handle tracking stop events."""
        self.root_log.info(f"Received {event_type} event for antenna {ant}")
        self._ant_tracking[ant] = False
        
        # Determine if we should stop recording
        should_stop_recording = self._should_stop_recording(ant, event_type, obs)
        
        if should_stop_recording and self.recorder.is_recording and self._recorder_started_for_obs:
            self._stop_automatic_recording()
        else:
            self.root_log.info(f"Not stopping recorder (not running or not started by this observation)")

    def _should_start_recording(self, ant: str, event_type: str, obs: Observation) -> bool:
        """Determine if recording should be started based on the tracking event."""
        ant_selection = obs.ant
        mode = obs.mode
        is_split_mode = obs.params.get('split_track', False)
        
        # For split mode scans, start when scan antenna emits start_scan
        if is_split_mode and mode in ['rasta', 'pointing_offsets']:
            if event_type == 'start_scan':
                self.root_log.info(f"Split mode scan started by scan antenna {ant}")
                return True
            else:
                self.root_log.info(f"Split mode: ignoring tracking start from scan antenna {ant}")
                return False
        
        # For regular operations, check antenna selection
        if ant_selection == 'both':
            if self._ant_tracking["N"] and self._ant_tracking["S"]:
                self.root_log.info(f"Both antennas are tracking")
                return True
        elif ant_selection in ["N", "S"] and self._ant_tracking[ant_selection]:
            self.root_log.info(f"Antenna {ant_selection} is tracking")
            return True
        
        self.root_log.info(f"Waiting for more antennas. Current: N={self._ant_tracking.get('N', False)}, S={self._ant_tracking.get('S', False)}")
        return False

    def _should_stop_recording(self, ant: str, event_type: str, obs: Observation) -> bool:
        """Determine if recording should be stopped based on the tracking event."""
        mode = obs.mode
        is_split_mode = obs.params.get('split_track', False)
        
        # For split mode scans, stop when scan antenna emits stop_scan
        if is_split_mode and mode in ['rasta', 'pointing_offsets']:
            scan_antenna = obs.ant
            # The 'stop_scan' event is only emitted by the scan antenna when all points are done.
            if ant == scan_antenna and event_type == 'stop_scan':
                self.root_log.info(f"Split mode scan ended by scan antenna {ant}. Stopping recorder.")
                return True
            # For any other stop event in split mode (e.g., tracking antenna stopping), don't stop recording.
            self.root_log.info(f"Split mode: ignoring stop event from {ant} (event: {event_type})")
            return False
            
        # For non-split mode, check if all active antennas have stopped tracking.
        active_antennas = self._get_active_antennas(obs.ant, mode, obs.params)
        if not active_antennas:
             self.root_log.info("No active antennas for observation, no reason to stop recording.")
             return False

        # Stop if ALL active antennas are no longer tracking
        all_stopped = all(not self._ant_tracking.get(a, False) for a in active_antennas)
        
        if all_stopped:
            self.root_log.info(f"All active antennas ({active_antennas}) have stopped tracking. Stopping recorder.")
            return True
        else:
            still_tracking = [a for a in active_antennas if self._ant_tracking.get(a, False)]
            self.root_log.info(f"Antenna {ant} stopped, but other antennas ({still_tracking}) are still active.")
            return False

    def _start_automatic_recording(self, obs: Observation):
        """Start recording automatically when tracking begins."""
        self.root_log.info(f"Starting recorder for observation")
        extra_metadata = self._gather_run_metadata(is_automatic_start=True)
        self._set_observation_name_from_metadata(extra_metadata)
        self.thr_rec_cmd.submit("start", fftshift=self.sp_fft.value(), acclen=self.sp_acc.value(), extra_metadata=extra_metadata)
        self.btn_start.setText("Stop")
        self.recorder_status_label.setText("Recording")
        self._recorder_started_for_obs = True

    def _stop_automatic_recording(self):
        """Stop recording automatically when tracking ends."""
        self.thr_rec_cmd.submit("stop")
        self.btn_start.setText("Start")
        self.recorder_status_label.setText("Stopping...")
        self._recorder_started_for_obs = False

    def _update_antenna_position_and_target(self, antenna: str, az: float, el: float, tar_az: float, tar_el: float):
        """Update both antenna position and target display together."""
        # Format position values, showing "--" if None
        pos_text = f"Position: Az {az:.2f}°, El {el:.2f}°" if az is not None and el is not None else "Position: --"
        target_text = f"Target: Az {tar_az:.2f}°, El {tar_el:.2f}°" if tar_az is not None and tar_el is not None else "Target: --"
        
        if antenna == "N":
            self.lbl_north_pos.setText(pos_text)
            self.lbl_north_target.setText(target_text)
        elif antenna == "S":
            self.lbl_south_pos.setText(pos_text)
            self.lbl_south_target.setText(target_text)

    def update_antenna_status(self, antenna: str, status: str):
        """Update antenna status display."""
        if antenna == "N":
            self.lbl_north_status.setText(f"North: {status}")
        elif antenna == "S":
            self.lbl_south_status.setText(f"South: {status}")

    def update_antenna_progress(self, antenna: str, progress: int):
        """Update antenna progress bar."""
        if antenna == "N":
            self.prog_north.setValue(progress)
        elif antenna == "S":
            self.prog_south.setValue(progress)

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
        """Reset tracker status to Idle when operation completes."""
        # Update antenna status to Idle
        self.update_antenna_status(ant, "Idle")

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