from __future__ import annotations
import sys
import time
import logging
import queue
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
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal

from recording import Recorder
from tracking import Source
from tracking.utils.progress import ProgressInfo as TrkProgInfo
from recording.utils.progress import ProgressInfo as RecProgInfo
from gui.widgets.coordinate_calculator_widget import CoordinateCalculatorWidget

from core.threads import (
    TrackerThread,
    RecorderCmdThread,
    RecorderStatusThread,
    AntennaPositionThread,
)
from core.bridges import (
    TrackerThreadBridge,
    RecorderCmdThreadBridge,
    RecorderStatusThreadBridge,
    AntennaPositionThreadBridge,
)
from gui.observation_panel import ObservationPanel, Observation
from gui.widgets.ftx_settings_widget import FTXPanel
from gui.widgets.data_plot_widget import DataPlotWidget
from gui.widgets.recording_control_widget import RecordingControlWidget
from gui.widgets.antenna_status_widget import AntennaStatusWidget
from gui.widgets.log_widget import LogWidget
from gui.widgets.fftshift_test_widget import FFTShiftTestWidget
from gui.services.metadata import MetadataBuilder
from gui.services.fftshift_runner import FFTShiftTestRunner



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

class QueueLogHandler(QObject):
    log_received = pyqtSignal(str)

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def poll_log_queue(self):
        while True:
            try:
                record = self.log_queue.get(block=False)
                # Format the record properly
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                msg = formatter.format(record)
                self.log_received.emit(msg)
            except queue.Empty:
                break

class MainWindow(QMainWindow):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.setWindowTitle("Antenna & Recorder Control")
        self.resize(900, 400)

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Track if we started the recorder for the current observation
        self._recorder_started_for_obs = False
        self._ant_tracking = {"N": False, "S": False}  # Track tracking state of both antennas

        # Initialize components
        self._setup_recorder()
        self._setup_bridges()
        self._setup_threads()
        self._build_ui()
        self._setup_logging(log_queue)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _setup_logging(self, log_queue: queue.Queue):
        """Setup logging queue handler."""
        self.log_handler = QueueLogHandler(log_queue)
        self.log_handler.log_received.connect(self._append_log)
        
        # Poll the queue every 100ms
        self.log_poll_timer = QTimer(self)
        self.log_poll_timer.timeout.connect(self.log_handler.poll_log_queue)
        self.log_poll_timer.start(100)

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

        # Recorder command bridge
        self.rec_cmd_bridge = RecorderCmdThreadBridge()
        self.rec_cmd_bridge.progress_signal.connect(self._on_rec_progress)

        # Recorder status bridge
        self.rec_status_bridge = RecorderStatusThreadBridge()
        self.rec_status_bridge.status_signal.connect(self._update_rec_status)
        self.rec_status_bridge.data_signal.connect(self._update_data_buffer)

        # Antenna position bridges
        self.pos_n_bridge = AntennaPositionThreadBridge()
        self.pos_s_bridge = AntennaPositionThreadBridge()
        self.pos_n_bridge.position_and_target_signal.connect(self._update_antenna_position_and_target)
        self.pos_s_bridge.position_and_target_signal.connect(self._update_antenna_position_and_target)

    def _setup_threads(self):
        """Initialize and start all worker threads (unified bridges)."""
        
        self.thr_trk_n = TrackerThread("N", self.trk_n_bridge, logging.getLogger(__name__))
        self.thr_trk_s = TrackerThread("S", self.trk_s_bridge, logging.getLogger(__name__))
        self.thr_rec_cmd = RecorderCmdThread(self.recorder, self.rec_cmd_bridge, logging.getLogger(__name__))
        self.thr_rec_status = RecorderStatusThread(self.recorder, self.rec_status_bridge, logging.getLogger(__name__))
        self.thr_pos_n = AntennaPositionThread("N", self.pos_n_bridge, logging.getLogger(__name__))
        self.thr_pos_s = AntennaPositionThread("S", self.pos_s_bridge, logging.getLogger(__name__))
        threads = [self.thr_trk_n, self.thr_trk_s, self.thr_rec_cmd, 
                  self.thr_rec_status, self.thr_pos_n, self.thr_pos_s]
        for thread in threads:
            thread.start()

    # ------------------------------------------------------------------
    # UI building
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setStyleSheet("""
            /* Base inspired by Firenze palette */
            QMainWindow {
                background-color: #FAEECF; /* lightest palette color */
            }
            QWidget {
                background-color: #FAEECF; /* make all widget backgrounds beige */
                color: #2C3E50;
                font-family: Arial, sans-serif;
            }
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                border: 1px solid #E8E2D0;
                border-radius: 5px;
                margin-top: 10px;
                background-color: #FAEECF; /* match app background */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #FAEECF; /* match app background */
                color: #2C3E50;
            }
            QLabel {
                font-size: 10pt;
                color: #2C3E50;
            }
            QPushButton {
                font-size: 10pt;
                padding: 8px;
                border-radius: 4px;
                background-color: #FFC857; /* default orange */
                border: 1px solid #D9A63F;
                color: #2C3E50;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E0B24A;
            }
            QPushButton:disabled {
                background-color: #D8D8D8;
                color: #EEEEEE;
            }
            QRadioButton, QCheckBox {
                font-size: 10pt;
                color: #2C3E50;
            }
            QTextEdit {
                background-color: #FEFAE9; /* keep inputs white */
                border: 1px solid #E8E2D0;
                border-radius: 4px;
                font-size: 9pt;
            }
            QDoubleSpinBox, QSpinBox {
                padding: 4px;
                border: 1px solid #E8E2D0;
                border-radius: 4px;
                background-color: #FEFAE9; /* keep inputs white */
            }
            QProgressBar {
                text-align: center;
                border: 1px solid #E8E2D0;
                border-radius: 4px;
                background-color: #FEFAE9; /* keep control white */
                color: #2C3E50;
            }
            QProgressBar::chunk {
                background-color: #4F8A69; /* green */
            }
            QSplitter::handle {
                background-color: #E8E2D0;
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
        self.observation_panel.setFixedWidth(350)  # Fixed width for observation panel
        left_layout.addWidget(self.observation_panel)
        
        # Middle column: Recording controls, antenna status, and coordinate calculations (fixed width)
        v_middle = QVBoxLayout()
        # Recording controls (modular widget)
        self.rec_panel = RecordingControlWidget()
        v_middle.addWidget(self.rec_panel)
        # Expose child controls on MainWindow for backward compatibility
        self.recorder_status_label = self.rec_panel.recorder_status_label
        self.sp_fft_p0 = self.rec_panel.sp_fft_p0
        self.sp_fft_p1 = self.rec_panel.sp_fft_p1
        self.sp_acc = self.rec_panel.sp_acc
        self.btn_start = self.rec_panel.btn_start
        self.btn_set = self.rec_panel.btn_set
        self.lbl_fft = self.rec_panel.lbl_fft
        self.lbl_acc = self.rec_panel.lbl_acc
        self.lbl_ovf = self.rec_panel.lbl_ovf
        # Wire high-level signals from the widget
        self.rec_panel.start_requested.connect(self._handle_rec_start_requested)
        self.rec_panel.stop_requested.connect(self._handle_rec_stop_requested)
        self.rec_panel.set_params_requested.connect(self._handle_rec_set_params_requested)

        # Antenna status (modular widget)
        self.ant_status = AntennaStatusWidget()
        v_middle.addWidget(self.ant_status)
        # Expose labels and progress bars for compatibility
        self.lbl_north_status = self.ant_status.lbl_north_status
        self.lbl_south_status = self.ant_status.lbl_south_status
        self.lbl_north_pos = self.ant_status.lbl_north_pos
        self.lbl_north_target = self.ant_status.lbl_north_target
        self.lbl_south_pos = self.ant_status.lbl_south_pos
        self.lbl_south_target = self.ant_status.lbl_south_target
        self.prog_north = self.ant_status.prog_north
        self.prog_south = self.ant_status.prog_south
        # Coordinate calculator widget (separate)
        self.coord_calc_widget = CoordinateCalculatorWidget()
        v_middle.addWidget(self.coord_calc_widget)

        # FFT Shift Test widget (modular)
        self.fftshift_widget = FFTShiftTestWidget()
        # Hide by default to match previous behavior; set to True if you want it visible
        self.fftshift_widget.setVisible(False)
        # Log messages from widget
        self.fftshift_widget.log_message.connect(self._append_log)
        # Provide backend so the widget can self-run the test sequence
        self.fftshift_widget.configure_backend(
            recorder=self.recorder,
            rec_cmd_thread=self.thr_rec_cmd,
            fft_params_getter=lambda: (self.sp_fft_p0.value(), self.sp_fft_p1.value(), self.sp_acc.value()),
            extra_metadata_getter=lambda: self._gather_run_metadata(is_automatic_start=True),
        )
        v_middle.addWidget(self.fftshift_widget)
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
        left_container.setFixedWidth(350 + 300 + 200)  # 800 total
        
        splitter.addWidget(left_container)

        # Right column: Plot and logs
        right_widget = QWidget()
        v_right = QVBoxLayout(right_widget)
        plot_box = QGroupBox("Data Plot")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self._create_plot_widget())
        v_right.addWidget(plot_box)
        # Log panel (modular widget)
        self.log_widget = LogWidget()
        self.txt_log = self.log_widget.txt_log
        v_right.addWidget(self.log_widget)
        splitter.addWidget(right_widget)

        # Set initial splitter sizes and configure splitter behavior
        splitter.setSizes([850, 500])  # Left side is now 800 pixels (fixed width), right side 400 pixels
        
        # Prevent the left side from being resizable by setting stretch factors
        # This ensures only the right side can resize when the window is resized
        splitter.setStretchFactor(0, 0)  # Left side (index 0) has stretch factor 0 (no stretch)
        splitter.setStretchFactor(1, 1)  # Right side (index 1) has stretch factor 1 (can stretch)
        
        h.addWidget(splitter)
        
        # Connect observation panel signals to main window slots
        self.observation_panel.observation_requested.connect(self._on_observation_requested)
        self.observation_panel.command_requested.connect(self._on_command_requested)
        self.observation_panel.log_message_requested.connect(self._append_log)
        
        # Connect FFT shift testing signals
        self.observation_panel.fftshift_test_start_requested.connect(self._on_fftshift_test_start)
        self.observation_panel.fftshift_test_stop_requested.connect(self._on_fftshift_test_stop)
        self.observation_panel.fftshift_set_requested.connect(self._on_fftshift_set)
        self.observation_panel.fftshift_recording_start_requested.connect(self._on_fftshift_recording_start)
        
        # Connect completion signal to queue logic after panel is created
        if hasattr(self, 'trk_n_bridge'): # Changed from br_completion to trk_n_bridge
            self.trk_n_bridge.completion_signal.connect(self.observation_panel._on_tracker_completion)
        
        if hasattr(self, 'trk_s_bridge'): # Also connect S antenna completion signal
            self.trk_s_bridge.completion_signal.connect(self.observation_panel._on_tracker_completion)
        
    def _build_rec_panel(self):
        """Builds the recorder control panel."""
        box = QGroupBox("Recorder")
        h = QVBoxLayout(box)
        
        # Status label row
        self.recorder_status_label = QLabel("Idle")
        self.recorder_status_label.setStyleSheet("color: #4F8A69; font-weight: bold;")
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        status_row.addWidget(self.recorder_status_label)
        status_row.addStretch()
        h.addLayout(status_row)
        
        # Top row controls
        row = QHBoxLayout()
        self.sp_fft_p0 = QSpinBox()
        self.sp_fft_p0.setRange(0, 4095)
        self.sp_fft_p0.setValue(1904)
        self.sp_fft_p1 = QSpinBox()
        self.sp_fft_p1.setRange(0, 4095)
        self.sp_fft_p1.setValue(1904)
        self.sp_acc = QSpinBox()
        self.sp_acc.setRange(1, 1_000_000)
        self.sp_acc.setValue(131072)
        
        row.addWidget(QLabel("fftshift S:"))
        row.addWidget(self.sp_fft_p0)
        row.addWidget(QLabel("N:"))
        row.addWidget(self.sp_fft_p1)
        row.addWidget(QLabel("acclen"))
        row.addWidget(self.sp_acc)
        h.addLayout(row)
        
        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet("background-color: #4F8A69; border: 1px solid #3E6E54; color: #FFFFFF;")
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



    def _build_antenna_status_panel(self):
        box = QGroupBox("Antenna Status")
        layout = QVBoxLayout(box)
        
        # North antenna status
        north_status = QVBoxLayout()
        
        # Status line
        north_status_line = QHBoxLayout()
        self.lbl_north_status = QLabel("North: Idle")
        self.lbl_north_status.setStyleSheet("color: #4F8A69; font-weight: bold;")
        north_status_line.addWidget(self.lbl_north_status)
        north_status_line.addStretch()
        north_status.addLayout(north_status_line)
        
        # Position and Target stacked vertically
        self.lbl_north_pos = QLabel("Position: --")
        self.lbl_north_pos.setStyleSheet("background-color: #FEFAE9; padding: 4px; border: 1px solid #E8E2D0; border-radius: 3px;")
        north_status.addWidget(self.lbl_north_pos)
        
        self.lbl_north_target = QLabel("Target: --")
        self.lbl_north_target.setStyleSheet("background-color: #FEFAE9; padding: 4px; border: 1px solid #E8E2D0; border-radius: 3px;")
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
        self.lbl_south_status.setStyleSheet("color: #4F8A69; font-weight: bold;")
        south_status_line.addWidget(self.lbl_south_status)
        south_status_line.addStretch()
        south_status.addLayout(south_status_line)
        
        # Position and Target stacked vertically
        self.lbl_south_pos = QLabel("Position: --")
        self.lbl_south_pos.setStyleSheet("background-color: #FEFAE9; padding: 4px; border: 1px solid #E8E2D0; border-radius: 3px;")
        south_status.addWidget(self.lbl_south_pos)
        
        self.lbl_south_target = QLabel("Target: --")
        self.lbl_south_target.setStyleSheet("background-color: #FEFAE9; padding: 4px; border: 1px solid #E8E2D0; border-radius: 3px;")
        south_status.addWidget(self.lbl_south_target)
        
        layout.addLayout(south_status)
        
        # South progress bar
        self.prog_south = QProgressBar()
        self.prog_south.setRange(0, 100)
        self.prog_south.setValue(0)
        layout.addWidget(self.prog_south)
        
        return box

    def _create_plot_widget(self):
        """Create the reusable plot widget and wire refresh callback."""
        self.data_plot = DataPlotWidget()
        # Wire the plot widget refresh signal to request data from the recorder
        self.data_plot.refresh_requested.connect(self._refresh_plot)
        return self.data_plot

    def _build_log_panels(self):
        layout = QVBoxLayout()

        # Combined Log
        box_log = QGroupBox("Log")
        v1 = QVBoxLayout(box_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        v1.addWidget(self.txt_log)
        layout.addWidget(box_log)

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
                max_distance_deg=params["max_dist_deg"], 
                steps_deg=params["step_deg"], 
                position_angle_deg=params["position_angle_deg"], 
                duration_hours=params["duration_hours"], 
                slew=params["slew"], 
                park=params["park"]
            )
        elif mode == "rtos":
            # Build two source objects
            src2 = Source(ra_hrs=params["ra2"], dec_deg=params["dec2"])
            self._target_thr(ant).submit(
                "rtos", 
                source1=src, 
                source2=src2, 
                number_of_points=params["number_of_points"], 
                duration_hours=params["duration_hours"], 
                slew=params["slew"], 
                park=params["park"]
            )
        elif mode == "pointing_scan":
            self._target_thr(ant).submit(
                "pointing_offsets", 
                source=src, 
                closest_distance_deg=params["closest_dist_deg"], 
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

            # Use the same 'park' value as the scan antenna (from params)
            park_value = params.get("park", True)
            track_params = {
                "ra": params["ra"],
                "dec": params["dec"],
                "duration_hours": track_duration,
                "slew": True, # Always slew for tracking part of split obs
                "park": park_value, # Use the same park value as the scan antenna
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
            self.logger.info(f"Set observation name to: {obs_name}")

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
        
        # Add FTX parameters to metadata
        try:
            ftx_params = self.ftx_panel.get_all_ftx_parameters()
            metadata.update(ftx_params)
            self.logger.info("Successfully added FTX parameters to metadata")
        except Exception as e:
            self.logger.warning(f"Failed to get FTX parameters for metadata: {e}")
            # Add placeholder values to indicate failure
            metadata.update({
                'ftx_yn_attenuation_db': None,
                'ftx_yn_rf_power_dbm': None,
                'ftx_yn_laser_current_ma': None,
                'ftx_xn_attenuation_db': None,
                'ftx_xn_rf_power_dbm': None,
                'ftx_xn_laser_current_ma': None,
                'ftx_ys_attenuation_db': None,
                'ftx_ys_rf_power_dbm': None,
                'ftx_ys_laser_current_ma': None,
                'ftx_xs_attenuation_db': None,
                'ftx_xs_rf_power_dbm': None,
                'ftx_xs_laser_current_ma': None,
                'ftx_query_timestamp': None,
                'ftx_query_error': str(e)
            })
        
        return metadata

    def _extract_observation_metadata(self, obs: Observation) -> dict:
        """Extract metadata from current observation (delegated to service)."""
        return MetadataBuilder.from_observation(obs)

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
                (self.observation_panel.rb_rtos, 'rtos'),
            ]:
                if rb.isChecked():
                    mode = m
                    break
            
            # Get coordinates from the coordinate calculator widget if available
            target_ra = None
            target_dec = None
            target_az = None
            target_el = None
            if hasattr(self, 'coord_calc_widget') and self.coord_calc_widget is not None:
                try:
                    coords = self.coord_calc_widget.get_current_coordinates()
                    target_ra = coords.get('ra')
                    target_dec = coords.get('dec')
                except Exception:
                    pass
            
            return {
                'mode': mode,
                'antenna': ant,
                'active_antennas': self._get_active_antennas(ant, mode, {}),
                'target_ra': float(target_ra) if target_ra is not None else None,
                'target_dec': float(target_dec) if target_dec is not None else None,
                'target_az': float(target_az) if target_az is not None else None,
                'target_el': float(target_el) if target_el is not None else None,
            }
        except Exception as e:
            self.logger.warning(f"Error gathering metadata from UI: {e}")
            return {'error': 'Failed to gather complete metadata'}

    def _get_active_antennas(self, ant: str, mode: str, params: dict) -> list:
        """Determine which antennas are active for an observation (delegated)."""
        return MetadataBuilder.active_antennas(ant, mode, params)

    

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
            # info.antenna is Antenna enum; convert to 'N'/'S' for UI
            ant_code = getattr(info.antenna, 'name', None)
            if ant_code == 'NORTH':
                ant_code = 'N'
            elif ant_code == 'SOUTH':
                ant_code = 'S'
            elif isinstance(info.antenna, str):
                ant_code = info.antenna
            else:
                ant_code = 'N'
            self.update_antenna_status(ant_code, status)
            self.update_antenna_progress(ant_code, int(info.percent_complete))
        
        # Recording stop is now handled by the delayed stop_run signal, not here

    def _on_rec_progress(self, info: RecProgInfo):
        pct = int(info.percent_complete)
        if info.is_complete:
            self.rec_panel.set_recording_state(False)
            self.rec_panel.set_status_text("Idle")
        else:
            self.rec_panel.set_recording_state(True)
            self.rec_panel.set_status_text(f"Recording ({pct}%)")

    def _update_rec_status(self, status: tuple):
        fft0, fft1, acc, ovf0, ovf1, rec = status
        self.rec_panel.update_status_values(fft0, fft1, acc, ovf0, ovf1)
        self.rec_panel.set_recording_state(rec)

    def _on_tracking_event(self, ant: str, event_type: str, source=None, source_idx=None):
        self.logger.info(f"Tracking event received: ant={ant}, event_type={event_type}, source_idx={source_idx}, current_obs={getattr(self.observation_panel, '_current_obs', None)}")
        obs = self.observation_panel._current_obs
        if not obs or not obs.params.get('record', False):
            self.logger.info(f"Ignoring tracking event for {ant} because recording is not enabled.")
            return

        def _delayed_handler(handler, delay_ms, *args, **kwargs):
            QTimer.singleShot(delay_ms, lambda: handler(*args, **kwargs))

        # New four-signal system with different delays to avoid race conditions
        if event_type == 'start_run':
            _delayed_handler(self._handle_tracking_start, 2500, ant, event_type, obs) #65000
        elif event_type == 'stop_run':
            _delayed_handler(self._handle_tracking_stop, 3500, ant, event_type, obs)
        elif event_type == 'start_point':
            # Add extra 500ms delay to ensure recording has started first
            _delayed_handler(self._handle_point_start, 3000, ant, source, source_idx, obs)
        elif event_type == 'stop_point':
            _delayed_handler(self._handle_point_completion, 3000, ant, source, source_idx, obs)
        else:
            pass

    def _handle_point_start(self, ant: str, source, source_idx: int, obs: Observation):
        """Handle point start for scan operations to notify the recorder. This does NOT affect recording state."""
        mode = obs.mode
        is_split_mode = obs.params.get('split_track', False)
        # In split mode, only process events from the scan antenna
        if is_split_mode and mode in ['rasta', 'pointing_offsets']:
            scan_antenna = obs.ant
            if ant != scan_antenna:
                return
        if self.recorder.is_recording and self._recorder_started_for_obs:
            format_dict = {
                'antenna': ant,
                'ra': source.ra_hrs if source else 0.0,
                'dec': source.dec_deg if source else 0.0,
                'timestamp': time.time()
            }
            self.thr_rec_cmd.submit(
                "start_point_recording",
                source_idx=source_idx,
                format_dict=format_dict
            )

    def _handle_point_completion(self, ant: str, source, source_idx: int, obs: Observation):
        """Handle point completion for scan operations. This does NOT affect recording state."""
        mode = obs.mode
        is_split_mode = obs.params.get('split_track', False)
        # In split mode, only process events from the scan antenna
        if is_split_mode and mode in ['rasta', 'pointing_offsets']:
            scan_antenna = obs.ant
            if ant != scan_antenna:
                return
        if self.recorder.is_recording and self._recorder_started_for_obs:
            self.thr_rec_cmd.submit(
                "stop_point_recording",
                source_idx=source_idx,
            )

    def _handle_tracking_start(self, ant: str, event_type: str, obs: Observation):
        """Handle tracking start events."""
        # self.root_log.info(f"Received {event_type} event for antenna {ant}")
        self._ant_tracking[ant] = True
        
        # Determine if we should start recording
        should_start_recording = self._should_start_recording(ant, event_type, obs)
        
        if should_start_recording and not self.recorder.is_recording:
            self._start_automatic_recording(obs)
        elif self.recorder.is_recording:
            # self.root_log.info(f"Recorder is already running, not starting again")
            pass

    def _handle_tracking_stop(self, ant: str, event_type: str, obs: Observation):
        """Handle tracking stop events."""
        # self.root_log.info(f"Received {event_type} event for antenna {ant}")
        self._ant_tracking[ant] = False
        
        # Determine if we should stop recording
        should_stop_recording = self._should_stop_recording(ant, event_type, obs)
        
        if should_stop_recording and self.recorder.is_recording and self._recorder_started_for_obs:
            self._stop_automatic_recording()
        else:
            # self.root_log.info(f"Not stopping recorder (not running or not started by this observation)")
            pass

    def _should_start_recording(self, ant: str, event_type: str, obs: Observation) -> bool:
        """
        Determine if recording should start based on the current event, antenna, and observation.

        --- Recording Trigger Logic Table ---
        | Mode           | Ant Selection | Split Mode | Event Type     | Antenna | Should Start Recording? |
        |----------------|--------------|------------|---------------|---------|------------------------|
        | track/rasta/pointing | N/S/both    | False      | start_run      | N/S     | Yes (when tracking starts for selected antenna(s)) |
        | track/rasta/pointing | both        | False      | start_run      | N/S     | Yes (when both are tracking) |
        | rasta/pointing      | N/S         | True       | start_run      | scan    | Yes (only when scan antenna emits start_run) |
        | rasta/pointing      | N/S         | True       | start_run      | track   | No |
        | (all others)        | any         | any        | any            | any     | No |
        """
        ant_selection = obs.ant
        mode = obs.mode
        is_split_mode = obs.params.get('split_track', False)

        # --- Split mode logic: Only scan antenna's start_run event triggers recording ---
        if is_split_mode and mode in ['rasta', 'pointing_scan']:
            scan_antenna = ant_selection  # The antenna selected for the scan
            if ant == scan_antenna and event_type == 'start_run':
                # self.root_log.info(f"Split mode: scan antenna {ant} emitted start_run, will start recording.")
                return True
            else:
                # self.root_log.info(f"Split mode: ignoring event (ant={ant}, event_type={event_type}) for recording trigger.")
                return False

        # --- Regular (non-split) logic ---
        if ant_selection == 'both':
            if self._ant_tracking["N"] and self._ant_tracking["S"]:
                # self.root_log.info(f"Both antennas are tracking")
                return True
        elif ant_selection in ["N", "S"] and self._ant_tracking[ant_selection]:
            # self.root_log.info(f"Antenna {ant_selection} is tracking")
            return True

        # self.root_log.info(f"Waiting for more antennas. Current: N={self._ant_tracking.get('N', False)}, S={self._ant_tracking.get('S', False)}")
        return False

    def _should_stop_recording(self, ant: str, event_type: str, obs: Observation, mode=None) -> bool:
        is_split_mode = obs.params.get('split_track', False)
        
        # --- Split-mode logic: only the SCAN antenna's stop_run ends recording ---
        if is_split_mode:
            scan_ant = obs.ant  # the antenna that is actually scanning
            # stop only when the scan antenna finishes
            if ant == scan_ant and event_type == 'stop_run':
                return True
            # any other stop_run (e.g. from the tracking antenna) is ignored
            return False
        
        # ---------- Non-split logic (unchanged) ----------
        active_antennas = self._get_active_antennas(obs.ant, mode, obs.params)
        if not active_antennas:
            return False
        all_stopped = all(not self._ant_tracking.get(a, False) for a in active_antennas)
        return all_stopped

    def _start_automatic_recording(self, obs: Observation):
        """Start recording automatically when tracking begins."""
        # self.root_log.info(f"Starting recorder for observation")
        extra_metadata = self._gather_run_metadata(is_automatic_start=True)
        self._set_observation_name_from_metadata(extra_metadata)
        p0, p1, acc = self.rec_panel.get_params()
        self.thr_rec_cmd.submit("start", fftshift_p0=p0, fftshift_p1=p1, acclen=acc, extra_metadata=extra_metadata)
        self._recorder_started_for_obs = True

    def _stop_automatic_recording(self):
        """Stop recording automatically when tracking ends."""
        self.thr_rec_cmd.submit("stop")
        self.recorder_status_label.setText("Stopping...")
        self._recorder_started_for_obs = False

    def _update_antenna_position_and_target(self, antenna: str, az: float, el: float, tar_az: float, tar_el: float):
        """Update both antenna position and target display together."""
        self.ant_status.update_position_and_target(antenna, az, el, tar_az, tar_el)

    def update_antenna_status(self, antenna: str, status: str):
        """Update antenna status display (delegated)."""
        if antenna not in ("N", "S"):
            self.logger.warning(f"Unknown antenna {antenna} in update_antenna_status")
            return
        self.ant_status.update_antenna_status(antenna, status)

    def update_antenna_progress(self, antenna: str, progress: int):
        """Update antenna progress bar (delegated)."""
        self.ant_status.update_antenna_progress(antenna, progress)

    def _update_data_buffer(self, data: list):
        """Update the data buffer display and redraw the plot."""
        if hasattr(self, 'data_plot'):
            self.data_plot.set_data(data)
        self._refresh_requested = False

    def _refresh_plot(self):
        """Refresh the plot with current data buffer."""
        # Always request fresh data buffer from recorder status thread
        self._refresh_requested = True  # Mark that we're expecting data
        self.thr_rec_status.request_data_buffer()

    

    def _append_log(self, msg: str):
        """Appends a log message to the unified log view."""
        if hasattr(self, 'log_widget'):
            self.log_widget.append(msg)
        else:
            self.txt_log.append(msg)

    # ------------------------------------------------------------------
    # Recorder control handlers from RecordingControlWidget
    # ------------------------------------------------------------------
    def _handle_rec_start_requested(self, p0: int, p1: int, acc: int):
        extra_metadata = self._gather_run_metadata(is_automatic_start=False)
        self._set_observation_name_from_metadata(extra_metadata)
        self.thr_rec_cmd.submit("start", fftshift_p0=p0, fftshift_p1=p1, acclen=acc, extra_metadata=extra_metadata)
        self.rec_panel.set_status_text("Recording")

    def _handle_rec_stop_requested(self):
        self.thr_rec_cmd.submit("stop")
        self.rec_panel.set_status_text("Stopping...")

    def _handle_rec_set_params_requested(self, p0: int, p1: int, acc: int):
        self.thr_rec_cmd.submit("set_params", fftshift_p0=p0, fftshift_p1=p1, acclen=acc)
        self.rec_panel.set_status_text("Setting parameters...")



    def _reset_tracker_status(self, ant: str):
        """Reset tracker status to Idle when operation completes."""
        # Update antenna status to Idle
        self.update_antenna_status(ant, "Idle")

    # ------------------------------------------------------------------
    # FFT Shift Testing Signal Handlers
    # ------------------------------------------------------------------
    
    def _on_fftshift_test_start(self, fftshift_list: list, current_attenuations: dict):
        """Handle FFT shift test start request using service runner."""
        try:
            import threading
            from utils.ftx import FTXController, Antenna, Polarization

            runner = FFTShiftTestRunner(
                atten_step_getter=lambda: self.observation_panel.fft_atten_step.value(),
                duration_getter=lambda: self.observation_panel.fft_test_duration.value(),
            )
            self._fft_runner = runner

            ftx_north = FTXController(Antenna.NORTH)
            ftx_south = FTXController(Antenna.SOUTH)

            def _set_attenuations(north: dict, south: dict) -> bool:
                ok = True
                try:
                    if not ftx_north.set_attenuation(Polarization.POL0, float(north.get(0, 0.0))):
                        ok = False
                    if not ftx_north.set_attenuation(Polarization.POL1, float(north.get(1, 0.0))):
                        ok = False
                    if not ftx_south.set_attenuation(Polarization.POL0, float(south.get(0, 0.0))):
                        ok = False
                    if not ftx_south.set_attenuation(Polarization.POL1, float(south.get(1, 0.0))):
                        ok = False
                except Exception:
                    ok = False
                return ok

            def _start_recording(obs_name: str, meta: dict) -> None:
                try:
                    self.recorder.set_observation_name(obs_name)
                    self.recorder.set_metadata(meta)
                    extra_metadata = self._gather_run_metadata(is_automatic_start=True)
                    extra_metadata.update(meta)
                    self.thr_rec_cmd.submit(
                        "start",
                        fftshift_p0=self.sp_fft_p0.value(),
                        fftshift_p1=self.sp_fft_p1.value(),
                        acclen=self.sp_acc.value(),
                        extra_metadata=extra_metadata,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"Error starting FFT shift recording: {exc}")

            def _stop_recording() -> None:
                self.thr_rec_cmd.submit("stop")

            def _start_point(idx: int, fmt: dict) -> None:
                self.thr_rec_cmd.submit("start_point_recording", source_idx=idx, format_dict=fmt)

            def _stop_point(idx: int) -> None:
                self.thr_rec_cmd.submit("stop_point_recording", source_idx=idx)

            def _set_fft(p0: int, p1: int) -> None:
                try:
                    self.recorder.set_fftshift(p0, p1)
                    self.logger.info(f"Set FFT shift: S={p0}, N={p1}")
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"Error setting FFT shift: {exc}")

            def _log(msg: str) -> None:
                self.logger.info(msg)

            self.fftshift_test_thread = threading.Thread(
                target=runner.run,
                args=(
                    fftshift_list,
                    current_attenuations,
                ),
                kwargs=dict(
                    set_fftshift=_set_fft,
                    start_recording=_start_recording,
                    stop_recording=_stop_recording,
                    start_point=_start_point,
                    stop_point=_stop_point,
                    set_attenuations=_set_attenuations,
                    log=_log,
                ),
                daemon=True,
            )
            self.fftshift_test_thread.start()
            self.logger.info("FFT shift test sequence started")
        except Exception as e:
            self.logger.error(f"Error starting FFT shift test: {e}")
    
    def _on_fftshift_test_stop(self):
        """Handle FFT shift test stop request."""
        if hasattr(self, "_fft_runner") and self._fft_runner:
            self._fft_runner.stop()
        self.logger.info("FFT shift test stop requested")
    
    def _on_fftshift_set(self, fftshift_p0: int, fftshift_p1: int):
        """Handle FFT shift set request."""
        try:
            # Delegate to widget which has backend configuration
            if hasattr(self, 'fftshift_widget'):
                self.fftshift_widget.set_fftshift(fftshift_p0, fftshift_p1)
            else:
                self.recorder.set_fftshift(fftshift_p0, fftshift_p1)
                self.logger.info(f"Set FFT shift: S={fftshift_p0}, N={fftshift_p1}")
        except Exception as e:
            self.logger.error(f"Error setting FFT shift: {e}")
    
    def _on_fftshift_recording_start(self, observation_name: str, metadata: dict):
        """Handle FFT shift recording start request."""
        try:
            if hasattr(self, 'fftshift_widget'):
                self.fftshift_widget.start_recording_with_name(observation_name, metadata)
            else:
                self.recorder.set_observation_name(observation_name)
                self.recorder.set_metadata(metadata)
                extra_metadata = self._gather_run_metadata(is_automatic_start=True)
                extra_metadata.update(metadata)
                self.thr_rec_cmd.submit(
                    "start",
                    fftshift_p0=self.sp_fft_p0.value(),
                    fftshift_p1=self.sp_fft_p1.value(),
                    acclen=self.sp_acc.value(),
                    extra_metadata=extra_metadata,
                )
                self.logger.info(f"Started FFT shift recording: {observation_name}")
        except Exception as e:
            self.logger.error(f"Error starting FFT shift recording: {e}")
    
    # Removed legacy inlined FFT shift runner methods in favor of service-based runner

    # ------------------------------------------------------------------
    def closeEvent(self, event):  # pylint: disable=invalid-name
        """Cleanup resources when closing the application."""
        # Stop any running FFT shift test
        if hasattr(self, "_fft_runner") and self._fft_runner:
            try:
                self._fft_runner.stop()
            except Exception:
                pass
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