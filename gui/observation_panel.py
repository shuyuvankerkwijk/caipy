from __future__ import annotations
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
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
    QRadioButton,
    QButtonGroup,
    QStackedWidget,
    QDialog,
    QDialogButtonBox,
    QDateTimeEdit,
)
from PyQt5.QtCore import QDateTime, QTimer, pyqtSignal
import pytz
from pytz import timezone

from tracking import Source

# -----------------------------------------------------------------------------
# ObservationPanel – unified observation control panel
# -----------------------------------------------------------------------------

observation_queue = []  # Global queue for scheduled/queued observations

class Observation:
    """
    Represents a queued or scheduled observation.
    Replaces the old observation dicts.
    """
    def __init__(self, ant: str, mode: str, params: dict, start: datetime = None, added_time: datetime = None):
        self.ant = ant
        self.mode = mode
        self.params = params
        self.start = start
        self.added_time = added_time or datetime.now()

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        return (self.ant == other.ant and self.mode == other.mode and self.params == other.params and
                self.start == other.start and self.added_time == other.added_time)

    def __repr__(self):
        return (f"Observation(ant={self.ant!r}, mode={self.mode!r}, params={self.params!r}, "
                f"start={self.start!r}, added_time={self.added_time!r})")

    def display_html(self, idx: int, running: bool = False, next_to_run: bool = False) -> str:
        """Return HTML for queue display."""
        params = self.params
        entry_html = ""
        if self.start:
            start_str = self.start.strftime('%Y-%m-%d %H:%M')
            if self.mode == "slew":
                entry_html += f"<b>{idx+1}. SCHEDULED:</b> SLEW {self.ant} @ {start_str}<br>"
                if 'ra' in params:
                    entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;RA={params['ra']:.2f}h Dec={params['dec']:.2f}°"
                else:
                    entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;Az={params['az']:.2f}° El={params['el']:.2f}°"
            else:
                entry_html += f"<b>{idx+1}. SCHEDULED:</b> {self.mode.upper()} {self.ant} @ {start_str}<br>"
                entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;RA={params['ra']:.2f}h Dec={params['dec']:.2f}°"
        else:
            if self.mode == "slew":
                entry_html += f"<b>{idx+1}. QUEUED:</b> SLEW {self.ant}<br>"
                if 'ra' in params:
                    entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;RA={params['ra']:.2f}h Dec={params['dec']:.2f}°"
                else:
                    entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;Az={params['az']:.2f}° El={params['el']:.2f}°"
            else:
                entry_html += f"<b>{idx+1}. QUEUED:</b> {self.mode.upper()} {self.ant}<br>"
                entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;RA={params['ra']:.2f}h Dec={params['dec']:.2f}°"
        # Highlight based on state
        if running:
            return f"<div style='background-color: #d4edda; padding: 2px; border-radius: 3px;'>{entry_html}</div><div style='height: 5px;'></div>"
        elif next_to_run:
            return f"<div style='background-color: #f8d7da; padding: 2px; border-radius: 3px;'>{entry_html}</div><div style='height: 5px;'></div>"
        else:
            return f"<div>{entry_html}</div><div style='height: 5px;'></div>"

class ObservationPanel(QWidget):
    # Signals to communicate with the main window
    observation_requested = pyqtSignal(object)  # Emits an Observation object
    command_requested = pyqtSignal(str, str) # Emits antenna and command (e.g., "park", "stop")
    log_message_requested = pyqtSignal(str) # Emits a string to be logged

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.queue_running = False  # Ensure this is always defined
        self._current_obs = None
        self._obs_remaining_ants = set()

        # Timer for periodic queue checks (for scheduled tasks)
        self.queue_check_timer = QTimer(self)
        self.queue_check_timer.timeout.connect(self._check_for_ready_observations)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        title = QLabel("Observation Panel")
        title.setStyleSheet("font-weight: bold; font-size: 16pt; color: #2c3e50;")
        layout.addWidget(title)

        # Antenna selection
        ant_group = QGroupBox("Antenna(s)")
        ant_layout = QHBoxLayout(ant_group)
        self.rb_north = QRadioButton("North")
        self.rb_south = QRadioButton("South")
        self.rb_both = QRadioButton("Both")
        self.rb_both.setChecked(True)
        self.ant_btn_group = QButtonGroup(self)
        self.ant_btn_group.addButton(self.rb_north)
        self.ant_btn_group.addButton(self.rb_south)
        self.ant_btn_group.addButton(self.rb_both)
        ant_layout.addWidget(self.rb_north)
        ant_layout.addWidget(self.rb_south)
        ant_layout.addWidget(self.rb_both)
        ant_layout.addStretch()
        layout.addWidget(ant_group)

        # Mode selection
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_group)
        self.rb_track = QRadioButton("TRACK")
        self.rb_slew = QRadioButton("SLEW")
        self.rb_rasta = QRadioButton("RASTA")
        self.rb_point = QRadioButton("OFFSETS")
        self.rb_track.setChecked(True)
        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.addButton(self.rb_track)
        self.mode_btn_group.addButton(self.rb_slew)
        self.mode_btn_group.addButton(self.rb_rasta)
        self.mode_btn_group.addButton(self.rb_point)
        mode_layout.addWidget(self.rb_track)
        mode_layout.addWidget(self.rb_slew)
        mode_layout.addWidget(self.rb_rasta)
        mode_layout.addWidget(self.rb_point)
        mode_layout.addStretch()
        layout.addWidget(mode_group)

        # Stacked widget for mode-specific parameters
        self.stacked = QStackedWidget()
        
        # TRACK parameters
        self.page_track = QWidget()
        track_layout = QVBoxLayout(self.page_track)
        track_params = QHBoxLayout()
        self.sp_track_ra = QDoubleSpinBox(); self.sp_track_ra.setRange(0, 24); self.sp_track_ra.setDecimals(4); self.sp_track_ra.setValue(12.0)
        self.sp_track_dec = QDoubleSpinBox(); self.sp_track_dec.setRange(-90, 90); self.sp_track_dec.setDecimals(4); self.sp_track_dec.setValue(45.0)
        self.sp_track_len = QDoubleSpinBox(); self.sp_track_len.setRange(0.0, 24.0); self.sp_track_len.setDecimals(4); self.sp_track_len.setValue(1.0)
        track_params.addWidget(QLabel("RA (h):")); track_params.addWidget(self.sp_track_ra)
        track_params.addWidget(QLabel("Dec (°):")); track_params.addWidget(self.sp_track_dec)
        track_params.addWidget(QLabel("Len (h):")); track_params.addWidget(self.sp_track_len)
        track_layout.addLayout(track_params)
        
        # TRACK options on new line
        track_options = QHBoxLayout()
        self.chk_track_no_slew = QCheckBox("No Slew")
        self.chk_track_no_park = QCheckBox("No Park")
        self.chk_track_record = QCheckBox("Record")
        self.chk_track_record.setChecked(True)
        track_options.addWidget(self.chk_track_no_slew)
        track_options.addWidget(self.chk_track_no_park)
        track_options.addWidget(self.chk_track_record)
        track_options.addStretch()
        track_layout.addLayout(track_options)
        track_layout.addStretch()
        
        # SLEW parameters
        self.page_slew = QWidget()
        slew_layout = QVBoxLayout(self.page_slew)
        
        # Coordinate type selection
        slew_coord_type = QHBoxLayout()
        self.rb_slew_radec = QRadioButton("RA/Dec")
        self.rb_slew_azel = QRadioButton("Az/El")
        self.rb_slew_radec.setChecked(True)
        slew_coord_group = QButtonGroup(self)
        slew_coord_group.addButton(self.rb_slew_radec)
        slew_coord_group.addButton(self.rb_slew_azel)
        slew_coord_type.addWidget(QLabel("Coordinates:"))
        slew_coord_type.addWidget(self.rb_slew_radec)
        slew_coord_type.addWidget(self.rb_slew_azel)
        slew_coord_type.addStretch()
        slew_layout.addLayout(slew_coord_type)
        
        # RA/Dec parameters
        self.slew_radec_widget = QWidget()
        slew_radec_layout = QHBoxLayout(self.slew_radec_widget)
        self.sp_slew_ra = QDoubleSpinBox(); self.sp_slew_ra.setRange(0, 24); self.sp_slew_ra.setDecimals(4); self.sp_slew_ra.setValue(12.0)
        self.sp_slew_dec = QDoubleSpinBox(); self.sp_slew_dec.setRange(-90, 90); self.sp_slew_dec.setDecimals(4); self.sp_slew_dec.setValue(45.0)
        slew_radec_layout.addWidget(QLabel("RA (h):")); slew_radec_layout.addWidget(self.sp_slew_ra)
        slew_radec_layout.addWidget(QLabel("Dec (°):")); slew_radec_layout.addWidget(self.sp_slew_dec)
        slew_radec_layout.addStretch()
        slew_layout.addWidget(self.slew_radec_widget)
        
        # Az/El parameters
        self.slew_azel_widget = QWidget()
        slew_azel_layout = QHBoxLayout(self.slew_azel_widget)
        self.sp_slew_az = QDoubleSpinBox(); self.sp_slew_az.setRange(0, 360); self.sp_slew_az.setDecimals(2); self.sp_slew_az.setValue(180.0)
        self.sp_slew_el = QDoubleSpinBox(); self.sp_slew_el.setRange(0, 90); self.sp_slew_el.setDecimals(2); self.sp_slew_el.setValue(45.0)
        slew_azel_layout.addWidget(QLabel("Az (°):")); slew_azel_layout.addWidget(self.sp_slew_az)
        slew_azel_layout.addWidget(QLabel("El (°):")); slew_azel_layout.addWidget(self.sp_slew_el)
        slew_azel_layout.addStretch()
        slew_layout.addWidget(self.slew_azel_widget)
        
        # Initially show RA/Dec parameters
        self.slew_azel_widget.setVisible(False)
        
        slew_layout.addStretch()
        
        # RASTA parameters
        self.page_rasta = QWidget()
        rasta_layout = QVBoxLayout(self.page_rasta)
        rasta_params = QHBoxLayout()
        self.sp_rasta_ra = QDoubleSpinBox(); self.sp_rasta_ra.setRange(0, 24); self.sp_rasta_ra.setDecimals(4); self.sp_rasta_ra.setValue(12.0)
        self.sp_rasta_dec = QDoubleSpinBox(); self.sp_rasta_dec.setRange(-90, 90); self.sp_rasta_dec.setDecimals(4); self.sp_rasta_dec.setValue(45.0)
        self.sp_rasta_len = QDoubleSpinBox(); self.sp_rasta_len.setRange(0.0, 24.0); self.sp_rasta_len.setDecimals(4); self.sp_rasta_len.setValue(1.0)
        rasta_params.addWidget(QLabel("RA (h):")); rasta_params.addWidget(self.sp_rasta_ra)
        rasta_params.addWidget(QLabel("Dec (°):")); rasta_params.addWidget(self.sp_rasta_dec)
        rasta_params.addWidget(QLabel("Length (h):")); rasta_params.addWidget(self.sp_rasta_len)
        rasta_layout.addLayout(rasta_params)
        
        # RASTA extra parameters
        rasta_extra = QHBoxLayout()
        self.sp_rasta_max = QDoubleSpinBox(); self.sp_rasta_max.setRange(0.1, 50.0); self.sp_rasta_max.setDecimals(2); self.sp_rasta_max.setValue(10.0)
        self.sp_rasta_step = QDoubleSpinBox(); self.sp_rasta_step.setRange(0.1, 10.0); self.sp_rasta_step.setDecimals(2); self.sp_rasta_step.setValue(0.5)
        self.sp_rasta_angle = QDoubleSpinBox(); self.sp_rasta_angle.setRange(0, 360); self.sp_rasta_angle.setDecimals(1); self.sp_rasta_angle.setValue(0.0)
        rasta_extra.addWidget(QLabel("Max Offset (°):")); rasta_extra.addWidget(self.sp_rasta_max)
        rasta_extra.addWidget(QLabel("Step (°):")); rasta_extra.addWidget(self.sp_rasta_step)
        rasta_extra.addWidget(QLabel("Angle (°):")); rasta_extra.addWidget(self.sp_rasta_angle)
        rasta_layout.addLayout(rasta_extra)

        # RASTA options on new line
        rasta_options = QHBoxLayout()
        self.chk_rasta_no_slew = QCheckBox("No Slew")
        self.chk_rasta_no_park = QCheckBox("No Park")
        self.chk_rasta_record = QCheckBox("Record")
        self.chk_rasta_record.setChecked(True)
        self.chk_rasta_split = QCheckBox("Other antenna tracks RA/Dec for same duration")
        self.chk_rasta_split.setVisible(False)
        rasta_options.addWidget(self.chk_rasta_no_slew)
        rasta_options.addWidget(self.chk_rasta_no_park)
        rasta_options.addWidget(self.chk_rasta_record)
        rasta_options.addWidget(self.chk_rasta_split)
        rasta_options.addStretch()
        rasta_layout.addLayout(rasta_options)
        
        rasta_layout.addStretch()
        
        # POINTING SCAN parameters
        self.page_point = QWidget()
        point_layout = QVBoxLayout(self.page_point)
        point_params = QHBoxLayout()
        self.sp_point_ra = QDoubleSpinBox(); self.sp_point_ra.setRange(0, 24); self.sp_point_ra.setDecimals(4); self.sp_point_ra.setValue(12.0)
        self.sp_point_dec = QDoubleSpinBox(); self.sp_point_dec.setRange(-90, 90); self.sp_point_dec.setDecimals(4); self.sp_point_dec.setValue(45.0)
        self.sp_point_len = QDoubleSpinBox(); self.sp_point_len.setRange(0.0, 24.0); self.sp_point_len.setDecimals(4); self.sp_point_len.setValue(1.0)
        point_params.addWidget(QLabel("RA (h):")); point_params.addWidget(self.sp_point_ra)
        point_params.addWidget(QLabel("Dec (°):")); point_params.addWidget(self.sp_point_dec)
        point_params.addWidget(QLabel("Length (h):")); point_params.addWidget(self.sp_point_len)
        point_layout.addLayout(point_params)
        
        # POINTING extra parameters
        point_extra = QHBoxLayout()
        self.sp_point_dist = QDoubleSpinBox(); self.sp_point_dist.setRange(0.01, 20.0); self.sp_point_dist.setDecimals(2); self.sp_point_dist.setValue(0.5)
        self.sp_point_npts = QSpinBox(); self.sp_point_npts.setRange(5, 13); self.sp_point_npts.setSingleStep(2); self.sp_point_npts.setValue(5)
        point_extra.addWidget(QLabel("Dist (°):")); point_extra.addWidget(self.sp_point_dist)
        point_extra.addWidget(QLabel("Npts (5,7,9,13):")); point_extra.addWidget(self.sp_point_npts)
        point_layout.addLayout(point_extra)

        # POINTING options on new line
        point_options = QHBoxLayout()
        self.chk_point_no_slew = QCheckBox("No Slew")
        self.chk_point_no_park = QCheckBox("No Park")
        self.chk_point_record = QCheckBox("Record")
        self.chk_point_record.setChecked(True)
        self.chk_point_split = QCheckBox("Other antenna tracks RA/Dec for same duration")
        self.chk_point_split.setVisible(False)
        point_options.addWidget(self.chk_point_no_slew)
        point_options.addWidget(self.chk_point_no_park)
        point_options.addWidget(self.chk_point_record)
        point_options.addWidget(self.chk_point_split)
        point_options.addStretch()
        point_layout.addLayout(point_options)
        
        point_layout.addStretch()
        
        # Add to stacked
        self.stacked.addWidget(self.page_track)
        self.stacked.addWidget(self.page_slew)
        self.stacked.addWidget(self.page_rasta)
        self.stacked.addWidget(self.page_point)
        layout.addWidget(QLabel("Mode Parameters"))
        layout.addWidget(self.stacked)

        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("\u25B6 RUN NOW")
        self.btn_run.setStyleSheet("background-color: #28a745;")
        self.btn_queue = QPushButton("＋ QUEUE")
        self.btn_sched = QPushButton("＋ SCHEDULE")
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_queue)
        btn_row.addWidget(self.btn_sched)
        layout.addLayout(btn_row)

        # Park/Stop buttons placed above the queue widget
        btn_row2 = QHBoxLayout()
        self.btn_park = QPushButton("PARK")
        self.btn_park.setStyleSheet("background-color: #ffc107; color: #333;")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet("background-color: #dc3545;")
        btn_row2.addWidget(self.btn_park)
        btn_row2.addWidget(self.btn_stop)
        layout.addLayout(btn_row2)
        
        # Queue and Schedule Display
        queue_box = QGroupBox("Queue")
        queue_layout = QVBoxLayout(queue_box)
        
        # Queue list
        self.queue_list = QTextEdit()
        self.queue_list.setReadOnly(True)
        self.queue_list.setMaximumHeight(200)  # Increased height for better visibility
        self.queue_list.setPlaceholderText("No queued or scheduled observations")
        queue_layout.addWidget(self.queue_list)

        # Queue controls
        queue_controls = QHBoxLayout()
        self.btn_clear_queue = QPushButton("Clear Queue")
        self.btn_run_next = QPushButton("Run Next")
        # New buttons for automatic queue execution
        self.btn_run_queue = QPushButton("Run Queue")
        self.btn_stop_queue = QPushButton("Stop Queue")
        queue_controls.addWidget(self.btn_clear_queue)
        queue_controls.addWidget(self.btn_run_next)
        queue_controls.addWidget(self.btn_run_queue)
        queue_controls.addWidget(self.btn_stop_queue)
        queue_controls.addStretch()
        queue_layout.addLayout(queue_controls)
        
        layout.addWidget(queue_box)

        layout.addStretch()

        # Connect mode switching
        self.rb_track.toggled.connect(self._on_mode_changed)
        self.rb_slew.toggled.connect(self._on_mode_changed)
        self.rb_rasta.toggled.connect(self._on_mode_changed)
        self.rb_point.toggled.connect(self._on_mode_changed)
        # Connect antenna selection to mode changed for split checkbox visibility
        self.rb_north.toggled.connect(self._on_mode_changed)
        self.rb_south.toggled.connect(self._on_mode_changed)
        self.rb_both.toggled.connect(self._on_mode_changed)
        
        # Connect slew coordinate selection
        self.rb_slew_radec.toggled.connect(self._on_slew_coord_changed)
        self.rb_slew_azel.toggled.connect(self._on_slew_coord_changed)
        
        self._on_mode_changed()  # Set initial page

        # Connect actions
        self.btn_run.clicked.connect(self._run_now)
        self.btn_queue.clicked.connect(self._add_to_queue)
        self.btn_sched.clicked.connect(self._schedule_obs)
        self.btn_park.clicked.connect(self._do_park)
        self.btn_stop.clicked.connect(self._do_stop)
        self.btn_clear_queue.clicked.connect(self._clear_queue)
        self.btn_run_next.clicked.connect(self._run_next)
        self.btn_run_queue.clicked.connect(self._run_queue)
        self.btn_stop_queue.clicked.connect(self._stop_queue)

    def _on_mode_changed(self):
        if self.rb_track.isChecked():
            self.stacked.setCurrentIndex(0)
        elif self.rb_slew.isChecked():
            self.stacked.setCurrentIndex(1)
        elif self.rb_rasta.isChecked():
            self.stacked.setCurrentIndex(2)
        elif self.rb_point.isChecked():
            self.stacked.setCurrentIndex(3)
            
        # --- Split checkbox visibility logic ---
        ant = self._get_selected_ant()
        if self.rb_rasta.isChecked() and ant != "both":
            self.chk_rasta_split.setVisible(True)
        else:
            self.chk_rasta_split.setVisible(False)
        if self.rb_point.isChecked() and ant != "both":
            self.chk_point_split.setVisible(True)
        else:
            self.chk_point_split.setVisible(False)
    
    def _on_slew_coord_changed(self):
        """Handle slew coordinate type selection."""
        if self.rb_slew_radec.isChecked():
            self.slew_radec_widget.setVisible(True)
            self.slew_azel_widget.setVisible(False)
        else:
            self.slew_radec_widget.setVisible(False)
            self.slew_azel_widget.setVisible(True)

    def _get_selected_ant(self):
        if self.rb_north.isChecked():
            return "N"
        elif self.rb_south.isChecked():
            return "S"
        else:
            return "both"

    def _get_selected_mode(self):
        if self.rb_track.isChecked():
            return "track"
        elif self.rb_slew.isChecked():
            return "slew"
        elif self.rb_rasta.isChecked():
            return "rasta"
        else:
            return "pointing_scan"

    def _collect_params(self):
        mode = self._get_selected_mode()
        params = {}
        
        if mode == "track":
            params = {
                "ra": self.sp_track_ra.value(),
                "dec": self.sp_track_dec.value(),
                "duration_hours": self.sp_track_len.value(),
                "slew": not self.chk_track_no_slew.isChecked(),
                "park": not self.chk_track_no_park.isChecked(),
                "record": self.chk_track_record.isChecked(),
            }
        elif mode == "slew":
            if self.rb_slew_radec.isChecked():
                params = {
                    "ra": self.sp_slew_ra.value(),  # Right Ascension (hours)
                    "dec": self.sp_slew_dec.value(), # Declination (deg)
                    "duration_hours": 0.1,
                }
            else:
                params = {
                    "az": self.sp_slew_az.value(),
                    "el": self.sp_slew_el.value(),
                    "duration_hours": 0.1,
                }
        elif mode == "rasta":
            params = {
                "ra": self.sp_rasta_ra.value(),
                "dec": self.sp_rasta_dec.value(),
                "duration_hours": self.sp_rasta_len.value(),
                "slew": not self.chk_rasta_no_slew.isChecked(),
                "park": not self.chk_rasta_no_park.isChecked(),
                "max_dist_deg": self.sp_rasta_max.value(),
                "step_deg": self.sp_rasta_step.value(),
                "position_angle_deg": self.sp_rasta_angle.value(),
                "record": self.chk_rasta_record.isChecked(),
                "split_track": self.chk_rasta_split.isChecked(),
            }
        elif mode == "pointing_scan":
            params = {
                "ra": self.sp_point_ra.value(),
                "dec": self.sp_point_dec.value(),
                "duration_hours": self.sp_point_len.value(),
                "slew": not self.chk_point_no_slew.isChecked(),
                "park": not self.chk_point_no_park.isChecked(),
                "closest_dist_deg": self.sp_point_dist.value(),
                "number_of_points": self.sp_point_npts.value(),
                "record": self.chk_point_record.isChecked(),
                "split_track": self.chk_point_split.isChecked(),
            }
        return params

    def _run_now(self):
        ant = self._get_selected_ant()
        mode = self._get_selected_mode()
        params = self._collect_params()
        obs = Observation(ant, mode, params)
        self.observation_requested.emit(obs)
        
        # Log the command once for both antennas
        if ant == "both":
            if mode == "track":
                self._log(f"RUN NOW: TRACK BOTH RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
            elif mode == "slew":
                if 'ra' in params:
                    self._log(f"RUN NOW: SLEW BOTH RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
                else:
                    self._log(f"RUN NOW: SLEW BOTH Az={params['az']:.2f}° El={params['el']:.2f}°")
            elif mode == "rasta":
                self._log(f"RUN NOW: RASTA BOTH RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
            elif mode == "pointing_scan":
                self._log(f"RUN NOW: POINTING BOTH RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
        else:
            if mode == "track":
                self._log(f"RUN NOW: TRACK {ant} RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
            elif mode == "slew":
                if 'ra' in params:
                    self._log(f"RUN NOW: SLEW {ant} RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
                else:
                    self._log(f"RUN NOW: SLEW {ant} Az={params['az']:.2f}° El={params['el']:.2f}°")
            elif mode == "rasta":
                self._log(f"RUN NOW: RASTA {ant} RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
            elif mode == "pointing_scan":
                self._log(f"RUN NOW: POINTING {ant} RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")

    def _add_to_queue(self):
        ant = self._get_selected_ant()
        mode = self._get_selected_mode()
        params = self._collect_params()
        entry = Observation(ant, mode, params)
        observation_queue.append(entry)
        print(f"DEBUG: Added to queue, queue now: {observation_queue}")
        # Log correct parameters for each mode
        if mode == "slew":
            if 'ra' in params:
                self._log(f"Added to queue: SLEW {ant} RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
            else:
                self._log(f"Added to queue: SLEW {ant} Az={params['az']:.2f}° El={params['el']:.2f}°")
        else:
            self._log(f"Added to queue: {mode.upper()} {ant} RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
        self._update_queue_display()

    def _schedule_obs(self):
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Schedule Observation")
        v = QVBoxLayout(dlg)
        
        # Create datetime edit with California timezone
        california_tz = pytz.timezone('America/Los_Angeles')
        # Use timezone-aware datetime object for current time
        current_ca_time = datetime.now(california_tz)
        dt_edit = QDateTimeEdit(current_ca_time)
        dt_edit.setCalendarPopup(True)
        dt_edit.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        
        v.addWidget(QLabel("Start time (California):"))
        v.addWidget(dt_edit)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        v.addWidget(btns)
        
        def accept():
            ant = self._get_selected_ant()
            mode = self._get_selected_mode()
            params = self._collect_params()
            
            # Get the selected datetime and interpret it as California time
            selected_dt = dt_edit.dateTime().toPyDateTime()
            start = california_tz.localize(selected_dt)
            
            entry = Observation(ant, mode, params, start)
            observation_queue.append(entry)
            self._log(f"Scheduled: {mode.upper()} {ant} @ {start.strftime('%Y-%m-%d %H:%M')} (California time)")
            self._update_queue_display()
            dlg.accept()
        
        btns.accepted.connect(accept)
        btns.rejected.connect(dlg.reject)
        dlg.setModal(False)
        dlg.show()

    def _do_park(self):
        self._do_stop()
        ant = self._get_selected_ant()
        self.command_requested.emit(ant, "park")
        self._log(f"PARK {ant}")

    def _do_stop(self):
        ant = self._get_selected_ant()
        self.command_requested.emit(ant, "stop")
        self._log(f"STOP {ant}")

    def _log(self, msg):
        # Write a one-line message into the Tracking Log area
        self.log_message_requested.emit(msg)
    
    def update_antenna_position(self, antenna: str, az: float, el: float):
        """Update antenna position display."""
        if antenna == "N":
            self.lbl_north_pos.setText(f"Position: Az {az:.2f}°, El {el:.2f}°")
        elif antenna == "S":
            self.lbl_south_pos.setText(f"Position: Az {az:.2f}°, El {el:.2f}°")
    
    def update_antenna_target(self, antenna: str, az: float, el: float):
        """Update target position display."""
        if antenna == "N":
            self.lbl_north_target.setText(f"Target: Az {az:.2f}°, El {el:.2f}°")
        elif antenna == "S":
            self.lbl_south_target.setText(f"Target: Az {az:.2f}°, El {el:.2f}°")
    
    def _update_queue_display(self):
        """Update the queue display."""
        if not observation_queue:
            self.queue_list.setHtml("No queued or scheduled observations")
            return
        
        # Sort queue: scheduled items first (by start time), then queued items (by added time)
        sorted_queue = sorted(observation_queue, key=lambda x: (x.start is None, x.start or x.added_time))
        
        # Identify the next item to run if the queue is stopped
        next_to_run_obs = None
        if not self.queue_running and sorted_queue:
            next_to_run_obs = sorted_queue[0]

        html_text = ""
        for i, entry in enumerate(sorted_queue):
            mode = entry.mode
            params = entry.params
            
            entry_html = ""
            if entry.start:
                start_str = entry.start.strftime('%Y-%m-%d %H:%M')
                if mode == "slew":
                    entry_html += f"<b>{i+1}. SCHEDULED:</b> SLEW {entry.ant} @ {start_str}<br>"
                    if 'ra' in params:
                        entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;RA={params['ra']:.2f}h Dec={params['dec']:.2f}°"
                    else:
                        entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;Az={params['az']:.2f}° El={params['el']:.2f}°"
                else:
                    entry_html += f"<b>{i+1}. SCHEDULED:</b> {mode.upper()} {entry.ant} @ {start_str}<br>"
                    entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;RA={params['ra']:.2f}h Dec={params['dec']:.2f}°"
            else:
                if mode == "slew":
                    entry_html += f"<b>{i+1}. QUEUED:</b> SLEW {entry.ant}<br>"
                    if 'ra' in params:
                        entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;RA={params['ra']:.2f}h Dec={params['dec']:.2f}°"
                    else:
                        entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;Az={params['az']:.2f}° El={params['el']:.2f}°"
                else:
                    entry_html += f"<b>{i+1}. QUEUED:</b> {mode.upper()} {entry.ant}<br>"
                    entry_html += f"&nbsp;&nbsp;&nbsp;&nbsp;RA={params['ra']:.2f}h Dec={params['dec']:.2f}°"

            # Highlight based on state
            is_running = self._current_obs and self._current_obs.added_time == entry.added_time
            is_next_to_run = next_to_run_obs and next_to_run_obs.added_time == entry.added_time

            html_text += entry.display_html(i, is_running, is_next_to_run)
            
        self.queue_list.setHtml(html_text)
    
    def _clear_queue(self):
        """Clear all queued and scheduled observations."""
        observation_queue.clear()
        self._stop_queue() # Also stop processing
        self._update_queue_display()
        self._log("Queue cleared")
    
    def _run_next(self):
        """Run the next observation from the queue (single step)."""
        if not observation_queue:
            self._log("No observations in queue")
            return
        
        # Use the same helper as automatic runner so logic stays identical
        ran = self._start_next_in_queue(single_step=True)
        if not ran:
            self._log("No observation ready to run (may be waiting for scheduled time or conflict)")
        self._update_queue_display()
    
    def _can_run_queued_observation(self, obs: Observation):
        """Check if a queued observation can run without conflicting with scheduled observations."""

        if obs.start:
            return True  # function only used for queued observations

        duration_hours = obs.params.get('duration_hours', 0.1)
        # Use timezone-aware object for all datetime comparisons
        tz_la = timezone('America/Los_Angeles')
        now_la = datetime.now(tz_la)
        # Ensure end time is also timezone-aware
        queued_end = now_la + timedelta(hours=duration_hours)

        for scheduled in observation_queue:
            if scheduled.start:
                # Estimate scheduled observation end
                sched_dur = scheduled.params.get('duration_hours', 0.1)
                sched_end = scheduled.start + timedelta(hours=sched_dur)
                # Overlap test
                if scheduled.start < queued_end and sched_end > now_la:
                    return False
        return True

    def _run_observation_from_queue(self, obs: Observation):
        self.observation_requested.emit(obs)
        
        # Log the command
        ant = obs.ant
        mode = obs.mode
        params = obs.params
        if mode == "slew":
            if obs.start:
                self._log(f"RUNNING SCHEDULED: SLEW {ant} Az={params['az']:.2f}° El={params['el']:.2f}°")
            else:
                self._log(f"RUNNING QUEUED: SLEW {ant} Az={params['az']:.2f}° El={params['el']:.2f}°")
        else:
            if obs.start:
                self._log(f"RUNNING SCHEDULED: {mode.upper()} {ant} RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
            else:
                self._log(f"RUNNING QUEUED: {mode.upper()} {ant} RA={params['ra']:.2f}h Dec={params['dec']:.2f}°")
        
    # ------------------------------------------------------------------
    # Automatic queue runner helpers (inside ObservationPanel)
    def _check_for_ready_observations(self):
        """Periodically called by a timer to check for scheduled tasks."""
        if not self.queue_running:
            return
        
        # Don't start a new task if one is already running
        if self._current_obs:
            return
        
        # If we are here, queue is running but idle, so check for next task
        self._start_next_in_queue()

    def _run_queue(self):
        """Begin running the entire queue sequentially."""
        if self.queue_running:
            return  # already running
        self.queue_running = True
        self.queue_check_timer.start(1000) # Check every second
        self._log("Queue processing started")
        self._update_queue_display()
        self._check_for_ready_observations()

    def _stop_queue(self):
        """Stop before the next observation starts."""
        if not self.queue_running and not self.queue_check_timer.isActive():
            return
            
        self.queue_running = False
        self.queue_check_timer.stop()
        
        if self._current_obs:
            self._log("Queue processing will stop after current observation.")
        else:
            self._log("Queue processing stopped.")
        self._update_queue_display()

    def _start_next_in_queue(self, *, single_step: bool = False):
        """Launch the next ready observation. If *single_step* is True, run only one.
        When automatic mode is active this function calls itself again when the
        running observation finishes."""

        # Check that both antennas are idle before starting next observation
        mainwin = self.window() if hasattr(self, 'window') else None
        if mainwin is not None:
            north_idle = "Idle" in mainwin.lbl_north_status.text()
            south_idle = "Idle" in mainwin.lbl_south_status.text()
            if not (north_idle and south_idle):
                print("DEBUG: Not starting next queue item, antennas not idle.")
                if self.queue_running and not self._current_obs:
                    self._log("Waiting for both antennas to be idle before starting next observation...")
                return False

        print(f"DEBUG: _start_next_in_queue called, queue: {observation_queue}")
        if not observation_queue:
            if self.queue_running:
                self._log("Queue empty — nothing more to run")
                self.queue_running = False
            return False

        # Sort: scheduled first by start time, then queued by added time
        sorted_queue = sorted(
            observation_queue,
            key=lambda x: (x.start is None, x.start or x.added_time)
        )

        tz_la = timezone('America/Los_Angeles')
        current_time = datetime.now(tz_la)
        for next_obs in sorted_queue:
            ready = True
            if next_obs.start:
                if current_time < next_obs.start:
                    ready = False
            else:
                ready = self._can_run_queued_observation(next_obs)
            print(f"DEBUG: Considering obs: {next_obs}, ready={ready}")
            if ready:
                # Set as current observation but DO NOT remove from queue yet
                self._current_obs = next_obs
                
                # Update display to highlight the running task
                self._update_queue_display()
                
                self._run_observation_from_queue(next_obs)
                ants = ["N", "S"] if next_obs.ant == "both" else [next_obs.ant]
                self._obs_remaining_ants = set(ants)
                if not self._obs_remaining_ants:
                    self._on_observation_finished()
                if single_step:
                    return True
                return True
        print("DEBUG: No ready observation found in queue.")
        if self.queue_running:
            if not self._current_obs: # only show waiting if truly idle
                self._log("No observation ready to run, waiting...")
        self._update_queue_display()
        return False

    def _on_tracker_completion(self, ant: str):
        """Called when an individual antenna signals completion."""
        print(f"DEBUG: Tracker completion received for {ant}. Remaining: {getattr(self, '_obs_remaining_ants', None)}")
        if self._current_obs is None:
            return
        if hasattr(self, '_obs_remaining_ants'):
            self._obs_remaining_ants.discard(ant)
            if not self._obs_remaining_ants:
                print("DEBUG: All antennas complete, starting next queue item if running.")
                self._on_observation_finished()

    def _on_observation_finished(self):
        """Current multi-antenna observation finished."""
        # Remove the completed observation from the queue
        finished_obs = self._current_obs
        if finished_obs:
            try:
                observation_queue.remove(finished_obs)
            except ValueError:
                # This can happen if queue was cleared manually
                print(f"DEBUG: Could not find finished observation in queue to remove: {finished_obs}")
        
        self._current_obs = None
        
        # Update display now that task is removed
        self._update_queue_display()
        
        if self.queue_running:
            # Immediately check for the next item without waiting for the timer
            self._start_next_in_queue()
        else:
            print("DEBUG: Queue stopped, not starting next observation.")
            self._log("Queue processing stopped.")
            self._update_queue_display() 