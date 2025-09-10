#!/usr/bin/env python3
"""
Coordinate Calculator Widget

A modular widget for converting RA/Dec to Az/El using the astro_pointing class.
Supports antenna selection (N/S) and optional datetime input in PDT.
"""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QDateTimeEdit,
    QCheckBox,
    QTextEdit,
    QRadioButton,
    QButtonGroup,
    QComboBox,
)
from PyQt5.QtCore import QDateTime, QDate, QTime, pyqtSignal
from datetime import datetime
from pytz import timezone as pytz_timezone

from tracking import Source
from tracking.utils.antenna import Antenna
from tracking.core.astro_pointing import AstroPointing


class CoordinateCalculatorWidget(QWidget):
    """Coordinate calculator widget for RA/Dec to Az/El conversion."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.astro_pointing = AstroPointing()
        self._build_ui()
        self._connect_signals()
    
    def _build_ui(self):
        """Build the user interface."""
        layout = QVBoxLayout(self)
        
        # Main group box
        box = QGroupBox("Coordinate Calculator")
        box_layout = QVBoxLayout(box)
        
        # Input row for RA/Dec
        input_row = QHBoxLayout()
        self.coord_ra = QDoubleSpinBox()
        self.coord_ra.setRange(0, 24)
        self.coord_ra.setDecimals(4)
        self.coord_ra.setValue(12.0)
        self.coord_dec = QDoubleSpinBox()
        self.coord_dec.setRange(-90, 90)
        self.coord_dec.setDecimals(4)
        self.coord_dec.setValue(45.0)
        
        input_row.addWidget(QLabel("RA (h):"))
        input_row.addWidget(self.coord_ra)
        input_row.addWidget(QLabel("Dec (°):"))
        input_row.addWidget(self.coord_dec)
        box_layout.addLayout(input_row)
        
        # More options dropdown
        more_options_row = QHBoxLayout()
        more_options_row.addWidget(QLabel("Options:"))
        self.combo_options = QComboBox()
        self.combo_options.addItems(["Basic", "Advanced"])
        self.combo_options.setCurrentText("Basic")
        more_options_row.addWidget(self.combo_options)
        more_options_row.addStretch()
        box_layout.addLayout(more_options_row)
        
        # Advanced options container (hidden by default)
        self.advanced_options_widget = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_options_widget)
        
        # Antenna selection row
        antenna_row = QHBoxLayout()
        antenna_row.addWidget(QLabel("Antenna:"))
        self.antenna_group = QButtonGroup()
        self.rb_north = QRadioButton("North")
        self.rb_south = QRadioButton("South")
        self.rb_north.setChecked(True)
        self.antenna_group.addButton(self.rb_north)
        self.antenna_group.addButton(self.rb_south)
        antenna_row.addWidget(self.rb_north)
        antenna_row.addWidget(self.rb_south)
        antenna_row.addStretch()
        advanced_layout.addLayout(antenna_row)
        
        # Optional time row (PDT)
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Time (PDT):"))
        self.coord_time = QDateTimeEdit()
        self.coord_time.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.coord_time.setCalendarPopup(True)
        # Initialize to current time in PDT for display
        _pdt_now = datetime.now(pytz_timezone('America/Los_Angeles'))
        self.coord_time.setDateTime(
            QDateTime(QDate(_pdt_now.year, _pdt_now.month, _pdt_now.day), QTime(_pdt_now.hour, _pdt_now.minute))
        )
        self.coord_use_now = QCheckBox("Now")
        self.coord_use_now.setChecked(True)
        # Disable time input unless custom time is desired
        self.coord_time.setEnabled(False)
        time_row.addWidget(self.coord_time)
        time_row.addWidget(self.coord_use_now)
        advanced_layout.addLayout(time_row)
        
        # Options row
        options_row = QHBoxLayout()
        self.chk_apply_corrections = QCheckBox("Corrections")
        self.chk_apply_corrections.setChecked(True)
        self.chk_apply_pointing_model = QCheckBox("Pointing model")
        self.chk_apply_pointing_model.setChecked(True)
        self.chk_clip_elevation = QCheckBox("Clip")
        self.chk_clip_elevation.setChecked(True)
        options_row.addWidget(self.chk_apply_corrections)
        options_row.addWidget(self.chk_apply_pointing_model)
        options_row.addWidget(self.chk_clip_elevation)
        options_row.addStretch()
        advanced_layout.addLayout(options_row)
        
        # Add advanced options to main layout and hide by default
        box_layout.addWidget(self.advanced_options_widget)
        self.advanced_options_widget.setVisible(False)
        
        # Buttons row
        button_row = QHBoxLayout()
        self.btn_get_azel = QPushButton("Get Az/El")
        self.btn_get_azel.setStyleSheet("background-color: #C25B4B; border: 1px solid #A3473A; color: #FFFFFF;")
        self.btn_clear_results = QPushButton("Clear")
        self.btn_clear_results.setStyleSheet("background-color: #C25B4B; border: 1px solid #A3473A; color: #FFFFFF;")
        
        button_row.addWidget(self.btn_get_azel)
        button_row.addWidget(self.btn_clear_results)
        box_layout.addLayout(button_row)
        
        # Results display
        self.txt_coord_results = QTextEdit()
        self.txt_coord_results.setReadOnly(True)
        self.txt_coord_results.setMaximumHeight(150)
        box_layout.addWidget(self.txt_coord_results)
        
        layout.addWidget(box)
    
    def _connect_signals(self):
        """Connect widget signals to slots."""
        self.btn_get_azel.clicked.connect(self._calculate_azel)
        self.btn_clear_results.clicked.connect(self._clear_coord_results)
        self.coord_use_now.toggled.connect(self._toggle_time_enabled)
        self.combo_options.currentTextChanged.connect(self._toggle_advanced_options)
    
    def _toggle_time_enabled(self, checked: bool):
        """Enable/disable time input based on 'Now' checkbox."""
        self.coord_time.setEnabled(not checked)
    
    def _toggle_advanced_options(self, option_text: str):
        """Show/hide advanced options based on dropdown selection."""
        self.advanced_options_widget.setVisible(option_text == "Advanced")
    
    def _calculate_azel(self):
        """Calculate Az/El for the given RA/Dec and display results."""
        try:
            ra = self.coord_ra.value()
            dec = self.coord_dec.value()
            
            # Create Source object
            source = Source(ra_hrs=ra, dec_deg=dec)
            
            # Determine antenna
            ant = Antenna.NORTH if self.rb_north.isChecked() else Antenna.SOUTH
            
            # Determine observation datetime
            pdt = pytz_timezone('America/Los_Angeles')
            if not self.coord_use_now.isChecked():
                # QDateTimeEdit returns a naive datetime; interpret it as PDT (America/Los_Angeles)
                local_naive = self.coord_time.dateTime().toPyDateTime()
                obs_local = pdt.localize(local_naive)
            else:
                # Use timezone-aware 'now' directly to avoid misinterpreting system local time
                obs_local = datetime.now(pdt)

            # Convert to UTC for astro_pointing (which expects UTC)
            obs_dt = obs_local.astimezone(pytz_timezone('UTC'))
            
            # Get options
            apply_corrections = self.chk_apply_corrections.isChecked()
            apply_pointing_model = self.chk_apply_pointing_model.isChecked()
            clip_elevation = self.chk_clip_elevation.isChecked()
            
            # Calculate Az/El using astro_pointing
            az, el = self.astro_pointing.radec2azel(
                source=source,
                ant=ant,
                obs_datetime=obs_dt,
                apply_corrections=apply_corrections,
                apply_pointing_model=apply_pointing_model,
                clip=clip_elevation
            )
            
            # Format results
            result_text = f"Azimuth/Elevation for RA={ra:.4f}h, Dec={dec:.4f}°:\n"
            result_text += f"Antenna: {'N' if ant == Antenna.NORTH else 'S'}\n"
            result_text += f"Azimuth: {az:.2f}°\n"
            result_text += f"Elevation: {el:.2f}°\n\n"
            
            # Convert back to PDT for display
            pdt_time = obs_dt.astimezone(pytz_timezone('America/Los_Angeles'))
            result_text += f"Time used (PDT): {pdt_time.strftime('%Y-%m-%d %H:%M')}\n"
            result_text += f"Time used (UTC): {obs_dt.strftime('%Y-%m-%d %H:%M')}\n\n"
            
            result_text += f"Options:\n"
            result_text += f"  Sky corrections: {'Yes' if apply_corrections else 'No'}\n"
            result_text += f"  Pointing model: {'Yes' if apply_pointing_model else 'No'}\n"
            result_text += f"  Clip elevation: {'Yes' if clip_elevation else 'No'}\n\n"
            
            result_text += f"Location: OVRO {ant} antenna\n"
            result_text += f"  Latitude: 37.23347717° (N) / 37.23330959° (S)\n"
            result_text += f"  Longitude: -118.2805309° (N) / -118.2805311° (S)\n"
            result_text += f"  Height: 0.0 m (antenna pad)\n"
            
            self.txt_coord_results.setText(result_text)
            
        except Exception as e:
            self.txt_coord_results.setText(f"Error calculating Az/El: {str(e)}")
    
    def _clear_coord_results(self):
        """Clear the coordinate results display."""
        self.txt_coord_results.clear()
    
    def get_current_coordinates(self):
        """Get the current RA/Dec values from the widget."""
        return {
            'ra': self.coord_ra.value(),
            'dec': self.coord_dec.value(),
            'antenna': 'N' if self.rb_north.isChecked() else 'S'
        }
