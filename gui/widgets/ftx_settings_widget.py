from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout, QDoubleSpinBox, QCheckBox, QPushButton, QComboBox
from PyQt5.QtCore import QTimer
from utils.ftx import FTXController, Antenna
from typing import Dict, Any, Optional

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
        title = QLabel("FTX Control")
        title.setStyleSheet("font-weight: bold; font-size: 14pt;")
        layout.addWidget(title)

        # -- Control Panel --
        control_group = QGroupBox("Set FTX Parameters")
        control_layout = QVBoxLayout(control_group)

        # FTX Selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Target FTX:"))
        self.ftx_selector = QComboBox()
        self.ftx_map = {
            "Y(N)": (Antenna.NORTH, 0),
            "X(N)": (Antenna.NORTH, 1),
            "Y(S)": (Antenna.SOUTH, 0),
            "X(S)": (Antenna.SOUTH, 1),
        }
        self.ftx_selector.addItems(self.ftx_map.keys())
        selector_layout.addWidget(self.ftx_selector)
        selector_layout.addStretch()
        control_layout.addLayout(selector_layout)

        # Vertical controls
        self.control_atten = QDoubleSpinBox(); self.control_atten.setRange(0, 31.75); self.control_atten.setSingleStep(0.25); self.control_atten.setDecimals(2)
        control_layout.addWidget(QLabel("Attenuation (dB):"))
        control_layout.addWidget(self.control_atten)

        self.control_laser = QDoubleSpinBox(); self.control_laser.setRange(0, 50); self.control_laser.setDecimals(2)
        control_layout.addWidget(QLabel("Laser (mA):"))
        control_layout.addWidget(self.control_laser)

        self.control_lna = QCheckBox("LNA Enable"); self.control_lna.setChecked(True)
        control_layout.addWidget(self.control_lna)
        
        # Set Button
        btn_set = QPushButton("Set")
        btn_set.clicked.connect(self.set_ftx)
        control_layout.addWidget(btn_set)
        
        layout.addWidget(control_group)
        
        # -- Status Section --
        self.blocks = {}
        status_group = QGroupBox("FTX Status")
        status_layout = QVBoxLayout(status_group)
        for ant in [Antenna.NORTH, Antenna.SOUTH]:
            for pol in [0, 1]:
                block = self._make_block(ant, pol)
                status_layout.addWidget(block['group'])
                self.blocks[(ant, pol)] = block
        layout.addWidget(status_group)

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
        return {'group': group, 'labels': labels}

    def refresh_all(self):
        for (ant, pol), block in self.blocks.items():
            try:
                ctrl = self.controllers[(ant, pol)]
                data = ctrl.get_monitor_data(pol)
                block['labels']['attenuation'].setText(f"Attenuation: {data.attenuation_db:.2f} dB")
                block['labels']['rf_power'].setText(f"RF Power: {data.rf_power_dbm:.2f} dBm")
                block['labels']['laser_current'].setText(f"Laser Current: {data.ld_current_ma:.2f} mA")
            except Exception as e:
                for lbl in block['labels'].values():
                    lbl.setText(f"Error: {e}")

    def set_ftx(self):
        # Get selected target
        target_key = self.ftx_selector.currentText()
        ant, pol = self.ftx_map[target_key]
        
        # Get values from controls
        atten_db = self.control_atten.value()
        laser_ma = self.control_laser.value()
        lna_enabled = self.control_lna.isChecked()

        ctrl = self.controllers[(ant, pol)]
        try:
            ctrl.set_attenuation(pol, atten_db)
            ctrl.set_laser_current(pol, laser_ma)
            ctrl.set_lna_enabled(pol, lna_enabled)
        except Exception as e:
            # Optionally show error in UI
            pass 

    def get_all_ftx_parameters(self) -> Dict[str, Any]:
        """
        Query all FTX parameters (attenuation, RF power, and laser current) 
        for all four FTX units (Y(N), X(N), Y(S), X(S)).
        
        Returns:
            Dictionary containing FTX parameters for all units, with structure:
            {
                'ftx_yn_attenuation_db': float,
                'ftx_yn_rf_power_dbm': float,
                'ftx_yn_laser_current_ma': float,
                'ftx_xn_attenuation_db': float,
                'ftx_xn_rf_power_dbm': float,
                'ftx_xn_laser_current_ma': float,
                'ftx_ys_attenuation_db': float,
                'ftx_ys_rf_power_dbm': float,
                'ftx_ys_laser_current_ma': float,
                'ftx_xs_attenuation_db': float,
                'ftx_xs_rf_power_dbm': float,
                'ftx_xs_laser_current_ma': float,
                'ftx_query_timestamp': str
            }
            
            If any FTX unit fails to respond, its values will be None.
        """
        import time
        from datetime import datetime
        
        ftx_params = {}
        ftx_units = {
            'yn': (Antenna.NORTH, 0),  # Y(N)
            'xn': (Antenna.NORTH, 1),  # X(N)
            'ys': (Antenna.SOUTH, 0),  # Y(S)
            'xs': (Antenna.SOUTH, 1),  # X(S)
        }
        
        for unit_name, (ant, pol) in ftx_units.items():
            try:
                ctrl = self.controllers[(ant, pol)]
                data = ctrl.get_monitor_data(pol)
                
                # Store the three key parameters
                ftx_params[f'ftx_{unit_name}_attenuation_db'] = data.attenuation_db
                ftx_params[f'ftx_{unit_name}_rf_power_dbm'] = data.rf_power_dbm
                ftx_params[f'ftx_{unit_name}_laser_current_ma'] = data.ld_current_ma
                
            except Exception as e:
                # If any FTX unit fails, set its values to None
                ftx_params[f'ftx_{unit_name}_attenuation_db'] = None
                ftx_params[f'ftx_{unit_name}_rf_power_dbm'] = None
                ftx_params[f'ftx_{unit_name}_laser_current_ma'] = None
                print(f"Warning: Failed to query {unit_name.upper()} FTX parameters: {e}")
        
        # Add timestamp
        ftx_params['ftx_query_timestamp'] = datetime.now().isoformat()
        
        return ftx_params 