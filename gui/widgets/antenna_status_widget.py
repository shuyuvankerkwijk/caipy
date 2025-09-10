from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QProgressBar


class AntennaStatusWidget(QWidget):
    """Encapsulates antenna status UI for North and South.

    Keeps public attributes compatible with MainWindow expectations:
    - lbl_north_status, lbl_south_status
    - lbl_north_pos, lbl_north_target
    - lbl_south_pos, lbl_south_target
    - prog_north, prog_south
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        box = QGroupBox("Antenna Status", self)
        layout = QVBoxLayout(box)

        # North antenna status
        north_status = QVBoxLayout()
        north_status_line = QHBoxLayout()
        self.lbl_north_status = QLabel("North: Idle")
        self.lbl_north_status.setStyleSheet("color: #4F8A69; font-weight: bold;")
        north_status_line.addWidget(self.lbl_north_status)
        north_status_line.addStretch()
        north_status.addLayout(north_status_line)
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
        south_status_line = QHBoxLayout()
        self.lbl_south_status = QLabel("South: Idle")
        self.lbl_south_status.setStyleSheet("color: #4F8A69; font-weight: bold;")
        south_status_line.addWidget(self.lbl_south_status)
        south_status_line.addStretch()
        south_status.addLayout(south_status_line)
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

        # Root layout
        root = QVBoxLayout(self)
        root.addWidget(box)

    # ---- Public API ----
    def update_antenna_status(self, antenna: str, status: str):
        if antenna == "N":
            self.lbl_north_status.setText(f"North: {status}")
        elif antenna == "S":
            self.lbl_south_status.setText(f"South: {status}")

    def update_antenna_progress(self, antenna: str, progress: int):
        if antenna == "N":
            self.prog_north.setValue(progress)
        elif antenna == "S":
            self.prog_south.setValue(progress)

    def update_position_and_target(self, antenna: str, az: float, el: float, tar_az: float, tar_el: float):
        pos_text = f"Position: Az {az:.2f}°, El {el:.2f}°" if az is not None and el is not None else "Position: --"
        target_text = f"Target: Az {tar_az:.2f}°, El {tar_el:.2f}°" if tar_az is not None and tar_el is not None else "Target: --"
        if antenna == "N":
            self.lbl_north_pos.setText(pos_text)
            self.lbl_north_target.setText(target_text)
        elif antenna == "S":
            self.lbl_south_pos.setText(pos_text)
            self.lbl_south_target.setText(target_text)


