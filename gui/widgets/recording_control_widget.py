from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QSpinBox
from PyQt5.QtCore import pyqtSignal


class RecordingControlWidget(QWidget):
    """Encapsulates the recorder control panel UI.

    Exposes child widgets as attributes for easy wiring from the main window:
    - sp_fft_p0, sp_fft_p1, sp_acc
    - btn_start, btn_set
    - recorder_status_label, lbl_fft, lbl_acc, lbl_ovf
    """

    # Signals for higher-level actions
    start_requested = pyqtSignal(int, int, int)  # fftshift_p0, fftshift_p1, acclen
    stop_requested = pyqtSignal()
    set_params_requested = pyqtSignal(int, int, int)  # fftshift_p0, fftshift_p1, acclen

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        box = QGroupBox("Recorder", self)
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

        # Wire button signals to high-level signals
        self.btn_start.clicked.connect(self._on_toggle_start)
        self.btn_set.clicked.connect(self._on_set_params)

        # Root layout
        root = QVBoxLayout(self)
        root.addWidget(box)

    # ---- Public API for MainWindow to control/update UI ----
    def set_recording_state(self, is_recording: bool) -> None:
        if is_recording:
            self.btn_start.setText("Stop")
            self.recorder_status_label.setText("Recording")
        else:
            self.btn_start.setText("Start")
            self.recorder_status_label.setText("Idle")

    def set_status_text(self, text: str) -> None:
        self.recorder_status_label.setText(text)

    def update_status_values(self, fft0: int, fft1: int, acc: int, ovf0: int, ovf1: int) -> None:
        self.lbl_fft.setText(f"FFT: {fft0}, {fft1}")
        self.lbl_acc.setText(f"AccLen: {acc}")
        self.lbl_ovf.setText(f"Overflow: {ovf0}, {ovf1}")

    def get_params(self) -> tuple:
        return (self.sp_fft_p0.value(), self.sp_fft_p1.value(), self.sp_acc.value())

    # ---- Internal handlers ----
    def _on_toggle_start(self):
        p0, p1, acc = self.get_params()
        if self.btn_start.text() == "Start":
            self.start_requested.emit(p0, p1, acc)
        else:
            self.stop_requested.emit()

    def _on_set_params(self):
        p0, p1, acc = self.get_params()
        self.set_params_requested.emit(p0, p1, acc)


