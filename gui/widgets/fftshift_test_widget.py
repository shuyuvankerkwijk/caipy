from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QTextEdit, QSpinBox, QDoubleSpinBox, QPushButton, QProgressBar
from PyQt5.QtCore import pyqtSignal
import threading
from typing import Callable, Optional
from gui.services.fftshift_runner import FFTShiftTestRunner


class FFTShiftTestWidget(QWidget):
    """Encapsulates the FFT Shift Testing UI and emits high-level signals.

    Signals:
    - start_requested(list[int], dict): fftshift_list, current_attenuations
    - stop_requested(): request to stop the test
    - log_message(str): optional log messages for the parent to display
    """

    start_requested = pyqtSignal(list, dict)
    stop_requested = pyqtSignal()
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()
        self.fft_test_running = False
        # Backend components provided by MainWindow
        self._recorder = None
        self._rec_cmd_thread = None
        self._get_fft_params: Optional[Callable[[], tuple]] = None
        self._get_extra_metadata: Optional[Callable[[], dict]] = None
        self._runner: Optional[FFTShiftTestRunner] = None
        self._runner_thread: Optional[threading.Thread] = None

    def configure_backend(self, recorder, rec_cmd_thread, fft_params_getter: Callable[[], tuple], extra_metadata_getter: Callable[[], dict]):
        self._recorder = recorder
        self._rec_cmd_thread = rec_cmd_thread
        self._get_fft_params = fft_params_getter
        self._get_extra_metadata = extra_metadata_getter

    def _build_ui(self):
        root = QVBoxLayout(self)

        box = QGroupBox("FFT Shift Testing")
        layout = QVBoxLayout(box)

        # FFT shift list input
        fftshift_row = QHBoxLayout()
        fftshift_row.addWidget(QLabel("FFT Shift Values:"))
        self.fftshift_list_edit = QTextEdit()
        self.fftshift_list_edit.setMaximumHeight(60)
        self.fftshift_list_edit.setPlaceholderText("Enter FFT shift values (one per line)\nExample:\n1000\n1500\n2000")
        fftshift_row.addWidget(self.fftshift_list_edit)
        layout.addLayout(fftshift_row)

        # Test parameters
        params_row = QHBoxLayout()
        self.fft_test_duration = QSpinBox()
        self.fft_test_duration.setRange(5, 3600)
        self.fft_test_duration.setValue(300)
        self.fft_test_duration.setSuffix(" s")
        self.fft_atten_step = QDoubleSpinBox()
        self.fft_atten_step.setRange(0.25, 5.0)
        self.fft_atten_step.setValue(0.5)
        self.fft_atten_step.setDecimals(2)
        self.fft_atten_step.setSuffix(" dB")
        params_row.addWidget(QLabel("Duration:"))
        params_row.addWidget(self.fft_test_duration)
        params_row.addWidget(QLabel("Atten Step:"))
        params_row.addWidget(self.fft_atten_step)
        layout.addLayout(params_row)

        # Status and progress
        self.fft_test_status = QLabel("Ready")
        self.fft_test_status.setStyleSheet("color: #4F8A69; font-weight: bold;")
        self.fft_test_progress = QProgressBar()
        self.fft_test_progress.setVisible(False)
        layout.addWidget(self.fft_test_status)
        layout.addWidget(self.fft_test_progress)

        # Control buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Test")
        self.btn_start.setStyleSheet("background-color: #FFC857; border: 1px solid #D9A63F; color: #2C3E50;")
        self.btn_stop = QPushButton("Stop Test")
        self.btn_stop.setStyleSheet("background-color: #9B2D14; border: 1px solid #7C230F; color: #FFFFFF;")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        root.addWidget(box)

    def _connect_signals(self):
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

    def _on_start_clicked(self):
        try:
            fftshift_text = self.fftshift_list_edit.toPlainText().strip()
            if not fftshift_text:
                self.log_message.emit("Please enter FFT shift values")
                return
            values = []
            for line in fftshift_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    v = int(line)
                except ValueError:
                    self.log_message.emit(f"Invalid FFT shift value: {line}")
                    return
                if 0 <= v <= 4095:
                    values.append(v)
                else:
                    self.log_message.emit(f"FFT shift value {v} must be between 0 and 4095")
                    return
            if not values:
                self.log_message.emit("No valid FFT shift values entered")
                return

            current_attenuations = self._get_current_attenuations()

            # Update UI state
            self.fft_test_running = True
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.fft_test_progress.setVisible(True)
            self.fft_test_progress.setValue(0)
            self.fft_test_status.setText("Starting test...")

            # Start service-runner locally to reduce MainWindow responsibilities
            self._start_runner(values, current_attenuations)
            # Also emit signal for external listeners if desired
            self.start_requested.emit(values, current_attenuations)
        except Exception as e:  # noqa: BLE001
            self.log_message.emit(f"Error starting FFT shift test: {e}")
            self._on_stop_clicked()

    def _on_stop_clicked(self):
        self.fft_test_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.fft_test_progress.setVisible(False)
        self.fft_test_status.setText("Test stopped")
        self.log_message.emit("FFT shift test stopped by user")
        # Stop local runner if active
        if self._runner:
            try:
                self._runner.stop()
            except Exception:
                pass
        self.stop_requested.emit()

    def _get_current_attenuations(self) -> dict:
        try:
            from utils.ftx import FTXController, Antenna, Polarization
            ftx_north = FTXController(Antenna.NORTH)
            ftx_south = FTXController(Antenna.SOUTH)
            current = {}
            for pol in [Polarization.POL0, Polarization.POL1]:
                data_n = ftx_north.get_monitor_data(pol)
                data_s = ftx_south.get_monitor_data(pol)
                if data_n:
                    current[f'north_pol{pol.value}'] = data_n.attenuation_db
                if data_s:
                    current[f'south_pol{pol.value}'] = data_s.attenuation_db
            return current
        except Exception as e:  # noqa: BLE001
            self.log_message.emit(f"Could not get current attenuations: {e}")
            return {
                'north_pol0': 0.0, 'north_pol1': 0.0,
                'south_pol0': 0.0, 'south_pol1': 0.0
            }

    # ---- Internal runner orchestration ----
    def _start_runner(self, fftshift_list, current_attenuations):
        if not (self._recorder and self._rec_cmd_thread and self._get_fft_params and self._get_extra_metadata):
            self.log_message.emit("FFTShift backend not configured; cannot start")
            return
        runner = FFTShiftTestRunner(
            atten_step_getter=lambda: self.fft_atten_step.value(),
            duration_getter=lambda: self.fft_test_duration.value(),
        )
        self._runner = runner

        def _set_attenuations(north: dict, south: dict) -> bool:
            try:
                from utils.ftx import FTXController, Antenna, Polarization
                ftx_north = FTXController(Antenna.NORTH)
                ftx_south = FTXController(Antenna.SOUTH)
                ok = True
                if not ftx_north.set_attenuation(Polarization.POL0, float(north.get(0, 0.0))):
                    ok = False
                if not ftx_north.set_attenuation(Polarization.POL1, float(north.get(1, 0.0))):
                    ok = False
                if not ftx_south.set_attenuation(Polarization.POL0, float(south.get(0, 0.0))):
                    ok = False
                if not ftx_south.set_attenuation(Polarization.POL1, float(south.get(1, 0.0))):
                    ok = False
                return ok
            except Exception:
                return False

        def _start_recording(obs_name: str, meta: dict) -> None:
            try:
                # Set observation name and metadata
                self._recorder.set_observation_name(obs_name)
                self._recorder.set_metadata(meta)
                extra = self._get_extra_metadata() or {}
                extra.update(meta)
                p0, p1, acc = self._get_fft_params()
                self._rec_cmd_thread.submit(
                    "start",
                    fftshift_p0=p0,
                    fftshift_p1=p1,
                    acclen=acc,
                    extra_metadata=extra,
                )
            except Exception as exc:
                self.log_message.emit(f"Error starting FFT shift recording: {exc}")

        def _stop_recording() -> None:
            self._rec_cmd_thread.submit("stop")

        def _start_point(idx: int, fmt: dict) -> None:
            self._rec_cmd_thread.submit("start_point_recording", source_idx=idx, format_dict=fmt)

        def _stop_point(idx: int) -> None:
            self._rec_cmd_thread.submit("stop_point_recording", source_idx=idx)

        def _set_fft(p0: int, p1: int) -> None:
            try:
                self._recorder.set_fftshift(p0, p1)
                self.log_message.emit(f"Set FFT shift: S={p0}, N={p1}")
            except Exception as exc:
                self.log_message.emit(f"Error setting FFT shift: {exc}")

        def _log(msg: str) -> None:
            self.log_message.emit(msg)

        self._runner_thread = threading.Thread(
            target=runner.run,
            args=(fftshift_list, current_attenuations),
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
        self._runner_thread.start()

    # Optional public helpers for external control
    def set_fftshift(self, p0: int, p1: int):
        try:
            if not self._recorder:
                raise RuntimeError("Recorder not configured")
            self._recorder.set_fftshift(p0, p1)
            self.log_message.emit(f"Set FFT shift: S={p0}, N={p1}")
        except Exception as e:
            self.log_message.emit(f"Error setting FFT shift: {e}")

    def start_recording_with_name(self, observation_name: str, metadata: dict):
        try:
            if not (self._recorder and self._rec_cmd_thread and self._get_fft_params and self._get_extra_metadata):
                self.log_message.emit("FFTShift backend not configured; cannot start recording")
                return
            self._recorder.set_observation_name(observation_name)
            self._recorder.set_metadata(metadata)
            extra = self._get_extra_metadata() or {}
            extra.update(metadata)
            p0, p1, acc = self._get_fft_params()
            self._rec_cmd_thread.submit(
                "start",
                fftshift_p0=p0,
                fftshift_p1=p1,
                acclen=acc,
                extra_metadata=extra,
            )
            self.log_message.emit(f"Started FFT shift recording: {observation_name}")
        except Exception as e:
            self.log_message.emit(f"Error starting FFT shift recording: {e}")


