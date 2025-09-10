from __future__ import annotations
from typing import List
from PyQt5.QtCore import pyqtSignal

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class DataPlotWidget(QWidget):
    """
    Reusable widget that encapsulates the plot and its refresh button.
    Exposes two minimal methods:
    - set_data(data_buffer): stores buffer and redraws
    - clear(): clears the plot area
    """

    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_data_buffer: List[np.ndarray] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.btn_refresh = QPushButton("Refresh Plot")
        self.btn_refresh.setStyleSheet("background-color: #C25B4B; border: 1px solid #A3473A; color: #FFFFFF;")
        layout.addWidget(self.btn_refresh)
        # Emit a high-level signal when refresh is requested
        self.btn_refresh.clicked.connect(self.refresh_requested.emit)

        fig = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # Set plot and canvas background to match UI color request
        try:
            fig.patch.set_facecolor("#FEFAE9")
            ax.set_facecolor("#FEFAE9")
            self.canvas.setStyleSheet("background-color: #FEFAE9;")
        except Exception:
            pass
        ax.set_title("Data Plot")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        layout.addWidget(self.canvas)

    # --- Public API ---
    def set_data(self, data_buffer: List[np.ndarray]) -> None:
        self._current_data_buffer = data_buffer
        self._plot_data(data_buffer)

    def clear(self) -> None:
        ax = self.canvas.figure.axes[0]
        ax.clear()
        self.canvas.draw()

    # --- Internal plotting logic (moved from MainWindow) ---
    def _plot_data(self, data_buffer: List[np.ndarray]) -> None:
        ax = self.canvas.figure.axes[0]
        ax.clear()

        if not data_buffer:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return

        try:
            specs = []
            tims = []
            total_time_samples = 0

            for data_array in data_buffer:
                d = data_array
                axis_len = d.shape[0]
                specs.append(d.reshape((int(axis_len/10), 10, 8195))[:,:,3:])
                tims.append(d.reshape((int(axis_len/10), 10, 8195))[:,:,0])
                total_time_samples += int(axis_len/10)

            if total_time_samples == 0:
                ax.text(0.5, 0.5, 'No valid data samples', ha='center', va='center', transform=ax.transAxes)
                self.canvas.draw()
                return

            spectra = np.zeros((total_time_samples, 10, 8192), dtype=np.complex64)
            times = np.zeros(total_time_samples, dtype=np.float64)
            cursor = 0
            for i in range(len(data_buffer)):
                d = specs[i]
                tt = d.shape[0]
                spectra[cursor:cursor+tt, :, :] = d
                times[cursor:cursor+tt] = np.real(tims[i][:, 0])
                cursor += tt

            autocorrs = np.real(spectra[:, [0, 4, 7, 9], :]).astype(np.float64)
            # crosscorrs = spectra[:, [2, 3, 5, 6], :]

            XX_1 = np.fft.fftshift(autocorrs[:, 0, :], axes=1)
            YY_1 = np.fft.fftshift(autocorrs[:, 1, :], axes=1)
            XX_2 = np.fft.fftshift(autocorrs[:, 2, :], axes=1)
            YY_2 = np.fft.fftshift(autocorrs[:, 3, :], axes=1)

            freqs_mhz = np.linspace(0, 2400, 8192)

            tmin = 0
            tmax = autocorrs.shape[0] - 1
            if tmax < tmin or autocorrs.shape[0] == 0:
                ax.text(0.5, 0.5, 'No valid autocorrelation data', ha='center', va='center', transform=ax.transAxes)
                self.canvas.draw()
                return

            labels = ["XX(S)", "YY(S)", "XX(N)", "YY(N)"]
            colors = ['blue', 'red', 'green', 'orange']

            plotted = False
            for i in range(4):
                if tmax >= tmin and autocorrs[tmin:tmax, i, :].size > 0:
                    y = np.mean(autocorrs[tmin:tmax, i, :], axis=0)
                    y = np.where(y > 0, y, 1e-10)
                    y = 10. * np.log10(y)
                    ax.plot(freqs_mhz, np.fft.fftshift(y), '-', label=labels[i], color=colors[i], linewidth=1, alpha=0.5)
                    plotted = True

            if plotted:
                ax.legend(fontsize=10)

            for freq in [1420.4, 1612.0, 1665.0, 1667.0, 1720.0]:
                ax.axvline(x=freq, color='black', linestyle='--', alpha=0.3)

            ax.set_xlabel("Frequency (MHz)", fontsize=12)
            ax.set_ylabel("Amplitude (arb. dB)", fontsize=12)
            ax.set_title(f"Autocorrs - {len(data_buffer)} time integrations", fontsize=12)
            ax.grid(True, alpha=0.3)
        except Exception as exc:  # noqa: BLE001
            ax.text(0.5, 0.5, f'Error processing data: {str(exc)}', ha='center', va='center', transform=ax.transAxes)

        self.canvas.draw()

