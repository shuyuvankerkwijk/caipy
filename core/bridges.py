from __future__ import annotations
from PyQt5.QtCore import QObject, pyqtSignal

class UILogBridge(QObject):
    """Bridge for logging from worker threads to the UI."""
    log_signal = pyqtSignal(str)

    def write(self, msg: str):
        self.log_signal.emit(msg.strip())

class TrackerThreadBridge(QObject):
    """Unified bridge for TrackerThread (progress, completion, events)."""
    progress_signal = pyqtSignal(object)  # TrkProgInfo
    completion_signal = pyqtSignal(str)   # antenna
    tracking_event_signal = pyqtSignal(str, str, object, int)  # antenna, event_type, source, source_idx
    log_signal = pyqtSignal(str)

    def emit_progress(self, info):
        self.progress_signal.emit(info)
    def emit_completion(self, ant: str):
        """Emit completion signal."""
        self.completion_signal.emit(ant)
    def emit_tracking_event(self, ant, event_type, source=None, source_idx=None):
        self.tracking_event_signal.emit(ant, event_type, source, source_idx)
    def write(self, msg: str):
        self.log_signal.emit(msg.strip())

class RecorderCmdThreadBridge(QObject):
    """Unified bridge for RecorderCmdThread (progress, log)."""
    progress_signal = pyqtSignal(object)  # RecProgInfo
    log_signal = pyqtSignal(str)

    def emit_progress(self, info):
        self.progress_signal.emit(info)
    def write(self, msg: str):
        self.log_signal.emit(msg.strip())

class RecorderStatusThreadBridge(QObject):
    """Unified bridge for RecorderStatusThread (status, data buffer)."""
    status_signal = pyqtSignal(tuple)
    data_signal = pyqtSignal(list)
    log_signal = pyqtSignal(str)

    def emit_status(self, status):
        self.status_signal.emit(status)
    def emit_data(self, data):
        self.data_signal.emit(data)
    def write(self, msg: str):
        self.log_signal.emit(msg.strip())

class AntennaPositionThreadBridge(QObject):
    """Unified bridge for AntennaPositionThread (position/target, log)."""
    position_and_target_signal = pyqtSignal(str, float, float, float, float)
    log_signal = pyqtSignal(str)

    def emit_position_and_target(self, antenna, az, el, tar_az, tar_el):
        self.position_and_target_signal.emit(antenna, az, el, tar_az, tar_el)
    def write(self, msg: str):
        self.log_signal.emit(msg.strip()) 