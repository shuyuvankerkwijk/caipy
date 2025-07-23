from __future__ import annotations
from PyQt5.QtCore import QObject, pyqtSignal

from tracking.utils.progress import ProgressInfo as TrkProgInfo
from recording.utils.progress import ProgressInfo as RecProgInfo

# -----------------------------------------------------------------------------
# Bridges for worker -> UI communication. Must be QObjects to have signals.
# -----------------------------------------------------------------------------

class UILogBridge(QObject):
    """Bridge for logging from worker threads to the UI."""
    log_signal = pyqtSignal(str)

    def write(self, msg: str):
        self.log_signal.emit(msg.strip())

class UIProgressBridge(QObject):
    """Bridge for progress updates from worker threads to the UI."""
    progress_signal = pyqtSignal(object)

    def __call__(self, info: TrkProgInfo | RecProgInfo):
        self.progress_signal.emit(info)

class RecorderStatusBridge(QObject):
    """Bridge for recorder status updates from worker threads to the UI."""
    status_signal = pyqtSignal(tuple)

    def emit_status(self, status: tuple):
        self.status_signal.emit(status)

class DataBufferBridge(QObject):
    """Bridge for data buffer updates from worker threads to the UI."""
    data_signal = pyqtSignal(list)

    def emit_data(self, data: list):
        self.data_signal.emit(data)

class AntennaPositionBridge(QObject):
    """Bridge for antenna position updates from worker threads to the UI."""
    position_and_target_signal = pyqtSignal(str, float, float, float, float)  # antenna, az, el, target_az, target_el

    def emit_position_and_target(self, antenna: str, az: float, el: float, tar_az: float, tar_el: float):
        """Emit both position and target together."""
        self.position_and_target_signal.emit(antenna, az, el, tar_az, tar_el)

class TrackerCompletionBridge(QObject):
    """Bridge for tracker completion signals from worker threads to the UI."""
    completion_signal = pyqtSignal(str)

    def emit_completion(self, ant: str):
        self.completion_signal.emit(ant)

class TrackingStatusBridge(QObject):
    """Bridge for tracking status updates from worker threads to the UI."""
    tracking_status_signal = pyqtSignal(str, bool)

    def emit_tracking_status(self, ant: str, is_tracking: bool):
        self.tracking_status_signal.emit(ant, is_tracking)

class TrackingEventsBridge(QObject):
    """Bridge for tracking events (start/stop) from worker threads to the UI."""
    tracking_event_signal = pyqtSignal(str, str, object, int) # antenna, event_type, source, source_idx

    def emit_event(self, ant: str, event_type: str, source=None, source_idx=None):
        print(f"DEBUG: TrackingEventsBridge.emit_event: ant={ant}, event_type={event_type}, source_idx={source_idx}")
        self.tracking_event_signal.emit(ant, event_type, source, source_idx) 

class TrackerThreadBridge(QObject):
    """Unified bridge for TrackerThread (progress, completion, events)."""
    progress_signal = pyqtSignal(object)  # TrkProgInfo
    completion_signal = pyqtSignal(str)   # antenna
    tracking_event_signal = pyqtSignal(str, str, object, int)  # antenna, event_type, source, source_idx
    log_signal = pyqtSignal(str)

    def emit_progress(self, info):
        self.progress_signal.emit(info)
    def emit_completion(self, ant):
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