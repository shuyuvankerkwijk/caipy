from __future__ import annotations
import signal
import queue
import threading
import logging
import time
from typing import Dict, Tuple, Optional, Callable

from tracking import Tracker, Source
from recording import Recorder
from core.bridges import (
    UIProgressBridge,
    RecorderStatusBridge,
    DataBufferBridge,
    AntennaPositionBridge,
    TrackerCompletionBridge,
    TrackingEventsBridge,
)

# -----------------------------------------------------------------------------
# Helper – safe constructor for Tracker in worker threads (handles signal.signal)
# -----------------------------------------------------------------------------

def _construct_safely(factory):
    """Call *factory* replacing signal.signal with a dummy inside this thread."""
    original = signal.signal

    def dummy(_signum, _handler):
        return None

    signal.signal = dummy
    try:
        return factory()
    finally:
        signal.signal = original

# -----------------------------------------------------------------------------
# Worker threads
# -----------------------------------------------------------------------------

# --- TrackerThread ---
class TrackerThread(threading.Thread):
    def __init__(self, ant: str, bridge, logger_: logging.Logger):
        super().__init__(daemon=True)
        self.ant = ant
        self.log = logger_.getChild(f"tracker_{ant}")
        self.bridge = bridge
        self.cmd_q: queue.Queue[Tuple[str, Dict]] = queue.Queue()
        self.running = True
        self.tracker: Tracker | None = None
        self._stop_requested = False  # Flag to request operation interruption
        self._shutting_down = False  # Flag to prevent logging during shutdown

    def submit(self, cmd: str, **kw):
        self.cmd_q.put((cmd, kw))

    def request_stop(self):
        self._stop_requested = True
        if not self._shutting_down:
            self.log.info("Stop requested for current operation")

    def cleanup_tracker(self):
        self._shutting_down = True
        if self.tracker:
            try:
                self.tracker.cleanup()
            except Exception as exc:
                pass

    def _on_track_start(self, source=None, source_idx=None):
        print(f"DEBUG: _on_track_start called with source={source}, source_idx={source_idx}")
        self.log.info("Tracking started, emitting start event.")
        if source is not None and source_idx is not None:
            self.log.info(f"Emitting per-point start event for source {source_idx}: RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}°")
            self.bridge.emit_tracking_event(self.ant, 'start_point', source=source, source_idx=source_idx)
        else:
            if hasattr(self, '_last_cmd') and self._last_cmd in ['rasta_scan', 'pointing_offsets']:
                self.log.info("Emitting overall scan start event")
                self.bridge.emit_tracking_event(self.ant, 'start_scan')
            else:
                self.log.info("Emitting overall start event")
                self.bridge.emit_tracking_event(self.ant, 'start')

    def _on_track_stop(self, source=None, source_idx=None):
        print(f"DEBUG: _on_track_stop called with source={source}, source_idx={source_idx}")
        self.log.info("Tracking stopped, emitting stop event.")
        if source is not None and source_idx is not None:
            self.log.info(f"Emitting per-point stop event for source {source_idx}: RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}°")
            self.bridge.emit_tracking_event(self.ant, 'stop_point', source=source, source_idx=source_idx)
        else:
            if hasattr(self, '_last_cmd') and self._last_cmd in ['rasta_scan', 'pointing_offsets']:
                self.log.info("Emitting overall scan stop event")
                self.bridge.emit_tracking_event(self.ant, 'stop_scan')
            else:
                self.log.info("Emitting overall stop event")
                self.bridge.emit_tracking_event(self.ant, 'stop')

    def _progress_callback_with_stop_check(self, progress_info):
        if self._stop_requested:
            self._stop_requested = False
            if self.tracker:
                self.tracker.stop()
            raise InterruptedError(f"Operation interrupted by user request")
        self.bridge.emit_progress(progress_info)

    def _init_tracker(self):
        try:
            self.tracker = _construct_safely(Tracker)
            self.log.info("Tracker initialised")
        except Exception as exc:
            self.log.error("Tracker init failed: %s", exc)
            self.tracker = None

    def run(self):
        self._init_tracker()
        while self.running:
            try:
                cmd, kw = self.cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.tracker:
                self.log.warning("Tracker unavailable – ignoring %s", cmd)
                continue

            self._stop_requested = False

            try:
                self._last_cmd = cmd
                if cmd == "slew":
                    self.tracker.run_slew(ant=self.ant, progress_callback=self._progress_callback_with_stop_check, auto_cleanup=False, **kw)
                    self.bridge.emit_completion(self.ant)
                elif cmd == "track":
                    self.tracker.run_track(
                        ant=self.ant,
                        progress_callback=self._progress_callback_with_stop_check,
                        auto_cleanup=False,
                        on_track_start=self._on_track_start,
                        on_track_stop=self._on_track_stop,
                        **kw
                    )
                    self.bridge.emit_completion(self.ant)
                elif cmd == "park":
                    self.tracker.run_park(ant=self.ant, progress_callback=self._progress_callback_with_stop_check, auto_cleanup=False)
                    self.bridge.emit_completion(self.ant)
                elif cmd == "rasta_scan":
                    source = kw.get('source')
                    max_distance_deg = kw.get('max_dist_deg')
                    steps_deg = kw.get('step_deg')
                    position_angle_deg = kw.get('position_angle_deg')
                    duration_hours = kw.get('duration_hours')
                    slew = kw.get('slew', True)
                    park = kw.get('park', True)
                    if source is None:
                        raise ValueError("source parameter is required")
                    if max_distance_deg is None:
                        raise ValueError("max_dist_deg parameter is required")
                    if steps_deg is None:
                        raise ValueError("step_deg parameter is required")
                    if position_angle_deg is None:
                        raise ValueError("position_angle_deg parameter is required")
                    if duration_hours is None:
                        raise ValueError("duration_hours parameter is required")
                    self.tracker.run_rasta_scan(
                        ant=self.ant,
                        source=source,
                        max_distance_deg=max_distance_deg,
                        steps_deg=steps_deg,
                        position_angle_deg=position_angle_deg,
                        duration_hours=duration_hours,
                        slew=slew,
                        park=park,
                        progress_callback=self._progress_callback_with_stop_check,
                        auto_cleanup=False,
                        on_track_start=self._on_track_start,
                        on_track_stop=self._on_track_stop,
                    )
                    self.bridge.emit_completion(self.ant)
                elif cmd == "pointing_offsets":
                    source = kw.get('source')
                    closest_dist_deg = kw.get('closest_dist_deg')
                    number_of_points = kw.get('number_of_points')
                    duration_hours = kw.get('duration_hours')
                    slew = kw.get('slew', True)
                    park = kw.get('park', True)
                    if source is None or closest_dist_deg is None or number_of_points is None or duration_hours is None:
                        raise ValueError("Missing parameters for pointing_offsets command")
                    self.tracker.run_pointing_offsets(
                        ant=self.ant,
                        source=source,
                        closest_distance_deg=closest_dist_deg,
                        number_of_points=number_of_points,
                        duration_hours=duration_hours,
                        slew=slew,
                        park=park,
                        progress_callback=self._progress_callback_with_stop_check,
                        auto_cleanup=False,
                        on_track_start=self._on_track_start,
                        on_track_stop=self._on_track_stop,
                    )
                    self.bridge.emit_completion(self.ant)
                elif cmd == "stop":
                    self.request_stop()
                else:
                    self.log.warning("Unknown cmd %s", cmd)
            except InterruptedError as exc:
                self.log.info("Operation interrupted: %s", exc)
                self.bridge.emit_completion(self.ant)
            except Exception as exc:
                self.log.error("Tracker %s failed: %s", cmd, exc)
                self.bridge.emit_completion(self.ant)

# --- RecorderCmdThread ---
class RecorderCmdThread(threading.Thread):
    def __init__(self, recorder: Recorder, bridge, logger_: logging.Logger):
        super().__init__(daemon=True)
        self.recorder = recorder
        self.bridge = bridge
        self.log = logger_.getChild("rec_cmd")
        self.cmd_q: queue.Queue[Tuple[str, Dict]] = queue.Queue()
        self.running = True
        self.rec_worker: threading.Thread | None = None
        self._shutting_down = False

    def submit(self, cmd: str, **kw):
        self.cmd_q.put((cmd, kw))

    def _start_worker(self, fftshift: int, acclen: int, extra_metadata: Optional[Dict] = None):
        def _task():
            try:
                self.recorder.set_fftshift(fftshift)
                self.recorder.set_acclen(acclen)
                if extra_metadata:
                    self.recorder.set_metadata(extra_metadata)
                self.recorder.start_recording(progress_callback=self.bridge.emit_progress)
            except Exception as exc:
                if not self._shutting_down:
                    self.log.error("Recording task error: %s", exc)
            finally:
                self.rec_worker = None
        self.rec_worker = threading.Thread(target=_task, daemon=True)
        self.rec_worker.start()

    def cleanup_recording(self):
        self._shutting_down = True
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        if self.rec_worker is not None:
            try:
                if self.rec_worker.is_alive():
                    self.rec_worker.join(timeout=5.0)
                    if self.rec_worker.is_alive():
                        if not self._shutting_down:
                            self.log.warning("Recording worker thread did not finish gracefully")
                    else:
                        if not self._shutting_down:
                            self.log.info("Recording worker thread finished gracefully")
                else:
                    if not self._shutting_down:
                        self.log.info("Recording worker thread already finished")
            except AttributeError:
                if not self._shutting_down:
                    self.log.info("Recording worker thread already finished")

    def run(self):
        while self.running:
            try:
                cmd, kw = self.cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if cmd == "start":
                    if self.recorder.is_recording:
                        if not self._shutting_down:
                            self.log.warning("Recorder already running")
                        continue
                    extra_metadata = kw.get("extra_metadata")
                    self._start_worker(kw["fftshift"], kw["acclen"], extra_metadata)
                elif cmd == "stop":
                    if self.recorder.is_recording:
                        self.recorder.stop_recording()
                elif cmd == "start_point_recording":
                    source_idx = kw.get("source_idx")
                    if source_idx is not None:
                        self.recorder.start_point_recording(source_idx)
                    else:
                        if not self._shutting_down:
                            self.log.warning("Missing parameters for start_point_recording command")
                elif cmd == "set_params":
                    self.recorder.set_fftshift(kw["fftshift"])
                    self.recorder.set_acclen(kw["acclen"])
                elif cmd == "log_point_completion":
                    source_ra = kw.get("source_ra")
                    source_dec = kw.get("source_dec")
                    source_idx = kw.get("source_idx")
                    antenna = kw.get("antenna")
                    if source_ra is not None and source_dec is not None and source_idx is not None:
                        self.recorder.log_point_to_file(source_ra, source_dec, source_idx, antenna)
                    else:
                        if not self._shutting_down:
                            self.log.warning("Missing parameters for log_point_completion command")
                else:
                    if not self._shutting_down:
                        self.log.warning("Unknown recorder cmd %s", cmd)
            except Exception as exc:
                if not self._shutting_down:
                    self.log.error("Recorder cmd %s failed: %s", cmd, exc)

# --- RecorderStatusThread ---
class RecorderStatusThread(threading.Thread):
    def __init__(self, recorder: Recorder, bridge, logger_: logging.Logger):
        super().__init__(daemon=True)
        self.recorder = recorder
        self.bridge = bridge
        self.log = logger_.getChild("rec_status")
        self.running = True
        self._shutting_down = False
        self._data_requested = False

    def request_data_buffer(self):
        self._data_requested = True
        try:
            data_buffer = self.recorder.get_data_buffer()
            self.bridge.emit_data(data_buffer)
            self._data_requested = False
        except Exception as exc:
            if not self._shutting_down:
                self.log.error("Immediate data buffer request failed: %s", exc)

    def run(self):
        while self.running:
            try:
                fft_p0, fft_p1 = self.recorder.get_fftshift()
                acclen = self.recorder.get_acclen()
                ovf_p0, ovf_p1 = self.recorder.get_overflow_cnt()
                status = (fft_p0, fft_p1, acclen, ovf_p0, ovf_p1, self.recorder.is_recording)
                self.bridge.emit_status(status)
                if self._data_requested:
                    data_buffer = self.recorder.get_data_buffer()
                    self.bridge.emit_data(data_buffer)
                    self._data_requested = False
            except Exception as exc:
                if not self._shutting_down:
                    self.log.error("Status poll error: %s", exc)
            time.sleep(1.0)

# --- AntennaPositionThread ---
class AntennaPositionThread(threading.Thread):
    def __init__(self, ant: str, bridge, logger_: logging.Logger):
        super().__init__(daemon=True)
        self.ant = ant
        self.bridge = bridge
        self.log = logger_.getChild(f"pos_{ant}")
        self.running = True
        self.mqtt_client = None
        self.mqtt_connected = False
        self._shutting_down = False
        self._current_az = None
        self._current_el = None
        self._tar_az = None
        self._tar_el = None

    def _setup_mqtt(self):
        try:
            import paho.mqtt.client as mqtt
            from tracking.utils.config import config
            if self.ant == "N":
                broker_ip = config.mqtt.north_broker_ip
            elif self.ant == "S":
                broker_ip = config.mqtt.south_broker_ip
            else:
                raise ValueError(f"Invalid antenna '{self.ant}'. Must be 'N' or 'S'.")
            self.mqtt_client = mqtt.Client(client_id=f"pos_monitor_{self.ant}")
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message
            self.mqtt_client.connect(broker_ip, config.mqtt.port, config.mqtt.connection_timeout)
            self.mqtt_client.loop_start()
            timeout = config.mqtt.connection_wait_timeout
            start_time = time.time()
            while not self.mqtt_connected and (time.time() - start_time) < timeout:
                time.sleep(config.mqtt.poll_sleep_interval)
            if not self.mqtt_connected:
                raise ConnectionError(f"Failed to connect to MQTT broker {broker_ip}:{config.mqtt.port} within {timeout}s")
            self.mqtt_client.subscribe(config.mqtt.topic_az_status)
            self.mqtt_client.subscribe(config.mqtt.topic_el_status)
            if not self._shutting_down:
                self.log.info(f"MQTT position monitoring started for antenna {self.ant}")
        except Exception as exc:
            if not self._shutting_down:
                self.log.error(f"MQTT setup failed for antenna {self.ant}: {exc}")
            self.mqtt_client = None

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.mqtt_connected = True
            if not self._shutting_down:
                self.log.info(f"MQTT connected for antenna {self.ant}")
        else:
            if not self._shutting_down:
                self.log.error(f"MQTT connection failed for antenna {self.ant}: {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            import json
            from tracking.utils.config import config
            data = json.loads(msg.payload.decode())
            if not data or 'v' not in data:
                return
            v = data['v']
            if msg.topic == config.mqtt.topic_az_status:
                if 'act_pos' in v:
                    self._current_az = float(v['act_pos'])
                if 'target_pos' in v:
                    self._tar_az = float(v['target_pos'])
                if 'set_pos' in v:
                    self._set_az = float(v['set_pos'])
            elif msg.topic == config.mqtt.topic_el_status:
                if 'act_pos' in v:
                    self._current_el = float(v['act_pos'])
                if 'target_pos' in v:
                    self._tar_el = float(v['target_pos'])
                if 'set_pos' in v:
                    self._set_el = float(v['set_pos'])
            if self._current_az is not None and self._current_el is not None:
                self.bridge.emit_position_and_target(
                    self.ant,
                    self._current_az if self._current_az is not None else None,
                    self._current_el if self._current_el is not None else None,
                    self._tar_az if self._tar_az is not None else None,
                    self._tar_el if self._tar_el is not None else None,
                )
        except Exception as exc:
            if not self._shutting_down:
                self.log.error(f"Error processing MQTT message: {exc}")

    def cleanup_mqtt(self):
        self._shutting_down = True
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception as exc:
                pass

    def run(self):
        self._setup_mqtt()
        while self.running:
            time.sleep(1.0)
        self.cleanup_mqtt() 