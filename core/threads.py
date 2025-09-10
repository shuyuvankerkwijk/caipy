from __future__ import annotations
import signal
import queue
import threading
import logging
import time
from typing import Dict, Tuple, Optional, Callable

from tracking import Tracker, Source
from tracking.utils.antenna import Antenna, parse_antenna
from recording import Recorder

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
        # Normalize to enum immediately
        try:
            self.ant = parse_antenna(ant)
        except Exception:
            # Default to NORTH if invalid, but log will show later when used
            self.ant = Antenna.NORTH
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

    def _emit_run_signal(self, event):
        if event == 'start_run' or event == 'stop_run':
            self.bridge.emit_tracking_event('N' if self.ant == Antenna.NORTH else 'S', event)

    def _emit_point_signal(self, event, source, idx):
        if event == 'start_point' or event == 'stop_point':
            self.bridge.emit_tracking_event('N' if self.ant == Antenna.NORTH else 'S', event, source=source, source_idx=idx)

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
                # Still emit completion even if tracker is unavailable
                self.bridge.emit_completion(self.ant)
                continue

            self._stop_requested = False

            try:
                self._last_cmd = cmd
                if cmd == "slew":
                    self.tracker.run_slew(ant=self.ant, progress_callback=self._progress_callback_with_stop_check, auto_cleanup=False, **kw)
                elif cmd == "track":
                    self.tracker.run_track(
                        ant=self.ant,
                        progress_callback=self._progress_callback_with_stop_check,
                        auto_cleanup=False,
                        on_run_signal=self._emit_run_signal,
                        **kw
                    )
                elif cmd == "park":
                    self.tracker.run_park(ant=self.ant, progress_callback=self._progress_callback_with_stop_check, auto_cleanup=False)
                elif cmd == "rasta_scan":
                    self.tracker.run_rasta_scan(
                        ant=self.ant,
                        progress_callback=self._progress_callback_with_stop_check,
                        auto_cleanup=False,
                        on_run_signal=self._emit_run_signal,
                        on_point_signal=self._emit_point_signal,
                        **kw
                    )
                elif cmd == "pointing_offsets":
                    self.tracker.run_pointing_offsets(
                        ant=self.ant,
                        progress_callback=self._progress_callback_with_stop_check,
                        auto_cleanup=False,
                        on_run_signal=self._emit_run_signal,
                        on_point_signal=self._emit_point_signal,
                        **kw
                    )
                elif cmd == "rtos":
                    self.tracker.run_rtos(
                        ant=self.ant,
                        progress_callback=self._progress_callback_with_stop_check,
                        auto_cleanup=False,
                        on_run_signal=self._emit_run_signal,
                        on_point_signal=self._emit_point_signal,
                        **kw
                    )
                else:
                    self.log.warning("Unknown command: %s", cmd)
                    
                # Always emit completion after successful command execution
                self.bridge.emit_completion('N' if self.ant == Antenna.NORTH else 'S')
                
            except InterruptedError as e:
                self.log.info(f"Operation {cmd} for antenna {'N' if self.ant == Antenna.NORTH else 'S'} was interrupted: {e}")
                # Still emit completion when interrupted
                self.bridge.emit_completion('N' if self.ant == Antenna.NORTH else 'S')
            except Exception as e:
                self.log.error(f"Error executing {cmd} for antenna {'N' if self.ant == Antenna.NORTH else 'S'}: {e}")
                # Still emit completion when there's an error
                self.bridge.emit_completion('N' if self.ant == Antenna.NORTH else 'S')

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

    def _start_worker(self, fftshift_p0: int, fftshift_p1: int, acclen: int, extra_metadata: Optional[Dict] = None):
        def _task():
            """Worker thread that starts recording."""
            try:
                self.recorder.set_fftshift(fftshift_p0, fftshift_p1)
                self.recorder.set_acclen(acclen)
                if extra_metadata:
                    self.recorder.set_metadata(extra_metadata)

                if not self._shutting_down:
                    success = self.recorder.start_recording(progress_callback=self.bridge.emit_progress)
                    if not success:
                        self.log.warning("Recorder either failed to start or was interrupted")
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
                    self._start_worker(kw["fftshift_p0"], kw["fftshift_p1"], kw["acclen"], extra_metadata)
                elif cmd == "stop":
                    if self.recorder.is_recording:
                        self.recorder.stop_recording()
                elif cmd == "start_point_recording":
                    source_idx = kw.get("source_idx")
                    format_dict = kw.get("format_dict")
                    if source_idx is not None:
                        self.recorder.start_point_recording(source_idx, format_dict)
                    else:
                        if not self._shutting_down:
                            self.log.warning("Missing parameters for start_point_recording command")
                elif cmd == "stop_point_recording":
                    source_idx = kw.get("source_idx")
                    if source_idx is not None:
                        self.recorder.stop_point_recording(source_idx)
                    else:
                        if not self._shutting_down:
                            self.log.warning("Missing parameters for stop_point_recording command")
                elif cmd == "set_params":
                    self.recorder.set_fftshift(kw["fftshift_p0"], kw["fftshift_p1"])
                    self.recorder.set_acclen(kw["acclen"])
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
        self._broker_ip = None
        self._broker_port = None
        self._az_topic = None
        self._el_topic = None

    def _setup_mqtt(self):
        try:
            import paho.mqtt.client as mqtt
            from tracking.utils.config import config
            import uuid
            if self.ant == "N":
                broker_ip = config.mqtt.north_broker_ip
            elif self.ant == "S":
                broker_ip = config.mqtt.south_broker_ip
            else:
                raise ValueError(f"Invalid antenna '{self.ant}'. Must be 'N' or 'S'.")
            self._broker_ip = broker_ip
            self._broker_port = config.mqtt.port
            self._az_topic = config.mqtt.topic_az_status
            self._el_topic = config.mqtt.topic_el_status
            self.mqtt_client = mqtt.Client(client_id=f"pos_monitor_{self.ant}_{uuid.uuid4().hex[:8]}")
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message
            self.mqtt_client.on_disconnect = self._on_disconnect
            self.mqtt_client.connect(broker_ip, config.mqtt.port, config.mqtt.connection_timeout)
            self.mqtt_client.loop_start()
            timeout = config.mqtt.connection_wait_timeout
            start_time = time.time()
            while not self.mqtt_connected and (time.time() - start_time) < timeout:
                time.sleep(config.mqtt.poll_sleep_interval)
            if not self.mqtt_connected:
                raise ConnectionError(f"Failed to connect to MQTT broker {broker_ip}:{config.mqtt.port} within {timeout}s")
            self.mqtt_client.subscribe(self._az_topic)
            self.mqtt_client.subscribe(self._el_topic)
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
            # Resubscribe to topics on reconnect
            if self._az_topic:
                client.subscribe(self._az_topic)
            if self._el_topic:
                client.subscribe(self._el_topic)
        else:
            if not self._shutting_down:
                self.log.error(f"MQTT connection failed for antenna {self.ant}: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.mqtt_connected = False
        if not self._shutting_down:
            self.log.warning(f"MQTT disconnected for antenna {self.ant} (rc={rc}). Will attempt to reconnect.")

    def _on_message(self, client, userdata, msg):
        try:
            import json
            from tracking.utils.config import config
            data = json.loads(msg.payload.decode())
            if not data or 'v' not in data:
                return
            v = data['v']
            if msg.topic == self._az_topic:
                if 'act_pos' in v:
                    self._current_az = float(v['act_pos'])
                if 'target_pos' in v:
                    self._tar_az = float(v['target_pos'])
                if 'set_pos' in v:
                    self._set_az = float(v['set_pos'])
            elif msg.topic == self._el_topic:
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

    def _ensure_mqtt_connected(self):
        # Try to reconnect if not connected
        if not self.mqtt_connected and self.mqtt_client is not None:
            try:
                self.log.info(f"Attempting MQTT reconnect for antenna {self.ant}")
                self.mqtt_client.reconnect()
            except Exception as exc:
                self.log.error(f"MQTT reconnect failed for antenna {self.ant}: {exc}")

    def run(self):
        self._setup_mqtt()
        while self.running:
            # Periodically check connection and try to reconnect if needed
            if self.mqtt_client is not None and not self.mqtt_connected:
                self._ensure_mqtt_connected()
            time.sleep(1.0)
        self.cleanup_mqtt() 