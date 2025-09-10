"""
Core recorder class for data collection.
"""
import os
import signal
import dsa_rfsoc4x2
import numpy as np
import datetime
import time
from typing import Optional, Tuple, Any, Dict
from ..utils.config import config
from ..utils.exceptions import (
    DeviceConnectionError, DeviceInitializationError, DataCollectionError,
    DataSaveError, InvalidParameterError, DirectoryError, StateError
)
from ..utils.colors import Colors
from recording.utils.progress import ProgressCallback
import threading

# Import MQTT for antenna position reading
try:
    import paho.mqtt.client as mqtt
    from tracking.utils.config import config as tracking_config
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None
    tracking_config = None

# Get logger for this module
import logging
logger = logging.getLogger(__name__)


class Recorder:
    """
    A recorder for collecting data from the DSA RFSoC4x2 device.
    
    This class provides a clean interface for configuring and recording data
    from the FPGA-based data acquisition system. It automatically tracks 
    antenna positions by subscribing to MQTT topics, saving az/el positions 
    every 10 lines (one time integration) to a 'positions.csv' file.
    
    Position tracking works by:
    - Automatically connecting to MQTT brokers for both antennas 
    - Subscribing to position status topics
    - Recording real positions if antennas are available
    - Recording NaN values if antennas are not available (shown as empty cells)
    - Creating positions.csv with columns: timestamp, north_az_deg, north_el_deg, south_az_deg, south_el_deg
    """
    
    def __init__(self):
        """Initialize the recorder and connect to the device."""
        # Core state
        self._waittime = config.recording.waittime
        self._save_dir: Optional[str] = None
        self._is_recording = False
        self._device = None
        self._interrupted = False
        
        # Data buffer for plotting, stores last N time samples
        self._data_buffer = []
        self._buffer_lock = threading.Lock()
        
        # External metadata and filename control
        self._external_metadata: Dict[str, Any] = {}
        self._observation_name: Optional[str] = None
        
        # Point recording state
        self._current_line_position = 0
        self._point_start_lines: Dict[int, int] = {}
        
        # Position tracking components
        self._positions_buffer = []
        self._positions_lock = threading.Lock()
        self._position_tracking_enabled = MQTT_AVAILABLE
        
        # Simple MQTT clients for position reading
        self._mqtt_clients: Dict[str, Optional[mqtt.Client]] = {'N': None, 'S': None}
        self._current_positions: Dict[str, Dict[str, Optional[float]]] = {
            'N': {'az': None, 'el': None},
            'S': {'az': None, 'el': None}
        }
        self._position_lock = threading.Lock()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Initialize device connection
        self._initialize_device()
        
        # Automatically enable position tracking if available
        if self._position_tracking_enabled:
            try:
                self.enable_position_tracking('both')
                logger.info("Automatically enabled position tracking for both antennas")
            except Exception as e:
                logger.info(f"Could not automatically enable position tracking: {e}")
                logger.info("Position tracking will record NaN values")

    # ============================================================================
    # PUBLIC PROPERTIES
    # ============================================================================
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    @property
    def save_directory(self) -> Optional[str]:
        """Get the current save directory."""
        return self._save_dir
    
    def get_current_save_dir(self) -> Optional[str]:
        """Get the current save directory for debugging."""
        logger.info(f"Current save directory: {self._save_dir}")
        return self._save_dir

    # ============================================================================
    # EXTERNAL INTERFACE METHODS
    # ============================================================================
    
    def set_observation_name(self, name: str) -> None:
        """Set the observation name for the next recording session."""
        self._observation_name = name
        logger.info(f"Observation name set to: {name}")

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set external metadata to be included in the recording session."""
        self._external_metadata = metadata.copy()
        logger.info(f"External metadata set: {len(metadata)} items")

    def clear_metadata(self) -> None:
        """Clear all external metadata."""
        self._external_metadata.clear()
        logger.info("External metadata cleared")

    # ============================================================================
    # ANTENNA POSITION TRACKING METHODS
    # ============================================================================
    
    def enable_position_tracking(self, antennas: str = 'both') -> None:
        """
        Enable antenna position tracking by subscribing to MQTT topics.
        
        Args:
            antennas: Which antennas to track ("N", "S", or "both")
        """
        if not self._position_tracking_enabled:
            logger.warning("Position tracking not available - MQTT module not found")
            return
            
        if antennas in ['N', 'S']:
            antenna_list = [antennas]
        elif antennas == 'both':
            antenna_list = ['N', 'S']
        else:
            raise InvalidParameterError(f"Invalid antenna identifier: {antennas}. Must be 'N', 'S', or 'both'")
        
        for ant in antenna_list:
            if self._mqtt_clients[ant] is None:
                self._setup_mqtt_client(ant)

    def _setup_mqtt_client(self, ant: str) -> None:
        """Set up a simple MQTT client to read antenna positions."""
        try:
            # Get broker IP based on antenna
            if ant == "N":
                broker_ip = tracking_config.mqtt.north_broker_ip
            elif ant == "S":
                broker_ip = tracking_config.mqtt.south_broker_ip
            else:
                raise ValueError(f"Invalid antenna: {ant}")
            
            # Create MQTT client
            import uuid
            client_id = f"recorder_position_{ant}_{uuid.uuid4().hex[:8]}"
            client = mqtt.Client(client_id=client_id)
            
            # Set up callbacks
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    logger.info(f"Connected to MQTT broker for antenna {ant}")
                    # Subscribe to position topics
                    az_topic = tracking_config.mqtt.topic_az_status.format(ant=ant) if '{ant}' in tracking_config.mqtt.topic_az_status else tracking_config.mqtt.topic_az_status
                    el_topic = tracking_config.mqtt.topic_el_status.format(ant=ant) if '{ant}' in tracking_config.mqtt.topic_el_status else tracking_config.mqtt.topic_el_status
                    client.subscribe(az_topic)
                    client.subscribe(el_topic)
                else:
                    logger.error(f"Failed to connect to MQTT broker for antenna {ant}: {rc}")
            
            def on_message(client, userdata, msg):
                self._handle_position_message(ant, msg.topic, msg.payload.decode())
            
            client.on_connect = on_connect
            client.on_message = on_message
            
            # Connect to broker
            client.connect(broker_ip, tracking_config.mqtt.port, 60)
            client.loop_start()
            
            self._mqtt_clients[ant] = client
            logger.info(f"Set up position tracking for antenna {ant}")
            
        except Exception as e:
            logger.error(f"Failed to setup MQTT client for antenna {ant}: {e}")
            self._mqtt_clients[ant] = None

    def _handle_position_message(self, ant: str, topic: str, payload: str) -> None:
        """Handle incoming MQTT position messages."""
        try:
            import json
            data = json.loads(payload)
            
            if 'v' in data and 'act_pos' in data['v']:
                position = float(data['v']['act_pos'])
                
                with self._position_lock:
                    if 'az' in topic.lower():
                        self._current_positions[ant]['az'] = position
                    elif 'el' in topic.lower():
                        self._current_positions[ant]['el'] = position
                        
        except Exception as e:
            logger.debug(f"Error parsing position message for {ant}: {e}")

    def get_current_antenna_positions(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """
        Get current positions for all antennas.
        
        Returns:
            Dict mapping antenna ID to (az, el) tuple, or (None, None) if unavailable
        """
        positions = {}
        
        with self._position_lock:
            for ant in ['N', 'S']:
                az = self._current_positions[ant]['az']
                el = self._current_positions[ant]['el']
                positions[ant] = (az, el)
        
        return positions

    def _record_antenna_positions(self) -> None:
        """Record current antenna positions to the buffer."""
        if not self._position_tracking_enabled:
            return
            
        current_time = time.time()
        positions = self.get_current_antenna_positions()
        
        # Create position record: [timestamp, N_az, N_el, S_az, S_el]
        position_record = [
            current_time,
            positions['N'][0] if positions['N'][0] is not None else np.nan,
            positions['N'][1] if positions['N'][1] is not None else np.nan,
            positions['S'][0] if positions['S'][0] is not None else np.nan,
            positions['S'][1] if positions['S'][1] is not None else np.nan
        ]
        
        with self._positions_lock:
            self._positions_buffer.append(position_record)
        
        # Log positions occasionally (every 10th record to avoid spam)
        if len(self._positions_buffer) % 10 == 1:
            n_az_str = f"{positions['N'][0]:.2f}" if positions['N'][0] is not None else 'N/A'
            n_el_str = f"{positions['N'][1]:.2f}" if positions['N'][1] is not None else 'N/A'
            s_az_str = f"{positions['S'][0]:.2f}" if positions['S'][0] is not None else 'N/A'
            s_el_str = f"{positions['S'][1]:.2f}" if positions['S'][1] is not None else 'N/A'
            
            logger.info(f"Recorded positions - N: ({n_az_str}, {n_el_str}), S: ({s_az_str}, {s_el_str})")

    def _save_positions_file(self) -> None:
        """Save the positions buffer to 'positions.csv' file."""
        if not self._position_tracking_enabled or self._save_dir is None:
            return
            
        with self._positions_lock:
            if not self._positions_buffer:
                logger.debug("No positions to save")
                return
                
            # Save to CSV file
            try:
                positions_file = os.path.join(self._save_dir, 'positions.csv')
                
                # Write CSV file with headers
                with open(positions_file, 'w', newline='') as csvfile:
                    import csv
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    writer.writerow(['timestamp', 'north_az_deg', 'north_el_deg', 'south_az_deg', 'south_el_deg'])
                    
                    # Write data rows
                    for record in self._positions_buffer:
                        # Format the row, replacing NaN with empty string for readability
                        formatted_record = []
                        for i, value in enumerate(record):
                            if i == 0:  # timestamp
                                formatted_record.append(f"{value:.6f}")
                            elif np.isnan(value):
                                formatted_record.append("")  # Empty for NaN values
                            else:
                                formatted_record.append(f"{value:.6f}")
                        writer.writerow(formatted_record)
                
                logger.info(f"Saved {len(self._positions_buffer)} position records to {positions_file}")
                if self._current_line_position > 0:
                    ratio = self._current_line_position / len(self._positions_buffer)
                    logger.info(f"Data lines: {self._current_line_position}, Position records: {len(self._positions_buffer)}, Ratio: {ratio:.1f} (expected: ~10.0)")
                else:
                    logger.info(f"Position records: {len(self._positions_buffer)}")
            except Exception as e:
                logger.error(f"Failed to save positions file: {e}")

    def clear_positions_buffer(self) -> None:
        """Clear the positions buffer."""
        with self._positions_lock:
            self._positions_buffer.clear()
        logger.info("Positions buffer cleared")

    def is_position_tracking_enabled(self) -> bool:
        """Check if position tracking is enabled and available."""
        return self._position_tracking_enabled

    def get_position_tracking_status(self) -> Dict[str, Any]:
        """Get detailed status of position tracking system."""
        status = {
            'tracking_available': self._position_tracking_enabled,
            'mqtt_clients': {},
            'buffer_size': len(self._positions_buffer) if self._position_tracking_enabled else 0,
            'current_positions': {}
        }
        
        if self._position_tracking_enabled:
            for ant in ['N', 'S']:
                client = self._mqtt_clients[ant]
                status['mqtt_clients'][ant] = {
                    'configured': client is not None,
                    'connected': client.is_connected() if client is not None else False
                }
                
                # Include current position data
                with self._position_lock:
                    status['current_positions'][ant] = {
                        'az': self._current_positions[ant]['az'],
                        'el': self._current_positions[ant]['el']
                    }
        
        return status

    # ============================================================================
    # DEVICE CONTROL METHODS
    # ============================================================================
    
    def get_status(self) -> str:
        """Get the current device status."""
        try:
            return self._device.print_status_all()
        except Exception as e:
            raise DataCollectionError(f"Failed to get device status: {e}")
        
    def get_overflow_cnt(self) -> Tuple[int, int]:
        """Get the current overflow count."""
        try:
            return (
                self._device.p0_pfb_nc.get_overflow_count(),
                self._device.p1_pfb_nc.get_overflow_count()
            )
        except Exception as e:
            raise DataCollectionError(f"Failed to get overflow count: {e}")

    def get_fftshift(self) -> Tuple[int, int]:
        """Get the current FFT shift values for both polarizations."""
        try:
            return (
                self._device.p0_pfb_nc.get_fft_shift(),
                self._device.p1_pfb_nc.get_fft_shift()
            )
        except Exception as e:
            raise DataCollectionError(f"Failed to get FFT shift values: {e}")

    def get_acclen(self) -> int:
        """Get the current accumulation length."""
        try:
            return self._device.cross_corr.get_acc_len()
        except Exception as e:
            raise DataCollectionError(f"Failed to get accumulation length: {e}")

    def get_waittime(self) -> float:
        """Get the current wait time between data collections."""
        return self._waittime

    def set_fftshift(self, fftshift_p0: int = None, fftshift_p1: int = None) -> None:
        """Set the FFT shift values for P0 and P1 polarizations.
        
        Args:
            fftshift_p0: FFT shift value for P0 polarization
            fftshift_p1: FFT shift value for P1 polarization (defaults to fftshift_p0 if None)
        """
        # Must provide both values
        if fftshift_p1 is None or fftshift_p0 is None:
            raise InvalidParameterError("Both FFT shift values must be provided")
            
        # Validate both values
        for pol, value in [("P0", fftshift_p0), ("P1", fftshift_p1)]:
            if not config.device.fftshift_min <= value <= config.device.fftshift_max:
                raise InvalidParameterError(
                    f"{pol} FFT shift must be between {config.device.fftshift_min} and {config.device.fftshift_max}"
                )
        
        try:
            self._device.p0_pfb_nc.set_fft_shift(fftshift_p0)
            self._device.p1_pfb_nc.set_fft_shift(fftshift_p1)
            logger.info(f"Set FFT shift: P0={fftshift_p0}, P1={fftshift_p1}")
        except Exception as e:
            raise DataCollectionError(f"Failed to set FFT shift: {e}")

    def set_acclen(self, acclen: int) -> None:
        """Set the accumulation length."""
        if acclen <= 0:
            raise InvalidParameterError("Accumulation length must be positive")
        
        try:
            self._device.cross_corr.set_acc_len(acclen)
        except Exception as e:
            raise DataCollectionError(f"Failed to set accumulation length: {e}")

    def set_waittime(self, waittime: float) -> None:
        """Set the wait time between data collections."""
        if waittime <= 0:
            raise InvalidParameterError("Wait time must be positive")
        self._waittime = waittime

    # ============================================================================
    # RECORDING CONTROL METHODS
    # ============================================================================
    
    def start_recording(
        self,
        duration_seconds: Optional[int] = None,
        progress_callback: Optional["ProgressCallback"] = None,
    ) -> bool:
        """Start recording data with an optional time limit."""
        # If a recording is already running, bail out early – do NOT touch the
        # current save directory or any other state.
        if self._is_recording:
            logger.warning("Recording is already in progress – call stop_recording() first")
            return False

        # Always create a fresh save directory for every new recording attempt.
        # This avoids re-using an old directory if observation_name has changed.
        self._save_dir = None
        self._setup_save_directory()

        logger.info(f"Starting recording (observation_name={self._observation_name}, save_dir={self._save_dir})")

        # Save metadata file with run and recorder settings
        self._save_metadata_file(duration_seconds)
        
        # Initialize recording state
        self._initialize_recording_state()
        
        # Clear the plotting buffer when starting new recording
        self.clear_data_buffer()
        
        # Clear any previous point recording state
        self._point_start_lines.clear()
        
        # Clear positions buffer when starting new recording
        self.clear_positions_buffer()

        # Start the recording loop
        return self._recording_loop(duration_seconds, progress_callback)

    def stop_recording(self) -> None:
        if not self._is_recording:
            logger.info("Cannot stop recording -- not currently recording.")
            return
        self._is_recording = False
        self._interrupted = True
        logger.info(f"Recording stopped (observation_name={self._observation_name}, save_dir={self._save_dir})")
        logger.info(f"{Colors.RED}Recording stopped{Colors.RESET}")

    # ============================================================================
    # POINT RECORDING METHODS
    # ============================================================================
    
    def start_point_recording(self, source_idx: int, format_dict: dict = None) -> None:
        """
        Called when a point starts to record the current line position and immediately writes to point_ranges.txt with 'inf' as the end line.
        
        Args:
            source_idx: Unique identifier for the point
            format_dict: Dictionary containing key-value pairs to log. All keys and values will be written to the log line.
        """
        if not self.is_recording or not self._save_dir:
            logger.warning("Cannot start point recording: recording is not active or save directory is not set.")
            return
        if source_idx in self._point_start_lines:
            logger.info(f"Point {source_idx} already has start line set to {self._point_start_lines[source_idx]}")
            return
        self._point_start_lines[source_idx] = self._current_line_position
        log_file_path = os.path.join(self._save_dir, "point_ranges.txt")
        
        if format_dict is None:
            format_dict = {}
        
        # Always start with p{source_idx}
        log_parts = [f"p{source_idx}"]
        
        # Add all key-value pairs from the dictionary
        for key, value in format_dict.items():
            if isinstance(value, float):
                log_parts.append(f"{key} {value:.2f}")
            else:
                log_parts.append(f"{key} {value}")
        
        # Add line numbers
        log_parts.append(f"lines {self._current_line_position}-inf")
        
        log_line = ", ".join(log_parts) + "\n"
        
        with open(log_file_path, "a") as f:
            f.write(log_line)
        logger.info(f"Started tracking point {source_idx} at line {self._current_line_position}")

    def stop_point_recording(self, source_idx: int) -> None:
        """
        Called when a point ends to update the corresponding line in point_ranges.txt, replacing 'inf' with the actual end line.
        
        Args:
            source_idx: Unique identifier for the point
        """
        if not self.is_recording or not self._save_dir:
            logger.warning("Cannot stop point recording: recording is not active or save directory is not set.")
            return
        log_file_path = os.path.join(self._save_dir, "point_ranges.txt")
        end_line = self._current_line_position - 1
        
        # Read all lines, update the relevant one
        try:
            with open(log_file_path, "r") as f:
                lines = f.readlines()
            updated = False
            
            # Always match p{source_idx} pattern
            pattern = f"p{source_idx}"
            
            for i, line in enumerate(lines):
                if line.startswith(pattern) and line.strip().endswith("-inf"):
                    # Replace 'inf' with the actual end line
                    lines[i] = line.replace("-inf", f"-{end_line}")
                    updated = True
                    break
                    
            if updated:
                with open(log_file_path, "w") as f:
                    f.writelines(lines)
                logger.info(f"Stopped tracking point {source_idx} at line {end_line}")
            else:
                logger.warning(f"No matching start line found for point {source_idx} to update end line.")
        except Exception as e:
            logger.error(f"Failed to update point_ranges.txt for point {source_idx}: {e}")

    def is_recording_active(self) -> bool:
        """Check if recording is currently active (for external monitoring)."""
        return self._is_recording and not self._interrupted

    # ============================================================================
    # DATA BUFFER MANAGEMENT
    # ============================================================================
    
    def get_data_buffer(self) -> list:
        """
        Get a copy of the current data buffer for plotting.
        
        Returns:
            List of data arrays (each array is a time sample)
        """
        with self._buffer_lock:
            buffer_copy = self._data_buffer.copy()
            return buffer_copy

    def get_buffer_size(self) -> int:
        """
        Get the current number of samples in the buffer.
        
        Returns:
            Number of samples currently in buffer
        """
        with self._buffer_lock:
            return len(self._data_buffer)

    def clear_data_buffer(self) -> None:
        """Clear the data buffer."""
        with self._buffer_lock:
            self._data_buffer.clear()

    # ============================================================================
    # CLEANUP AND SHUTDOWN
    # ============================================================================
    
    def cleanup(self) -> None:
        """
        Comprehensive cleanup method for all shutdown scenarios.
        
        This method handles:
        - Stopping any active recording (ensuring partial data is saved)
        - Clearing data buffers
        - Closing device connections
        - Resetting internal state
        
        Safe to call multiple times.
        """
        logger.info(f"{Colors.BLUE}Cleaning up recorder resources...{Colors.RESET}")
        
        # Stop recording if active
        if self._is_recording:
            self.stop_recording()
            # Wait a bit for the recording loop to exit and trigger the finally block
            self._wait_for_recording_stop()
        
        # Clear data buffer
        self.clear_data_buffer()
        
        # Save any remaining positions and close MQTT clients
        self._cleanup_position_tracking()
        
        # Close device connection
        self._close_device_connection()
        
        # Reset internal state
        self._reset_internal_state()
        
        logger.info(f"{Colors.BLUE}Recorder cleanup complete{Colors.RESET}")

    # ============================================================================
    # PRIVATE INITIALIZATION METHODS
    # ============================================================================
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame: Any) -> None:
            logger.warning(f"{Colors.RED}Received signal {signum}, initiating graceful shutdown{Colors.RESET}")
            self.cleanup()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _initialize_device(self) -> None:
        """Initialize the device connection and set default parameters."""
        try:
            self._device = dsa_rfsoc4x2.DsaRfsoc4x2(config.device.hostname, config.device.fpgfile)
            self._device.initialize()
        except Exception as e:
            raise DeviceConnectionError(f"Failed to connect to device at {config.device.hostname}: {e}")
        
        # Set default FPGA parameters
        try:
            self._device.cross_corr.set_acc_len(config.device.default_acclen)
            self._device.p0_pfb_nc.set_fft_shift(config.device.default_fftshift)
            self._device.p1_pfb_nc.set_fft_shift(config.device.default_fftshift)
        except Exception as e:
            raise DeviceInitializationError(f"Failed to initialize device parameters: {e}")

    def _setup_save_directory(self) -> None:
        try:
            base_dir = str(config.recording.base_directory)
            run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            if self._observation_name:
                save_dir = os.path.join(base_dir, f'{self._observation_name}_{run_timestamp}')
            else:
                save_dir = os.path.join(base_dir, f'observation_{run_timestamp}')
            os.makedirs(save_dir, exist_ok=True)
            self._save_dir = save_dir
            logger.info(f"Created new save directory: {self._save_dir}")
        except Exception as e:
            raise DirectoryError(f"Failed to create save directory: {e}")
        logger.info(f"Saving to: {self._save_dir}")

    def _save_metadata_file(self, duration_seconds: Optional[int]) -> None:
        """Save metadata file with run and recorder settings."""
        try:
            # Auto-set observation name from metadata if not already set
            if not self._observation_name and self._external_metadata:
                # Try to create a meaningful observation name from metadata
                mode = self._external_metadata.get('mode', 'unknown')
                antenna = self._external_metadata.get('antenna', 'unknown')
                if antenna == 'both':
                    antenna = 'NS'
                elif antenna in ['N', 'S']:
                    antenna = antenna
                else:
                    antenna = 'unknown'
                
                # Create observation name from mode and antenna
                self._observation_name = f"{mode}_{antenna}"
                logger.info(f"Auto-set observation name to: {self._observation_name}")
            
            meta = {
                "observation_name": self._observation_name,
                "start_time": datetime.datetime.now().isoformat(),
                "duration_seconds": duration_seconds,
                "fftshift": self.get_fftshift(),
                "acclen": self.get_acclen(),
                "waittime": self.get_waittime(),
                "save_dir": self._save_dir,
            }
            
            # Add external metadata
            if self._external_metadata:
                meta.update(self._external_metadata)
            
            with open(os.path.join(self._save_dir, "run_settings.txt"), "w") as f:
                for k, v in meta.items():
                    f.write(f"{k}: {v}\n")
                    
            logger.info(f"Saved metadata file with {len(meta)} items")
        except Exception as e:
            logger.warning(f"Could not write run_settings.txt: {e}")

    def _initialize_recording_state(self) -> None:
        """Initialize the recording state variables."""
        self._is_recording = True
        self._interrupted = False

    # ============================================================================
    # PRIVATE RECORDING METHODS
    # ============================================================================
    
    def _recording_loop(
        self, 
        duration_seconds: Optional[int], 
        progress_callback: Optional["ProgressCallback"]
    ) -> bool:
        """Main recording loop."""
        # Preallocate buffer
        data = np.zeros(config.recording.data_buffer_shape, dtype=np.complex128)
        idx = 0
        batch_num = 1
        start_time = time.time()
        
        # Reset line position counter
        self._current_line_position = 0
        
        # Log recording start
        if duration_seconds is None:
            logger.info(f"{Colors.BLUE}Starting continuous data collection (will generate new files every 2000 lines)...{Colors.RESET}")
        else:
            logger.info(f"{Colors.BLUE}Starting data collection for {duration_seconds} seconds...{Colors.RESET}")

        try:
            while True:
                # Check if interrupted
                if self._interrupted:
                    if duration_seconds is None:
                        # For continuous recordings, user interruption is successful completion
                        logger.info(f"{Colors.GREEN}Continuous recording stopped by user{Colors.RESET}")
                        return True
                    else:
                        # For timed recordings, user interruption is a failure
                        logger.warning(f"{Colors.RED}Timed recording interrupted by user{Colors.RESET}")
                        return False
                
                # Check if duration_seconds is set and time is up
                if duration_seconds is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= duration_seconds:
                        logger.info(f"Reached duration of {duration_seconds} seconds. Stopping recording.")
                        break

                # Collect data
                self._collect_data_sample(data, idx)
                idx += config.recording.cross_correlations_per_batch
                
                # Update cumulative line position
                self._current_line_position += config.recording.cross_correlations_per_batch
                
                # Check if batch is complete
                if idx >= config.recording.rows_per_batch:
                    # Save to regular batch file
                    self._save_batch(data, batch_num, start_time, duration_seconds, progress_callback)
                    batch_num += 1
                    idx = 0
                
                time.sleep(self._waittime)
                    
        except KeyboardInterrupt:
            logger.warning(f"{Colors.RED}Keyboard interrupt detected, saving partial data up to line {idx}{Colors.RESET}")
            return False
        finally:
            self._finalize_recording(data, idx, batch_num, start_time, duration_seconds, progress_callback)
        
        return True

    def _collect_data_sample(self, data: np.ndarray, idx: int) -> None:
        """Collect a single data sample from the device."""
        try:
            cross_corrs = self._device.cross_corr.get_new_spectra(wait_on_new=True, get_timestamp=True)
            time_col = np.full((config.recording.cross_correlations_per_batch, 1), cross_corrs[2])
            data[idx:idx+config.recording.cross_correlations_per_batch] = np.concatenate([
                time_col, cross_corrs[0], cross_corrs[1]
            ], axis=1)
            logger.info(f"Wrote {config.recording.cross_correlations_per_batch} lines to buffer position {idx}")
            
            # Add data to plotting buffer
            self._add_to_buffer(data[idx:idx+config.recording.cross_correlations_per_batch])
            
            # Record antenna positions every 10 lines (one time integration)
            try:
                self._record_antenna_positions()
            except Exception as e:
                logger.warning(f"Failed to record antenna positions: {e}")
            
        except Exception as e:
            raise DataCollectionError(f"Error during data collection: {e}")

    def _save_batch(
        self, 
        data: np.ndarray, 
        batch_num: int, 
        start_time: float, 
        duration_seconds: Optional[int], 
        progress_callback: Optional["ProgressCallback"]
    ) -> None:
        """Save a complete batch of data."""
        logger.info(f"{Colors.GREEN}Completed batch {batch_num} with {config.recording.rows_per_batch} lines, saving file...{Colors.RESET}")
        
        # Save the batch
        self._save_array(data, batch_num=batch_num)
        
        # Update progress
        if progress_callback:
            self._update_progress(start_time, duration_seconds, batch_num, progress_callback)

    def _finalize_recording(
        self, 
        data: np.ndarray, 
        idx: int, 
        batch_num: int, 
        start_time: float, 
        duration_seconds: Optional[int], 
        progress_callback: Optional["ProgressCallback"]
    ) -> None:
        """Finalize the recording session."""
        self._is_recording = False
        
        # Save any remaining data
        if idx > 0:
            logger.info(f"Saving partial batch with {idx} lines...")
            self._save_array(data[:idx], suffix="partial")
            logger.info(f"Partial data saved successfully")
        else:
            logger.info("No remaining data to save.")
        
        # Send completion progress
        if progress_callback:
            self._send_completion_progress(duration_seconds, progress_callback)
        
        # Save positions file
        self._save_positions_file()
        
        # Log completion
        if duration_seconds is None:
            logger.info(f"{Colors.GREEN}Continuous recording completed - saved {batch_num-1} complete batches{Colors.RESET}")
        else:
            logger.info(f"{Colors.GREEN}Recording completed successfully{Colors.RESET}")
        
        logger.info(f"Total lines recorded: {self._current_line_position}")

    def _update_progress(
        self, 
        start_time: float, 
        duration_seconds: Optional[int], 
        batch_num: int, 
        progress_callback: "ProgressCallback"
    ) -> None:
        """Update progress callback."""
        elapsed = time.time() - start_time
        total_steps = 0 if duration_seconds is None else duration_seconds
        pct = 0.0 if duration_seconds is None else min(elapsed / duration_seconds * 100.0, 100.0)
        
        from recording.utils.progress import ProgressInfo, OperationType
        progress_callback(
            ProgressInfo(
                operation_type=OperationType.RECORD,
                current_step=int(elapsed),
                total_steps=total_steps,
                percent_complete=pct,
                message=f"Saved batch {batch_num-1} ({elapsed:.1f}s elapsed, continuous recording)" if duration_seconds is None else f"Saved batch {batch_num-1} ({elapsed:.1f}s elapsed)",
            )
        )

    def _send_completion_progress(
        self, 
        duration_seconds: Optional[int], 
        progress_callback: "ProgressCallback"
    ) -> None:
        """Send completion progress callback."""
        from recording.utils.progress import ProgressInfo, OperationType
        progress_callback(
            ProgressInfo(
                operation_type=OperationType.RECORD,
                current_step=duration_seconds or 0,
                total_steps=duration_seconds or 0,
                percent_complete=100.0,
                message="Recording completed" if not self._interrupted else "Recording stopped",
                is_complete=True,
            )
        )

    # ============================================================================
    # PRIVATE DATA MANAGEMENT METHODS
    # ============================================================================
    
    def _save_array(self, array: np.ndarray, batch_num: Optional[int] = None, suffix: Optional[str] = None) -> None:
        """Save the array to a .npy file in the current save directory."""
        if self._save_dir is None:
            raise StateError("No save directory set. Call start_recording first.")
        
        try:
            save_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Get current FFT shift and accumulation length values
            fftshift_p0, fftshift_p1 = self.get_fftshift()
            acclen = self.get_acclen()
            
            # Create standard filename with FFT shift and accumulation length
            if batch_num is not None:
                filename = os.path.join(self._save_dir, f'data_batch_{batch_num:04d}_{save_timestamp}_fft{fftshift_p0}_{fftshift_p1}_acc{acclen}.npy')
            elif suffix:
                filename = os.path.join(self._save_dir, f'data_{save_timestamp}_{suffix}_fft{fftshift_p0}_{fftshift_p1}_acc{acclen}.npy')
            else:
                filename = os.path.join(self._save_dir, f'data_{save_timestamp}_fft{fftshift_p0}_{fftshift_p1}_acc{acclen}.npy')
            
            np.save(filename, array)
            logger.info(f"Saved array of shape {array.shape} to {filename}")
        except Exception as e:
            raise DataSaveError(f"Failed to save array: {e}")

    def _add_to_buffer(self, data: np.ndarray) -> None:
        """
        Add data to the buffer, maintaining only the last N samples.
        
        Args:
            data: Data array to add to buffer
        """
        with self._buffer_lock:
            self._data_buffer.append(data.copy())
            # Keep only the last N samples
            if len(self._data_buffer) > config.recording.N:
                self._data_buffer.pop(0)

    # ============================================================================
    # PRIVATE CLEANUP METHODS
    # ============================================================================
    
    def _wait_for_recording_stop(self) -> None:
        """Wait for the recording loop to stop gracefully."""
        wait_time = 0.0
        while self._is_recording and wait_time < 2.0:  # Wait up to 2 seconds
            time.sleep(0.1)
            wait_time += 0.1

    def _close_device_connection(self) -> None:
        """Close the device connection."""
        if self._device is not None:
            try:
                self._device = None
                logger.info(f"{Colors.BLUE}Recorder device connection closed{Colors.RESET}")
            except Exception as e:
                logger.warning(f"{Colors.RED}Error closing recorder device: {e}{Colors.RESET}")

    def _cleanup_position_tracking(self) -> None:
        """Clean up position tracking MQTT clients and save any remaining positions."""
        # Save final positions if recording was active and positions haven't been saved yet
        # (This is mainly for emergency cleanup scenarios)
        if self._save_dir is not None and self._is_recording:
            logger.info("Emergency cleanup: saving remaining positions")
            self._save_positions_file()
        
        # Close MQTT clients
        for ant in ['N', 'S']:
            client = self._mqtt_clients[ant]
            if client is not None:
                try:
                    client.loop_stop()
                    client.disconnect()
                    self._mqtt_clients[ant] = None
                    logger.info(f"{Colors.BLUE}Closed MQTT client for antenna {ant}{Colors.RESET}")
                except Exception as e:
                    logger.warning(f"{Colors.RED}Error closing MQTT client for antenna {ant}: {e}{Colors.RESET}")

    def _reset_internal_state(self) -> None:
        """Reset internal state variables."""
        self._is_recording = False
        self._interrupted = False
        self._save_dir = None
        # Clear positions buffer
        self.clear_positions_buffer() 