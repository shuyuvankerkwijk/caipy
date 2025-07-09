"""
Core recorder class for data collection.
"""
import os
import signal
import dsa_rfsoc4x2
import numpy as np
import datetime
import time
from typing import Optional, Tuple, Any
from ..utils.config import config
from ..utils.exceptions import (
    DeviceConnectionError, DeviceInitializationError, DataCollectionError,
    DataSaveError, InvalidParameterError, DirectoryError, StateError
)
from ..utils.colors import Colors
from recording.utils.progress import ProgressCallback
import threading

# Get logger for this module
import logging
logger = logging.getLogger(__name__)


class Recorder:
    """
    A recorder for collecting data from the DSA RFSoC4x2 device.
    
    This class provides a clean interface for configuring and recording data
    from the FPGA-based data acquisition system.
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
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Initialize device connection
        self._initialize_device()

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

    def set_fftshift(self, fftshift: int) -> None:
        """Set the FFT shift value for both polarizations."""
        if not config.device.fftshift_min <= fftshift <= config.device.fftshift_max:
            raise InvalidParameterError(
                f"FFT shift must be between {config.device.fftshift_min} and {config.device.fftshift_max}"
            )
        
        try:
            self._device.p0_pfb_nc.set_fft_shift(fftshift)
            self._device.p1_pfb_nc.set_fft_shift(fftshift)
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
        observation_name: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        new: bool = True,
        progress_callback: Optional["ProgressCallback"] = None,
    ) -> bool:
        """
        Start recording data.
        
        Args:
            observation_name: Optional name for the observation directory
            duration_seconds: Optional duration limit in seconds (None for continuous recording)
            new: Whether to create a new directory (True) or use existing (False)
            progress_callback: Optional callback to report progress
        
        Returns:
            True if recording completed successfully, False if interrupted
        
        Raises:
            StateError: If already recording
            DirectoryError: If directory creation fails
            DataCollectionError: If data collection fails
        """
        if self._is_recording:
            raise StateError("Already recording. Stop current recording first.")
        
        # Setup save directory
        self._setup_save_directory(observation_name, new)
        
        # Initialize recording state
        self._initialize_recording_state()
        
        # Clear the plotting buffer when starting new recording
        self.clear_data_buffer()

        # Start the recording loop
        return self._recording_loop(duration_seconds, progress_callback)

    def stop_recording(self) -> None:
        """Stop the current recording session."""
        if not self._is_recording:
            logger.info("Not currently recording.")
            return
        
        self._is_recording = False
        self._interrupted = True
        logger.info(f"{Colors.RED}Recording stopped{Colors.RESET}")

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

    def _setup_save_directory(self, observation_name: Optional[str], new: bool) -> None:
        """Setup the save directory for recording."""
        if new:
            try:
                base_dir = str(config.recording.base_directory)
                run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                if observation_name:
                    save_dir = os.path.join(base_dir, f'{observation_name}_{run_timestamp}')
                else:
                    save_dir = os.path.join(base_dir, f'run_{run_timestamp}')
                os.makedirs(save_dir, exist_ok=True)
                self._save_dir = save_dir
            except Exception as e:
                raise DirectoryError(f"Failed to create save directory: {e}")
        else:
            if not self._save_dir:
                raise StateError("No save directory specified. Set new=False only when a directory is already set.")

        logger.info(f"Saving to: {self._save_dir}")

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
        
        # Log recording start
        if duration_seconds is None:
            logger.info(f"{Colors.BLUE}Starting continuous data collection (will generate new files every 2000 lines)...{Colors.RESET}")
        else:
            logger.info(f"{Colors.BLUE}Starting data collection for {duration_seconds} seconds...{Colors.RESET}")

        try:
            while True:
                # Check if interrupted
                if self._interrupted:
                    logger.warning(f"{Colors.RED}Recording interrupted by user{Colors.RESET}")
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
                
                # Check if batch is complete
                if idx >= config.recording.rows_per_batch:
                    self._save_batch(data, batch_num, start_time, duration_seconds, progress_callback)
                    idx = 0
                    batch_num += 1
                
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
        
        # Log completion
        if duration_seconds is None:
            logger.info(f"{Colors.GREEN}Continuous recording completed successfully - saved {batch_num-1} complete batches{Colors.RESET}")
        else:
            logger.info(f"{Colors.GREEN}Recording completed successfully{Colors.RESET}")

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
            if batch_num is not None:
                filename = os.path.join(self._save_dir, f'data_batch_{batch_num:04d}_{save_timestamp}.npy')
            elif suffix:
                filename = os.path.join(self._save_dir, f'data_{save_timestamp}_{suffix}.npy')
            else:
                filename = os.path.join(self._save_dir, f'data_{save_timestamp}.npy')
            
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

    def _reset_internal_state(self) -> None:
        """Reset internal state variables."""
        self._is_recording = False
        self._interrupted = False
        self._save_dir = None 