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
        self._waittime = config.recording.waittime
        self._save_dir: Optional[str] = None
        self._is_recording = False
        self._device = None
        self._interrupted = False
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
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

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame: Any) -> None:
            logger.warning(f"{Colors.RED}Received signal {signum}, initiating graceful shutdown{Colors.RESET}")
            self._interrupted = True
            self.stop_recording()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup device connection."""
        self.close()

    def close(self):
        """Close the device connection."""
        if self._device is not None:
            try:
                self._device = None
            except Exception:
                pass

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    @property
    def save_directory(self) -> Optional[str]:
        """Get the current save directory."""
        return self._save_dir

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
        if duration_seconds is None:
            logger.info(f"{Colors.BLUE}Starting continuous data collection (will generate new files every 2000 lines)...{Colors.RESET}")
        else:
            logger.info(f"{Colors.BLUE}Starting data collection for {duration_seconds} seconds...{Colors.RESET}")

        # Preallocate buffer
        data = np.zeros(config.recording.data_buffer_shape, dtype=np.complex128)
        idx = 0
        batch_num = 1
        self._is_recording = True
        self._interrupted = False

        try:
            start_time = time.time()
            while True:
                # Check if interrupted
                if self._interrupted:
                    logger.warning(f"{Colors.RED}Recording interrupted by user{Colors.RESET}")
                    return False
                
                # Check if duration_seconds is set and time is up
                if duration_seconds is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= duration_seconds:
                        logger.info(
                            f"Reached duration of {duration_seconds} seconds. Stopping recording."
                        )
                        break

                try:
                    cross_corrs = self._device.cross_corr.get_new_spectra(wait_on_new=True, get_timestamp=True)
                    time_col = np.full((config.recording.cross_correlations_per_batch, 1), cross_corrs[2])
                    data[idx:idx+config.recording.cross_correlations_per_batch] = np.concatenate([
                        time_col, cross_corrs[0], cross_corrs[1]
                    ], axis=1)
                    logger.info(f"Wrote {config.recording.cross_correlations_per_batch} lines to buffer position {idx}")
                    idx += config.recording.cross_correlations_per_batch
                    
                    if idx >= config.recording.rows_per_batch:
                        logger.info(f"{Colors.GREEN}Completed batch {batch_num} with {config.recording.rows_per_batch} lines, saving file...{Colors.RESET}")
                        self._save_array(data, batch_num=batch_num)
                        idx = 0
                        batch_num += 1
                        
                        # progress update on each batch save
                        if progress_callback:
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
                    
                    time.sleep(self._waittime)
                except Exception as e:
                    raise DataCollectionError(f"Error during data collection: {e}")
                    
        except KeyboardInterrupt:
            logger.warning(f"{Colors.RED}Keyboard interrupt detected, saving partial data up to line {idx}{Colors.RESET}")
            return False
        finally:
            self._is_recording = False
            # Save any remaining data
            if idx > 0:
                logger.info(f"Saving partial batch with {idx} lines...")
                self._save_array(data[:idx], suffix="partial")
            else:
                logger.info("No remaining data to save.")
        
        # Send completion progress
        if progress_callback:
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
        
        if duration_seconds is None:
            logger.info(f"{Colors.GREEN}Continuous recording completed successfully - saved {batch_num-1} complete batches{Colors.RESET}")
        else:
            logger.info(f"{Colors.GREEN}Recording completed successfully{Colors.RESET}")
        return True

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