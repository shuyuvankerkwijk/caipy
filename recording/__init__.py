"""
Recording Package

A modular and standalone Python package for recording data from DSA RFSoC4x2 devices.

Main API:
    Recorder: Core recording class for data collection operations
    RecordingError: Base exception for recording errors
"""

import logging
import sys
from pathlib import Path

# Configure logging
def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration for the recording package."""
    logger = logging.getLogger("recording")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Setup default logging
logger = setup_logging()

from .core import Recorder
from .utils.exceptions import (
    RecordingError, ConfigurationError, DeviceConnectionError,
    DeviceInitializationError, DataCollectionError, DataSaveError,
    InvalidParameterError, DirectoryError, TimeoutError, StateError
)
from .utils.config import config, RecorderConfig
from .utils.colors import Colors

__version__ = "1.0.0"
__all__ = [
    "Recorder", "logger", "setup_logging",
    "RecordingError", "ConfigurationError", "DeviceConnectionError",
    "DeviceInitializationError", "DataCollectionError", "DataSaveError",
    "InvalidParameterError", "DirectoryError", "TimeoutError", "StateError",
    "config", "RecorderConfig", "Colors"
]
