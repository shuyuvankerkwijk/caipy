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

# Setup default logging
logger = logging.getLogger(__name__)

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
    "Recorder", "logger",
    "RecordingError", "ConfigurationError", "DeviceConnectionError",
    "DeviceInitializationError", "DataCollectionError", "DataSaveError",
    "InvalidParameterError", "DirectoryError", "TimeoutError", "StateError",
    "config", "RecorderConfig", "Colors"
]
