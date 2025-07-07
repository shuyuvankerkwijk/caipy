"""
Telescope Tracking Package

Based on the Tracker.m MATLAB code by S. Padin (5/12/25)

A complete telescope control and tracking system for astronomical observations.

Main API:
    Tracker: Core tracking class for controlling telescope operations
    Source: Astronomical source representation
    State: Tracking state enumeration
    TrackingError: Base exception for tracking errors
"""

import logging
import sys
from pathlib import Path

# Configure logging
def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration for the tracking package."""
    logger = logging.getLogger("tracking")
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

from .core.tracker import Tracker, State
from .utils.source import Source
from .utils.exceptions import (
    TrackingError, SafetyError, MQTTError, PointingError, 
    ValidationError, ConfigurationError, OperationError, TimeoutError
)
from .utils.config import config, TrackingConfig, TelescopeConfig, MQTTConfig
from .utils.colors import Colors

__version__ = "1.0.0"
__all__ = [
    "Tracker", "Source", "State", "logger", "setup_logging",
    "TrackingError", "SafetyError", "MQTTError", "PointingError",
    "ValidationError", "ConfigurationError", "OperationError", "TimeoutError",
    "config", "TrackingConfig", "TelescopeConfig", "MQTTConfig", "Colors"
]