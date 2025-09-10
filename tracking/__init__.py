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

# Setup default logging
logger = logging.getLogger(__name__)

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
    "Tracker", "Source", "State", "logger",
    "TrackingError", "SafetyError", "MQTTError", "PointingError",
    "ValidationError", "ConfigurationError", "OperationError", "TimeoutError",
    "config", "TrackingConfig", "TelescopeConfig", "MQTTConfig", "Colors"
]