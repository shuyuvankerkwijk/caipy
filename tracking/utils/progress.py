"""
progress.py - Progress callback interfaces for tracking operations
"""
from typing import Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum


class OperationType(Enum):
    """Types of tracking operations."""
    SLEW = "slew"
    TRACK = "track"
    PARK = "park"
    RASTA_SCAN = "rasta_scan"
    POINTING_OFFSETS = "pointing_offsets"


@dataclass
class ProgressInfo:
    """Progress information for tracking operations."""
    operation_type: OperationType
    antenna: str
    percent_complete: float = 0.0
    message: str = ""
    is_complete: bool = False
    error: Optional[str] = None


class ProgressCallback:
    """Base class for progress callbacks."""
    
    def __call__(self, progress_info: ProgressInfo) -> None:
        """Handle progress update."""
        raise NotImplementedError("Subclasses must implement __call__")


class LoggingProgressCallback(ProgressCallback):
    """Progress callback that logs to the standard logger."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def __call__(self, progress_info: ProgressInfo) -> None:
        """Log progress information."""
        if progress_info.error:
            self.logger.error(f"{progress_info.antenna} {progress_info.operation_type.value}: {progress_info.error}")
        elif progress_info.is_complete:
            self.logger.info(f"{progress_info.antenna} {progress_info.operation_type.value}: Complete")
        else:
            self.logger.info(f"{progress_info.antenna} {progress_info.operation_type.value}: {progress_info.percent_complete:.1f}% - {progress_info.message}")


class SimpleProgressCallback(ProgressCallback):
    """Simple progress callback that calls a function with percent and message."""
    
    def __init__(self, callback_func: Callable[[str, float, str], None]):
        self.callback_func = callback_func
    
    def __call__(self, progress_info: ProgressInfo) -> None:
        """Call the callback function with progress information."""
        if not progress_info.is_complete and not progress_info.error:
            self.callback_func(progress_info.antenna, progress_info.percent_complete, progress_info.message)


def create_progress_callback(callback_type: str = "logging", **kwargs) -> Optional[ProgressCallback]:
    """
    Create a progress callback based on the specified type.
    
    Args:
        callback_type: Type of callback ("logging", "simple", or None)
        **kwargs: Additional arguments for the callback
        
    Returns:
        ProgressCallback instance or None
    """
    if callback_type == "logging":
        import logging
        logger = logging.getLogger(__name__)
        return LoggingProgressCallback(logger)
    elif callback_type == "simple" and "callback_func" in kwargs:
        return SimpleProgressCallback(kwargs["callback_func"])
    else:
        return None 