"""
Custom exceptions for the recording package.
"""


class RecordingError(Exception):
    """Base exception for recording package errors."""
    pass


class ConfigurationError(RecordingError):
    """Raised when there's an issue with configuration settings."""
    pass


class DeviceConnectionError(RecordingError):
    """Raised when unable to connect to the recording device."""
    pass


class DeviceInitializationError(RecordingError):
    """Raised when the device fails to initialize properly."""
    pass


class DataCollectionError(RecordingError):
    """Raised when there's an error during data collection."""
    pass


class DataSaveError(RecordingError):
    """Raised when there's an error saving data to disk."""
    pass


class InvalidParameterError(RecordingError):
    """Raised when an invalid parameter is provided."""
    pass


class DirectoryError(RecordingError):
    """Raised when there's an issue with directory operations."""
    pass


class TimeoutError(RecordingError):
    """Raised when an operation times out."""
    pass


class StateError(RecordingError):
    """Raised when the recorder is in an invalid state for the requested operation."""
    pass 