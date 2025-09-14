"""
Custom exceptions.
"""

class TrackingError(Exception):
    """Base exception for tracking package errors."""
    pass

class SafetyError(TrackingError):
    """Raised when safety checks fail."""
    pass

class MQTTError(TrackingError):
    """Raised when MQTT communication fails."""
    pass

class PointingError(TrackingError):
    """Raised when pointing calculations fail."""
    pass

class ValidationError(TrackingError):
    """Raised when input validation fails."""
    pass

class ConfigurationError(TrackingError):
    """Raised when configuration is invalid or missing."""
    pass

class OperationError(TrackingError):
    """Raised when a telescope operation fails."""
    pass

class TimeoutError(TrackingError):
    """Raised when an operation times out."""
    pass 