"""
recording/config.py - Configuration settings for the recording package

This module provides standalone configuration for the recording package,
making it independent of the main application configuration.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from .exceptions import ConfigurationError


@dataclass
class DeviceConfig:
    """Device configuration parameters."""
    # Connection settings
    hostname: str = '10.10.1.11'
    fpgfile: str = "/home/sprite/TEST_ARRAY/dsa-rfsoc-firmware/firmware/dsa_ta_rfsoc4x2/outputs/dsa_ta_rfsoc4x2_2025-05-21_1326.fpg"
    
    # FPGA settings
    default_fftshift: int = 1170
    default_acclen: int = 131072
    fftshift_min: int = 0
    fftshift_max: int = 4095
    
    def __post_init__(self):
        """Validate device configuration."""
        if not os.path.exists(self.fpgfile):
            raise ConfigurationError(f"FPGA file not found: {self.fpgfile}")
        
        if self.default_fftshift < self.fftshift_min or self.default_fftshift > self.fftshift_max:
            raise ConfigurationError(f"default_fftshift must be between {self.fftshift_min} and {self.fftshift_max}")
        
        if self.default_acclen <= 0:
            raise ConfigurationError("default_acclen must be positive")


@dataclass
class RecordingConfig:
    """Recording configuration parameters."""
    # Timing
    waittime: float = 0.2  # seconds
    
    # Data parameters
    rows_per_batch: int = 2000
    data_buffer_shape: tuple = (2000, 8195)  # (rows, columns)
    cross_correlations_per_batch: int = 10
    
    # Buffer settings
    N: int = 4  # Number of time samples to keep in buffer for plotting
    
    # File paths
    base_directory: Optional[Path] = None
    
    def __post_init__(self):
        """Validate recording configuration."""
        if self.waittime <= 0:
            raise ConfigurationError("waittime must be positive")
        
        if self.rows_per_batch <= 0:
            raise ConfigurationError("rows_per_batch must be positive")
        
        if self.cross_correlations_per_batch <= 0:
            raise ConfigurationError("cross_correlations_per_batch must be positive")
        
        if self.cross_correlations_per_batch > self.rows_per_batch:
            raise ConfigurationError("cross_correlations_per_batch cannot be greater than rows_per_batch")
        
        if self.N <= 0:
            raise ConfigurationError("N must be positive")
        
        if self.base_directory is None:
            self.base_directory = Path.home() / "vikram/testarray/data"
        
        # Ensure base directory exists
        try:
            self.base_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigurationError(f"Failed to create base directory {self.base_directory}: {e}")


@dataclass
class RecorderConfig:
    """Main recording package configuration."""
    # Component configs
    device: DeviceConfig = DeviceConfig()
    recording: RecordingConfig = RecordingConfig()
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Package settings
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Post-initialization validation."""
        pass


# Global configuration instance
config = RecorderConfig()