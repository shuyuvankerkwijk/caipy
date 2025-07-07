"""
config.py - Configuration settings for the interferometer control system
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import astropy.units as u
from astropy.coordinates import EarthLocation

@dataclass
class TelescopeConfig:
    """Telescope configuration parameters (deprecated - use tracking.config.TelescopeConfig)."""
    # Note: This class is deprecated. Use tracking.config.TelescopeConfig instead.
    # Kept for backward compatibility only.
    pass

@dataclass
class ObservationConfig:
    """Observation configuration parameters."""
    # Duration limits
    min_duration_minutes: int = 0.5 # 30 seconds
    max_duration_minutes: int = 1440  # 24 hours
    default_duration_minutes: int = 10
    
    # Validation
    ra_min: float = 0.0
    ra_max: float = 24.0
    dec_min: float = -90.0
    dec_max: float = 90.0
    
    # Execution
    executor_sleep_interval: float = 5.0  # seconds when no tasks
    status_update_interval: float = 1.0  # seconds


@dataclass
class QueueConfig:
    """Queue configuration parameters."""
    save_file: Path = Path.home() / ".interferometer_queue.json"
    max_log_lines: int = 1000


@dataclass
class UIConfig:
    """UI configuration parameters."""
    # Window settings
    window_width: int = 1200
    window_height: int = 800
    window_x: int = 100
    window_y: int = 100
    
    # Update intervals
    display_update_interval_ms: int = 500
    
    # Styling
    button_min_height: int = 30
    font_size_normal: str = "11pt"
    font_size_large: str = "12pt"
    
    # Colors
    button_success_color: str = "#4CAF50"
    button_primary_color: str = "#2196F3"
    button_danger_color: str = "#f44336"
    button_warning_color: str = "#FF9800"


@dataclass
class MQTTConfig:
    """MQTT configuration parameters (deprecated - use tracking.config.MQTTConfig)."""
    # Note: This class is deprecated. Use tracking.config.MQTTConfig instead.
    # Kept for backward compatibility only.
    pass


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    # Component configs
    # recorder: RecorderConfig = RecorderConfig()  # Now in recording/config.py
    observation: ObservationConfig = ObservationConfig()
    queue: QueueConfig = QueueConfig()
    ui: UIConfig = UIConfig()
    
    # Application settings
    log_max_lines: int = 1000
    update_interval_ms: int = 500


# Global configuration instance
config = ApplicationConfig()


def load_config(config_file: Optional[Path] = None) -> ApplicationConfig:
    """
    Load configuration from file.
    
    Args:
        config_file: Path to configuration file (JSON or YAML)
        
    Returns:
        Loaded configuration
    """
    # For now, just return default config
    # In future, implement loading from file
    return config


def save_config(config: ApplicationConfig, config_file: Path) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        config_file: Path to save configuration to
    """
    # TODO: Implement configuration saving
    pass