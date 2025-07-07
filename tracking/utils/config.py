"""
tracking/config.py - Configuration settings for the telescope tracking package

This module provides standalone configuration for the tracking package,
making it independent of the main application configuration.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import astropy.units as u
from astropy.coordinates import EarthLocation

@dataclass
class TelescopeConfig:
    """Telescope configuration parameters."""
    # Location (OVRO)
    latitude: float = 37.23335 # degrees
    longitude: float = -118.28065 # degrees
    height: float = 1222.0 # meters
    location: EarthLocation = None
    
    # Limits
    elevation_min: float = 11.0 # degrees
    elevation_max: float = 89.0 # degrees

    # Sky offsets
    sky_el: float = 0.0 # degrees
    sky_xel: float = 0.0 # degrees
    
    # Safety
    sun_safety_radius: float = 20.0 # degrees
    sun_future_check_interval: float = 0.1  # hours between future checks
    
    # Path finding
    detour_base_padding: float = 30.0 # degrees beyond safety radius for detour paths
    detour_padding_step: float = 10.0 # degrees to increase padding by each iteration
    detour_max_padding: float = 180.0 # maximum padding to try before giving up
    safe_elevation_candidates: list = None # list of elevations to try for detour paths
    
    # Movement
    start_tracking_tolerance: float = 10.0 # degrees
    slew_simulation_steps: int = 50 # number of steps in the slew simulation
    
    # Axis mode settle/poll
    axis_mode_settle_timeout: float = 15.0 # seconds to wait for mode change
    axis_mode_poll_interval: float = 0.1 # seconds between polls
    
    # Progress updates
    slew_progress_update_interval: float = 2.0 # seconds between slewing progress updates
    
    # Park position
    park_azimuth: float = 0.0 # degrees
    park_elevation: float = 70.0 # degrees
    
    # Position update
    position_update_interval: float = 0.5 # seconds
    
    # Weather
    pressure_mb: float = 101.0 # mb
    temperature_C: float = 24.0 # C
    relative_humidity: float = 20 # %
    
    # Pointing model
    #                  flex sin, flex cos, az tilt ha, az tilt lat, el tilt, collim x, collim y, az zero, el zero, az sin, az cos
    #n_ppar = [0.047662, -0.019278, 0.420949, 0.216930, -0.023133, 0.000000, 0.000000, -0.031568, 0.000000, 0.011365, -0.008912]
    #s_ppar = [0.000000, -0.076122, 0.054547, -0.044196, 0.005655, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
    n_ppar = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
    s_ppar = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]

    def __post_init__(self):
        if self.location is None:
            self.location = EarthLocation(lat=self.latitude * u.deg, 
                                        lon=self.longitude * u.deg, 
                                        height=self.height * u.m)
        
        if self.safe_elevation_candidates is None:
            self.safe_elevation_candidates = [15, 13, 20, 18, 25, 30, 35, 40, 45, 50]


@dataclass
class MQTTConfig:
    """MQTT configuration parameters."""
    # Broker settings
    north_broker_ip: str = "192.168.65.60" #"000.000.00.00"
    south_broker_ip: str = "192.168.65.50" #"000.000.00.01"
    port: int = 1883
    client_id: str = "spmac"
    connection_timeout: int = 60
    connection_wait_timeout: float = 10.0
    
    # Specific topics
    topic_az_status: str = "mtexControls/Motion/MotionAxis/Azimuth/status"
    topic_az_cmd: str = "mtexControls/Motion/MotionAxis/Azimuth/execute"
    topic_el_status: str = "mtexControls/Motion/MotionAxis/Elevation/status"
    topic_el_cmd: str = "mtexControls/Motion/MotionAxis/Elevation/execute"
    topic_prg_trk_cmd: str = "mtexControls/Setpoints/ProgramTrack/PrgTr1/execute"
    topic_act_time: str = "mtexControls/TimeSync/TimeManager/Instance/status/actual_time"
    topic_start_time: str = "mtexControls/Setpoints/ProgramTrack/PrgTr1/status/start_time"
    
    # Command parameters
    id_client_motion: int = 31
    id_client_tracking: int = 30
    source: str = "caltech"
    destination_az: str = "Motion.MotionAxis.Azimuth"
    destination_el: str = "Motion.MotionAxis.Elevation"
    destination_prg_trk: str = "Setpoints.ProgramTrack.PrgTr1"
    
    # Timeouts and intervals
    read_timeout: float = 1.0
    poll_sleep_interval: float = 0.01
    track_sleep_interval: float = 0.0001
    
    # Mode mapping from integer to string
    mode_mapping: dict = None
    
    def __post_init__(self):
        if self.mode_mapping is None:
            self.mode_mapping = {
                1: "off",
                6: "track",
                3: "stop", 
                5: "position"
            }

@dataclass
class TrackingConfig:
    """Main tracking package configuration."""
    # Component configs
    telescope: TelescopeConfig = field(default_factory=TelescopeConfig)
    mqtt: MQTTConfig = field(default_factory=MQTTConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Package settings
    version: str = "1.0.0"


# Global configuration instance
config = TrackingConfig()


def load_config(config_file: Optional[Path] = None) -> TrackingConfig:
    """
    Load configuration from file.
    
    Args:
        config_file: Path to configuration file (JSON or YAML)
        
    Returns:
        Loaded configuration
    """
    #TODO
    return config


def save_config(config: TrackingConfig, config_file: Path) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        config_file: Path to save configuration to
    """
    # TODO
    pass 