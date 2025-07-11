#!/usr/bin/env python3
import os
import signal
import numpy as np
from time import sleep, time
from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, Any, Union, List
import logging
from astropy.coordinates import SkyCoord
from astropy import units as u


from tracking.utils.helpers import xel2az, sync_to_half_second, angular_separation
from tracking.core.astro_pointing import AstroPointing
from tracking.core.mqtt_controller import MqttController
from tracking.core.safety_checker import SafetyChecker
from tracking.utils.config import config
from tracking.utils.source import Source
from tracking.utils.exceptions import (
    TrackingError, SafetyError, MQTTError, ValidationError, 
    ConfigurationError, OperationError, TimeoutError
)
from tracking.utils.colors import Colors
from tracking.utils.progress import ProgressCallback, ProgressInfo, OperationType
from tracking.utils.helpers import d2m
from tracking.utils.coordinate_utils import get_targets_offsets

# Get logger for this module
logger = logging.getLogger(__name__)

class State(Enum):
    IDLE = "idle"
    SLEWING = "slewing"
    TRACKING = "tracking"
    STOP = "stop"

class Tracker:
    """
    A class for tracking celestial objects using a telescope mount via MQTT communication.
    Handles one or both antennas in parallel, with graceful interruption.
    
    Public API:
        run_track(): Complete tracking operation with optional slewing and parking
        run_slew(): Slew to a target position
        run_park(): Park the telescope
        stop(): Stop current operation
        get_current_position(): Get current telescope position
        cleanup(): Clean up resources and disconnect
    """
    
    def __init__(self) -> None:
        """
        Initialize the Tracker with default parameters.
        """
        self.safety_checker = SafetyChecker()
        self.pointing = AstroPointing()
        self.mqtt = MqttController()
        self.ant: Optional[str] = None
        self.state: State = State.IDLE
        self._is_initialized: bool = False
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    # ============================================================================
    # PUBLIC API METHODS
    # ============================================================================

    def run_track(self, ant: str, source: Source, duration_hours: float, 
                  slew: bool = True, park: bool = True, 
                  progress_callback: Optional[ProgressCallback] = None,
                  auto_cleanup: bool = True) -> bool:
        """
        Run the tracker for one antenna, in ["N", "S"]. Handles slewing, tracking, and parking in sequence.
        
        Args:
            ant: Antenna identifier ("N" or "S")
            source: Source object with coordinates
            duration_hours: Duration to track in hours
            slew: Whether to slew to target before tracking (default: True)
            park: Whether to park after tracking (default: True)
            progress_callback: Optional ProgressCallback for detailed progress updates
            auto_cleanup: Whether to automatically cleanup and disconnect MQTT (default: True)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If input parameters are invalid
            SafetyError: If safety checks fail
            MQTTError: If MQTT communication fails
            OperationError: If tracking operation fails
        """
        logger.info(f"Starting tracking run for antenna {ant}")
        logger.info(f"Target: RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}°")
        logger.info(f"Duration: {duration_hours:.4f} hours")

        try:
            if not self._setup(ant):
                raise OperationError("Failed to setup tracker")

            # Validate target coordinates
            self._validate_target(source, ant, duration_hours)

            # Slew to target if requested
            if slew:
                logger.info(f"{Colors.BLUE}Slewing to target...{Colors.RESET}")
                if not self._slew(source=source, progress_callback=progress_callback):
                    raise OperationError("Slewing failed")

            # Track source
            logger.info(f"{Colors.BLUE}Starting tracking...{Colors.RESET}")
            if progress_callback:
                progress_info = ProgressInfo(
                    operation_type=OperationType.TRACK,
                    antenna=self.ant,
                    percent_complete=0.0,
                    message="Starting tracking operation"
                )
                progress_callback(progress_info)
            
            if not self._track(source, duration_hours, progress_callback=progress_callback):
                raise OperationError("Tracking failed")

            # Park if requested
            if park:
                logger.info(f"{Colors.BLUE}Parking telescope...{Colors.RESET}")
                if not self._park(progress_callback=progress_callback):
                    raise OperationError("Parking failed")
            
            logger.info(f"{Colors.GREEN}Tracking run completed successfully{Colors.RESET}")
            return True
            
        except (SafetyError, MQTTError, ValidationError, OperationError) as e:
            logger.error(f"{Colors.RED}Tracking failed: {e}{Colors.RESET}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error in tracking run: {e}"
            logger.error(f"{Colors.RED}{error_msg}{Colors.RESET}")
            raise OperationError(error_msg)
        finally:
            self._cleanup(auto_cleanup)

    def run_slew(self, ant: str, source: Optional[Source] = None, 
                 az: Optional[float] = None, el: Optional[float] = None, 
                 progress_callback: Optional[ProgressCallback] = None,
                 auto_cleanup: bool = True) -> bool:
        """
        Slew to a target position. Returns True if successful, False otherwise.
        
        Args:
            ant: Antenna identifier ("N" or "S")
            source: Optional Source object with coordinates
            az: Optional azimuth in degrees
            el: Optional elevation in degrees
            progress_callback: Optional ProgressCallback for detailed progress updates
            auto_cleanup: Whether to automatically cleanup and disconnect MQTT (default: True)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If input parameters are invalid
            SafetyError: If safety checks fail
            MQTTError: If MQTT communication fails
            OperationError: If slewing operation fails
        """
        logger.info(f"{Colors.BLUE}Starting slew operation for antenna {ant}{Colors.RESET}")
        
        try:
            if not self._setup(ant):
                raise OperationError("Failed to setup tracker")

            # Validate and slew to target
            if source is not None:
                self._validate_target(source, ant)
                logger.info(f"Target coordinates: RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}°")
                if not self._slew(source=source, progress_callback=progress_callback):
                    raise OperationError("Slewing failed")

            elif az is not None and el is not None:
                self._validate_target_coordinates(az, el, ant)
                logger.info(f"Target coordinates: AZ={az:.2f}°, EL={el:.2f}°")
                if not self._slew(az=az, el=el, progress_callback=progress_callback):
                    raise OperationError("Slewing failed")
            else:
                raise ValidationError("No target or coordinates provided")
            
            logger.info(f"{Colors.GREEN}Slewing completed successfully{Colors.RESET}")
            return True

        except (SafetyError, MQTTError, ValidationError, OperationError) as e:
            logger.error(f"{Colors.RED}Slewing failed: {e}{Colors.RESET}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error in slew operation: {e}"
            logger.error(f"{Colors.RED}{error_msg}{Colors.RESET}")
            raise OperationError(error_msg)
        finally:
            self._cleanup(auto_cleanup)
    
    def run_park(self, ant: str, 
                 progress_callback: Optional[ProgressCallback] = None,
                 auto_cleanup: bool = True) -> bool:
        """
        Park the telescope. Returns True if successful, False otherwise.
        
        Args:
            ant: Antenna identifier ("N" or "S")
            progress_callback: Optional ProgressCallback for detailed progress updates
            auto_cleanup: Whether to automatically cleanup and disconnect MQTT (default: True)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If input parameters are invalid
            MQTTError: If MQTT communication fails
            OperationError: If parking operation fails
        """
        logger.info(f"{Colors.BLUE}Starting park operation for antenna {ant}{Colors.RESET}")
        
        try:
            if not self._setup(ant):
                raise OperationError("Failed to setup tracker")

            if not self._park(progress_callback=progress_callback):
                raise OperationError("Parking failed")

            logger.info(f"{Colors.GREEN}Parking completed successfully{Colors.RESET}")
            return True

        except (MQTTError, ValidationError, OperationError) as e:
            logger.error(f"{Colors.RED}Parking failed: {e}{Colors.RESET}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error in park operation: {e}"
            logger.error(f"{Colors.RED}{error_msg}{Colors.RESET}")
            raise OperationError(error_msg)
        finally:
            self._cleanup(auto_cleanup)

    def stop(self) -> bool:
        """
        Interrupts all ongoing operations and stops motion for the specified antenna(s) or all if None.
        """
        try:
            self.state = State.STOP
            if hasattr(self, 'mqtt') and self.mqtt is not None:
                self.mqtt.set_axis_mode_stop()
                current_pos = self.mqtt.get_current_position()
                if current_pos[0] is not None and current_pos[1] is not None:
                    logger.info(f"{Colors.RED}Antenna {self.ant} stopped at: ({current_pos[0]:.2f}, {current_pos[1]:.2f}){Colors.RESET}")
                else:
                    logger.info(f"{Colors.RED}Antenna {self.ant} stopped (position unknown){Colors.RESET}")
            else:
                logger.warning(f"{Colors.RED}MQTT not available for stop command{Colors.RESET}")
        except Exception as e:
            logger.error(f"{Colors.RED}Error stopping motion: {e}{Colors.RESET}")
            return False
        return True

    def get_current_position(self) -> tuple:
        """
        Get the current telescope position.
        
        Returns:
            tuple: (azimuth, elevation) in degrees, or (None, None) if not available
        """
        try:
            if hasattr(self, 'mqtt') and self.mqtt is not None:
                return self.mqtt.get_current_position()
            else:
                return (None, None)
        except Exception as e:
            logger.warning(f"Error getting current position: {e}")
            return (None, None)
    
    def cleanup(self) -> None:
        """
        Clean up resources and safely disconnect from MQTT.
        
        Sends stop commands to both axes and properly disconnects the MQTT client.
        Should be called when tracking is complete or interrupted.
        """
        logger.info(f"{Colors.BLUE}Cleaning up resources...{Colors.RESET}")
        
        # Stop motion if MQTT is available
        if hasattr(self, 'mqtt') and self.mqtt is not None:
            try:
                self.stop()
            except Exception as e:
                logger.warning(f"{Colors.RED}Error stopping motion: {e}{Colors.RESET}")
            
            # Disconnect MQTT if client exists
            if hasattr(self.mqtt, 'mqtt_client') and self.mqtt.mqtt_client is not None:
                try:
                    self.mqtt.mqtt_client.loop_stop()
                    self.mqtt.mqtt_client.disconnect()
                    logger.info(f"{Colors.BLUE}MQTT disconnected{Colors.RESET}")
                except Exception as e:
                    logger.warning(f"{Colors.RED}Error disconnecting MQTT: {e}{Colors.RESET}")
            else:
                logger.warning(f"{Colors.RED}MQTT client not initialized{Colors.RESET}")
        else:
            logger.warning(f"{Colors.RED}MQTT controller not initialized{Colors.RESET}")

        # Reset state
        if hasattr(self, 'ant') and self.ant is not None:
            self.ant = None
        if hasattr(self, 'state') and self.state is not None:
            self.state = State.IDLE
        self._is_initialized = False

        logger.info(f"{Colors.GREEN}Cleanup complete{Colors.RESET}")

    # ============================================================================
    # SPECIALTY METHODS
    # ============================================================================

    def run_rasta_scan(self, ant: str, source: Source, max_distance_deg: float, steps_deg: float, position_angle_deg: float, duration_hours: float, 
                       slew: bool = True, park: bool = True,
                       progress_callback: Optional[ProgressCallback] = None,
                       auto_cleanup: bool = True) -> bool:
        """
        Run a Rasta scan. Returns True if successful, False otherwise.
        
        Args:
            ant: Antenna identifier ("N" or "S")
            source: Source object with coordinates
            max_distance_deg: Maximum distance from center in degrees
            steps_deg: Step size in degrees
            position_angle_deg: Position angle for the scan in degrees
            duration_hours: Duration to track at each position in hours
            slew: Whether to slew to target before scanning (default: True)
            park: Whether to park after scanning (default: True)
            progress_callback: Optional ProgressCallback for detailed progress updates
            auto_cleanup: Whether to automatically cleanup and disconnect MQTT (default: True)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If input parameters are invalid
            SafetyError: If safety checks fail
            MQTTError: If MQTT communication fails
            OperationError: If tracking operation fails
        """
        logger.info(f"{Colors.BLUE}Starting Rasta scan for antenna {ant}{Colors.RESET}")
        logger.info(f"Target: RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}°")
        logger.info(f"Max distance: {max_distance_deg:.2f}°, Steps: {steps_deg:.2f}°")
        logger.info(f"Duration: {duration_hours:.2f} hours at each step")

        # Validate input parameters
        if max_distance_deg <= 0:
            raise ValidationError(f"max_distance_deg must be positive, got {max_distance_deg}")
        if steps_deg <= 0:
            raise ValidationError(f"steps_deg must be positive, got {steps_deg}")
        if duration_hours <= 0:
            raise ValidationError(f"duration_hours must be positive, got {duration_hours}")
        if steps_deg > max_distance_deg:
            raise ValidationError(f"steps_deg ({steps_deg}) cannot be larger than max_distance_deg ({max_distance_deg})")

        # Make a list of steps
        steps = np.arange(-max_distance_deg, max_distance_deg + steps_deg, steps_deg)
        if len(steps) == 0:
            raise ValidationError("No scan positions generated. Check max_distance_deg and steps_deg values.")
            
        center_coord = SkyCoord(ra=source.ra_hrs*u.hourangle, dec=source.dec_deg*u.deg)
        sources = []

        for step in steps:
            offset_coord = center_coord.directional_offset_by(
                position_angle=position_angle_deg*u.deg,
                separation=step*u.deg
            )
            sources.append(Source(ra_hrs=offset_coord.ra.hour, dec_deg=offset_coord.dec.deg))

        logger.info(f"Generated {len(sources)} scan positions")

        try:
            if not self._setup(ant):
                raise OperationError("Failed to setup tracker")

            # Validate each source individually with its own duration
            total_duration = duration_hours * len(sources)
            for i, scan_source in enumerate(sources):
                logger.debug(f"Validating scan position {i+1}/{len(sources)}")
                self._validate_target(scan_source, ant, duration_hours)

            # Also perform a total duration safety check with the center source
            logger.info(f"Performing total duration safety check ({total_duration:.2f} hours)")
            safety_result = self.safety_checker.check_run_safety(source.ra_hrs, source.dec_deg, total_duration)
            if not safety_result.is_safe:
                raise SafetyError(f"Total duration safety check failed: {safety_result.message}")

            # Slew to first target position if requested
            if slew:
                logger.info(f"{Colors.BLUE}Slewing to first scan position...{Colors.RESET}")
                if not self._slew(source=sources[0], progress_callback=progress_callback):
                    raise OperationError("Slewing to first position failed")
            
            # Track source at every offset
            if not self._rasta_scan(sources, duration_hours, progress_callback):
                raise OperationError("Rasta scan failed")

            logger.info(f"{Colors.GREEN}Rasta scan completed successfully{Colors.RESET}")

            # Park the telescope if requested
            if park:
                logger.info(f"{Colors.BLUE}Parking telescope...{Colors.RESET}")
                if not self._park(progress_callback=progress_callback):
                    raise OperationError("Parking failed")

            return True
            
        except (SafetyError, MQTTError, ValidationError, OperationError) as e:
            logger.error(f"{Colors.RED}Rasta scan failed: {e}{Colors.RESET}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error in rasta scan: {e}"
            logger.error(f"{Colors.RED}{error_msg}{Colors.RESET}")
            raise OperationError(error_msg)
        finally:
            self._cleanup(auto_cleanup)

    def run_pointing_offsets(self, ant: str, source: Source, closest_distance_deg: float, number_of_points: int,
                             duration_hours: float, slew: bool = True, park: bool = True,
                             progress_callback: Optional[ProgressCallback] = None,
                             auto_cleanup: bool = True) -> bool:
        """Run a pointing-offset pattern (cross / triangle etc.) around *source*.

        This helper generates *number_of_points* offsets (5, 7, 9, 13 supported – see
        ``coordinate_utils.get_targets_offsets``) at *closest_distance_deg* from the
        centre and tracks each position for *duration_hours*.

        Except for the way the offset list is generated this is identical to
        :py:meth:`run_rasta_scan` and therefore re-uses the same validation, slew,
        tracking and parking logic.
        """
        logger.info(f"{Colors.BLUE}Starting pointing-offset scan for antenna {ant}{Colors.RESET}")
        logger.info(f"Target: RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}°")
        logger.info(f"Closest distance: {closest_distance_deg:.2f}°, Points: {number_of_points}")
        logger.info(f"Duration: {duration_hours:.2f} hours at each point")

        # Basic validation of arguments
        if closest_distance_deg <= 0:
            raise ValidationError(f"closest_distance_deg must be positive, got {closest_distance_deg}")
        if number_of_points <= 0:
            raise ValidationError(f"number_of_points must be positive, got {number_of_points}")
        if duration_hours <= 0:
            raise ValidationError(f"duration_hours must be positive, got {duration_hours}")

        try:
            ra_list, dec_list = get_targets_offsets(closest_distance_deg, number_of_points,
                                                    source.ra_hrs, source.dec_deg)
        except NotImplementedError as e:
            raise ValidationError(str(e)) from e

        sources = [Source(ra_hrs=ra, dec_deg=dec) for ra, dec in zip(ra_list, dec_list)]
        logger.info(f"Generated {len(sources)} pointing-offset positions")

        # Total duration for safety check
        total_duration = duration_hours * len(sources)

        try:
            if not self._setup(ant):
                raise OperationError("Failed to setup tracker")

            # Validate each individual source
            for idx, scan_source in enumerate(sources):
                logger.debug(f"Validating offset position {idx+1}/{len(sources)}")
                self._validate_target(scan_source, ant, duration_hours)

            # Perform combined safety check at centre for total duration
            logger.info(f"Performing total duration safety check ({total_duration:.2f} hours)")
            safety_result = self.safety_checker.check_run_safety(source.ra_hrs, source.dec_deg, total_duration)
            if not safety_result.is_safe:
                raise SafetyError(f"Total duration safety check failed: {safety_result.message}")

            # Optional slew to first point
            if slew:
                logger.info(f"{Colors.BLUE}Slewing to first offset position...{Colors.RESET}")
                if not self._slew(source=sources[0], progress_callback=progress_callback):
                    raise OperationError("Slewing to first position failed")

            # Track each offset using existing _rasta_scan helper
            if not self._rasta_scan(sources, duration_hours, progress_callback):
                raise OperationError("Pointing-offset scan failed")

            logger.info(f"{Colors.GREEN}Pointing-offset scan completed successfully{Colors.RESET}")

            # Optional park after completion
            if park:
                logger.info(f"{Colors.BLUE}Parking telescope...{Colors.RESET}")
                if not self._park(progress_callback=progress_callback):
                    raise OperationError("Parking failed")

            return True

        except (SafetyError, MQTTError, ValidationError, OperationError) as e:
            logger.error(f"{Colors.RED}Pointing-offset scan failed: {e}{Colors.RESET}")
            raise
        except Exception as e:
            err_msg = f"Unexpected error in pointing-offset scan: {e}"
            logger.error(f"{Colors.RED}{err_msg}{Colors.RESET}")
            raise OperationError(err_msg)
        finally:
            self._cleanup(auto_cleanup)

    # ============================================================================
    # PRIVATE HELPER METHODS
    # ============================================================================

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame: Any) -> None:
            logger.warning(f"{Colors.RED}Received signal {signum}, initiating graceful shutdown{Colors.RESET}")
            self.stop()
            
            if signum == signal.SIGINT:
                raise KeyboardInterrupt("Telescope operation interrupted")
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup(self, ant: str) -> bool:
        """
        Setup the tracker for an operation.
        
        Args:
            ant: Antenna identifier
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Validate antenna
            if ant not in ["N", "S"]:
                raise ValidationError(f"Invalid antenna '{ant}'. Must be 'N' or 'S'.")
            
            # Check if we need to reinitialize (different antenna or not initialized)
            antenna_changed = hasattr(self, 'ant') and self.ant != ant
            needs_mqtt_setup = not self._is_initialized or antenna_changed
            
            # Set antenna
            self.ant = ant
            
            # Connect to MQTT only if needed
            if needs_mqtt_setup:
                logger.info(f"{Colors.BLUE}Setting up MQTT connection for antenna {ant}...{Colors.RESET}")
                self.mqtt.setup_mqtt(ant)
                self._is_initialized = True
            else:
                logger.debug(f"MQTT already initialized for antenna {ant}, skipping setup")
            
            # Always ensure axes are stopped when starting an operation
            self.mqtt.set_axis_mode_stop()
            
            return True
            
        except Exception as e:
            if isinstance(e, (ValidationError, MQTTError)):
                raise
            raise MQTTError(f"Setup failed: {e}")

    def _validate_target(self, source: Source, ant: str, duration_hours: Optional[float] = None) -> None:
        """Validate target coordinates and safety."""
        logger.info(f"{Colors.BLUE}Performing target validation...{Colors.RESET}")
        validation_result = self.safety_checker.validate_target(pointing=self.pointing, source=source, ant=ant)
        if not validation_result.is_safe:
            raise SafetyError(f"Target validation failed: {validation_result.message}")

        if duration_hours is not None:
            logger.info(f"{Colors.BLUE}Performing sun safety check...{Colors.RESET}")
            safety_result = self.safety_checker.check_run_safety(source.ra_hrs, source.dec_deg, duration_hours)
            if not safety_result.is_safe:
                raise SafetyError(f"Sun safety check failed: {safety_result.message}")

    def _validate_target_coordinates(self, az: float, el: float, ant: str) -> None:
        """Validate target coordinates."""
        logger.info(f"{Colors.BLUE}Performing target validation...{Colors.RESET}")
        validation_result = self.safety_checker.validate_target(az=az, el=el, ant=ant)
        if not validation_result.is_safe:
            raise SafetyError(f"Target validation failed: {validation_result.message}")

    def _rasta_scan(self, sources: List[Source], duration_hours: float, progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Track multiple sources for a given duration. Returns True if successful, False otherwise.
        
        Args:
            sources: List of Source objects to track
            duration_hours: Duration to track each source in hours
            progress_callback: Optional ProgressCallback for detailed progress updates
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If sources list is empty or invalid
            OperationError: If tracking operation fails
        """
        logger.info(f"{Colors.BLUE}Starting rasta scan for {len(sources)} sources{Colors.RESET}")

        # Validate sources list
        if not sources:
            raise ValidationError("Sources list cannot be empty")
        
        # Check if we're in the right state
        if self.state != State.IDLE:
            raise OperationError(f"Operation not completed. Current state: {self.state.value}. Please wait for current operation to complete.")
        
        # Check if we're close to the first target
        offset = self._get_current_offset(sources[0]) 
        if offset > config.telescope.start_tracking_tolerance:
            raise OperationError(f"Too far away from first target. Offset: {offset:.2f} degrees.")

        # Setup tracking
        num_points = int(duration_hours * 3600 / config.telescope.position_update_interval)
        logger.info(f"Setting up rasta scan for {num_points} points ({duration_hours:.2f} hours) for each of {len(sources)} sources")

        self.state = State.TRACKING
        self.mqtt.set_axis_mode_track()
        self.mqtt.setup_program_track()
        logger.info(f"{Colors.BLUE}MQTT setup complete{Colors.RESET}")

        # Send zero start time to wake up broker
        dzero = '0001-01-01T00:00:00.000'
        self.mqtt.send_track_start_time(dzero)

        # Send start time
        logger.info(f"{Colors.BLUE}Synchronizing start time...{Colors.RESET}")
        sync_to_half_second()
        d0 = datetime.now(timezone.utc)
        d0s = d0.microsecond / 1e6
        if d0s > 0.5:
            d0s = d0s - 0.5
        ds = d0 - timedelta(seconds=d0s)
        sst = ds + timedelta(seconds=3) + timedelta(seconds=1.173)
        self.mqtt.send_track_start_time(d2m(sst))

        logger.info(f"{Colors.BLUE}Starting tracking loop for {num_points} points ({duration_hours:.4f} hours){Colors.RESET}")
        
        total_sources = len(sources)
        rt_offset = 0  # Track cumulative time across all sources

        for source_idx, source in enumerate(sources):
            for i in range(num_points):
                # Check if motion stop requested
                if self.state == State.STOP:
                    raise OperationError("Tracking interrupted.")
                
                # Get time
                ds = ds + timedelta(seconds=0.5)
                d = ds + timedelta(seconds=2.5)
                rt = rt_offset + i * 0.5  # Continuous relative time

                # Get azel coordinates
                mnt_az, mnt_el = self.pointing.radec2azel(source, self.ant, d, apply_corrections=True, apply_pointing_model=True, clip=True)
                
                # Send position command
                self.mqtt.send_track_position(rt, mnt_az, mnt_el, ds)
                
                # Update progress - calculate overall progress across all sources
                total_points = total_sources * num_points
                current_point = source_idx * num_points + i + 1
                percent = current_point / total_points * 100.0
                
                if progress_callback and (i + 1) % 1 == 0:  # Update every 1 points (0.5 seconds)
                    progress_info = ProgressInfo(
                        operation_type=OperationType.TRACK,
                        antenna=self.ant,
                        percent_complete=percent,
                        message=f"Source {source_idx+1}/{total_sources}, Point {i+1}/{num_points} - AZ={mnt_az:.2f}°, EL={mnt_el:.2f}°, time={rt:.1f}s"
                    )
                    progress_callback(progress_info)
                
                # Log progress every 1 points (0.5 seconds)
                if (i + 1) % 1 == 0:
                    logger.info(f"{Colors.BLUE}Track progress: Source {source_idx+1}/{total_sources}, Point {i+1}/{num_points} - AZ={mnt_az:.2f}°, EL={mnt_el:.2f}°, time={rt:.1f}s{Colors.RESET}")

                        # Update rt_offset for next source
            rt_offset += num_points * 0.5
            
            # Log completion of this source
            logger.info(f"{Colors.GREEN}Completed tracking source {source_idx+1}/{total_sources}: {num_points} points processed{Colors.RESET}")
        
        # Send completion progress
        if progress_callback:
            progress_info = ProgressInfo(
                operation_type=OperationType.TRACK,
                antenna=self.ant,
                percent_complete=100.0,
                message="Rasta scan completed",
                is_complete=True
            )
            progress_callback(progress_info)
        
        # Stop tracking
        logger.info(f"{Colors.RED}Stopping scan{Colors.RESET}")
        self.mqtt.set_axis_mode_stop()
        self.state = State.IDLE
        return True

    def _track(self, source: Source, duration_hours: float, 
               progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Track a source for a given duration. Returns True if successful, False otherwise.
        """
        logger.info(f"{Colors.BLUE}Starting track_source{Colors.RESET}")
        
        # Check if we're in the right state
        if self.state != State.IDLE:
            raise OperationError(f"Operation not completed. Current state: {self.state.value}. Please wait for current operation to complete.")
        
        # Check if we're close to the target
        offset = self._get_current_offset(source) 
        if offset > config.telescope.start_tracking_tolerance:
            raise OperationError(f"Too far away from target. Offset: {offset:.2f} degrees.")

        # Setup tracking
        num_points = int(duration_hours * 3600 / config.telescope.position_update_interval)
        logger.info(f"Setting up tracking for {num_points} points ({duration_hours:.2f} hours)")
        
        self.state = State.TRACKING
        self.mqtt.set_axis_mode_track()
        self.mqtt.setup_program_track()
        logger.info(f"{Colors.BLUE}MQTT setup complete{Colors.RESET}")

        # Send zero start time to wake up broker
        dzero = '0001-01-01T00:00:00.000'
        self.mqtt.send_track_start_time(dzero)

        # Send start time
        logger.info(f"{Colors.BLUE}Synchronizing start time...{Colors.RESET}")
        sync_to_half_second() # wait until next half second boundary (x.0, x.5)
        d0 = datetime.now(timezone.utc) # get current time
        d0s = d0.microsecond / 1e6 # get fractional part of current second
        if d0s > 0.5: 
            d0s = d0s - 0.5
        ds = d0 - timedelta(seconds=d0s) # get most recent 0.0 or 0.5 second boundary
        sst = ds + timedelta(seconds=2.5) + timedelta(seconds=0.5) + timedelta(seconds=1.173) # add 3 seconds and 1.173 seconds to get scheduled start time
        self.mqtt.send_track_start_time(d2m(sst))

        # Track source
        logger.info(f"{Colors.BLUE}Starting tracking loop for {num_points} points ({duration_hours:.4f} hours){Colors.RESET}")
        for i in range(num_points):

            # Check if motion stop requested
            if self.state == State.STOP:
                raise OperationError("Tracking interrupted.")
            
            # Get time
            ds = ds + timedelta(seconds=0.5) # sending time
            d = ds + timedelta(seconds=2.5) # sky time (calculate position 2.5 seconds ahead)
            rt = i * 0.5 # relative time to sst

            # Get azel with detailed logging for first few points
            if i < 3:  # Log details for first 3 points
                # Get coordinates without corrections first
                raw_az, raw_el = self.pointing.radec2azel(source, self.ant, d, apply_corrections=False, apply_pointing_model=False, clip=False)
                logger.info(f"Point {i+1} raw coordinates: AZ={raw_az:.4f}°, EL={raw_el:.4f}°")
                
                # Get coordinates with corrections but no pointing model
                corr_az, corr_el = self.pointing.radec2azel(source, self.ant, d, apply_corrections=True, apply_pointing_model=False, clip=False)
                logger.info(f"Point {i+1} with corrections: AZ={corr_az:.4f}°, EL={corr_el:.4f}°")
                
                # Get final coordinates with everything applied
                mnt_az, mnt_el = self.pointing.radec2azel(source, self.ant, d, apply_corrections=True, apply_pointing_model=True, clip=True)
                logger.info(f"Point {i+1} final coordinates: AZ={mnt_az:.4f}°, EL={mnt_el:.4f}°")
            else:
                # Get final coordinates for remaining points
                mnt_az, mnt_el = self.pointing.radec2azel(source, self.ant, d, apply_corrections=True, apply_pointing_model=True, clip=True)
            
            # Send position command
            self.mqtt.send_track_position(rt, mnt_az, mnt_el, ds)
            
            # Update progress
            percent = (i + 1) / num_points * 100.0
            if progress_callback and (i + 1) % 10 == 0:  # Update every 10 points (5 seconds)
                progress_info = ProgressInfo(
                    operation_type=OperationType.TRACK,
                    antenna=self.ant,
                    percent_complete=percent,
                    message=f"Point {i+1}/{num_points} - AZ={mnt_az:.2f}°, EL={mnt_el:.2f}°, time={rt:.1f}s"
                )
                progress_callback(progress_info)
            
            # Log progress every 10 points (5 seconds)
            if (i + 1) % 10 == 0:
                logger.info(f"{Colors.BLUE}Track progress: Point {i+1}/{num_points} - AZ={mnt_az:.2f}°, EL={mnt_el:.2f}°, time={rt:.1f}s{Colors.RESET}")

        # Log final progress
        logger.info(f"{Colors.GREEN}Track completed: {num_points} points processed{Colors.RESET}")
        
        # Send completion progress
        if progress_callback:
            progress_info = ProgressInfo(
                operation_type=OperationType.TRACK,
                antenna=self.ant,
                percent_complete=100.0,
                message="Tracking completed",
                is_complete=True
            )
            progress_callback(progress_info)
        
        # Stop tracking
        logger.info(f"{Colors.RED}Stopping tracking{Colors.RESET}")
        self.mqtt.set_axis_mode_stop()
        self.state = State.IDLE
        return True

    def _park(self, progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Park the telescope. Returns True if successful, False if interrupted or error.
        """
        return self._slew(az=config.telescope.park_azimuth, el=config.telescope.park_elevation, progress_callback=progress_callback)

    def _slew(self, source: Optional[Source] = None, az: Optional[float] = None, el: Optional[float] = None, 
              progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Slew to a target position. Returns True if successful, False if interrupted or error.
        
        Args:
            source: Optional Source object with coordinates
            az: Optional azimuth in degrees. If None, uses source.
            el: Optional elevation in degrees. If None, uses source.
            progress_callback: Optional ProgressCallback for detailed progress updates.
        """
        # Use provided coordinates or calculate from source
        if az is None and el is None:
            if source is None:
                raise ValidationError("Either source or az/el coordinates must be provided")
            target_az, target_el = self.pointing.radec2azel(source, self.ant, datetime.now(timezone.utc), apply_corrections=True, apply_pointing_model=True)
        else:
            target_az = az
            target_el = el

        logger.info(f"Target coordinates: AZ={target_az:.2f}°, EL={target_el:.2f}°")

        # Get current position
        curr_az, curr_el = self.mqtt.get_current_position()
        if curr_az is None or curr_el is None:
            raise MQTTError("Could not get current telescope position")
        logger.info(f"Current position: AZ={curr_az:.2f}°, EL={curr_el:.2f}°")
        logger.info(f"Position difference: ΔAZ={target_az-curr_az:.2f}°, ΔEL={target_el-curr_el:.2f}°")

        # Get safe path
        logger.info(f"{Colors.BLUE}Calculating safe path...{Colors.RESET}")
        az_list, el_list = self.safety_checker.get_safe_path(curr_az, curr_el, target_az, target_el)
        
        # Check if direct path was taken or if we need a detour
        if len(az_list) == 1:
            logger.info(f"{Colors.GREEN}Direct path is safe - proceeding with single step{Colors.RESET}")
        else:
            logger.info(f"Multi-step path required: {len(az_list)} waypoints")
            # Print the path and safety information
            self.safety_checker._show_path([curr_az] + az_list, [curr_el] + el_list)
        
        # Slew to target
        self.state = State.SLEWING
        for i in range(len(az_list)):
            logger.info(f"Step {i+1}/{len(az_list)}: AZ={az_list[i]:.2f}°, EL={el_list[i]:.2f}°")

            # Check if motion stop requested
            if self.state == State.STOP:
                raise OperationError("Slew interrupted.")
            
            mnt_az = az_list[i]
            mnt_el = el_list[i]

            self.mqtt.set_axis_mode_position(mnt_az, mnt_el)

            # Wait for slewing to complete with progress callback or print
            logger.info(f"{Colors.BLUE}Waiting for axes to reach position...{Colors.RESET}")
            az_mode, el_mode = self.mqtt.get_current_mode()
            while az_mode == "position" or el_mode == "position":
                sleep(0.5)
                az_mode, el_mode = self.mqtt.get_current_mode()
                percent = (i+1) / len(az_list) * 100.0
                
                if progress_callback:
                    progress_info = ProgressInfo(
                        operation_type=OperationType.SLEW,
                        antenna=self.ant,
                        percent_complete=percent,
                        message=f"Step {i+1}/{len(az_list)}: AZ → {mnt_az:.2f}°, EL → {mnt_el:.2f}°"
                    )
                    progress_callback(progress_info)
                else:
                    # Log slewing progress every slew_progress_update_interval seconds (configurable)
                    if hasattr(self, '_last_slew_log_time'):
                        if time.time() - self._last_slew_log_time >= config.telescope.slew_progress_update_interval:
                            logger.info(f"{Colors.BLUE}Slewing progress: Step {i+1}/{len(az_list)} - AZ → {mnt_az:.2f}°, EL → {mnt_el:.2f}°{Colors.RESET}")
                            self._last_slew_log_time = time.time()
                    else:
                        self._last_slew_log_time = time.time()
                        logger.info(f"{Colors.BLUE}Slewing progress: Step {i+1}/{len(az_list)} - AZ → {mnt_az:.2f}°, EL → {mnt_el:.2f}°{Colors.RESET}")
        
        # Stop slewing
        logger.info(f"{Colors.RED}Stopping axes{Colors.RESET}")
        self.mqtt.set_axis_mode_stop()
        self.state = State.IDLE
        
        # Verify final position
        final_az, final_el = self.mqtt.get_current_position()
        if final_az is not None and final_el is not None:
            offset = angular_separation(final_az, final_el, target_az, target_el)
            logger.info(f"{Colors.GREEN}Final position: AZ={final_az:.2f}°, EL={final_el:.2f}° (offset: {offset:.2f}°){Colors.RESET}")
        
        # Send completion progress
        if progress_callback:
            progress_info = ProgressInfo(
                operation_type=OperationType.SLEW,
                antenna=self.ant,
                percent_complete=100.0,
                message="Slewing completed",
                is_complete=True
            )
            progress_callback(progress_info)
        
        return True
        
    def _get_current_offset(self, source: Source) -> float:
        """
        Get the difference between the current position and the target position in deg.
        
        Args:
            source: Source object with target coordinates
            
        Returns:
            float: Offset in degrees
        """
        az_target, el_target = self.pointing.radec2azel(source, self.ant, datetime.now(timezone.utc), apply_corrections=True, apply_pointing_model=True)
        
        az_current, el_current = self.mqtt.get_current_position()
        
        if az_current is None or el_current is None:
            raise MQTTError("Can't get current position")
            
        offset = angular_separation(az_current, el_current, az_target, el_target)        
        return offset
    
    def _cleanup(self, auto_cleanup: bool) -> None:
        """
        Handles cleanup after a successful or failed operation.
        
        Args:
            auto_cleanup: Whether to automatically cleanup and disconnect MQTT.
        """
        if not hasattr(self, '_is_initialized') or not self._is_initialized:
            return
            
        if auto_cleanup:
            # Full cleanup: disconnect MQTT and reset everything
            self.cleanup()
        else:
            # Partial cleanup: just stop motion, keep connection and state for reuse
            self.stop()
            # Reset state to idle so next operation can start
            self.state = State.IDLE
