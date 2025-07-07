#!/usr/bin/env python3
import os
import signal
from time import sleep, time
from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, Any, Union
import logging

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
    """
    
    def __init__(self) -> None:
        """
        Initialize the StarTracker with default parameters.
        
        Sets up sky offsets and elevation limits.
        """
        self.safety_checker = SafetyChecker()
        self.pointing = AstroPointing()
        self.mqtt = MqttController()
        self.ant: Optional[str] = None
        self.target: Optional[Source] = None
        self.state: State = State.IDLE
        self._is_initialized: bool = False
        self._interrupted: bool = False
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame: Any) -> None:
            logger.warning(f"{Colors.RED}Received signal {signum}, initiating graceful shutdown{Colors.RESET}")
            self._interrupted = True
            self._stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def run_track(self, ant: str, source: Source, duration_hours: float, 
                  slew: bool = True, park: bool = True, 
                  progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Run the tracker for one antenna, in ["N", "S"]. Handles slewing, tracking, and parking in sequence.
        
        Args:
            ant: Antenna identifier ("N" or "S")
            source: Source object with coordinates
            duration_hours: Duration to track in hours
            slew: Whether to slew to target before tracking (default: True)
            park: Whether to park after tracking (default: True)
            progress_callback: Optional ProgressCallback for detailed progress updates
            
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
            # Setup
            if not self._setup(ant, source):
                raise OperationError("Failed to setup tracker")

            # Validate target coordinates using safety checker
            logger.info(f"{Colors.BLUE}Performing target validation...{Colors.RESET}")
            validation_result = self.safety_checker.validate_target(pointing=self.pointing, source=source, ant=ant)
            if not validation_result.is_safe:
                raise SafetyError(f"Target validation failed: {validation_result.message}")

            # Check run safety
            logger.info(f"{Colors.BLUE}Performing sun safety check...{Colors.RESET}")
            safety_result = self.safety_checker.check_run_safety(source.ra_hrs, source.dec_deg, duration_hours)
            if not safety_result.is_safe:
                raise SafetyError(f"Sun safety check failed: {safety_result.message}")

            # If requested, safely slew to target
            if slew:
                logger.info(f"{Colors.BLUE}Slewing to target...{Colors.RESET}")
                slew_success = self._slew(progress_callback=progress_callback)
                if not slew_success:
                    raise OperationError("Slewing failed")

            # Track source
            logger.info(f"{Colors.BLUE}Starting tracking...{Colors.RESET}")
            if progress_callback:
                progress_info = ProgressInfo(
                    operation_type=OperationType.TRACK,
                    antenna=self.ant,
                    current_step=0,
                    total_steps=1,
                    message="Starting tracking operation",
                    percent_complete=0.0
                )
                progress_callback(progress_info)
            
            track_success = self._track(duration_hours, progress_callback=progress_callback)
            if not track_success:
                raise OperationError("Tracking failed")

            # If requested, safely park
            if park:
                logger.info(f"{Colors.BLUE}Parking telescope...{Colors.RESET}")
                park_success = self._park(progress_callback=progress_callback)
                if not park_success:
                    raise OperationError("Parking failed")
            
            logger.info(f"{Colors.GREEN}Tracking run completed successfully{Colors.RESET}")
            return True
            
        except (SafetyError, MQTTError, ValidationError, OperationError) as e:
            # Log the specific error before cleanup
            logger.error(f"{Colors.RED}Tracking failed: {e}{Colors.RESET}")
            self._cleanup()
            raise
        except Exception as e:
            # Convert unexpected exceptions to OperationError
            error_msg = f"Unexpected error in tracking run: {e}"
            logger.error(f"{Colors.RED}{error_msg}{Colors.RESET}")
            self._cleanup()
            raise OperationError(error_msg)
        finally:
            # Only cleanup if we haven't already (in case of specific exceptions)
            if hasattr(self, '_is_initialized') and self._is_initialized:
                self._cleanup()

    def run_slew(self, ant: str, source: Optional[Source] = None, 
                 az: Optional[float] = None, el: Optional[float] = None, 
                 progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Slew to a target position. Returns True if successful, False otherwise.
        
        Args:
            ant: Antenna identifier ("N" or "S")
            source: Optional Source object with coordinates
            az: Optional azimuth in degrees
            el: Optional elevation in degrees
            progress_callback: Optional ProgressCallback for detailed progress updates
            
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
            # Setup
            if not self._setup(ant, source):
                raise OperationError("Failed to setup tracker")

            # Validate target coordinates using safety checker
            if source is not None:
                logger.info(f"{Colors.BLUE}Performing target validation...{Colors.RESET}")
                validation_result = self.safety_checker.validate_target(pointing=self.pointing, source=source, ant=ant)
                if not validation_result.is_safe:
                    raise SafetyError(f"Target validation failed: {validation_result.message}")

                logger.info(f"Target coordinates: RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}°")
                slew_success = self._slew(progress_callback=progress_callback)
                if not slew_success:
                    raise OperationError("Slewing failed")

            elif az is not None and el is not None:
                logger.info(f"{Colors.BLUE}Performing target validation...{Colors.RESET}")
                validation_result = self.safety_checker.validate_target(az=az, el=el, ant=ant)
                if not validation_result.is_safe:
                    raise SafetyError(f"Target validation failed: {validation_result.message}")
                
                logger.info(f"Target coordinates: AZ={az:.2f}°, EL={el:.2f}°")
                slew_success = self._slew(az=az, el=el, progress_callback=progress_callback)
                if not slew_success:
                    raise OperationError("Slewing failed")
            else:
                raise ValidationError("No target or coordinates provided")
            
            logger.info(f"{Colors.GREEN}Slewing completed successfully{Colors.RESET}")
            return True

        except (SafetyError, MQTTError, ValidationError, OperationError) as e:
            logger.error(f"{Colors.RED}Slewing failed: {e}{Colors.RESET}")
            self._cleanup()
            raise
        except Exception as e:
            error_msg = f"Unexpected error in slew operation: {e}"
            logger.error(f"{Colors.RED}{error_msg}{Colors.RESET}")
            self._cleanup()
            raise OperationError(error_msg)
        finally:
            if hasattr(self, '_is_initialized') and self._is_initialized:
                self._cleanup()
    
    def run_park(self, ant: str, 
                 progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Park the telescope. Returns True if successful, False otherwise.
        
        Args:
            ant: Antenna identifier ("N" or "S")
            progress_callback: Optional ProgressCallback for detailed progress updates
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValidationError: If input parameters are invalid
            MQTTError: If MQTT communication fails
            OperationError: If parking operation fails
        """
        logger.info(f"{Colors.BLUE}Starting park operation for antenna {ant}{Colors.RESET}")
        
        try:
            # Setup
            if not self._setup(ant):
                raise OperationError("Failed to setup tracker")

            park_success = self._park(progress_callback=progress_callback)
            if not park_success:
                raise OperationError("Parking failed")

            logger.info(f"{Colors.GREEN}Parking completed successfully{Colors.RESET}")
            return True

        except (MQTTError, ValidationError, OperationError) as e:
            # Log the specific error before cleanup
            logger.error(f"{Colors.RED}Parking failed: {e}{Colors.RESET}")
            self._cleanup()
            raise
        except Exception as e:
            # Convert unexpected exceptions to OperationError
            error_msg = f"Unexpected error in park operation: {e}"
            logger.error(f"{Colors.RED}{error_msg}{Colors.RESET}")
            self._cleanup()
            raise OperationError(error_msg)
        finally:
            # Only cleanup if we haven't already (in case of specific exceptions)
            if hasattr(self, '_is_initialized') and self._is_initialized:
                self._cleanup()

    def _setup(self, ant: str, source: Optional[Source] = None) -> bool:
        """
        Setup the tracker for an operation.
        
        Args:
            ant: Antenna identifier
            source: Optional source object
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Validate antenna
            if ant not in ["N", "S"]:
                raise ValidationError(f"Invalid antenna '{ant}'. Must be 'N' or 'S'.")
            
            # Set antenna and source
            self.ant = ant
            if source is not None:
                self.target = source
            
            # Connect to MQTT
            logger.info(f"{Colors.BLUE}Setting up MQTT connection...{Colors.RESET}")
            self.mqtt.setup_mqtt(ant)
            self.mqtt.set_axis_mode_stop()
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            if isinstance(e, (ValidationError, MQTTError)):
                raise
            raise MQTTError(f"Setup failed: {e}")

    def _track(self, duration_hours: float, 
               progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Track a source for a given duration. Returns True if successful, False otherwise.
        """
        logger.info(f"{Colors.BLUE}Starting track_source{Colors.RESET}")
        
        # Check if we're in the right state
        if self.state != State.IDLE:
            raise OperationError(f"Operation not completed. Current state: {self.state.value}. Please wait for current operation to complete.")
        
        # Check if we're close to the target
        offset = self._get_current_offset() 
        if offset > config.telescope.start_tracking_tolerance:
            raise OperationError(f"Too far away from target. Offset: {offset:.2f} degrees.")

        # Setup tracking
        num_points = int(duration_hours * 3600 / config.telescope.position_update_interval)
        logger.info(f"Setting up tracking for {num_points} points ({duration_hours:.2f} hours)")
        
        self.state = State.TRACKING
        self.mqtt.set_axis_mode_track()
        self.mqtt.setup_program_track()
        logger.info(f"{Colors.BLUE}MQTT setup complete{Colors.RESET}")

        # Send start time
        logger.info(f"{Colors.BLUE}Synchronizing start time...{Colors.RESET}")
        sync_to_half_second()
        d0 = datetime.now(timezone.utc)
        d0s = d0.microsecond / 1e6
        if d0s > 0.5:
            d0s = d0s - 0.5
        ds = d0 - timedelta(seconds=d0s)
        sto = 1.173
        sst = ds + timedelta(seconds=3) + timedelta(seconds=sto)
        self.mqtt.send_track_start_time(sst)

        # Track source
        logger.info(f"{Colors.BLUE}Starting tracking loop for {num_points} points ({duration_hours:.4f} hours){Colors.RESET}")
        for i in range(num_points):

            # Check if motion stop requested
            if self.state == State.STOP:
                raise OperationError("Tracking interrupted.")
            
            # Get time
            ds = ds + timedelta(seconds=0.5)
            d = ds + timedelta(seconds=2.5)
            rt = i * 0.5

            # Get azel with detailed logging for first few points
            if i < 3:  # Log details for first 3 points
                # Get coordinates without corrections first
                raw_az, raw_el = self.pointing.radec2azel(self.target, self.ant, d, apply_corrections=False, apply_pointing_model=False, clip=False)
                logger.info(f"Point {i+1} raw coordinates: AZ={raw_az:.4f}°, EL={raw_el:.4f}°")
                
                # Get coordinates with corrections but no pointing model
                corr_az, corr_el = self.pointing.radec2azel(self.target, self.ant, d, apply_corrections=True, apply_pointing_model=False, clip=False)
                logger.info(f"Point {i+1} with corrections: AZ={corr_az:.4f}°, EL={corr_el:.4f}°")
                
                # Get final coordinates with everything applied
                mnt_az, mnt_el = self.pointing.radec2azel(self.target, self.ant, d, apply_corrections=True, apply_pointing_model=True, clip=True)
                logger.info(f"Point {i+1} final coordinates: AZ={mnt_az:.4f}°, EL={mnt_el:.4f}°")
            else:
                # Get final coordinates for remaining points
                mnt_az, mnt_el = self.pointing.radec2azel(self.target, self.ant, d, apply_corrections=True, apply_pointing_model=True, clip=True)
            
            d2 = datetime.now(timezone.utc)

            # Send position command
            self.mqtt.send_track_position(rt, mnt_az, mnt_el, ds)
            
            # Update progress
            percent = (i + 1) / num_points * 100.0
            if progress_callback and (i + 1) % 10 == 0:  # Update every 10 points (5 seconds)
                progress_info = ProgressInfo(
                    operation_type=OperationType.TRACK,
                    antenna=self.ant,
                    current_step=i+1,
                    total_steps=num_points,
                    current_az=mnt_az,
                    current_el=mnt_el,
                    percent_complete=percent,
                    message=f"Point {i+1}/{num_points} - AZ={mnt_az:.2f}°, EL={mnt_el:.2f}°, time={rt:.1f}s"
                )
                progress_callback(progress_info)
            elif (i + 1) % 10 == 0:
                # Log progress every 10 points (5 seconds)
                logger.info(f"{Colors.BLUE}Track progress: Point {i+1}/{num_points} - AZ={mnt_az:.2f}°, EL={mnt_el:.2f}°, time={rt:.1f}s{Colors.RESET}")

        # Log final progress
        logger.info(f"{Colors.GREEN}Track completed: {num_points} points processed{Colors.RESET}")
        
        # Send completion progress
        if progress_callback:
            progress_info = ProgressInfo(
                operation_type=OperationType.TRACK,
                antenna=self.ant,
                current_step=num_points,
                total_steps=num_points,
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

    def _slew(self, az: Optional[float] = None, el: Optional[float] = None, 
              progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Slew to a target position. Returns True if successful, False if interrupted or error.
        
        Args:
            az: Optional azimuth in degrees. If None, uses self.target.
            el: Optional elevation in degrees. If None, uses self.target.
            progress_callback: Optional ProgressCallback for detailed progress updates.
        """
        # Use provided coordinates or calculate from self.target
        if az is None and el is None:
            target_az, target_el = self.pointing.radec2azel(self.target, self.ant, datetime.now(timezone.utc), apply_corrections=True, apply_pointing_model=True)
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
                curr_pos_az, curr_pos_el = self.mqtt.get_current_position()
                percent = (i+1) / len(az_list) * 100.0
                
                if progress_callback:
                    progress_info = ProgressInfo(
                        operation_type=OperationType.SLEW,
                        antenna=self.ant,
                        current_step=i+1,
                        total_steps=len(az_list),
                        current_az=curr_pos_az,
                        current_el=curr_pos_el,
                        target_az=mnt_az,
                        target_el=mnt_el,
                        percent_complete=percent,
                        message=f"Step {i+1}/{len(az_list)}: AZ {curr_pos_az:.2f}° → {mnt_az:.2f}°, EL {curr_pos_el:.2f}° → {mnt_el:.2f}°"
                    )
                    progress_callback(progress_info)
                else:
                    # Log slewing progress every slew_progress_update_interval seconds (configurable)
                    if hasattr(self, '_last_slew_log_time'):
                        if time.time() - self._last_slew_log_time >= config.telescope.slew_progress_update_interval:
                            logger.info(f"{Colors.BLUE}Slewing progress: Step {i+1}/{len(az_list)} - AZ: {curr_pos_az:.2f}° → {mnt_az:.2f}°, EL: {curr_pos_el:.2f}° → {mnt_el:.2f}°{Colors.RESET}")
                            self._last_slew_log_time = time.time()
                    else:
                        self._last_slew_log_time = time.time()
                        logger.info(f"{Colors.BLUE}Slewing progress: Step {i+1}/{len(az_list)} - AZ: {curr_pos_az:.2f}° → {mnt_az:.2f}°, EL: {curr_pos_el:.2f}° → {mnt_el:.2f}°{Colors.RESET}")
        
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
                current_step=len(az_list),
                total_steps=len(az_list),
                current_az=final_az,
                current_el=final_el,
                target_az=target_az,
                target_el=target_el,
                percent_complete=100.0,
                message="Slewing completed",
                is_complete=True
            )
            progress_callback(progress_info)
        
        return True
        
    def _stop(self) -> bool:
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

    def _get_current_offset(self) -> float:
        """
        Get the difference between the current position and the target position in deg.
        """
        if self.target is None:
            raise ValidationError("No target set")
            
        az_target, el_target = self.pointing.radec2azel(self.target, self.ant, datetime.now(timezone.utc), apply_corrections=True, apply_pointing_model=True)
        az_current, el_current = self.mqtt.get_current_position()
        
        if az_current is None or el_current is None:
            raise MQTTError("Can't get current position")
            
        offset = angular_separation(az_current, el_current, az_target, el_target)        
        return offset
    
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
    
    def _cleanup(self) -> None:
        """
        Clean up resources and safely disconnect from MQTT.
        
        Sends stop commands to both axes and properly disconnects the MQTT client.
        Should be called when tracking is complete or interrupted.
        """
        logger.info(f"{Colors.BLUE}Cleaning up resources...{Colors.RESET}")
        
        # Stop motion if MQTT is available
        if hasattr(self, 'mqtt') and self.mqtt is not None:
            try:
                self._stop()
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
        if hasattr(self, 'target') and self.target is not None:
            self.target = None
        if hasattr(self, 'ant') and self.ant is not None:
            self.ant = None
        if hasattr(self, 'state') and self.state is not None:
            self.state = State.IDLE
        self._is_initialized = False

        logger.info(f"{Colors.GREEN}Cleanup complete{Colors.RESET}")