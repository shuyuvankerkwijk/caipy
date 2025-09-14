"""
General helper functions.
"""

import math
from datetime import datetime, timezone
import time
import numpy as np
from typing import Tuple

def angular_separation(az1: float, el1: float, az2: float, el2: float) -> float:
        """
        Compute angular separation between two alt/az points in degrees.
        """
        # Convert degrees to radians
        az1_rad = np.radians(az1)
        el1_rad = np.radians(el1)
        az2_rad = np.radians(az2)
        el2_rad = np.radians(el2)
        # Spherical law of cosines
        cos_sep = (
            np.sin(el1_rad) * np.sin(el2_rad) +
            np.cos(el1_rad) * np.cos(el2_rad) * np.cos(az1_rad - az2_rad)
        )
        sep_rad = np.arccos(np.clip(cos_sep, -1, 1))
        return np.degrees(sep_rad)

def xel2az(xel: float, el: float) -> float:
    """
    Convert cross-elevation offset to azimuth offset at given elevation.
    
    Calculates the change in azimuth corresponding to a sky cross-elevation
    distance at a specific elevation. This accounts for the fact that the same
    angular distance in the sky corresponds to different azimuth changes at
    different elevations.
    
    Args:
        xel (float): Cross-elevation offset in degrees (positive = right)
        el (float): Elevation angle in degrees
        
    Returns:
        float: Corresponding azimuth offset in degrees
    """
    # Convert to radians
    xel_rad = math.radians(xel)
    el_rad = math.radians(el)
    
    az = math.acos(1 + (math.cos(xel_rad) - 1) / (math.cos(el_rad)**2))
    
    # Handle case where xel > 2*(90-el)
    if isinstance(az, complex):
        az = az.real
        
    if xel < 0:
        az = -az
        
    return math.degrees(az)

def d2m(dt: datetime) -> str:
    """
    Convert Python datetime to MTEX time format string.
    
    Converts a datetime object to the specific string format required by the
    MTEX control system: 'YYYY-MM-DDTHH:MM:SS.sssZ'
    
    Args:
        dt (datetime): Python datetime object (should be UTC)
        
    Returns:
        str: Time string in MTEX format with millisecond precision
    """
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def m2d(m: str) -> datetime:
    """
    Convert MTEX time format string to Python datetime.
    
    Parses an MTEX format time string and returns a timezone-aware datetime
    object in UTC.
    
    Args:
        m (str): Time string in MTEX format 'YYYY-MM-DDTHH:MM:SS.sssZ'
        
    Returns:
        datetime: Timezone-aware datetime object in UTC
    """
    return datetime.strptime(m, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
        
def sync_to_half_second():
    """
    Synchronize execution to the next 0.5 second boundary (UTC clock).
    
    Waits until the system clock reaches a half-second boundary (0.0s or 0.5s).
    If close to the end of a minute (>55s), waits 5 seconds first to avoid
    minute boundary issues. Ensures the next boundary is at least 0.5s in the future.
    """
    now = datetime.now(timezone.utc)
    cs = now.second + now.microsecond / 1e6
    
    # Wait if we are close to the end of a minute
    if cs > 55:
        time.sleep(5)
        now = datetime.now(timezone.utc)
        cs = now.second + now.microsecond / 1e6
        
    # Find the next 0.5s tick that is at least 0.5s in the future
    if (cs - math.floor(cs)) > 0.5:
        target_cs = math.floor(cs) + 1.5
    else:
        target_cs = math.floor(cs) + 1
        
    # Wait for the tick
    while True:
        now = datetime.now(timezone.utc)
        current_cs = now.second + now.microsecond / 1e6
        if current_cs >= target_cs:
            break
        time.sleep(0.001)    

def shortest_azimuth_distance(az1: float, az2: float) -> float:
    """
    Calculate the shortest azimuth distance between two azimuths.
    Considers the no-wrap constraint of the telescope.
    
    Args:
        az1: First azimuth (0-360)
        az2: Second azimuth (0-360)
        
    Returns:
        Shortest distance in degrees (positive or negative)
    """
    diff = az2 - az1
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff
    
def generate_telescope_path(current_az: float, current_el: float,
                            target_az: float, target_el: float,
                            num_steps: int,
                            az_vel_max: float = 3.75,
                            el_vel_max: float = 1.00,
                            az_acc_max: float = 2.00,
                            el_acc_max: float = 1.00) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a realistic telescope path considering velocity and acceleration limits.
        
        Each axis follows a trapezoidal velocity profile:
        - Accelerate at max acceleration until reaching max velocity or midpoint
        - Maintain max velocity (if reached)
        - Decelerate at max acceleration to stop at target
        
        Args:
            current_az, current_el: Starting position (degrees)
            target_az, target_el: Target position (degrees)
            num_steps: Number of points to generate along the path
            az_vel_max, el_vel_max: Maximum velocities (deg/s)
            az_acc_max, el_acc_max: Maximum accelerations (deg/s²)
        
        Returns:
            az_path, el_path: Arrays of positions along the path
        """
        
        def compute_axis_trajectory(start: float, end: float, vel_max: float, acc_max: float, num_points: int) -> np.ndarray:
            """Compute trajectory for a single axis with trapezoidal velocity profile."""
            distance = end - start
            direction = np.sign(distance)
            abs_distance = abs(distance)
            
            # Time to accelerate to max velocity
            t_acc = vel_max / acc_max
            
            # Distance covered during acceleration
            d_acc = 0.5 * acc_max * t_acc**2
            
            # Check if we reach max velocity
            if 2 * d_acc > abs_distance:
                # Triangle profile - we don't reach max velocity
                # We accelerate halfway then decelerate
                t_acc = np.sqrt(abs_distance / acc_max)
                t_total = 2 * t_acc
                t_const = 0
            else:
                # Trapezoidal profile - we reach max velocity
                # Distance at constant velocity
                d_const = abs_distance - 2 * d_acc
                t_const = d_const / vel_max
                t_total = 2 * t_acc + t_const
            
            # Generate time points
            times = np.linspace(0, t_total, num_points)
            positions = np.zeros_like(times)
            
            for i, t in enumerate(times):
                if t <= t_acc:
                    # Acceleration phase
                    positions[i] = start + direction * 0.5 * acc_max * t**2
                elif t <= t_acc + t_const:
                    # Constant velocity phase
                    t_at_const = t - t_acc
                    positions[i] = start + direction * (d_acc + vel_max * t_at_const)
                else:
                    # Deceleration phase
                    t_dec = t - t_acc - t_const
                    # Position at end of constant phase + deceleration distance
                    d_at_const_end = d_acc + vel_max * t_const
                    positions[i] = start + direction * (d_at_const_end + vel_max * t_dec - 0.5 * acc_max * t_dec**2)
            
            return positions
        
        def compute_axis_time(start: float, end: float, vel_max: float, acc_max: float) -> float:
            """Compute time needed for axis to complete motion."""
            distance = abs(end - start)
            t_acc = vel_max / acc_max
            d_acc = 0.5 * acc_max * t_acc**2
            
            if 2 * d_acc > distance:
                # Triangle profile
                return 2 * np.sqrt(distance / acc_max)
            else:
                # Trapezoidal profile
                d_const = distance - 2 * d_acc
                return 2 * t_acc + d_const / vel_max
        
        # Find which axis takes longer (this determines total slew time)
        az_time = compute_axis_time(current_az, target_az, az_vel_max, az_acc_max)
        el_time = compute_axis_time(current_el, target_el, el_vel_max, el_acc_max)
        total_time = max(az_time, el_time)
        
        # Generate synchronized paths
        # Both axes start and stop together, but each follows its own profile
        az_path = compute_axis_trajectory(current_az, target_az, az_vel_max, az_acc_max, num_steps)
        el_path = compute_axis_trajectory(current_el, target_el, el_vel_max, el_acc_max, num_steps)
        
        # If one axis finishes before the other, we need to resample its trajectory
        # to match the slower axis timing
        if abs(az_time - el_time) > 1e-6:  # Not equal
            if az_time < el_time:
                # Azimuth finishes first, need to stretch its trajectory
                # Create time array for original azimuth trajectory
                az_times_orig = np.linspace(0, az_time, num_steps)
                # Create time array for synchronized trajectory
                times_sync = np.linspace(0, total_time, num_steps)
                # Interpolate azimuth positions to synchronized times
                # After az_time, azimuth stays at target position
                az_interp = np.interp(times_sync, 
                                    np.append(az_times_orig, total_time),
                                    np.append(az_path, target_az))
                az_path = az_interp
            else:
                # Elevation finishes first
                el_times_orig = np.linspace(0, el_time, num_steps)
                times_sync = np.linspace(0, total_time, num_steps)
                el_interp = np.interp(times_sync,
                                    np.append(el_times_orig, total_time),
                                    np.append(el_path, target_el))
                el_path = el_interp
        
        return az_path, el_path