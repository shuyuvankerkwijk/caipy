import math
from datetime import datetime, timezone
import time
import numpy as np



def angular_separation(az1: float, el1: float, az2: float, el2: float) -> float:
        """
        Compute angular separation between two alt/az points (degrees).
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
    Synchronize execution to the next 0.5 second boundary.
    
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
    