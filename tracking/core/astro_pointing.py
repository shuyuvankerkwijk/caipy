import math
import numpy as np
from datetime import datetime, timezone, timedelta
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import astropy.units as u
from typing import Optional
import logging

from tracking.utils.config import config
from tracking.utils.source import Source
from tracking.utils.helpers import xel2az
from tracking.utils.colors import Colors

# Get logger for this module
logger = logging.getLogger(__name__)

class AstroPointing:
    def __init__(self, location : Optional[EarthLocation] = None, temp_C : Optional[float] = None, pressure : Optional[float] = None, rh : Optional[float] = None, n_ppar : Optional[list[float]] = None, s_ppar : Optional[list[float]] = None):
        """
        Pointing model object for a given antenna and source location.

        Args:
            ant (str): Antenna name, in ["N", "S"]
            location (EarthLocation): Observatory location
            temp_C (float): Temperature in Celsius
            pressure (float): Pressure in millibars
            rh (float): Relative humidity in %
            n_ppar (list): North pointing model parameters
            s_ppar (list): South pointing model parameters
        """

        self.location = location or config.telescope.location
        self.temp_C = temp_C or config.telescope.temperature_C
        self.pressure = pressure or config.telescope.pressure_mb
        self.rh = rh or config.telescope.relative_humidity

        # Pointing model
        self.n_ppar = n_ppar or config.telescope.n_ppar
        self.s_ppar = s_ppar or config.telescope.s_ppar

    def radec2azel(self, source : Source, ant : str, obs_datetime : datetime = None, apply_corrections : bool = True, apply_pointing_model : bool = True, clip : bool = True) -> tuple[float, float]:
        # Validate antenna parameter
        if ant not in ["N", "S"]:
            raise ValueError(f"Invalid antenna: {ant}. Must be 'N' or 'S'")
        """
        Calculate star position in horizontal coordinates using Astropy.
        
        Converts celestial coordinates (RA/Dec) to horizontal coordinates (Alt/Az)
        for a specific observer location and time, including proper motion and
        atmospheric refraction corrections.
        
        Args:
            ra_hrs (float): Right ascension in hours
            dec_deg (float): Declination in degrees
            pm_ra (float): Proper motion in RA in mas/yr
            pm_dec (float): Proper motion in Dec in mas/yr
            plx (float): Parallax in mas
            radvel (float): Radial velocity in km/s
            obs_datetime (datetime): Observation time (must be timezone-aware, UTC)
        Returns:
            tuple: (azimuth_deg, elevation_deg)
        """

        # Ensure obs_datetime is timezone-aware and in UTC
        if obs_datetime.tzinfo is None or obs_datetime.tzinfo.utcoffset(obs_datetime) is None:
            raise ValueError("obs_datetime must be timezone-aware and in UTC")
        if obs_datetime.tzinfo != timezone.utc:
            obs_datetime = obs_datetime.astimezone(timezone.utc)
        obs_time = Time(obs_datetime)

        # Create coordinate (ICRS frame)
        coord = SkyCoord(
            ra=source.ra_hrs * u.hourangle,
            dec=source.dec_deg * u.deg,
            pm_ra_cosdec=source.pm_ra * u.mas/u.yr,
            pm_dec=source.pm_dec * u.mas/u.yr,
            distance=1000/source.plx * u.pc,
            radial_velocity=source.radvel * u.km/u.s,
            obstime='J2000'
        )

        # Calculate alt/az
        altaz = coord.transform_to(AltAz(
            location=self.location,
            obstime=obs_time,
            pressure=self.pressure * u.mbar,
            temperature=self.temp_C * u.deg_C
        ))
        az = altaz.az.deg
        el = altaz.alt.deg

        # Apply corrections if requested
        if apply_corrections:
            el = el + config.telescope.sky_el
            az = az + xel2az(config.telescope.sky_xel, el)

        # Apply pointing model if requested
        if apply_pointing_model:
            az, el = self.pointing_model(ant, az, el)
        
        # Check for invalid values
        if np.isnan(el) or np.isnan(az):
            raise ValueError(f"Warning: Invalid az el result.")
        
        # Clip to min/max elevation if requested
        if clip:
            if el < config.telescope.elevation_min:
                el = config.telescope.elevation_min
                logger.warning(f"{Colors.RED}EL below minimum limit: {el:.15f}. Setting to {config.telescope.elevation_min}.{Colors.RESET}")
            if el > config.telescope.elevation_max:
                el = config.telescope.elevation_max
                logger.warning(f"{Colors.RED}EL above maximum limit: {el:.15f}. Setting to {config.telescope.elevation_max}.{Colors.RESET}")

        return az, el
    
    def pointing_model(self, ant, ide_az, ide_el):
        """
        Apply pointing model to convert ideal topocentric coordinates to mount coordinates.
        Applies an 11-parameter pointing model that accounts for various mechanical
        imperfections in the telescope mount.
        [0] - flexure sin
        [1] - flexure cos
        [2] - az tilt y (ha)
        [3] - az tilt x (lat)
        [4] - el tilt
        [5] - collimation x (cross-el)
        [6] - collimation y (el) same as en encdr zero
        [7] - encdr zero az
        [8] - encdr zero el
        [9] - az sin
        [10] - az cos
        
        Args:
            self.model (list): 11-element list of pointing model parameters in degrees
            ide_az (float): Ideal azimuth in degrees
            ide_el (float): Ideal elevation in degrees
            
        Returns:
            tuple: (mount_az, mount_el) - Mount coordinates in degrees
        """ 

        if ant == "N":
            ppar = self.n_ppar
        elif ant == "S":
            ppar = self.s_ppar
        else:
            raise ValueError(f"Invalid antenna: {ant}")

        # Convert parameters and az/el to radians
        par = np.array(ppar, dtype=np.float64) * np.pi / 180
        az = ide_az * np.pi / 180
        el = ide_el * np.pi / 180
        
        # 1. Flexure
        el = el - par[0] * np.sin(el) - par[1] * np.cos(el)
        
        # 2. Az tilt
        x = par[3]  # az tilt x (lat)
        y = par[2]  # az tilt y (ha)
        
        # Precompute trig terms
        cos_x = np.cos(x)
        sin_x = np.sin(x)
        ycos_x = y * cos_x
        cos_ycosx = np.cos(ycos_x)
        sin_ycosx = np.sin(ycos_x)
        
        # Compute the azimuth of the tilt
        top = sin_ycosx
        bot = cos_ycosx * sin_x
        tilt_az = 0 - np.pi - np.arctan2(top, bot)
        
        # Compute the magnitude of the tilt
        tilt_mag = np.arccos(cos_x * cos_ycosx)
        
        # Compute the direction between the azimuth of the source and the azimuth
        # of the axis around which the tilt is directed
        w = tilt_az - np.pi/2 - az
        
        # Precompute trig terms
        sin_w = np.sin(w)
        cos_w = np.cos(w)
        sin_mag = np.sin(tilt_mag)
        cos_mag = np.cos(tilt_mag)
        
        # Compute the new target elevation
        sin_el = np.sin(el) * cos_mag - np.cos(el) * sin_mag * sin_w
        el = np.arcsin(sin_el)
        cos_el = np.cos(el)
        
        # Compute the new target azimuth
        top = cos_w * cos_el
        bot = -(cos_mag * sin_w * cos_el + sin_mag * sin_el)
        az = tilt_az - np.arctan2(top, bot)
        
        # 3. El tilt
        # Check for invalid values that would cause domain errors
        sin_el_over_cos_par4 = np.sin(el) / np.cos(par[4])
        if abs(sin_el_over_cos_par4) <= 1:
            el = np.arcsin(sin_el_over_cos_par4)
            
            # Check for division by zero in tan calculation
            if np.cos(el) != 0:
                tan_par4_sin_el_over_cos_el = np.tan(par[4]) * np.sin(el) / np.cos(el)
                if abs(tan_par4_sin_el_over_cos_el) <= 1:
                    az = az - np.arcsin(tan_par4_sin_el_over_cos_el)
                else:
                    # Return NaN for both if invalid
                    return np.nan, np.nan
            else:
                return np.nan, np.nan
        else:
            # Return NaN for both if invalid
            return np.nan, np.nan
        
        # 4. El collimation (same as el encoder zero point)
        el = el + par[6]
        
        # 5. Cross-el Collimation
        if np.cos(el) != 0:
            az = az + np.arctan(np.tan(par[5]) / np.cos(el))
            
            sin_el_cos_par5 = np.sin(el) * np.cos(par[5])
            if abs(sin_el_cos_par5) <= 1:
                el = np.arcsin(sin_el_cos_par5)
            else:
                return np.nan, np.nan
        else:
            return np.nan, np.nan
        
        # 6. Encoder zero points
        az = az + par[7]
        el = el + par[8]
        
        # 7. Az encoder alignment error
        az = az + par[9] * np.sin(az) + par[10] * np.cos(az)
        
        # Put az into 0 to 2pi range
        while az < 0:
            az += 2 * np.pi
        while az > 2 * np.pi:
            az -= 2 * np.pi
        
        # Convert back to degrees
        return az * 180 / np.pi, el * 180 / np.pi
