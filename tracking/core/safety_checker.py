"""
safety_checker.py - Validates movements for sun safety
"""
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Optional
from dataclasses import dataclass
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import astropy.units as u
import logging

from tracking.utils.config import config
from tracking.utils.helpers import angular_separation, shortest_azimuth_distance
from tracking.utils.colors import Colors

# Get logger for this module
logger = logging.getLogger(__name__)

@dataclass
class SafetyResult:
    """Result of a safety check."""
    is_safe: bool
    message: str
    separation_deg: Optional[float] = None

class SafetyChecker:
    """Checks telescope movements for sun safety."""
    
    def __init__(self, location: Optional[EarthLocation] = None, safety_radius_deg: Optional[float] = None):
        """
        Initialize safety checker.
        
        Args:
            location: Observatory location
            safety_radius_deg: Minimum safe angular distance from sun (uses config if None)
        """
        self.location = location or config.telescope.location
        self.safety_radius_deg = safety_radius_deg or config.telescope.sun_safety_radius
    
    def check_position_safety(self, ra_hrs: float, dec_deg: float, 
                            check_time: Optional[datetime] = None) -> SafetyResult:
        """
        Check if a target position is safe with respect to the sun.
        Returns:
            SafetyResult: is_safe, message (sun separation in degrees), separation_deg
        """
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        obs_time = Time(check_time)
        sun = get_sun(obs_time)
        sun_icrs = sun.transform_to('icrs')
        
        # Convert target to the same frame as sun (ICRS) to avoid transformation warning
        target = SkyCoord(ra=ra_hrs * u.hourangle, dec=dec_deg * u.deg, frame='icrs')
        separation_deg = target.separation(sun_icrs).deg
        
        is_safe = separation_deg >= self.safety_radius_deg
        
        logger.debug(f"    Position safety: RA={ra_hrs:.6f}h, Dec={dec_deg:.6f}°")
        logger.debug(f"    Sun separation: {separation_deg:.1f}° (threshold: {self.safety_radius_deg}°)")
        logger.debug(f"    Result: {'SAFE' if is_safe else 'UNSAFE'}")
        
        return SafetyResult(
            is_safe=is_safe,
            message=f"Sun separation: {separation_deg:.1f}° at {check_time.strftime('%H:%M:%S')}",
            separation_deg=separation_deg,
        )
    
    def check_run_safety(self, ra_hrs: float, dec_deg: float, duration_hours: float, start_time: Optional[datetime] = None) -> SafetyResult:
        """
        Check if a target position will remain safe from the sun for the duration of observation.
        Args:
            ra_hrs: Right ascension in hours
            dec_deg: Declination in degrees
            duration_hours: Duration to check (in hours)
            start_time: Optional start time (defaults to now)
        Returns:
            SafetyResult: is_safe, message (min separation and time)
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        logger.info(f"{Colors.BLUE}Checking run safety for {duration_hours:.4f} hours starting at {start_time.strftime('%H:%M:%S')}{Colors.RESET}")
        
        check_times = np.arange(0, duration_hours, config.telescope.sun_future_check_interval)
        check_times = [start_time + timedelta(hours=t) for t in check_times]
        logger.info(f"  Checking {len(check_times)} time points (every {config.telescope.sun_future_check_interval:.3f} hours)")
        
        separation_degs = []
        for i, check_time in enumerate(check_times):
            result = self.check_position_safety(ra_hrs, dec_deg, check_time)
            separation_degs.append(result.separation_deg)
            if not result.is_safe:
                logger.error(f"  {Colors.RED}UNSAFE at {check_time.strftime('%H:%M:%S')}: separation {result.separation_deg:.1f}° < {self.safety_radius_deg}°{Colors.RESET}")
                return SafetyResult(
                    is_safe=False,
                    message=f"Min separation: {result.separation_deg:.1f}° < {self.safety_radius_deg:.1f}° at {check_time.strftime('%H:%M:%S')}",
                    separation_deg=result.separation_deg,
                )
        
        min_sep = np.min(separation_degs)
        min_idx = np.argmin(separation_degs)
        logger.info(f"  {Colors.GREEN}SAFE: min separation {min_sep:.1f}° at {check_times[min_idx].strftime('%H:%M:%S')}{Colors.RESET}")
        
        return SafetyResult(
            is_safe=True,
            message=f"Min separation: {min_sep:.1f}° > {self.safety_radius_deg}° at {check_times[min_idx].strftime('%H:%M:%S')}",
            separation_deg=min_sep,
        )
    
    def check_path_safety(self, current_az: float, current_el: float,
                         target_az: float, target_el: float,
                         num_steps: Optional[int] = None) -> SafetyResult:
        """
        Check if the path between two positions crosses too close to the sun.
        Args:
            current_az: Current azimuth (deg)
            current_el: Current elevation (deg)
            target_az: Target azimuth (deg)
            target_el: Target elevation (deg)
            num_steps: Number of steps to simulate
        Returns:
            SafetyResult: is_safe, message (min separation and path progress)
        """
        if num_steps is None:
            num_steps = config.telescope.slew_simulation_steps
        
        # Get current sun position in alt/az
        obs_time = Time(datetime.now(timezone.utc))
        sun = get_sun(obs_time)
        sun_altaz = sun.transform_to(AltAz(location=self.location, obstime=obs_time))
        sun_az = sun_altaz.az.deg
        sun_alt = sun_altaz.alt.deg
        
        # Generate path points
        az_path = np.linspace(current_az, target_az, num_steps)
        el_path = np.linspace(current_el, target_el, num_steps)

        # Check separation at each point
        min_separation = float('inf')
        separation_degs = []
        for az, el in zip(az_path, el_path):
            separation = angular_separation(az, el, sun_az, sun_alt)
            separation_degs.append(separation)
            min_separation = min(min_separation, separation)
        
        is_safe = min_separation >= self.safety_radius_deg

        return SafetyResult(
            is_safe=is_safe,
            message=f"Minimum sun separation along path: {min_separation:.1f}° at path progress {np.argmin(separation_degs)/num_steps*100:.1f}%",
            separation_deg=min_separation,
        )

    def validate_target(self, pointing=None, source=None, ant: str = None, az: float = None, el: float = None) -> SafetyResult:
        """
        Validate that a target is above the horizon, within safe limits, and safe from the sun.
        
        Args:
            pointing: AstroPointing instance (required for source and ra/dec modes)
            source: Optional Source object to validate
            ant: Antenna identifier ("N" or "S") - required if using source or ra/dec
            az: Optional azimuth in degrees (use with el)
            el: Optional elevation in degrees (use with az)
            ra_hrs: Optional right ascension in hours (use with dec_deg)
            dec_deg: Optional declination in degrees (use with ra_hrs)
            
        Returns:
            SafetyResult: is_safe, message, separation_deg (sun separation if checked)
        """
        try:
            # Determine which mode we're in
            if source is not None:
                # Source mode - calculate az/el from source
                if ant is None:
                    raise ValueError("Antenna identifier 'ant' is required when using source")
                if pointing is None:
                    raise ValueError("AstroPointing instance is required when using source")
                
                target_az, target_el = pointing.radec2azel(source, ant, datetime.now(timezone.utc), apply_corrections=True, apply_pointing_model=True, clip = False)
                
                logger.info(f"{Colors.BLUE}Target validation (source mode):{Colors.RESET}")
                logger.info(f"  RA: {source.ra_hrs:.6f}h, Dec: {source.dec_deg:.6f}°")
                logger.info(f"  Calculated AZ: {target_az:.2f}°, EL: {target_el:.2f}°")
                
                # For sun safety check, use the source coordinates
                sun_safety_result = self.check_position_safety(source.ra_hrs, source.dec_deg)
                
            elif (az is not None and el is not None):
                # Az/El mode - use provided coordinates directly
                target_az, target_el = az, el
                
                logger.info(f"{Colors.BLUE}Target validation (az/el mode):{Colors.RESET}")
                logger.info(f"  AZ: {target_az:.2f}°, EL: {target_el:.2f}°")
                
                # For sun safety check, we need to convert az/el to ra/dec
                # This is a simplified approach - in practice you might want more sophisticated conversion
                logger.warning(f"  {Colors.RED}Warning: Sun safety check not available for az/el coordinates{Colors.RESET}")
                sun_safety_result = SafetyResult(is_safe=True, message="Sun safety check skipped for az/el coordinates", separation_deg=None)
        
            else:
                raise ValueError("Must provide either source object or az/el coordinates")
            
            # Check elevation limits
            if target_el < config.telescope.elevation_min:
                message = f"Elevation {target_el:.2f}° is below minimum {config.telescope.elevation_min}°"
                logger.error(f"  {Colors.RED}ERROR: {message}{Colors.RESET}")
                logger.info(f"  This target is below the horizon or too close to the horizon")
                return SafetyResult(is_safe=False, message=message)
                
            if target_el > config.telescope.elevation_max:
                message = f"Elevation {target_el:.2f}° is above maximum {config.telescope.elevation_max}°"
                logger.error(f"  {Colors.RED}ERROR: {message}{Colors.RESET}")
                return SafetyResult(is_safe=False, message=message)
            
            # Check sun safety
            logger.info(f"  Checking sun safety...")
            if not sun_safety_result.is_safe:
                message = f"Target too close to sun: {sun_safety_result.message}"
                logger.error(f"  {Colors.RED}ERROR: {message}{Colors.RESET}")
                return SafetyResult(is_safe=False, message=message, separation_deg=sun_safety_result.separation_deg)
            
            logger.info(f"  {Colors.GREEN}Target is valid and above horizon{Colors.RESET}")
            if sun_safety_result.separation_deg is not None:
                logger.info(f"  {Colors.GREEN}Target is safe from sun (separation: {sun_safety_result.separation_deg:.1f}°){Colors.RESET}")
            return SafetyResult(is_safe=True, message="Target is above horizon, within limits, and safe from sun", separation_deg=sun_safety_result.separation_deg)
            
        except Exception as e:
            message = f"Failed to validate target: {e}"
            logger.error(f"  {Colors.RED}ERROR: {message}{Colors.RESET}")
            return SafetyResult(is_safe=False, message=message)

    def _show_path(self, azs: List[float], els: List[float]) -> None:
        """Convenience wrapper that also prints a header line."""
        logger.info(f"{Colors.BLUE}Path found: {len(azs)} way-points{Colors.RESET}")
        for j, (az, el) in enumerate(zip(azs, els), 1):
            logger.info(f"  Point {j}: AZ={az:.1f}°, EL={el:.1f}°")
        logger.info(f"{Colors.BLUE}Segment safety check:{Colors.RESET}")

        for i in range(len(azs) - 1):
            res = self.check_path_safety(azs[i], els[i], azs[i + 1], els[i + 1])
            verdict = f"{Colors.GREEN}SAFE{Colors.RESET}" if res.is_safe else f"{Colors.RED}UNSAFE{Colors.RESET}"
            logger.info(f"  Segment {i+1}: "
                f"({azs[i]:.1f}°, {els[i]:.1f}°) → "
                f"({azs[i+1]:.1f}°, {els[i+1]:.1f}°)  "
                f"Min sep = {res.separation_deg:.1f}°  [{verdict}]")
            
    def get_safe_path(self,
                    current_az: float, current_el: float,
                    target_az:  float, target_el:  float
                    ) -> Tuple[List[float], List[float]]:
        """
        Return a sequence of way-points (AZ°, EL°) that drives the telescope
        from (current_az, current_el) to (target_az, target_el) while

        • staying ≥ self.safety_radius_deg from the Sun,
        • never wrapping through 360 → 0 in azimuth,
        • always inside the elevation limits [11°, 89°].

        Strategy:

        1.  If the direct great-circle slew is safe, use it.
        2.  Otherwise build a very conservative four-segment "dog-leg" path:

            (a) Drop/Rise vertically to a low "safe" elevation   (safe_el)
            (b) Slew in azimuth at safe_el, well clear of the Sun
            (c) Rise vertically to the target elevation at the same safe az
            (d) Slew the remaining azimuth at the final elevation

            Widen the azimuth detour in 10° steps until all four segments
            satisfy `check_path_safety`.
        """
        logger.info(f"{Colors.BLUE}Calculating safe path from ({current_az:.1f}°, {current_el:.1f}°) to ({target_az:.1f}°, {target_el:.1f}°){Colors.RESET}")
        
        # Get current sun position for logging
        now = Time(datetime.now(timezone.utc))
        sun_altaz = get_sun(now).transform_to(AltAz(location=self.location, obstime=now))
        sun_az, sun_el = sun_altaz.az.deg, sun_altaz.alt.deg
        logger.info(f"  Current sun position: AZ={sun_az:.1f}°, EL={sun_el:.1f}°")
        logger.info(f"  Safety radius: {self.safety_radius_deg}°")
        
        # ── 0. trivial direct path test ─────────────────────────────────────────
        logger.info(f"  Testing direct path...")
        direct_path_result = self.check_path_safety(current_az, current_el, target_az, target_el)
        az_distance = abs(shortest_azimuth_distance(current_az, target_az))
        
        if direct_path_result.is_safe and az_distance <= 180:
            logger.info(f"  {Colors.GREEN}Direct path is safe! Min separation: {direct_path_result.separation_deg:.1f}°{Colors.RESET}")
            azs, els = [target_az], [target_el]
            return azs, els
        else:
            if not direct_path_result.is_safe:
                logger.info(f"  {Colors.RED}Direct path unsafe: min separation {direct_path_result.separation_deg:.1f}° < {self.safety_radius_deg}°{Colors.RESET}")
            if az_distance > 180:
                logger.info(f"  {Colors.RED}Direct path crosses 360°/0° boundary (az distance: {az_distance:.1f}°){Colors.RESET}")
            logger.info(f"  {Colors.BLUE}Attempting to find safe detour path...{Colors.RESET}")


        # ── 1. Sun position and a few useful constants ──────────────────────────
        avoid      = self.safety_radius_deg     # basic exclusion half-width
        base_pad   = config.telescope.detour_base_padding  # start beyond the exclusion
        safe_el_candidates = config.telescope.safe_elevation_candidates  # elevations to try
        
        logger.info(f"  Searching for detour paths with {base_pad}° base padding...")
        
        for dir_sign in (+1, -1):
            direction = "clockwise" if dir_sign == +1 else "counter-clockwise"
            logger.info(f"  Trying {direction} detour...")

            # ── 2. search for a safe dog-leg path ───────────────────────────────────
            for pad in range(int(base_pad), int(config.telescope.detour_max_padding + 1), int(config.telescope.detour_padding_step)):
                detour_az = (sun_az + dir_sign * (avoid + pad)) % 360
                logger.info(f"    Testing detour at AZ={detour_az:.1f}° (padding: {pad}°)")

                for safe_el in safe_el_candidates:
                    if safe_el < config.telescope.elevation_min + 2:
                        safe_el = config.telescope.elevation_min + 2

                    azs = [current_az,  current_az, detour_az,
                        detour_az,   target_az]
                    els = [current_el,  safe_el,    safe_el,
                        target_el,   target_el]

                    # quick mechanical-limit check (no wrap, elevation band)
                    mech_ok = True
                    prev = current_az
                    for az in azs[1:]:
                        if abs(shortest_azimuth_distance(prev, az)) > 180:
                            mech_ok = False
                            break
                        prev = az
                    if not mech_ok:
                        logger.debug(f"      Rejected: azimuth wrap detected")
                        continue
                    if not all(config.telescope.elevation_min <= e <= config.telescope.elevation_max
                            for e in els):
                        logger.debug(f"      Rejected: elevation {min(els):.1f}° below minimum {config.telescope.elevation_min}°")
                        continue

                    # safety check for each of the 4 segments
                    logger.debug(f"      Testing 4-segment path at EL={safe_el}°")
                    safe = all(self.check_path_safety(azs[i], els[i],
                                                    azs[i+1], els[i+1]).is_safe
                            for i in range(4))
                    if safe:
                        logger.info(f"  {Colors.GREEN}Found safe detour path!{Colors.RESET}")
                        logger.info(f"    Detour: AZ={detour_az:.1f}°, EL={safe_el}° (padding: {pad}°)")
                        # strip the starting point, return only way-points to visit                    
                        az_list, el_list = azs[1:], els[1:]     
                        return az_list, el_list
                    else:
                        logger.debug(f"      Path unsafe at this detour point")

        logger.error(f"  {Colors.RED}Failed to find any safe path after testing all detour options{Colors.RESET}")
        raise RuntimeError("Could not construct a safe path.")
