"""
Validates movements for sun safety
"""

import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
import logging
from dataclasses import dataclass
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u
from tracking.core.astro_pointing import AstroPointing
from tracking.utils.antenna import Antenna, parse_antenna
from tracking.utils.colors import Colors
from tracking.utils.config import config
from tracking.utils.source import Source
from tracking.utils.helpers import angular_separation, shortest_azimuth_distance, generate_telescope_path

# Get logger for this module
logger = logging.getLogger(__name__)

@dataclass
class SafetyResult:
    """Result of a safety check."""

    is_safe: bool
    message: str
    separation_deg: Optional[float] = None
    details: Optional[dict] = None

    def log_result(self, logger, operation_name: str = "Safety check") -> None:
        """
        Log the safety result in a clean, consistent format.

        Args:
            logger: Logger instance to use
            operation_name: Name of the operation being checked
        """
        if self.is_safe:
            logger.info(
                f"{Colors.GREEN}{operation_name} PASSED: {self.message}{Colors.RESET}"
            )
            if self.separation_deg is not None:
                logger.info(f"  Sun separation: {self.separation_deg:.1f}°")
        else:
            logger.error(
                f"{Colors.RED}{operation_name} FAILED: {self.message}{Colors.RESET}"
            )
            if self.separation_deg is not None:
                logger.error(f"  Sun separation: {self.separation_deg:.1f}°")

        # Log additional details if available
        if self.details:
            if "target_az" in self.details and "target_el" in self.details:
                logger.info(
                    f"  Target coordinates: AZ={self.details['target_az']:.2f}°, EL={self.details['target_el']:.2f}°"
                )
            if "min_separation_time" in self.details:
                logger.info(
                    f"  Minimum separation time: {self.details['min_separation_time'].strftime('%H:%M:%S')}"
                )
            if "unsafe_time" in self.details:
                logger.error(
                    f"  {Colors.RED}Unsafe at time: {self.details['unsafe_time'].strftime('%H:%M:%S')}{Colors.RESET}"
                )
            if "path_progress_percent" in self.details:
                logger.error(
                    f"  {Colors.RED}Unsafe at path progress: {self.details['path_progress_percent']:.1f}%{Colors.RESET}"
                )


class SafetyChecker:
    """Checks telescope movements for sun safety."""

    def __init__(
        self,
        location: Optional[EarthLocation] = None,
        safety_radius_deg: Optional[float] = None,
    ):
        """
        Initialize safety checker.

        Args:
            location: Observatory location
            safety_radius_deg: Minimum safe angular distance from sun (uses config if None)
        """
        self.location = location or config.telescope.location_m
        self.safety_radius_deg = safety_radius_deg or config.telescope.sun_safety_radius

    def check_position_safety(
        self,
        ra_hrs: Optional[float] = None,
        dec_deg: Optional[float] = None,
        az: Optional[float] = None,
        el: Optional[float] = None,
        check_time: Optional[datetime] = None,
    ) -> SafetyResult:
        """
        Check if a target position is safe with respect to the sun.
        Args:
            ra_hrs: Optional right ascension in hours
            dec_deg: Optional declination in degrees
            az: Optional azimuth in degrees (use with el)
            el: Optional elevation in degrees (use with az)
            check_time: Optional observation time (defaults to now)
        Returns:
            SafetyResult: is_safe, message, separation_deg, details
        """

        if check_time is None:
            check_time = datetime.now(timezone.utc)

        if az is not None and el is not None:
            # Alt/Az mode – compute separation in horizontal frame
            target_az, target_el = az, el
            obs_time = Time(check_time)
            sun_altaz = get_sun(obs_time).transform_to(
                AltAz(location=self.location, obstime=obs_time)
            )
            sun_az = sun_altaz.az.deg
            sun_el = sun_altaz.alt.deg

            separation_deg = angular_separation(target_az, target_el, sun_az, sun_el)
            is_safe = separation_deg >= self.safety_radius_deg

            return SafetyResult(
                is_safe=is_safe,
                message=f"Position AZ={target_az:.1f}°, EL={target_el:.1f}° is {'SAFE' if is_safe else 'UNSAFE'}: separation {separation_deg:.1f}°",
                separation_deg=separation_deg,
                details={
                    "target_az": target_az,
                    "target_el": target_el,
                    "sun_az": sun_az,
                    "sun_el": sun_el,
                    "check_time": check_time,
                },
            )

        elif ra_hrs is not None and dec_deg is not None:
            obs_time = Time(check_time)
            sun = get_sun(obs_time)
            sun_icrs = sun.transform_to("icrs")

            target = SkyCoord(
                ra=ra_hrs * u.hourangle, dec=dec_deg * u.deg, frame="icrs"
            )
            separation_deg = target.separation(sun_icrs).deg
            is_safe = separation_deg >= self.safety_radius_deg

            return SafetyResult(
                is_safe=is_safe,
                message=f"Position RA={ra_hrs:.6f}h, Dec={dec_deg:.6f}° is {'SAFE' if is_safe else 'UNSAFE'}: separation {separation_deg:.1f}°",
                separation_deg=separation_deg,
                details={
                    "ra_hrs": ra_hrs,
                    "dec_deg": dec_deg,
                    "check_time": check_time,
                },
            )

        else:
            raise ValueError("Must provide either ra/dec or az/el coordinates")

    def check_run_safety(
        self,
        ra_hrs: float,
        dec_deg: float,
        duration_hours: float,
        start_time: Optional[datetime] = None,
    ) -> SafetyResult:
        """
        Check if a target position will remain safe from the sun for the duration of observation.
        Args:
            ra_hrs: Right ascension in hours
            dec_deg: Declination in degrees
            duration_hours: Duration to check (in hours)
            start_time: Optional start time (defaults to now)
        Returns:
            SafetyResult: is_safe, message, separation_deg, details
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)

        check_times = np.arange(
            0, duration_hours, config.telescope.sun_future_check_interval
        )
        check_times = [start_time + timedelta(hours=t) for t in check_times]

        separation_degs = []
        for check_time in check_times:
            result = self.check_position_safety(
                ra_hrs=ra_hrs, dec_deg=dec_deg, check_time=check_time
            )
            separation_degs.append(result.separation_deg)
            if not result.is_safe:
                return SafetyResult(
                    is_safe=False,
                    message=f"RUN IS UNSAFE at {check_time.strftime('%H:%M:%S')}: separation {result.separation_deg:.1f}°",
                    separation_deg=result.separation_deg,
                    details={
                        "unsafe_time": check_time,
                        "duration_hours": duration_hours,
                        "start_time": start_time,
                    },
                )

        min_sep = np.min(separation_degs)
        min_idx = np.argmin(separation_degs)

        return SafetyResult(
            is_safe=True,
            message=f"RUN IS SAFE: min separation {min_sep:.1f}° at {check_times[min_idx].strftime('%H:%M:%S')}",
            separation_deg=min_sep,
            details={
                "min_separation_time": check_times[min_idx],
                "duration_hours": duration_hours,
                "start_time": start_time,
            },
        )

    def check_path_safety(
        self,
        current_az: float,
        current_el: float,
        target_az: float,
        target_el: float,
        num_steps: Optional[int] = None,
    ) -> SafetyResult:
        """
        Check if the path between two positions crosses too close to the sun.
        Args:
            current_az: Current azimuth (deg)
            current_el: Current elevation (deg)
            target_az: Target azimuth (deg)
            target_el: Target elevation (deg)
            num_steps: Number of steps to simulate
        Returns:
            SafetyResult: is_safe, message, separation_deg, details
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
        az_path, el_path = generate_telescope_path(
            current_az,
            current_el,
            target_az,
            target_el,
            num_steps,
            az_vel_max=3.75,
            el_vel_max=1.00,
            az_acc_max=2.00,
            el_acc_max=1.00,
        )

        # Check separation at each point
        separations = angular_separation(az_path, el_path, sun_az, sun_alt)
        min_separation = separations.min()
        min_idx = separations.argmin()

        is_safe = min_separation >= self.safety_radius_deg

        return SafetyResult(
            is_safe=is_safe,
            message="PATH IS SAFE"
            if is_safe
            else f"PATH IS UNSAFE at path progress {min_idx / num_steps * 100:.1f}%: separation {min_separation:.1f}°",
            separation_deg=min_separation,
            details={
                "min_separation_idx": min_idx,
                "path_progress_percent": min_idx / num_steps * 100,
                "num_steps": num_steps,
                "sun_az": sun_az,
                "sun_el": sun_alt,
            },
        )

    def validate_target(
        self,
        pointing: AstroPointing = None,
        source: Source = None,
        ant = None,
        az: float = None,
        el: float = None,
        ra_hrs: float = None,
        dec_deg: float = None,
        check_time: Optional[datetime] = None,
    ) -> SafetyResult:
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
            check_time: Optional datetime for sun safety check

        Returns:
            SafetyResult: is_safe, message, separation_deg, details
        """
        try:
            # Determine which mode we're in
            if source is not None:
                # Source mode - calculate az/el from source
                if ant is None:
                    raise ValueError(
                        "Antenna identifier 'ant' is required when using source"
                    )
                # Normalize antenna early (accepts 'N'/'S'/Antenna enum)
                _ = parse_antenna(ant)
                if pointing is None:
                    raise ValueError(
                        "AstroPointing instance is required when using source"
                    )

                target_az, target_el = pointing.radec2azel(
                    source,
                    ant,
                    datetime.now(timezone.utc),
                    apply_corrections=True,
                    apply_pointing_model=True,
                    clip=False,
                )

                # For sun safety check, use the source coordinates
                sun_safety_result = self.check_position_safety(
                    ra_hrs=source.ra_hrs, dec_deg=source.dec_deg, az=None, el=None, check_time=check_time
                )

            elif az is not None and el is not None:
                # Az/El mode - use provided coordinates directly
                target_az, target_el = az, el

                # For sun safety check, use the provided coordinates
                sun_safety_result = self.check_position_safety(
                    ra_hrs=None, dec_deg=None, az=target_az, el=target_el, check_time=check_time
                )

            else:
                raise ValueError(
                    "Must provide either source object or az/el coordinates"
                )

            # Check elevation limits
            if target_el < config.telescope.elevation_min:
                return SafetyResult(
                    is_safe=False,
                    message=f"Elevation {target_el:.2f}° is below minimum {config.telescope.elevation_min}°",
                    details={
                        "target_az": target_az,
                        "target_el": target_el,
                        "elevation_min": config.telescope.elevation_min,
                        "elevation_max": config.telescope.elevation_max,
                    },
                )

            if target_el > config.telescope.elevation_max:
                return SafetyResult(
                    is_safe=False,
                    message=f"Elevation {target_el:.2f}° is above maximum {config.telescope.elevation_max}°",
                    details={
                        "target_az": target_az,
                        "target_el": target_el,
                        "elevation_min": config.telescope.elevation_min,
                        "elevation_max": config.telescope.elevation_max,
                    },
                )

            # Check sun safety
            if not sun_safety_result.is_safe:
                return SafetyResult(
                    is_safe=False,
                    message=f"Target too close to sun: {sun_safety_result.message}",
                    separation_deg=sun_safety_result.separation_deg,
                    details={
                        "target_az": target_az,
                        "target_el": target_el,
                        "sun_safety_details": sun_safety_result.details,
                    },
                )

            return SafetyResult(
                is_safe=True,
                message="Target is above horizon, within limits, and safe from sun",
                separation_deg=sun_safety_result.separation_deg,
                details={
                    "target_az": target_az,
                    "target_el": target_el,
                    "sun_safety_details": sun_safety_result.details,
                },
            )

        except Exception as e:
            message = f"Failed to validate target: {e}"
            return SafetyResult(
                is_safe=False, message=message, details={"error": str(e)}
            )

    def log_path(self, azs: List[float], els: List[float], logger) -> None:
        """
        Analyze path segments and log safety information.

        Args:
            azs: List of azimuth coordinates (including start and end points)
            els: List of elevation coordinates (including start and end points)
            logger: Logger instance to use
        """
        # Log the path waypoints
        logger.info(f"{Colors.BLUE}Path waypoints:{Colors.RESET}")
        for j, (az, el) in enumerate(zip(azs, els), 1):
            logger.info(f"  Point {j}: AZ={az:.1f}°, EL={el:.1f}°")

        # Analyze and log each segment
        logger.info(f"{Colors.BLUE}Segment safety analysis:{Colors.RESET}")
        for i in range(len(azs) - 1):
            res = self.check_path_safety(azs[i], els[i], azs[i + 1], els[i + 1])
            verdict = (
                f"{Colors.GREEN}SAFE{Colors.RESET}"
                if res.is_safe
                else f"{Colors.RED}UNSAFE{Colors.RESET}"
            )
            logger.info(
                f"  Segment {i + 1}: "
                f"({azs[i]:.1f}°, {els[i]:.1f}°) → "
                f"({azs[i + 1]:.1f}°, {els[i + 1]:.1f}°)  "
                f"Min sep = {res.separation_deg:.1f}°  [{verdict}]"
            )

    def get_safe_path(
        self, current_az: float, current_el: float, target_az: float, target_el: float
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate a safe path from current position to target position.

        This method returns a sequence of waypoints that safely moves the telescope
        from (current_az, current_el) to (target_az, target_el) while:

        • Maintaining minimum safe distance from the Sun (≥ safety_radius_deg)
        • Avoiding azimuth wrapping through 360° → 0° boundary
        • Staying within elevation limits [elevation_min, elevation_max]

        Strategy:
        1. First try a direct path - if safe and doesn't cross 360°/0° boundary, use it
        2. If direct path is unsafe, calculate a detour path with multiple waypoints:
           - Move to a safe elevation away from the sun
           - Slew in azimuth at that safe elevation
           - Move to target elevation
           - Complete the azimuth movement to target

        Args:
            current_az: Current azimuth position (degrees)
            current_el: Current elevation position (degrees)
            target_az: Target azimuth position (degrees)
            target_el: Target elevation position (degrees)

        Returns:
            Tuple of (azimuth_list, elevation_list) - waypoints to visit (excluding start point)

        Raises:
            RuntimeError: If no safe path can be found after trying all detour options
        """
        # Get current sun position for logging
        now = Time(datetime.now(timezone.utc))
        sun_altaz = get_sun(now).transform_to(
            AltAz(location=self.location, obstime=now)
        )
        sun_az, sun_el = sun_altaz.az.deg, sun_altaz.alt.deg

        # Trivial direct path test
        direct_path_result = self.check_path_safety(
            current_az, current_el, target_az, target_el
        )
        az_distance = abs(shortest_azimuth_distance(current_az, target_az))

        if direct_path_result.is_safe and az_distance <= 180:
            azs, els = [target_az], [target_el]
            return azs, els

        # Sun position and a few useful constants
        avoid = self.safety_radius_deg  # basic exclusion half-width
        base_pad = config.telescope.detour_base_padding  # start beyond the exclusion
        safe_el_candidates = config.telescope.safe_elevation_candidates

        logger.debug(
            f"Direct path unsafe or crosses 360°/0° boundary, searching for detour path"
        )

        for dir_sign in (+1, -1):
            direction = "clockwise" if dir_sign == +1 else "counter-clockwise"

            # Search for a safe dog-leg path
            for pad in range(
                int(base_pad),
                int(config.telescope.detour_max_padding + 1),
                int(config.telescope.detour_padding_step),
            ):
                detour_az = (sun_az + dir_sign * (avoid + pad)) % 360

                for safe_el in safe_el_candidates:
                    if safe_el < config.telescope.elevation_min + 2:
                        safe_el = config.telescope.elevation_min + 2

                    azs = [current_az, current_az, detour_az, detour_az, target_az]
                    els = [current_el, safe_el, safe_el, target_el, target_el]

                    # quick mechanical-limit check (no wrap, elevation band)
                    mech_ok = True
                    prev = current_az
                    for az in azs[1:]:
                        if abs(shortest_azimuth_distance(prev, az)) > 180:
                            mech_ok = False
                            break
                        prev = az
                    if not mech_ok:
                        continue
                    if not all(
                        config.telescope.elevation_min
                        <= e
                        <= config.telescope.elevation_max
                        for e in els
                    ):
                        continue

                    # safety check for each of the 4 segments
                    safe = all(
                        self.check_path_safety(
                            azs[i], els[i], azs[i + 1], els[i + 1]
                        ).is_safe
                        for i in range(4)
                    )
                    if safe:
                        logger.debug(
                            f"Found safe detour path: AZ={detour_az:.1f}°, EL={safe_el}° (padding: {pad}°)"
                        )
                        # strip the starting point, return only way-points to visit
                        az_list, el_list = azs[1:], els[1:]
                        return az_list, el_list

        logger.error(f"Failed to find any safe path after testing all detour options")
        raise RuntimeError("Could not construct a safe path.")
