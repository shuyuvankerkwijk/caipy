#!/usr/bin/env python3
"""
Command-line interface for the telescope tracking system.

This module provides CLI tools for tracking, slewing, and parking operations.
"""

import argparse
import sys
import logging
from tracking.core.tracker import Tracker
from tracking.utils.source import Source
from tracking.utils.progress import ProgressCallback, ProgressInfo, OperationType
from tracking.utils.exceptions import SafetyError, MQTTError, ValidationError, OperationError, ConfigurationError, TimeoutError
from tracking.utils.antenna import Antenna, parse_antenna, antenna_letter

# Get logger for this module
logger = logging.getLogger(__name__)


class CLIProgressCallback(ProgressCallback):
    """Progress callback for CLI output."""
    
    def __init__(self, antenna: Antenna):
        self.antenna = antenna
        self.last_percent = 0
    
    def __call__(self, progress_info: ProgressInfo) -> None:
        """Print progress to console."""
        if progress_info.antenna == self.antenna:
            ant_code = antenna_letter(self.antenna)
            if progress_info.is_complete:
                print(f"\n{ant_code} {progress_info.operation_type.value}: Complete")
            elif progress_info.error:
                print(f"\n{ant_code} {progress_info.operation_type.value}: Error - {progress_info.error}")
            else:
                # Only print if percent changed significantly
                if progress_info.percent_complete - self.last_percent >= 5.0:
                    print(f"{ant_code} {progress_info.operation_type.value}: {progress_info.percent_complete:.1f}% - {progress_info.message}")
                    self.last_percent = progress_info.percent_complete

def track_and_park(ant: str, ra_hrs: float = None, dec_deg: float = None, pm_ra: float = 0, pm_dec: float = 0, 
                  plx: float = 0.0001, radvel: float = 0, duration_points: int = 3000, slew: bool = True, park: bool = True,
                  source: Source = None, progress_callback: ProgressCallback = None):
    """
    Track a source at the given RA/Dec coordinates and optionally park the telescope.
    
    Args:
        ant: Antenna identifier ("N" or "S")
        ra_hrs: Right ascension in hours (required if source not provided)
        dec_deg: Declination in degrees (required if source not provided)
        pm_ra: Proper motion in RA (mas/yr)
        pm_dec: Proper motion in Dec (mas/yr)
        plx: Parallax (mas)
        radvel: Radial velocity (km/s)
        duration_points: Number of 0.5s updates to queue (default 3000)
        slew: Whether to slew to source before tracking (default True)
        park: Whether to park the telescope after tracking (default True)
        source: Source object (alternative to providing ra_hrs/dec_deg)
    
    Returns:
        bool: True if tracking was successful, False otherwise
    """
    try:
        ant_enum = parse_antenna(ant)
    except Exception as e:
        logger.error(f"Error: {e}")
        return False
    
    tracker = Tracker()
    
    try:
        # Create source object if not provided
        if source is None:
            if ra_hrs is None or dec_deg is None:
                logger.error("Error: Either source object or both ra_hrs and dec_deg must be provided")
                return False
            source = Source(ra_hrs=ra_hrs, dec_deg=dec_deg, pm_ra=pm_ra, pm_dec=pm_dec, plx=plx, radvel=radvel)
        
        # Calculate duration in hours
        duration_hours = float(duration_points) * 0.5 / 3600
        
        # Create CLI progress callback if none provided
        if progress_callback is None:
            progress_callback = CLIProgressCallback(ant_enum)
        
        # Run tracking with optional slewing and parking
        success = tracker.run_track(ant_enum, source, duration_hours, slew=slew, park=park, progress_callback=progress_callback)
        
        if success:
            logger.info(f"Successfully tracked source at RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}° for {duration_hours:.2f} hours")
            if park:
                logger.info("Telescope parked after tracking")
        else:
            logger.error("Tracking failed")
            
        return success
        
    except KeyboardInterrupt:
        logger.warning("Tracking interrupted by user")
        return False
    except (SafetyError, MQTTError, ValidationError, OperationError, ConfigurationError, TimeoutError) as e:
        logger.error(f"Error during tracking: {e}")
        return False
    except Exception as exc:
        logger.error(f"Unexpected error during tracking: {exc}")
        if __debug__:
            import traceback
            traceback.print_exc()
        return False

def slew_to_position(ant: str, ra_hrs: float = None, dec_deg: float = None, pm_ra: float = 0, pm_dec: float = 0, 
                    plx: float = 0.0001, radvel: float = 0, az: float = None, el: float = None, source: Source = None,
                    progress_callback: ProgressCallback = None):
    """
    Slew the telescope to a specific position. Can use either RA/Dec coordinates or Az/El coordinates.
    
    Args:
        ant: Antenna identifier ("N" or "S")
        ra_hrs: Right ascension in hours (required if using RA/Dec mode and source not provided)
        dec_deg: Declination in degrees (required if using RA/Dec mode and source not provided)
        pm_ra: Proper motion in RA (mas/yr)
        pm_dec: Proper motion in Dec (mas/yr)
        plx: Parallax (mas)
        radvel: Radial velocity (km/s)
        az: Azimuth in degrees (required if using Az/El mode)
        el: Elevation in degrees (required if using Az/El mode)
        source: Source object (alternative to providing ra_hrs/dec_deg)
    
    Returns:
        bool: True if slewing was successful, False otherwise
    """
    try:
        ant_enum = parse_antenna(ant)
    except Exception as e:
        logger.error(f"Error: {e}")
        return False
    
    tracker = Tracker()
    
    try:
        # Create CLI progress callback if none provided
        if progress_callback is None:
            progress_callback = CLIProgressCallback(ant_enum)
        
        # Determine which mode to use
        if az is not None and el is not None:
            # Az/El mode
            logger.info(f"Slewing to Az={az:.2f}°, El={el:.2f}°")
            success = tracker.run_slew(ant_enum, az=az, el=el, progress_callback=progress_callback)
        elif source is not None or (ra_hrs is not None and dec_deg is not None):
            # RA/Dec mode
            if source is None:
                source = Source(ra_hrs=ra_hrs, dec_deg=dec_deg, pm_ra=pm_ra, pm_dec=pm_dec, plx=plx, radvel=radvel)
            logger.info(f"Slewing to RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}°")
            success = tracker.run_slew(ant_enum, source=source, progress_callback=progress_callback)
        else:
            logger.error("Error: Must provide either az/el coordinates or ra_hrs/dec_deg coordinates or source object")
            return False
        
        if success:
            if az is not None and el is not None:
                logger.info(f"Successfully slewed to Az={az:.2f}°, El={el:.2f}°")
            else:
                logger.info(f"Successfully slewed to RA={source.ra_hrs:.6f}h, Dec={source.dec_deg:.6f}°")
        else:
            logger.error("Slewing failed")
            
        return success
        
    except KeyboardInterrupt:
        logger.warning("Slewing interrupted by user")
        return False
    except (SafetyError, MQTTError, ValidationError, OperationError, ConfigurationError, TimeoutError) as e:
        logger.error(f"Error during slewing: {e}")
        return False
    except Exception as exc:
        logger.error(f"Unexpected error during slewing: {exc}")
        if __debug__:
            import traceback
            traceback.print_exc()
        return False

def park_telescope(ant: str, progress_callback: ProgressCallback = None):
    """
    Park the telescope to the configured park position.
    
    Args:
        ant: Antenna identifier ("N" or "S")
    
    Returns:
        bool: True if parking was successful, False otherwise
    """
    try:
        ant_enum = parse_antenna(ant)
    except Exception as e:
        logger.error(f"Error: {e}")
        return False
    
    tracker = Tracker()
    
    try:
        # Create CLI progress callback if none provided
        if progress_callback is None:
            progress_callback = CLIProgressCallback(ant_enum)
        
        # Park the telescope
        success = tracker.run_park(ant_enum, progress_callback=progress_callback)
        
        if success:
            logger.info("Successfully parked telescope")
        else:
            logger.error("Parking failed")
            
        return success
        
    except KeyboardInterrupt:
        logger.warning("Parking interrupted by user")
        return False
    except (SafetyError, MQTTError, ValidationError, OperationError, ConfigurationError, TimeoutError) as e:
        logger.error(f"Error during parking: {e}")
        return False
    except Exception as exc:
        logger.error(f"Unexpected error during parking: {exc}")
        if __debug__:
            import traceback
            traceback.print_exc()
        return False


def track_multiple_positions(
    ant: str,
    positions: list,
    pm_ra: float = 0,
    pm_dec: float = 0,
    plx: float = 0.0001,
    radvel: float = 0,
    duration_points: int = 3000,
    slew: bool = True,
    park: bool = True,
    progress_callback: ProgressCallback = None,
):
    """
    Track multiple sources sequentially for a fixed duration each.

    Args:
        ant: Antenna identifier ("N" or "S")
        positions: List of (ra_hrs, dec_deg) tuples
        pm_ra: Proper motion in RA (mas/yr) applied to all sources
        pm_dec: Proper motion in Dec (mas/yr) applied to all sources
        plx: Parallax (mas) applied to all sources
        radvel: Radial velocity (km/s) applied to all sources
        duration_points: Number of 0.5s updates to queue per source
        slew: Whether to slew to first source before tracking
        park: Whether to park telescope after tracking
        progress_callback: Optional progress callback

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        ant_enum = parse_antenna(ant)
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

    if not positions:
        logger.error("Error: At least one position must be provided")
        return False

    tracker = Tracker()

    try:
        sources = [
            Source(ra_hrs=float(ra), dec_deg=float(dec), pm_ra=pm_ra, pm_dec=pm_dec, plx=plx, radvel=radvel)
            for (ra, dec) in positions
        ]

        duration_hours = float(duration_points) * 0.5 / 3600

        if progress_callback is None:
            progress_callback = CLIProgressCallback(ant_enum)

        success = tracker.run_track_multiple(
            ant_enum,
            sources,
            duration_hours,
            slew=slew,
            park=park,
            operation_type=OperationType.TRACK,
            progress_callback=progress_callback,
        )

        if success:
            logger.info(
                f"Successfully tracked {len(sources)} sources for {duration_hours:.2f} hours each"
            )
            if park:
                logger.info("Telescope parked after tracking")
        else:
            logger.error("Track multiple failed")

        return success

    except KeyboardInterrupt:
        logger.warning("Multiple tracking interrupted by user")
        return False
    except (SafetyError, MQTTError, ValidationError, OperationError, ConfigurationError, TimeoutError) as e:
        logger.error(f"Error during multiple tracking: {e}")
        return False
    except Exception as exc:
        logger.error(f"Unexpected error during multiple tracking: {exc}")
        if __debug__:
            import traceback
            traceback.print_exc()
        return False

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Telescope control and tracking system")
    
    # Global logging options
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--quiet', action='store_true', help='Only show errors')

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Track command
    track_parser = subparsers.add_parser('track', help='Track a source at RA/Dec coordinates')
    track_parser.add_argument("-ant", type=str, required=True, help="N (North Antenna) or S (South Antenna)")
    track_parser.add_argument("-ra_hrs", type=float, required=True,help="Right ascension (hours)")
    track_parser.add_argument("-dec_deg", type=float, required=True,help="Declination (degrees)")
    track_parser.add_argument("-pm_ra", type=float, default=0, help="Proper motion in RA (mas/yr)")
    track_parser.add_argument("-pm_dec", type=float, default=0, help="Proper motion in Dec (mas/yr)")
    track_parser.add_argument("-plx", type=float, default=0.0001, help="Parallax (mas)")
    track_parser.add_argument("-radvel", type=float, default=0, help="Radial velocity (km/s)")
    track_parser.add_argument("-n", "--duration_points", type=int, default=3000, help="Number of 0.5s updates to queue (default 3000)")
    track_parser.add_argument("--no-slew", action="store_true", help="Don't slew to source")
    track_parser.add_argument("--no-park", action="store_true", help="Don't park the telescope after tracking")
    
    # Slew command
    slew_parser = subparsers.add_parser('slew', help='Slew to a position (RA/Dec or Az/El coordinates)')
    slew_parser.add_argument("-ant", type=str, required=True, help="N (North Antenna) or S (South Antenna)")
    
    # Create a mutually exclusive group for coordinate systems
    coord_group = slew_parser.add_mutually_exclusive_group(required=True)
    coord_group.add_argument("-ra_hrs", type=float, help="Right ascension (hours) - use with -dec_deg")
    coord_group.add_argument("-az", type=float, help="Azimuth in degrees - use with -el")
    
    slew_parser.add_argument("-dec_deg", type=float, help="Declination (degrees) - use with -ra_hrs")
    slew_parser.add_argument("-el", type=float, help="Elevation in degrees - use with -az")
    slew_parser.add_argument("-pm_ra", type=float, default=0, help="Proper motion in RA (mas/yr) - only for RA/Dec mode")
    slew_parser.add_argument("-pm_dec", type=float, default=0, help="Proper motion in Dec (mas/yr) - only for RA/Dec mode")
    slew_parser.add_argument("-plx", type=float, default=0.0001, help="Parallax (mas) - only for RA/Dec mode")
    slew_parser.add_argument("-radvel", type=float, default=0, help="Radial velocity (km/s) - only for RA/Dec mode")
    
    # Track multiple command
    tm_parser = subparsers.add_parser('track_multiple', help='Track multiple RA/Dec positions sequentially')
    tm_parser.add_argument("-ant", type=str, required=True, help="N (North Antenna) or S (South Antenna)")
    tm_parser.add_argument(
        "-pos",
        "--position",
        action="append",
        nargs=2,
        type=float,
        metavar=("RA_HRS", "DEC_DEG"),
        required=True,
        help="Add a position as RA(hours) DEC(deg); repeat for multiple positions",
    )
    tm_parser.add_argument("-pm_ra", type=float, default=0, help="Proper motion in RA (mas/yr)")
    tm_parser.add_argument("-pm_dec", type=float, default=0, help="Proper motion in Dec (mas/yr)")
    tm_parser.add_argument("-plx", type=float, default=0.0001, help="Parallax (mas)")
    tm_parser.add_argument("-radvel", type=float, default=0, help="Radial velocity (km/s)")
    tm_parser.add_argument("-n", "--duration_points", type=int, default=3000, help="Number of 0.5s updates to queue per source (default 3000)")
    tm_parser.add_argument("--no-slew", action="store_true", help="Don't slew to the first source")
    tm_parser.add_argument("--no-park", action="store_true", help="Don't park the telescope after tracking")

    # Park command
    park_parser = subparsers.add_parser('park', help='Park the telescope')
    park_parser.add_argument("-ant", type=str, required=True, help="N (North Antenna) or S (South Antenna)")
    
    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format='[%(levelname)s] %(message)s')
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    try:
        if args.command == 'track':
            # Validate that either ra_hrs/dec_deg are provided together
            if (args.ra_hrs is not None) != (args.dec_deg is not None):
                logger.error("Error: Both -ra_hrs and -dec_deg must be provided together")
                sys.exit(1)
            
            success = track_and_park(
                ant=args.ant,
                ra_hrs=args.ra_hrs,
                dec_deg=args.dec_deg,
                pm_ra=args.pm_ra,
                pm_dec=args.pm_dec,
                plx=args.plx,
                radvel=args.radvel,
                duration_points=args.duration_points,
                slew=not args.no_slew,
                park=not args.no_park
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'slew':
            # Validate coordinate pairs
            if args.ra_hrs is not None and args.dec_deg is None:
                logger.error("Error: -dec_deg must be provided with -ra_hrs")
                sys.exit(1)
            if args.dec_deg is not None and args.ra_hrs is None:
                logger.error("Error: -ra_hrs must be provided with -dec_deg")
                sys.exit(1)
            if args.az is not None and args.el is None:
                logger.error("Error: -el must be provided with -az")
                sys.exit(1)
            if args.el is not None and args.az is None:
                logger.error("Error: -az must be provided with -el")
                sys.exit(1)
            
            success = slew_to_position(
                ant=args.ant,
                ra_hrs=args.ra_hrs,
                dec_deg=args.dec_deg,
                pm_ra=args.pm_ra,
                pm_dec=args.pm_dec,
                plx=args.plx,
                radvel=args.radvel,
                az=args.az,
                el=args.el
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'park':
            success = park_telescope(ant=args.ant)
            sys.exit(0 if success else 1)
        
        elif args.command == 'track_multiple':
            # Ensure at least one position is present
            if not args.position:
                logger.error("Error: At least one -pos/--position must be provided")
                sys.exit(1)
            success = track_multiple_positions(
                ant=args.ant,
                positions=args.position,
                pm_ra=args.pm_ra,
                pm_dec=args.pm_dec,
                plx=args.plx,
                radvel=args.radvel,
                duration_points=args.duration_points,
                slew=not args.no_slew,
                park=not args.no_park,
            )
            sys.exit(0 if success else 1)
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if __debug__:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 