"""
Command-line interface for the recording package.
"""
import argparse
import sys
import signal
import logging
from ..core import Recorder
from ..utils.exceptions import RecordingError, ConfigurationError, DeviceConnectionError, DeviceInitializationError, DataCollectionError, DataSaveError, InvalidParameterError, DirectoryError, StateError
from ..utils.colors import Colors

# Get logger for this module
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.warning(f"{Colors.RED}Received interrupt signal. Stopping recording...{Colors.RESET}")
    sys.exit(0)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data recording tool for DSA RFSoC4x2 device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global logging options
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--quiet', action='store_true', help='Only show errors')

    # Recording options
    parser.add_argument(
        '--name', '-n',
        type=str,
        help='Observation name for the recording session'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        help='Recording duration in seconds (if not specified, records continuously until interrupted)'
    )
    parser.add_argument(
        '--waittime', '-w',
        type=float,
        help='Wait time between data collections in seconds (default: 0.2)'
    )
    
    # Device configuration
    parser.add_argument(
        '--fftshift',
        type=int,
        help='FFT shift value (0-4095)'
    )
    parser.add_argument(
        '--acclen',
        type=int,
        help='Accumulation length'
    )
    
    # Information options
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show device status and exit'
    )
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show current configuration and exit'
    )
    
    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format='[%(levelname)s] %(message)s')
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    recorder = None
    try:
        # Initialize recorder
        recorder = Recorder()
        
        # Handle status-only mode
        if args.status:
            status = recorder.get_status()
            print("Device Status:")
            print(status)
            return 0
        
        # Handle config-only mode
        if args.config:
            print("Current Configuration:")
            print(f"  FFT Shift: {recorder.get_fftshift()}")
            print(f"  Accumulation Length: {recorder.get_acclen()}")
            print(f"  Wait Time: {recorder.get_waittime()} seconds")
            print(f"  Save Directory: {recorder.save_directory}")
            return 0
        
        # Configure device parameters
        if args.fftshift is not None:
            logger.info(f"Setting FFT shift to {args.fftshift} (both polarizations)")
            recorder.set_fftshift(args.fftshift, args.fftshift)
        
        if args.acclen is not None:
            logger.info(f"Setting accumulation length to {args.acclen}")
            recorder.set_acclen(args.acclen)
        
        if args.waittime is not None:
            logger.info(f"Setting wait time to {args.waittime} seconds")
            recorder.set_waittime(args.waittime)
        
        # Show configuration
        logger.info("Current Configuration:")
        logger.info(f"  FFT Shift: {recorder.get_fftshift()}")
        logger.info(f"  Accumulation Length: {recorder.get_acclen()}")
        logger.info(f"  Wait Time: {recorder.get_waittime()} seconds")
        logger.info("")
        
        # Set observation name if provided
        if args.name:
            recorder.set_observation_name(args.name)
            logger.info(f"Observation name set to: {args.name}")
        
        # Start recording
        if args.duration:
            logger.info(f"Starting recording for {args.duration} seconds...")
        else:
            logger.info("Starting continuous recording (will create new files every 2000 lines, press Ctrl+C to stop)...")
        
        success = recorder.start_recording(duration_seconds=args.duration)
        
        if success:
            logger.info(f"{Colors.GREEN}Recording completed successfully{Colors.RESET}")
        else:
            logger.info("Recording was interrupted.")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning(f"{Colors.RED}Recording interrupted by user{Colors.RESET}")
        return 130
    except (RecordingError, ConfigurationError, DeviceConnectionError, DeviceInitializationError, DataCollectionError, DataSaveError, InvalidParameterError, DirectoryError, StateError) as e:
        logger.error(f"{Colors.RED}Recording error: {e}{Colors.RESET}")
        return 1
    except Exception as e:
        logger.error(f"{Colors.RED}Unexpected error: {e}{Colors.RESET}")
        if __debug__:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        if recorder is not None:
            try:
                recorder.cleanup()
            except Exception:
                pass


if __name__ == '__main__':
    sys.exit(main()) 