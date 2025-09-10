# Telescope Tracking Package

Telescope control and tracking system for astronomical observations.

Based on the Tracker.m MATLAB code by S. Padin (5/12/25)

## Example

```python
from tracking import Tracker, Source
from tracking.utils.antenna import Antenna

# Create tracker and source
tracker = Tracker()
source = Source(ra_hrs=12.0, dec_deg=45.0)

# Complete tracking operation (slew, track, park)
success = tracker.run_track(ant=Antenna.NORTH, source=source, duration_hours=1.0)
```

## API Reference

### Tracker Class

**Main Methods:**

#### `run_track(ant, source, duration_hours, slew=True, park=True, progress_callback=None)`
Complete tracking operation with optional slewing and parking.

**Parameters:**
- `ant` (Antenna): Antenna enum (`Antenna.NORTH` or `Antenna.SOUTH`)
- `source` (Source): Source object with coordinates
- `duration_hours` (float): Duration to track in hours
- `slew` (bool): Whether to slew to target before tracking (default: True)
- `park` (bool): Whether to park after tracking (default: True)
- `progress_callback` (ProgressCallback): Optional callback for detailed progress updates

**Returns:** bool - True if successful, False otherwise

#### `run_slew(ant, source=None, az=None, el=None, progress_callback=None)`
Slew to a target position. Can use either RA/Dec coordinates (via source) or Az/El coordinates directly.

**Parameters:**
- `ant` (Antenna): Antenna enum (`Antenna.NORTH` or `Antenna.SOUTH`)
- `source` (Source, optional): Source object with RA/Dec coordinates
- `az` (float, optional): Azimuth in degrees (use with el for Az/El mode)
- `el` (float, optional): Elevation in degrees (use with az for Az/El mode)
- `progress_callback` (ProgressCallback or callable): Optional callback for detailed progress updates

**Returns:** bool - True if successful, False otherwise

**Note:** Must provide either `source` OR both `az` and `el` parameters.

#### `run_park(ant, progress_callback=None)`
Park the telescope to the configured park position.

**Parameters:**
- `ant` (Antenna): Antenna enum (`Antenna.NORTH` or `Antenna.SOUTH`)
- `progress_callback` (ProgressCallback or callable): Optional callback for detailed progress updates

**Returns:** bool - True if successful, False otherwise

### Progress Callbacks

```python
from tracking.utils.progress import create_progress_callback, LoggingProgressCallback
from tracking.utils.antenna import Antenna

# Logging callback (default)
logging_callback = create_progress_callback("logging")

# Simple callback (antenna is an Antenna enum)
def my_callback(antenna: Antenna, percent: float, message: str):
    print(f"{antenna.name}: {percent}% - {message}")

simple_callback = create_progress_callback("simple", callback_func=my_callback)
```

### Source Class

```python
source = Source(
    ra_hrs=14.5,      # Right ascension in hours (0-24)
    dec_deg=-60.0,    # Declination in degrees (-90 to +90)
    pm_ra=100.0,      # Proper motion in RA (mas/yr, default: 0)
    pm_dec=-50.0,     # Proper motion in Dec (mas/yr, default: 0)
    plx=50.0,         # Parallax (mas, default: 0.0001)
    radvel=20.0       # Radial velocity (km/s, default: 0)
)
```

### Error Handling

```python
from tracking import (
    TrackingError, SafetyError, MQTTError, ValidationError,
    ConfigurationError, OperationError, TimeoutError
)
from tracking.utils.antenna import Antenna

try:
    success = tracker.run_track(ant=Antenna.NORTH, source=source, duration_hours=1.0)
except SafetyError as e:
    print(f"Safety violation: {e}")
except MQTTError as e:
    print(f"Communication error: {e}")
except TrackingError as e:
    print(f"General error: {e}")
```

## Command Line Interface

The CLI can be run using Python module syntax. Make sure you're in the project root directory.

### Track Command
Track a source at RA/Dec coordinates with optional slewing and parking.

```bash
# Basic tracking
python -m tracking.cli.cli track -ant N -ra_hrs 12.0 -dec_deg 45.0

# Track with proper motion and parallax
python -m tracking.cli.cli track -ant S -ra_hrs 6.0 -dec_deg 30.0 -pm_ra 100.0 -pm_dec -50.0 -plx 50.0

# Track without slewing or parking
python -m tracking.cli.cli track -ant N -ra_hrs 12.0 -dec_deg 45.0 --no-slew --no-park

# Track with custom duration (number of 0.5s updates)
python -m tracking.cli.cli track -ant N -ra_hrs 12.0 -dec_deg 45.0 -n 6000 # 3000 seconds
```

**Track Command Options:**
- `-ant`: Antenna ("N" or "S") - **Required**
- `-ra_hrs`: Right ascension in hours - **Required**
- `-dec_deg`: Declination in degrees - **Required**
- `-pm_ra`: Proper motion in RA (mas/yr, default: 0)
- `-pm_dec`: Proper motion in Dec (mas/yr, default: 0)
- `-plx`: Parallax (mas, default: 0.0001)
- `-radvel`: Radial velocity (km/s, default: 0)
- `-n, --duration_points`: Number of 0.5s updates to queue (default: 3000)
- `--no-slew`: Skip slewing to source before tracking
- `--no-park`: Skip parking the telescope after tracking

### Slew Command
Slew to a position using either RA/Dec coordinates or Az/El coordinates directly.

```bash
# Slew to RA/Dec coordinates
python -m tracking.cli.cli slew -ant N -ra_hrs 12.0 -dec_deg 45.0

# Slew to Az/El coordinates directly
python -m tracking.cli.cli slew -ant S -az 180.0 -el 45.0

# Slew to RA/Dec with proper motion
python -m tracking.cli.cli slew -ant N -ra_hrs 6.0 -dec_deg 30.0 -pm_ra 100.0 -pm_dec -50.0
```

**Slew Command Options:**
- `-ant`: Antenna ("N" or "S") - **Required**
- **Coordinate System (choose one):**
  - `-ra_hrs`: Right ascension in hours (use with -dec_deg)
  - `-dec_deg`: Declination in degrees (use with -ra_hrs)
  - `-az`: Azimuth in degrees (use with -el)
  - `-el`: Elevation in degrees (use with -az)
- **RA/Dec Mode Options (only used with -ra_hrs/-dec_deg):**
  - `-pm_ra`: Proper motion in RA (mas/yr, default: 0)
  - `-pm_dec`: Proper motion in Dec (mas/yr, default: 0)
  - `-plx`: Parallax (mas, default: 0.0001)
  - `-radvel`: Radial velocity (km/s, default: 0)

### Park Command
Park the telescope to the configured park position.

```bash
# Park telescope
python -m tracking.cli.cli park -ant N
```

**Park Command Options:**
- `-ant`: Antenna ("N" or "S") - **Required**

## Usage Examples

### Python API Examples

```python
from tracking import Tracker, Source
from tracking.utils.antenna import Antenna

tracker = Tracker()

# Track a source with all options
source = Source(
    ra_hrs=14.5,
    dec_deg=-60.0,
    pm_ra=100.0,
    pm_dec=-50.0,
    plx=50.0,
    radvel=20.0
)

# Complete tracking operation
success = tracker.run_track(
    ant=Antenna.NORTH,
    source=source,
    duration_hours=2.0,
    slew=True,
    park=True
)

# Slew to RA/Dec coordinates
success = tracker.run_slew(ant=Antenna.NORTH, source=source)

# Slew to Az/El coordinates directly
success = tracker.run_slew(ant=Antenna.SOUTH, az=180.0, el=45.0)

# Park telescope
success = tracker.run_park(ant=Antenna.NORTH)
```

### Command Line Examples

```bash
# Track a bright star for 1 hour
python -m tracking.cli.cli track -ant N -ra_hrs 14.5 -dec_deg -60.0 -n 7200

# Slew to a specific azimuth/elevation for maintenance
python -m tracking.cli.cli slew -ant S -az 90.0 -el 80.0

# Park both antennas
python -m tracking.cli.cli park -ant N
python -m tracking.cli.cli park -ant S

# Track with high proper motion object
python -m tracking.cli.cli track -ant N -ra_hrs 12.0 -dec_deg 45.0 -pm_ra 500.0 -pm_dec -200.0
```