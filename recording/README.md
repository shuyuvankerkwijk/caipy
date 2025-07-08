# Recording Package

A modular and standalone Python package for recording data from DSA RFSoC4x2 devices.

## Features

- **Modular Design**: Clean separation of concerns with core, utils, and CLI components
- **Error Handling**: Comprehensive exception handling with graceful error recovery
- **CLI Interface**: Full command-line interface for easy data collection
- **Context Manager**: Safe resource management with context manager support
- **Configuration**: Flexible configuration system with validation
- **Logging**: Structured logging with color-coded output
- **Signal Handling**: Graceful Ctrl+C handling for safe interruption

## Installation

```bash
pip install -e .
```

## Usage

### Python API

```python
from recording import Recorder, logger

# Basic usage with context manager (continuous recording)
with Recorder() as recorder:
    recorder.start_recording(observation_name="continuous_run")  # Records until stopped

# Timed recording
with Recorder() as recorder:
    recorder.start_recording(observation_name="test_run", duration_seconds=60)

# Advanced usage with custom parameters
recorder = Recorder()
recorder.set_fftshift(1000)
recorder.set_acclen(65536)
recorder.set_waittime(0.1)
recorder.start_recording(observation_name="calibration")  # Continuous recording
```

### Command Line Interface

```bash
# Record for 60 seconds
recording --duration 60

# Continuous recording (default behavior - creates new files every 2000 lines)
recording --name "continuous_observation"

# Record with custom name
recording --name "test_observation" --duration 120

# Set FPGA parameters and record continuously
recording --fftshift 1000 --acclen 65536

# Show device status
recording --status

# Show current configuration
recording --config

# Verbose output
recording --verbose --duration 30

# Quiet mode
recording --quiet --duration 60
```

## Package Structure

```
recording/
├── __init__.py          # Main package exports with logging setup
├── core/
│   ├── __init__.py      # Core functionality exports
│   └── recorder.py      # Main Recorder class with signal handling
├── utils/
│   ├── __init__.py      # Utilities exports
│   ├── config.py        # Configuration management with validation
│   ├── exceptions.py    # Custom exception hierarchy
│   └── colors.py        # ANSI color codes for output
└── cli/
    ├── __init__.py      # CLI exports
    └── cli.py          # Command-line interface with error handling
```

## Configuration

The package uses a hierarchical configuration system:

### Device Configuration
- **Hostname**: Device IP address
- **FPGA File**: Path to FPGA bitstream
- **FFT Shift**: Range 0-4095
- **Accumulation Length**: Must be positive

### Recording Configuration
- **Wait Time**: Time between data collections
- **Batch Parameters**: Data buffer and batch sizes
- **Base Directory**: Automatic directory creation

## Error Handling

The package includes comprehensive error handling with specific exceptions:

- `DeviceConnectionError`: Connection issues
- `DeviceInitializationError`: Device setup problems
- `DataCollectionError`: Data collection failures
- `DataSaveError`: File saving issues
- `InvalidParameterError`: Invalid parameter values
- `StateError`: Invalid operation states
- `DirectoryError`: Directory operation issues
- `ConfigurationError`: Configuration problems