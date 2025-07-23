#!/usr/bin/env python3
"""
complicatedDataRecorder.py. Last updated 2025-06-30. Last updated by Shuyu.
This script should be run in the 'rfsoc' conda environment.
It will automatically call the MS conversion in the 'casa' environment.
"""
import argparse
import signal
import sys
import os
import time
import subprocess
from datetime import datetime
import numpy as np
import dsa_rfsoc4x2

#CONSTANTS
WAITTIME = 0.2  # seconds
FPGFILE = "/home/sprite/TEST_ARRAY/dsa-rfsoc-firmware/firmware/dsa_ta_rfsoc4x2/outputs/dsa_ta_rfsoc4x2_2025-05-21_1326.fpg"
HOSTNAME = '10.10.1.11'
ROWS_PER_BATCH = 2000  # 10 rows per timestamp
ACC_LEN = 131072
FFT_SHIFT_p0 = 1904
FFT_SHIFT_p1 = 1904

# Global variable to handle graceful shutdown
stop_recording = False

def signal_handler(sig, frame):
    """Handle Ctrl+C signal."""
    global stop_recording
    print("\nReceived Ctrl+C, stopping data recording...")
    stop_recording = True

def record_data():
    """Record data from the RFSOC, returns the folder name where data was saved."""
    global stop_recording
    
    # Initialize RFSOC
    print("Initializing RFSOC...")
    d = dsa_rfsoc4x2.DsaRfsoc4x2(HOSTNAME, FPGFILE)
    d.initialize()
    d.cross_corr.set_acc_len(ACC_LEN)
    d.p0_pfb_nc.set_fft_shift(FFT_SHIFT_p0)
    d.p1_pfb_nc.set_fft_shift(FFT_SHIFT_p1)
    d.print_status_all()
    
    # Create save directory
    base_dir = os.path.expanduser('~/vikram/testarray/data')
    os.makedirs(base_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f'run_{run_timestamp}'
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")
    print("Starting data collection...")
    print("Press Ctrl+C to stop recording and proceed with MS conversion")
    
    # Initialize data array
    data = np.zeros((ROWS_PER_BATCH, 8195), dtype=np.complex128)
    idx = 0
    
    def savearray(array):
        """Save array to disk."""
        save_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_dir, f'data_{save_timestamp}.npy')
        np.save(filename, array)
        print(f"Saved array of shape {array.shape} to {filename}")
    
    # Main recording loop
    try:
        while not stop_recording:
            cross_corrs = d.cross_corr.get_new_spectra(wait_on_new=True, get_timestamp=True)
            time_col = np.full((10,1), cross_corrs[2])
            data[idx:idx+10] = np.concatenate([time_col, cross_corrs[0], cross_corrs[1]], axis=1)
            print(f"Wrote 10 lines to {idx}")
            idx += 10
            if idx >= ROWS_PER_BATCH:
                savearray(data)
                idx = 0
            time.sleep(WAITTIME)
    except Exception as e:
        print(f"Error during data recording: {e}")
        # Save any remaining data
        if idx > 0:
            savearray(data[:idx])
    finally:
        # Always save remaining data if any
        if idx > 0:
            print(f"Saving remaining data up to line {idx}")
            savearray(data[:idx])
        else:
            print("Nothing remaining to saveeeee")
    
    return folder_name

def create_measurement_sets_subprocess(folder_name, ra=None, dec=None):
    """Create two measurement sets using subprocess to run in casa environment."""
    
    # Default to North Pole if RA/Dec not provided
    if ra is None:
        ra = 0.0  # North Pole RA
    if dec is None:
        dec = 90.0  # North Pole Dec
    
    print(f"\nCreating measurement sets for folder: {folder_name}")
    print(f"Using RA={ra}, Dec={dec} for fringestopping")
    
    # Get the path to the conda executable
    conda_base = os.environ.get('CONDA_PREFIX', '').replace('/envs/rfsoc', '')
    if not conda_base:
        # Try to find conda base from conda info
        try:
            result = subprocess.run(['conda', 'info', '--base'], 
                                  capture_output=True, text=True, check=True)
            conda_base = result.stdout.strip()
        except:
            print("Warning: Could not determine conda base directory")
            conda_base = os.path.expanduser('~/miniconda3')  # Default guess
    
    # Build the command to run in casa environment
    casa_python = os.path.join(conda_base, 'envs', 'casa', 'bin', 'python')
    
    # Create a temporary Python script for MS conversion
    ms_converter_script = f"""
import sys
sys.path.append('/home/sprite/vikram/testarray')
from preprocessing.raw2ms import raw2ms

# Convert without fringestopping
print("Creating MS without fringestopping...")
raw2ms(filename='{folder_name}', ra_deg={ra}, dec_deg={dec}, 
       fringestop_bool=False, check_bool=False)

# Convert with fringestopping  
print("Creating MS with fringestopping...")
raw2ms(filename='{folder_name}', ra_deg={ra}, dec_deg={dec}, 
       fringestop_bool=True, check_bool=False)
"""
    
    # Write temporary script
    temp_script = 'temp_ms_converter.py'
    with open(temp_script, 'w') as f:
        f.write(ms_converter_script)
    
    try:
        # First, create MS without fringestopping
        print("\n1. Creating MS without fringestopping...")
        cmd = [casa_python, '-c', 
               f"import sys; sys.path.append('/home/sprite/vikram/testarray'); "
               f"from preprocessing.raw2ms import raw2ms; "
               f"raw2ms(filename='{folder_name}', ra_deg={ra}, dec_deg={dec}, "
               f"fringestop_bool=False, check_bool=False)"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error creating MS without fringestopping: {result.stderr}")
            return False
        print("Successfully created MS without fringestopping")
        
        # Second, create MS with fringestopping
        print("\n2. Creating MS with fringestopping...")
        cmd = [casa_python, '-c', 
               f"import sys; sys.path.append('/home/sprite/vikram/testarray'); "
               f"from preprocessing.raw2ms import raw2ms; "
               f"raw2ms(filename='{folder_name}', ra_deg={ra}, dec_deg={dec}, "
               f"fringestop_bool=True, check_bool=False)"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error creating MS with fringestopping: {result.stderr}")
            return False
        print("Successfully created MS with fringestopping")
        
        return True
        
    except Exception as e:
        print(f"Error running MS conversion: {e}")
        return False
    finally:
        # Clean up temporary script if it exists
        if os.path.exists(temp_script):
            os.remove(temp_script)

def create_wrapper_script():
    """Create a wrapper script that handles the environment switching."""
    wrapper_content = '''#!/bin/bash
# Wrapper script for complicatedDataRecorder.py
# This script ensures we're in the correct conda environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate rfsoc environment and run the main script
eval "$(conda shell.bash hook)"
conda activate rfsoc

python3 "$SCRIPT_DIR/complicatedDataRecorder.py" "$@"
'''
    
    wrapper_path = 'run_complicated_data_recorder.sh'
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)
    print(f"Created wrapper script: {wrapper_path}")

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Record data and convert to measurement sets')
    parser.add_argument('-ra_deg', type=float, default=None, 
                       help='Right Ascension in degrees (default: North Pole)')
    parser.add_argument('-dec_deg', type=float, default=None, 
                       help='Declination in degrees (default: North Pole)')
    parser.add_argument('--create-wrapper', action='store_true',
                       help='Create a bash wrapper script for easy execution')
    args = parser.parse_args()
    
    # Option to create wrapper script
    if args.create_wrapper:
        create_wrapper_script()
        return 0
    
    # Check if we're in the rfsoc environment
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env != 'rfsoc':
        print(f"Warning: Current conda environment is '{current_env}', expected 'rfsoc'")
        print("Please run: conda activate rfsoc")
        print("Or use: ./run_complicated_data_recorder.sh (after creating it with --create-wrapper)")
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Record data
    try:
        folder_name = record_data()
        print(f"\nData recording completed. Folder: {folder_name}")
    except Exception as e:
        print(f"Error during data recording: {e}")
        return 1
    
    # Give a moment for files to be written
    time.sleep(1)
    
    # Create measurement sets in casa environment
    try:
        if create_measurement_sets_subprocess(folder_name, args.ra_deg, args.dec_deg):
            print("\nSuccessfully created both measurement sets!")
            base_path = f"data/{folder_name}/{folder_name}"
            print(f"  - Without fringestopping: {base_path}.ms")
            print(f"  - With fringestopping: {base_path}_fs.ms")
            return 0
        else:
            print("\nFailed to create measurement sets")
            return 1
    except Exception as e:
        print(f"Error creating measurement sets: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# """
# complicatedDataRecorder.py. Last updated 2025-06-30. Last updated by Shuyu.
# """
# import argparse
# import signal
# import sys
# import os
# import time
# from datetime import datetime
# import numpy as np
# import dsa_rfsoc4x2

# from preprocessing.raw2ms import raw2ms

# #CONSTANTS
# WAITTIME = 0.2  # seconds
# FPGFILE = "/home/sprite/TEST_ARRAY/dsa-rfsoc-firmware/firmware/dsa_ta_rfsoc4x2/outputs/dsa_ta_rfsoc4x2_2025-05-21_1326.fpg"
# HOSTNAME = '10.10.1.11'
# ROWS_PER_BATCH = 2000  # 10 rows per timestamp
# ACC_LEN = 131072
# FFT_SHIFT_p0 = 1904
# FFT_SHIFT_p1 = 1904

# # Global variable to handle graceful shutdown
# stop_recording = False

# def signal_handler(sig, frame):
#     """Handle Ctrl+C signal."""
#     global stop_recording
#     print("\nReceived Ctrl+C, stopping data recording...")
#     stop_recording = True

# def record_data():
#     """Record data from the RFSOC, returns the folder name where data was saved."""
#     global stop_recording
    
#     # Initialize RFSOC
#     print("Initializing RFSOC...")
#     d = dsa_rfsoc4x2.DsaRfsoc4x2(HOSTNAME, FPGFILE)
#     d.initialize()
#     d.cross_corr.set_acc_len(ACC_LEN)
#     d.p0_pfb_nc.set_fft_shift(FFT_SHIFT_p0)
#     d.p1_pfb_nc.set_fft_shift(FFT_SHIFT_p1)
#     d.print_status_all()
    
#     # Create save directory
#     base_dir = os.path.expanduser('~/vikram/testarray/data')
#     run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     folder_name = f'run_{run_timestamp}'
#     save_dir = os.path.join(base_dir, folder_name)
#     os.makedirs(save_dir, exist_ok=True)
#     print(f"Saving to: {save_dir}")
#     print("Starting data collection...")
#     print("Press Ctrl+C to stop recording and proceed with MS conversion")
    
#     # Initialize data array
#     data = np.zeros((ROWS_PER_BATCH, 8195), dtype=np.complex128)
#     idx = 0
    
#     def savearray(array):
#         """Save array to disk."""
#         save_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         filename = os.path.join(save_dir, f'data_{save_timestamp}.npy')
#         np.save(filename, array)
#         print(f"Saved array of shape {array.shape} to {filename}")
    
#     # Main recording loop
#     try:
#         while not stop_recording:
#             cross_corrs = d.cross_corr.get_new_spectra(wait_on_new=True, get_timestamp=True)
#             time_col = np.full((10,1), cross_corrs[2])
#             data[idx:idx+10] = np.concatenate([time_col, cross_corrs[0], cross_corrs[1]], axis=1)
#             print(f"Wrote 10 lines to {idx}")
#             idx += 10
#             if idx >= ROWS_PER_BATCH:
#                 savearray(data)
#                 idx = 0
#             time.sleep(WAITTIME)
#     except Exception as e:
#         print(f"Error during data recording: {e}")
#         # Save any remaining data
#         if idx > 0:
#             savearray(data[:idx])
#     finally:
#         # Always save remaining data if any
#         if idx > 0:
#             print(f"Saving remaining data up to line {idx}")
#             savearray(data[:idx])
#         else:
#             print("Nothing remaining to saveeeee")
    
#     return folder_name

# def create_measurement_sets(folder_name, ra=None, dec=None):
#     """Create two measurement sets: one without and one with fringestopping."""
    
#     # Default to North Pole if RA/Dec not provided
#     if ra is None:
#         ra = 0.0  # North Pole RA
#     if dec is None:
#         dec = 90.0  # North Pole Dec
    
#     print(f"\nCreating measurement sets for folder: {folder_name}")
#     print(f"Using RA={ra}, Dec={dec} for fringestopping")
    
#     # First, create MS without fringestopping
#     print("\n1. Creating MS without fringestopping...")
#     raw2ms(filename=folder_name, ra_deg=ra, dec_deg=dec, fringestop_bool=False, check_bool=False)
    
#     # Second, create MS with fringestopping
#     print("\n2. Creating MS with fringestopping...")
#     raw2ms(filename=folder_name, ra_deg=ra, dec_deg=dec, fringestop_bool=True, check_bool=False)
    
#     return True

# def main():
#     """Main execution function."""

#     parser = argparse.ArgumentParser(description='Record data and convert to measurement sets')
#     parser.add_argument('-ra_deg', type=float, default=None, 
#                        help='Right Ascension in degrees (default: North Pole)')
#     parser.add_argument('-dec_deg', type=float, default=None, 
#                        help='Declination in degrees (default: North Pole)')
#     args = parser.parse_args()
    
#     # Set up signal handler for Ctrl+C
#     signal.signal(signal.SIGINT, signal_handler)
    
#     # Record data
#     try:
#         folder_name = record_data()
#         print(f"\nData recording completed. Folder: {folder_name}")
#     except Exception as e:
#         print(f"Error during data recording: {e}")
#         return 1
    
#     # Give a moment for files to be written
#     time.sleep(1)
    
#     # Create measurement sets
#     try:
#         if create_measurement_sets(folder_name, args.ra_deg, args.dec_deg):
#             print("\nSuccessfully created both measurement sets!")
#             base_path = f"data/{folder_name}/{folder_name}"
#             print(f"  - Without fringestopping: {base_path}.ms")
#             print(f"  - With fringestopping: {base_path}_fs.ms")
#             return 0
#         else:
#             print("\nFailed to create measurement sets")
#             return 1
#     except Exception as e:
#         print(f"Error creating measurement sets: {e}")
#         return 1

# if __name__ == "__main__":
#     sys.exit(main())