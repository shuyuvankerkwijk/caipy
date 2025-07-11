#!/usr/bin/env python3
"""
FTX utility module for reading monitoring data and writing control commands.

This module provides functions to interact with the FTX system via HTTP requests.
"""

import requests
import logging
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

# Get logger for this module
logger = logging.getLogger(__name__)


class Polarization(Enum):
    """FTX polarization options."""
    POL0 = 0
    POL1 = 1


class Antenna(Enum):
    """FTX antenna options."""
    NORTH = "N"
    SOUTH = "S"


@dataclass
class FTXMonitorData:
    """Data structure for FTX monitoring data."""
    uid: int
    temperature_c: float
    attenuation_db: float
    rf_power_dbm: float
    pd_current_ua: float
    ld_current_ma: float
    lna_current_ma: float
    lna_voltage_v: float
    vdd_voltage_v: float
    vdda_voltage_v: float


class FTXController:
    """
    Controller class for FTX system operations.
    
    Provides methods to read monitoring data and write control commands.
    """
    
    def __init__(self, antenna: Union[Antenna, str] = Antenna.NORTH, base_url: Optional[str] = None):
        """
        Initialize FTX controller.
        
        Args:
            antenna: Antenna to control (Antenna.NORTH, Antenna.SOUTH, "N", or "S")
            base_url: Optional custom base URL. If None, uses default based on antenna
        """
        # Convert antenna to enum if it's a string
        if isinstance(antenna, str):
            antenna = Antenna.NORTH if antenna.upper() == "N" else Antenna.SOUTH
        
        # Set base URL based on antenna if not provided
        if base_url is None:
            if antenna == Antenna.NORTH:
                base_url = "http://192.168.65.62"
            else:  # SOUTH
                base_url = "http://192.168.65.52"
        
        self.antenna = antenna
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set reasonable timeout for requests
        self.session.timeout = 10.0
        
        logger.info(f"FTX controller initialized for {antenna.value} antenna with base URL: {self.base_url}")
    
    def get_monitor_data(self, polarization: Union[Polarization, int]) -> Optional[FTXMonitorData]:
        """
        Get monitoring data from FTX system.
        
        Args:
            polarization: Polarization to query (0 or 1, or Polarization enum)
            
        Returns:
            FTXMonitorData object if successful, None if failed
            
        Raises:
            requests.exceptions.RequestException: If HTTP request fails
        """
        # Convert polarization to int if it's an enum
        pol = polarization.value if isinstance(polarization, Polarization) else polarization
        
        if pol not in [0, 1]:
            raise ValueError(f"Invalid polarization: {pol}. Must be 0 or 1.")
        
        url = f"{self.base_url}/monitor/{pol}"
        
        try:
            logger.debug(f"Fetching monitor data from: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Convert to dataclass
            monitor_data = FTXMonitorData(
                uid=data['uid'],
                temperature_c=data['temperature_c'],
                attenuation_db=data['attenuation_db'],
                rf_power_dbm=data['rf_power_dbm'],
                pd_current_ua=data['pd_current_ua'],
                ld_current_ma=data['ld_current_ma'],
                lna_current_ma=data['lna_current_ma'],
                lna_voltage_v=data['lna_voltage_v'],
                vdd_voltage_v=data['vdd_voltage_v'],
                vdda_voltage_v=data['vdda_voltage_v']
            )
            
            logger.debug(f"Successfully retrieved monitor data for polarization {pol}")
            return monitor_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching monitor data for polarization {pol}: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing expected field in monitor response: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting monitor data: {e}")
            raise
    
    def set_laser_current(self, polarization: Union[Polarization, int], current_ma: float) -> bool:
        """
        Set laser current for specified polarization.
        
        Args:
            polarization: Polarization to control (0 or 1, or Polarization enum)
            current_ma: Laser current in milliamps (0-50)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If current is out of range
            requests.exceptions.RequestException: If HTTP request fails
        """
        if not 0 <= current_ma <= 50:
            raise ValueError(f"Laser current must be between 0 and 50 mA, got: {current_ma}")
        
        return self._send_control_command(polarization, "Laser", {"current_ma": current_ma})
    
    def set_attenuation(self, polarization: Union[Polarization, int], atten_db: float) -> bool:
        """
        Set attenuation for specified polarization.
        
        Args:
            polarization: Polarization to control (0 or 1, or Polarization enum)
            atten_db: Attenuation in dB (0-31.75 in steps of 0.25)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If attenuation is out of range or not in valid steps
            requests.exceptions.RequestException: If HTTP request fails
        """
        if not 0 <= atten_db <= 31.75:
            raise ValueError(f"Attenuation must be between 0 and 31.75 dB, got: {atten_db}")
        
        # Check if attenuation is in valid steps of 0.25
        if atten_db % 0.25 != 0:
            raise ValueError(f"Attenuation must be in steps of 0.25 dB, got: {atten_db}")
        
        return self._send_control_command(polarization, "Atten", {"atten_db": atten_db})
    
    def set_lna_enabled(self, polarization: Union[Polarization, int], enabled: bool) -> bool:
        """
        Enable or disable LNA for specified polarization.
        
        Args:
            polarization: Polarization to control (0 or 1, or Polarization enum)
            enabled: Whether to enable LNA (True/False)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            requests.exceptions.RequestException: If HTTP request fails
        """
        return self._send_control_command(polarization, "Lna", {"enabled": enabled})
    
    def _send_control_command(self, polarization: Union[Polarization, int], 
                            control_type: str, params: Dict[str, Any]) -> bool:
        """
        Send control command to FTX system.
        
        Args:
            polarization: Polarization to control (0 or 1, or Polarization enum)
            control_type: Type of control (Laser, Atten, Lna)
            params: Parameters for the control command
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            requests.exceptions.RequestException: If HTTP request fails
        """
        # Convert polarization to int if it's an enum
        pol = polarization.value if isinstance(polarization, Polarization) else polarization
        
        if pol not in [0, 1]:
            raise ValueError(f"Invalid polarization: {pol}. Must be 0 or 1.")
        
        url = f"{self.base_url}/control/{pol}"
        payload = {control_type: params}
        
        try:
            logger.debug(f"Sending control command to {url}: {payload}")
            response = self.session.put(url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Successfully sent {control_type} control command for polarization {pol}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending {control_type} control command for polarization {pol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error sending control command: {e}")
            raise
    
    def get_status_summary(self, polarization: Union[Polarization, int]) -> Dict[str, Any]:
        """
        Get a summary of current status for specified polarization.
        
        Args:
            polarization: Polarization to query (0 or 1, or Polarization enum)
            
        Returns:
            Dictionary with status summary
        """
        try:
            monitor_data = self.get_monitor_data(polarization)
            
            return {
                "polarization": polarization.value if isinstance(polarization, Polarization) else polarization,
                "temperature_c": monitor_data.temperature_c,
                "attenuation_db": monitor_data.attenuation_db,
                "rf_power_dbm": monitor_data.rf_power_dbm,
                "laser_current_ma": monitor_data.ld_current_ma,
                "lna_current_ma": monitor_data.lna_current_ma,
                "lna_voltage_v": monitor_data.lna_voltage_v,
                "pd_current_ua": monitor_data.pd_current_ua,
                "vdd_voltage_v": monitor_data.vdd_voltage_v,
                "vdda_voltage_v": monitor_data.vdda_voltage_v
            }
            
        except Exception as e:
            logger.error(f"Error getting status summary for polarization {polarization}: {e}")
            return {"error": str(e)}
    
    def test_connection(self) -> bool:
        """
        Test connection to FTX system.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get monitor data for polarization 0
            self.get_monitor_data(0)
            logger.info("FTX connection test successful")
            return True
        except Exception as e:
            logger.error(f"FTX connection test failed: {e}")
            return False


# Convenience functions for simple operations
def get_ftx_monitor_data(polarization: int, antenna: Union[Antenna, str] = Antenna.NORTH, 
                         base_url: Optional[str] = None) -> Optional[FTXMonitorData]:
    """
    Convenience function to get FTX monitor data.
    
    Args:
        polarization: Polarization to query (0 or 1)
        antenna: Antenna to query (Antenna.NORTH, Antenna.SOUTH, "N", or "S")
        base_url: Optional custom base URL. If None, uses default based on antenna
        
    Returns:
        FTXMonitorData object if successful, None if failed
    """
    controller = FTXController(antenna, base_url)
    return controller.get_monitor_data(polarization)


def set_ftx_laser_current(polarization: int, current_ma: float, 
                         antenna: Union[Antenna, str] = Antenna.NORTH,
                         base_url: Optional[str] = None) -> bool:
    """
    Convenience function to set FTX laser current.
    
    Args:
        polarization: Polarization to control (0 or 1)
        current_ma: Laser current in milliamps (0-50)
        antenna: Antenna to control (Antenna.NORTH, Antenna.SOUTH, "N", or "S")
        base_url: Optional custom base URL. If None, uses default based on antenna
        
    Returns:
        True if successful, False otherwise
    """
    controller = FTXController(antenna, base_url)
    return controller.set_laser_current(polarization, current_ma)


def set_ftx_attenuation(polarization: int, atten_db: float, 
                       antenna: Union[Antenna, str] = Antenna.NORTH,
                       base_url: Optional[str] = None) -> bool:
    """
    Convenience function to set FTX attenuation.
    
    Args:
        polarization: Polarization to control (0 or 1)
        atten_db: Attenuation in dB (0-31.75 in steps of 0.25)
        antenna: Antenna to control (Antenna.NORTH, Antenna.SOUTH, "N", or "S")
        base_url: Optional custom base URL. If None, uses default based on antenna
        
    Returns:
        True if successful, False otherwise
    """
    controller = FTXController(antenna, base_url)
    return controller.set_attenuation(polarization, atten_db)


def set_ftx_lna_enabled(polarization: int, enabled: bool, 
                       antenna: Union[Antenna, str] = Antenna.NORTH,
                       base_url: Optional[str] = None) -> bool:
    """
    Convenience function to enable/disable FTX LNA.
    
    Args:
        polarization: Polarization to control (0 or 1)
        enabled: Whether to enable LNA (True/False)
        antenna: Antenna to control (Antenna.NORTH, Antenna.SOUTH, "N", or "S")
        base_url: Optional custom base URL. If None, uses default based on antenna
        
    Returns:
        True if successful, False otherwise
    """
    controller = FTXController(antenna, base_url)
    return controller.set_lna_enabled(polarization, enabled)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test both antennas
    for antenna_name, antenna_enum in [("North", Antenna.NORTH), ("South", Antenna.SOUTH)]:
        print(f"\n=== Testing {antenna_name} Antenna ===")
        
        # Create controller for this antenna
        controller = FTXController(antenna_enum)
        
        # Test connection
        if not controller.test_connection():
            print(f"Failed to connect to {antenna_name} FTX system")
            continue
        
        print(f"{antenna_name} FTX connection successful!")
        
        # Get monitor data for both polarizations
        for pol in [0, 1]:
            try:
                data = controller.get_monitor_data(pol)
                print(f"\nPolarization {pol} monitor data:")
                print(f"  Temperature: {data.temperature_c:.2f}°C")
                print(f"  Attenuation: {data.attenuation_db:.2f} dB")
                print(f"  RF Power: {data.rf_power_dbm:.2f} dBm")
                print(f"  Laser Current: {data.ld_current_ma:.2f} mA")
                print(f"  LNA Current: {data.lna_current_ma:.2f} mA")
            except Exception as e:
                print(f"Error getting data for polarization {pol}: {e}") 