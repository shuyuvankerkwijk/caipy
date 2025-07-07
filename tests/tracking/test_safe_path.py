#!/usr/bin/env python3
"""
Test script for the improved get_safe_path method.
Demonstrates sun avoidance path finding with various scenarios.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tracking'))

from safety_checker import SafetyChecker
from helpers import shortest_azimuth_distance
from datetime import datetime, timezone
from astropy.time import Time
from astropy.coordinates import get_sun, AltAz

def test_safe_path():
    """Test the get_safe_path method with various scenarios."""
    
    # Initialize safety checker
    checker = SafetyChecker()
    
    print("=== Testing Improved get_safe_path Method ===\n")
    
    # Get current sun position
    obs_time = Time(datetime.now(timezone.utc))
    sun = get_sun(obs_time)
    sun_altaz = sun.transform_to(AltAz(location=checker.location, obstime=obs_time))
    sun_az = sun_altaz.az.deg
    sun_alt = sun_altaz.alt.deg
    
    print(f"Current sun position: AZ={sun_az:.1f}°, EL={sun_alt:.1f}°")
    print(f"Safety radius: {checker.safety_radius_deg}°")
    print(f"Elevation limits: {checker.location.lat.deg}° (min: 11°, max: 89°)")
    print()
    
    # Test scenarios
    test_cases = [
        {
            "name": "Direct path (should be safe)",
            "current": (sun_az + 60, 45),  # 60° away from sun
            "target": (sun_az + 120, 60),  # 120° away from sun
        },
        {
            "name": "Path through sun (should find avoidance)",
            "current": (sun_az - 30, 45),  # 30° away from sun
            "target": (sun_az + 30, 60),   # 30° away from sun (opposite side)
        },
        {
            "name": "Short azimuth distance (no wrap)",
            "current": (350, 45),  # Near 360°
            "target": (10, 60),    # Near 0°
        },
        {
            "name": "Long azimuth distance (no wrap)",
            "current": (10, 45),   # Near 0°
            "target": (350, 60),   # Near 360°
        },
        {
            "name": "Low elevation target",
            "current": (sun_az + 60, 45),
            "target": (sun_az + 120, 15),  # Near minimum elevation
        },
        {
            "name": "High elevation target",
            "current": (sun_az + 60, 45),
            "target": (sun_az + 120, 85),  # Near maximum elevation
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 50)
        
        current_az, current_el = test_case["current"]
        target_az, target_el = test_case["target"]
        
        print(f"Current: AZ={current_az:.1f}°, EL={current_el:.1f}°")
        print(f"Target:  AZ={target_az:.1f}°, EL={target_el:.1f}°")
        
        # Calculate shortest azimuth distance using helper function
        az_diff = shortest_azimuth_distance(current_az, target_az)
        print(f"Azimuth distance: {az_diff:.1f}°")
        
        # Get safe path
        try:
            path_az, path_el = checker.get_safe_path(current_az, current_el, target_az, target_el)
            
            print(f"Path found: {len(path_az)} waypoints")
            for j, (az, el) in enumerate(zip(path_az, path_el)):
                sep = shortest_azimuth_distance(az, sun_az)
                print(f"  Point {j+1}: AZ={az:.1f}°, EL={el:.1f}° (az distance from sun: {abs(sep):.1f}°)")
            
            # Validate path safety
            print("Validating path safety...")
            for j in range(len(path_az) - 1):
                seg_result = checker.check_path_safety(
                    path_az[j], path_el[j], 
                    path_az[j+1], path_el[j+1]
                )
                print(f"  Segment {j+1}: {'SAFE' if seg_result.is_safe else 'UNSAFE'} "
                      f"(min separation: {seg_result.separation_deg:.1f}°)")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    
    checker = SafetyChecker()
    
    print("=== Testing Edge Cases ===\n")
    
    # Get current sun position
    obs_time = Time(datetime.now(timezone.utc))
    sun = get_sun(obs_time)
    sun_altaz = sun.transform_to(AltAz(location=checker.location, obstime=obs_time))
    sun_az = sun_altaz.az.deg
    sun_alt = sun_altaz.alt.deg
    
    edge_cases = [
        {
            "name": "Very close to sun (challenging case)",
            "current": (sun_az + 35, 45),  # Just outside safety radius
            "target": (sun_az - 35, 60),   # Just outside safety radius
        },
        {
            "name": "Minimum elevation path",
            "current": (sun_az + 60, 11),  # At minimum elevation
            "target": (sun_az + 120, 11),  # At minimum elevation
        },
        {
            "name": "Maximum elevation path",
            "current": (sun_az + 60, 89),  # At maximum elevation
            "target": (sun_az + 120, 89),  # At maximum elevation
        },
    ]
    
    for i, test_case in enumerate(edge_cases, 1):
        print(f"Edge Case {i}: {test_case['name']}")
        print("-" * 50)
        
        current_az, current_el = test_case["current"]
        target_az, target_el = test_case["target"]
        
        print(f"Current: AZ={current_az:.1f}°, EL={current_el:.1f}°")
        print(f"Target:  AZ={target_az:.1f}°, EL={target_el:.1f}°")
        
        try:
            path_az, path_el = checker.get_safe_path(current_az, current_el, target_az, target_el)
            print(f"Path found: {len(path_az)} waypoints")
            
            # Check if all waypoints respect elevation limits
            for j, (az, el) in enumerate(zip(path_az, path_el)):
                if el < 11 or el > 89:
                    print(f"  WARNING: Point {j+1} elevation {el:.1f}° outside limits!")
                else:
                    print(f"  Point {j+1}: AZ={az:.1f}°, EL={el:.1f}°")
                    
        except Exception as e:
            print(f"Error: {e}")
        
        print()

if __name__ == "__main__":
    test_safe_path()
    test_edge_cases() 