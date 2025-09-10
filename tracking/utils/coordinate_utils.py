#!/usr/bin/env python3
"""
Coordinate utility functions.
"""

import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
from datetime import datetime
from typing import List, Tuple, Optional

def get_targets_offsets(amp_offsets_deg: float, num_offsets: int, ra_hrs: float, dec_deg: float) -> Tuple[List[float], List[float]]:
    """Generate RA/Dec offset lists around a center.

    Supports patterns with 5, 7, 9, or 13 points as used by pointing-offset scans.

    Args:
        amp_offsets_deg: Closest/first ring angular offset in degrees
        num_offsets: Number of offsets in the pattern (5, 7, 9, or 13)
        ra_hrs: Center right ascension in hours
        dec_deg: Center declination in degrees

    Returns:
        Two lists of equal length: (ra_hrs_list, dec_deg_list)

    Raises:
        NotImplementedError: If the requested pattern size is not supported
    """
    center_coord = SkyCoord(ra=ra_hrs * u.hourangle, dec=dec_deg * u.deg)

    if num_offsets == 5:
        ra_hrs_list = [center_coord.ra.hour]
        dec_deg_list = [center_coord.dec.deg]
        for angle in range(0, 360, 90):
            offset_coord = center_coord.directional_offset_by(position_angle=angle * u.deg, separation=amp_offsets_deg * u.deg)
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        return ra_hrs_list, dec_deg_list

    elif num_offsets == 7:
        ra_hrs_list = [center_coord.ra.hour]
        dec_deg_list = [center_coord.dec.deg]
        for angle in range(0, 360, 120):
            offset_coord = center_coord.directional_offset_by(position_angle=angle * u.deg, separation=amp_offsets_deg * u.deg)
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        for angle in range(60, 420, 120):
            offset_coord = center_coord.directional_offset_by(position_angle=angle * u.deg, separation=2 * amp_offsets_deg * u.deg)
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        return ra_hrs_list, dec_deg_list

    elif num_offsets == 9:
        ra_hrs_list = [center_coord.ra.hour]
        dec_deg_list = [center_coord.dec.deg]
        for angle in range(0, 360, 45):
            offset_coord = center_coord.directional_offset_by(position_angle=angle * u.deg, separation=amp_offsets_deg * u.deg)
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        return ra_hrs_list, dec_deg_list

    elif num_offsets == 13:
        ra_hrs_list = [center_coord.ra.hour]
        dec_deg_list = [center_coord.dec.deg]
        
        # Create concentric triangular patterns for optimal 2D Gaussian fitting
        # Pattern: center + 4 rings of 3 points each at increasing distances
        
        # Ring 1: 3 points at 1×amp_offsets_deg, angles 0°, 120°, 240°
        for angle in [0, 120, 240]:
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle * u.deg, 
                separation=amp_offsets_deg * u.deg
            )
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        # Ring 2: 3 points at 2×amp_offsets_deg, angles 60°, 180°, 300° (rotated 60° from ring 1)
        for angle in [60, 180, 300]:
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle * u.deg, 
                separation=2 * amp_offsets_deg * u.deg
            )
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        # Ring 3: 3 points at 3×amp_offsets_deg, angles 0°, 120°, 240° (aligned with ring 1)
        for angle in [0, 120, 240]:
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle * u.deg, 
                separation=3 * amp_offsets_deg * u.deg
            )
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        # Ring 4: 3 points at 4×amp_offsets_deg, angles 60°, 180°, 300° (rotated 60° from ring 3)
        for angle in [60, 180, 300]:
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle * u.deg, 
                separation=4 * amp_offsets_deg * u.deg
            )
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        return ra_hrs_list, dec_deg_list

    else:
        raise NotImplementedError("Pattern with {num_offsets} offsets not implemented. Available: 5, 7, 9, 13.")