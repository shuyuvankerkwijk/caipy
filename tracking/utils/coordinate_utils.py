#!/usr/bin/env python3
"""
Coordinate utility functions for antenna control and pointing calculations.
Moved from utils.coordinate_utils to tracking.utils.coordinate_utils.
"""
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
from typing import List, Tuple, Optional

# --- OVRO Location Definition ---
OVRO_LOCATION = EarthLocation(
    lat=37.233889 * u.deg,
    lon=-118.282222 * u.deg,
    height=1222 * u.m
)

# Copy of functions -----------------------------------------------------------

def get_targets_offsets(amp_offsets_deg: float, num_offsets: int, ra_hrs: float, dec_deg: float) -> Tuple[List[float], List[float]]:
    """Generate offset target list (see original docstring)."""
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
        for angle in [0, 90, 180, 270]:
            offset_coord = center_coord.directional_offset_by(position_angle=angle * u.deg, separation=amp_offsets_deg * u.deg)
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        for angle in range(0, 360, 45):
            offset_coord = center_coord.directional_offset_by(position_angle=angle * u.deg, separation=2 * amp_offsets_deg * u.deg)
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        return ra_hrs_list, dec_deg_list

    else:
        raise NotImplementedError("Pattern with {num_offsets} offsets not implemented. Available: 5, 7, 9, 13.")


def radec_to_azel(ra_hrs: float, dec_deg: float, time_offset_minutes: Optional[float] = None) -> Tuple[float, float]:
    celestial_object = SkyCoord(ra=ra_hrs, dec=dec_deg, unit=(u.hourangle, u.deg), frame='icrs')
    observation_time = Time.now()
    if time_offset_minutes is not None:
        observation_time += TimeDelta(time_offset_minutes * u.minute)
    altaz_frame = AltAz(obstime=observation_time, location=OVRO_LOCATION)
    object_altaz = celestial_object.transform_to(altaz_frame)
    return object_altaz.az.deg, object_altaz.alt.deg


def get_ovro_location_info() -> str:
    return f"""Owens Valley Radio Observatory Location:\n  Latitude: {OVRO_LOCATION.lat.to_string(unit=u.deg, sep=':', precision=0)}\n  Longitude: {OVRO_LOCATION.lon.to_string(unit=u.deg, sep=':', precision=0)}\n  Altitude: {OVRO_LOCATION.height:.0f} m""" 