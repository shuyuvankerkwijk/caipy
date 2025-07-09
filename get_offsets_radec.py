#!/usr/bin/env python3
"""
Get n offsets from a given point (ra_hrs, dec_deg) in order to do radio pointing. S. Vankerkwijk (6/19/25)
"""
import argparse
import astropy.units as u
from astropy.coordinates import SkyCoord

def get_targets_offsets(amp_offsets_deg: float, num_offsets: int, ra_hrs: float, dec_deg: float):
    """
    Generate a list of targets with offsets from a given point (ra_hrs, dec_deg)
    
    This function creates a cross pattern of observations for radio interferometry
    pointing calibration. The pattern consists of:
    - One measurement at the center (suspected source position)
    - num_offsets-1 measurements centered on the suspected source position
    
    Args: 
        amp_offsets_deg: float, offset in degrees from the center position
        num_offsets: int, number of offsets to generate
        ra_hrs: float, source right ascension in hours
        dec_deg: float, source declination in degrees

    Returns:
        ra_hrs_list: list of floats, RA values in hours
        dec_deg_list: list of floats, Dec values in degrees
    """
    # Create the central coordinate
    center_coord = SkyCoord(ra=ra_hrs*u.hourangle, dec=dec_deg*u.deg)
    
    if num_offsets == 5:
        # 5-point pattern: center + 4 directions (N, E, S, W)
        ra_hrs_list = [center_coord.ra.hour]
        dec_deg_list = [center_coord.dec.deg]
        
        # Generate 4 points around the circle
        for angle in range(0, 360, 90):
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle*u.deg,
                separation=amp_offsets_deg*u.deg
            )
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        return ra_hrs_list, dec_deg_list

    elif num_offsets == 7:
        # 7-point pattern: center + inner triangle + outer triangle at double the offset
        ra_hrs_list = [center_coord.ra.hour]
        dec_deg_list = [center_coord.dec.deg]

        for angle in range(0, 360, 120):
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle*u.deg,
                separation=amp_offsets_deg*u.deg
            )
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        for angle in range(60, 420, 120):
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle*u.deg,
                separation=2*amp_offsets_deg*u.deg)
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        return ra_hrs_list, dec_deg_list
    
    elif num_offsets == 9:
        # 9-point pattern: center + 8 directions (N, NE, E, SE, S, SW, W, NW)
        ra_hrs_list = [center_coord.ra.hour]
        dec_deg_list = [center_coord.dec.deg]
        
        # Generate 8 points around the circle
        for angle in range(0, 360, 45):
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle*u.deg,
                separation=amp_offsets_deg*u.deg
            )
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        return ra_hrs_list, dec_deg_list
    
    elif num_offsets == 13:
        # 13-point pattern: center + inner ring (4 points) + outer ring (8 points)
        ra_hrs_list = [center_coord.ra.hour]
        dec_deg_list = [center_coord.dec.deg]
        
        # Inner ring at offset distance
        for angle in [0, 90, 180, 270]:
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle*u.deg,
                separation=amp_offsets_deg*u.deg
            )
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        # Outer ring at 2*offset distance
        for angle in range(0, 360, 45):
            offset_coord = center_coord.directional_offset_by(
                position_angle=angle*u.deg,
                separation=2*amp_offsets_deg*u.deg
            )
            ra_hrs_list.append(offset_coord.ra.hour)
            dec_deg_list.append(offset_coord.dec.deg)
        
        return ra_hrs_list, dec_deg_list
    
    else:
        raise NotImplementedError(f"Pattern with {num_offsets} offsets not implemented. "
                                  "Available options: 5, 9, 7, or 13 offsets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get n offsets from a given point (ra_hrs, dec_deg) in order to do radio pointing.")
    parser.add_argument("-ra_hrs", type=float, required=True, help="Right ascension (hours)")
    parser.add_argument("-dec_deg", type=float, required=True, help="Declination (degrees)")
    parser.add_argument("-amp_offsets_deg", type=float, required=True, help="Smallest offset in degrees from the center of the array")
    parser.add_argument("-num_offsets", type=int, required=True, help="Number of offsets to generate")
    parser.add_argument("-save", action="store_true", default=False, help="If set, save the offsets to a file")
    args = parser.parse_args()

    ra_hrs_list, dec_deg_list = get_targets_offsets(args.amp_offsets_deg, args.num_offsets, args.ra_hrs, args.dec_deg)

    if args.save:
        with open("offsets.txt", "w") as f:
            for ra_hrs, dec_deg in zip(ra_hrs_list, dec_deg_list):
                f.write(f"{ra_hrs} {dec_deg}\n")

    print("ra_hrs_list", ra_hrs_list)
    print("dec_deg_list", dec_deg_list)