import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
import sys

# --- 1. Define the Owens Valley Radio Observatory (OVRO) location ---
# Coordinates for Owens Valley Radio Observatory (OVRO)
# Latitude: 37°14′02″N
# Longitude: 118°16′56″W
# Altitude: 1222 meters
ovro_location = EarthLocation(
    lat=37.233889 * u.deg,  # 37°14′02″N converted to decimal degrees
    lon=-118.282222 * u.deg, # 118°16′56″W converted to decimal degrees (negative for West)
    height=1222 * u.m
)

print(f"Owens Valley Radio Observatory Location:")
print(f"  Latitude: {ovro_location.lat.to_string(unit=u.deg, sep=':', precision=0)}")
print(f"  Longitude: {ovro_location.lon.to_string(unit=u.deg, sep=':', precision=0)}")
print(f"  Altitude: {ovro_location.height:.0f}\n")

# --- 2. Define the celestial object's RA and Dec ---
# Example: A star with known RA and Dec (e.g., Vega)
# RA: 18h 36m 56.33635s
# Dec: +38° 47′ 01.291″
object_ra = sys.argv[1]
object_dec = sys.argv[2]
t = None
if len(sys.argv)>3:
    t = float(sys.argv[3])

celestial_object = SkyCoord(ra=object_ra, dec=object_dec, unit=(u.hourangle,u.deg), frame='icrs')

print(f"Celestial Object (ICRS Frame):")
print(f"  RA: {celestial_object.ra.to_string(unit=u.hourangle, sep=':', precision=2)}")
print(f"  Dec: {celestial_object.dec.to_string(unit=u.deg, sep=':', precision=2)}\n")

# --- 3. Define the observation time ---
# Using the current time for demonstration. You can change this to any specific time.
observation_time = Time.now()
if t is not None:
    observation_time += TimeDelta(t * u.minute)
# For a specific time, uncomment and use:
# observation_time = Time('2024-05-29 20:00:00', scale='utc')

print(f"Observation Time (UTC): {observation_time.utc.iso}\n")

# --- 4. Create the AltAz frame for the observation location and time ---
altaz_frame = AltAz(obstime=observation_time, location=ovro_location)

# --- 5. Transform the celestial object's coordinates to AltAz frame ---
object_altaz = celestial_object.transform_to(altaz_frame)

# --- 6. Print the results ---
print(f"Transformed Azimuth and Elevation for the celestial object at OVRO:")
print(f"  Azimuth (Az): {object_altaz.az.deg}")
print(f"  Elevation (El): {object_altaz.alt.deg}")

# You can also access the values directly:
# print(f"  Azimuth (degrees): {object_altaz.az.deg:.4f}")
# print(f"  Elevation (degrees): {object_altaz.alt.deg:.4f}")
