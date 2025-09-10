"""
Astronomical source representation.
"""

from dataclasses import dataclass
from .exceptions import ValidationError

@dataclass
class Source:
    """
    Represents an astronomical source with coordinates and proper motion.
    
    Attributes:
        ra_hrs: Right ascension in hours (0-24)
        dec_deg: Declination in degrees (-90 to +90)
        pm_ra: Proper motion in RA direction (mas/yr, default: 0)
        pm_dec: Proper motion in Dec direction (mas/yr, default: 0)
        plx: Parallax (mas, default: 0.0001)
        radvel: Radial velocity (km/s, default: 0)
    """
    
    ra_hrs: float
    dec_deg: float
    pm_ra: float = 0.0
    pm_dec: float = 0.0
    plx: float = 0.0001
    radvel: float = 0.0
    
    def __post_init__(self):
        """Validate source parameters after initialization."""
        if not (0 <= self.ra_hrs <= 24):
            raise ValidationError(f"RA must be between 0 and 24 hours, got {self.ra_hrs}")
        
        if not (-90 <= self.dec_deg <= 90):
            raise ValidationError(f"Dec must be between -90 and +90 degrees, got {self.dec_deg}")
    
    def __str__(self) -> str:
        """String representation of the source."""
        return f"Source(RA={self.ra_hrs:.6f}h, Dec={self.dec_deg:.6f}°)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Source(ra_hrs={self.ra_hrs}, dec_deg={self.dec_deg}, "
                f"pm_ra={self.pm_ra}, pm_dec={self.pm_dec}, "
                f"plx={self.plx}, radvel={self.radvel})")