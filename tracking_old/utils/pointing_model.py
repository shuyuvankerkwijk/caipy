"""
11-parameter pointing model for telescope mount corrections.
"""

from dataclasses import dataclass

@dataclass
class PointingModel:
    """
    11-parameter pointing model for telescope mount corrections.

    This model accounts for various mechanical imperfections in the telescope mount:
    - Flexure corrections (sin and cos components)
    - Azimuth tilt corrections (in HA and latitude directions)
    - Elevation tilt correction
    - Collimation errors (cross-elevation and elevation)
    - Encoder zero point offsets
    - Azimuth encoder alignment errors (sin and cos components)
    """

    flex_sin: float = 0.0  # [0] - flexure sin component (degrees)
    flex_cos: float = 0.0  # [1] - flexure cos component (degrees)
    az_tilt_y: float = 0.0  # [2] - az tilt y (ha) (degrees)
    az_tilt_x: float = 0.0  # [3] - az tilt x (lat) (degrees)
    el_tilt: float = 0.0  # [4] - el tilt (degrees)
    collim_x: float = 0.0  # [5] - collimation x (cross-el) (degrees)
    collim_y: float = 0.0  # [6] - collimation y (el) / el encoder zero (degrees)
    az_zero: float = 0.0  # [7] - encoder zero az (degrees)
    el_zero: float = 0.0  # [8] - encoder zero el (degrees)
    az_sin: float = 0.0  # [9] - az sin alignment error (degrees)
    az_cos: float = 0.0  # [10] - az cos alignment error (degrees)

    @classmethod
    def from_list(cls, params: list[float]) -> "PointingModel":
        """Create PointingModel from a list of 11 parameters."""
        if len(params) != 11:
            raise ValueError(f"Expected 11 parameters, got {len(params)}")

        return cls(
            flex_sin=params[0],
            flex_cos=params[1],
            az_tilt_y=params[2],
            az_tilt_x=params[3],
            el_tilt=params[4],
            collim_x=params[5],
            collim_y=params[6],
            az_zero=params[7],
            el_zero=params[8],
            az_sin=params[9],
            az_cos=params[10],
        )

    def to_list(self) -> list[float]:
        """Convert PointingModel to a list of 11 parameters."""
        return [
            self.flex_sin,
            self.flex_cos,
            self.az_tilt_y,
            self.az_tilt_x,
            self.el_tilt,
            self.collim_x,
            self.collim_y,
            self.az_zero,
            self.el_zero,
            self.az_sin,
            self.az_cos,
        ]