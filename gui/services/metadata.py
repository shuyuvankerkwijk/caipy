from __future__ import annotations

from typing import Dict, List


class MetadataBuilder:
    """
    Builds observation metadata dictionaries. Extracted from MainWindow for clarity.
    This object contains no Qt types and can be unit-tested easily.
    """

    @staticmethod
    def active_antennas(ant: str, mode: str | None, params: Dict) -> List[str]:
        if ant == "both":
            return ["N", "S"]
        if ant in ["N", "S"]:
            if params.get("split_track", False) and mode in ["rasta", "pointing_scan"]:
                other = "N" if ant == "S" else "S"
                return [ant, other]
            return [ant]
        return []

    @staticmethod
    def from_observation(obs) -> Dict:
        params = obs.params
        ant = obs.ant
        mode = obs.mode

        meta = {
            'mode': mode,
            'antenna': ant,
            'active_antennas': MetadataBuilder.active_antennas(ant, mode, params),
            'target_ra': params.get('ra'),
            'target_dec': params.get('dec'),
            'target_az': params.get('az'),
            'target_el': params.get('el'),
            'duration_hours': params.get('duration_hours'),
            'slew': params.get('slew'),
            'park': params.get('park'),
            'record': params.get('record'),
        }

        if mode == 'rasta':
            meta.update({
                'max_dist_deg': params.get('max_dist_deg'),
                'step_deg': params.get('step_deg'),
                'position_angle_deg': params.get('position_angle_deg'),
            })
        elif mode == 'pointing_scan':
            meta.update({
                'closest_dist_deg': params.get('closest_dist_deg'),
                'number_of_points': params.get('number_of_points'),
            })
        elif mode == 'rtos':
            meta.update({
                'ra2': params.get('ra2'),
                'dec2': params.get('dec2'),
                'number_of_points': params.get('number_of_points'),
            })

        if params.get('split_track', False) and ant in ["N", "S"] and mode in ["rasta", "pointing_scan"]:
            track_ant = "N" if ant == "S" else "S"
            meta.update({
                'split_track': True,
                'split_mode': True,
                'scan_antenna': ant,
                'track_antenna': track_ant,
                'scan_mode': mode,
                'track_mode': 'track',
            })
        else:
            meta['split_track'] = params.get('split_track', False)

        return meta

