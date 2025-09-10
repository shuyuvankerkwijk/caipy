from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Callable


class FFTShiftTestRunner:
    """
    Encapsulates the FFT shift testing sequence. This service does not own Qt widgets
    and exposes a minimal callback interface for integration with the GUI.

    Required callbacks:
    - set_fftshift(p0, p1)
    - start_recording(observation_name: str, extra_metadata: Dict)
    - stop_recording()
    - start_point(source_idx: int, format_dict: Dict)
    - stop_point(source_idx: int)
    - log(msg: str)
    - set_attenuations(north: Dict[int, float], south: Dict[int, float]) -> bool
    """

    def __init__(self, *, atten_step_getter: Callable[[], float], duration_getter: Callable[[], int]):
        self._atten_step_getter = atten_step_getter
        self._duration_getter = duration_getter
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self,
            fftshift_list: List[int],
            base_attenuations: Dict[str, float],
            *,
            set_fftshift: Callable[[int, int], None],
            start_recording: Callable[[str, Dict], None],
            stop_recording: Callable[[], None],
            start_point: Callable[[int, Dict], None],
            stop_point: Callable[[int], None],
            set_attenuations: Callable[[Dict[int, float], Dict[int, float]], bool],
            log: Callable[[str], None]) -> None:

        self._running = True

        obs_name = f"fftshift_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metadata = {
            'test_type': 'fftshift_attenuation_test',
            'fftshift_list': fftshift_list,
            'atten_step_db': self._atten_step_getter(),
            'duration_per_fftshift': self._duration_getter(),
            'base_attenuations': base_attenuations,
        }

        # Start recording
        start_recording(obs_name, metadata)
        time.sleep(1)

        try:
            atten_step = self._atten_step_getter()
            duration_per_fftshift = self._duration_getter()
            for i, fftshift in enumerate(fftshift_list):
                if not self._running:
                    break

                set_fftshift(fftshift, fftshift)
                time.sleep(0.5)

                # First half at current attenuation
                first_half = duration_per_fftshift // 2
                log(f"Recording at current attenuation for {first_half} seconds...")
                start_point(i * 2, {
                    'fftshift_p0': fftshift,
                    'fftshift_p1': fftshift,
                    'att_YN': base_attenuations.get('north_pol0', 0.0),
                    'att_XN': base_attenuations.get('north_pol1', 0.0),
                    'att_YS': base_attenuations.get('south_pol0', 0.0),
                    'att_XS': base_attenuations.get('south_pol1', 0.0),
                    'phase': 'before',
                })
                time.sleep(first_half)

                if not self._running:
                    break

                # Increase attenuations for second half
                inc = {
                    'north_pol0': base_attenuations.get('north_pol0', 0.0) + atten_step,
                    'north_pol1': base_attenuations.get('north_pol1', 0.0) + atten_step,
                    'south_pol0': base_attenuations.get('south_pol0', 0.0) + atten_step,
                    'south_pol1': base_attenuations.get('south_pol1', 0.0) + atten_step,
                }

                # Enforce maximum bounds (31.75 dB) and warn via callback log
                for key in list(inc.keys()):
                    if inc[key] > 31.75:
                        log(f"Attenuation {inc[key]} dB exceeds maximum, clamping to 31.75 dB")
                        inc[key] = 31.75

                # Apply increased attenuations
                north = {0: inc['north_pol0'], 1: inc['north_pol1']}
                south = {0: inc['south_pol0'], 1: inc['south_pol1']}
                ok = set_attenuations(north, south)
                if not ok:
                    log("Failed to set one or more attenuations; continuing")

                stop_point(i * 2)
                start_point(i * 2 + 1, {
                    'fftshift_p0': fftshift,
                    'fftshift_p1': fftshift,
                    'att_YN': inc['north_pol0'],
                    'att_XN': inc['north_pol1'],
                    'att_YS': inc['south_pol0'],
                    'att_XS': inc['south_pol1'],
                    'phase': 'after',
                })

                second_half = duration_per_fftshift - first_half
                log(f"Recording at increased attenuation for {second_half} seconds...")
                time.sleep(second_half)

                stop_point(i * 2 + 1)

                # Reset attenuations to base
                north_reset = {0: base_attenuations.get('north_pol0', 0.0), 1: base_attenuations.get('north_pol1', 0.0)}
                south_reset = {0: base_attenuations.get('south_pol0', 0.0), 1: base_attenuations.get('south_pol1', 0.0)}
                set_attenuations(north_reset, south_reset)

            log(f"FFT shift test {'completed' if self._running else 'stopped by user'}. Total FFT shifts tested: {len(fftshift_list)}")
        except Exception as exc:  # noqa: BLE001
            log(f"Error in FFT shift test sequence: {exc}")
        finally:
            stop_recording()
            self._running = False

