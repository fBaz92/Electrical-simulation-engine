from __future__ import annotations
import numpy as np
from typing import Tuple

def phasor(magnitude: float, phase_deg: float = 0.0) -> complex:
    """
    Convenience helper to create a complex phasor.

    Args:
        magnitude: Amplitude of the sinusoid.
        phase_deg: Phase in degrees (default: 0).

    Returns:
        Complex number representing the phasor.
    """
    return magnitude * np.exp(1j * np.deg2rad(phase_deg))


def polar(value: complex) -> Tuple[float, float]:
    """
    Convert a complex number into magnitude/phase (degrees).
    """
    return float(np.abs(value)), float(np.rad2deg(np.angle(value)))
