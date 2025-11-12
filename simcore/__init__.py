"""
Top-level namespace for the Electrical Simulation Engine.

The project now exposes two independent solvers:
- simcore.dynamic: time-domain, nonlinear circuit simulator.
- simcore.static: steady-state solver for DC/AC (phasor) analysis.
"""

from . import dynamic  # noqa: F401
from . import static  # noqa: F401

__all__ = ["dynamic", "static"]
