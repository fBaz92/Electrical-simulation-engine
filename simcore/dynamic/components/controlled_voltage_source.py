from dataclasses import dataclass
from .base import BranchComponent, EvalContext
import numpy as np

Array = np.ndarray

@dataclass
class ControlledVoltageSource(BranchComponent):
    V: float = 0.0
    """
    Controlled voltage source component.
    """
    V: float = 0.0
    R_internal: float = 1e-6
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the current through the controlled voltage source.
        """
        return (ctx.v_branch - self.V) / self.R_internal
    
    def dI_dv(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the derivative of current with respect to voltage.
        """
        return 1.0 / self.R_internal