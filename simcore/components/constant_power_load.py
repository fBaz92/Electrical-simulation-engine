from dataclasses import dataclass
from typing import Callable
from .base import BranchComponent, EvalContext

@dataclass
class ConstantPowerLoad(BranchComponent):
    """
    A constant power load component where power consumption is time-dependent.
    
    The load maintains constant power P(t) = V * I, meaning current adjusts
    inversely with voltage. A minimum voltage threshold (vmin) prevents
    numerical issues at low voltages.
    
    Attributes:
        P_of_t: Function that returns the power demand as a function of time.
        vmin: Minimum voltage threshold (default: 2.0 V) to prevent division by zero.
    """
    P_of_t: Callable[[float], float]
    vmin: float = 2.0

    def current(self, ctx: EvalContext, z_next=None) -> float:
        """
        Calculate the current drawn by the constant power load.
        
        The current is computed as I = P(t) / V, where P(t) is the time-dependent
        power demand. The voltage is clamped to a minimum value (vmin) to ensure
        numerical stability and prevent division by zero.
        
        Args:
            ctx: Evaluation context containing branch voltage, time, and other simulation parameters.
            z_next: Optional state vector for future time step (not used for constant power loads).
        
        Returns:
            Current drawn by the load: I = P(t) / max(V, vmin)
        """
        P = self.P_of_t(ctx.t_next)
        v = max(ctx.v_branch, self.vmin)
        return P / v

    def dI_dv(self, ctx: EvalContext, z_next=None) -> float:
        """
        Calculate the derivative of current with respect to voltage.
        
        For a constant power load, dI/dV = -P/V², as current decreases when voltage
        increases to maintain constant power. When voltage is below the minimum
        threshold, the derivative is set to zero since the voltage is clamped.
        
        Used by the solver for Newton-Raphson iterations.
        
        Args:
            ctx: Evaluation context containing branch voltage, time, and other simulation parameters.
            z_next: Optional state vector for future time step (not used for constant power loads).
        
        Returns:
            Derivative: dI/dV = -P/V² if V > vmin, else 0.0
        """
        P = self.P_of_t(ctx.t_next)
        v = ctx.v_branch
        return -P/(v*v) if v > self.vmin else 0.0
