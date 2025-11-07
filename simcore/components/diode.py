from dataclasses import dataclass
from .base import BranchComponent, EvalContext
import numpy as np

@dataclass
class Diode(BranchComponent):
    """
    Diode component using a softplus-based model for numerical stability.
    
    This diode implementation uses the softplus function instead of the traditional
    exponential Shockley diode equation. The softplus model provides:
    - Numerical stability: avoids overflow issues with large forward voltages
    - Smooth behavior: differentiable everywhere, suitable for Newton-Raphson solvers
    - Similar characteristics: approximates exponential behavior for practical purposes
    
    The current-voltage relationship is:
    I = Is * log(1 + exp(v / V_T))
    
    where log(1 + exp(x)) is the softplus function, implemented using np.log1p(np.exp(x))
    for numerical stability.
    
    Attributes:
        Is: Saturation current in Amperes (A). Default is 1e-12 A.
        V_T: Thermal voltage in Volts (V). Default is 0.02585 V (approximately 25.85 mV
             at room temperature, equivalent to kT/q).
    """
    Is: float = 1e-12
    V_T: float = 0.02585

    def current(self, ctx: EvalContext, z_next=None) -> float:
        """
        Calculate the current through the diode using the softplus model.
        
        The current is computed as:
        I = Is * log(1 + exp(v / V_T))
        
        This uses the numerically stable softplus function log(1 + exp(x)) implemented
        via np.log1p(np.exp(x)), which avoids overflow issues that occur with the
        traditional exponential diode model at large forward voltages.
        
        Args:
            ctx: Evaluation context containing branch voltage v_branch.
            z_next: Not used for diode (no internal states). Can be None.
        
        Returns:
            Current through the diode in Amperes (A).
        """
        x = ctx.v_branch / self.V_T
        # numerically stable softplus
        return self.Is * np.log1p(np.exp(x))

    def dI_dv(self, ctx: EvalContext, z_next=None) -> float:
        """
        Calculate the derivative of current with respect to branch voltage.
        
        The derivative of the softplus-based diode model is:
        dI/dv = Is * σ(v / V_T) / V_T
        
        where σ(x) = 1 / (1 + exp(-x)) is the logistic (sigmoid) function, which is
        the derivative of the softplus function. This provides a smooth, bounded
        derivative that is suitable for Newton-Raphson solvers.
        
        Args:
            ctx: Evaluation context containing branch voltage v_branch.
            z_next: Not used for diode (no internal states). Can be None.
        
        Returns:
            Derivative of current w.r.t. voltage in Amperes per Volt (A/V).
        """
        x = ctx.v_branch / self.V_T
        # logistic function σ(x) = 1 / (1 + exp(-x))
        sigma = 1.0 / (1.0 + np.exp(-x))
        return self.Is * sigma / self.V_T
