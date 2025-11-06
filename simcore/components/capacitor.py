from dataclasses import dataclass
from .base import BranchComponent, EvalContext

@dataclass
class Capacitor(BranchComponent):
    C: float
    V_init: float | None = None # if the initial voltage is not provided, the capacitor voltage is set to the volltage of the node it is attached to. This avoids initial transients.

    def current(self, ctx: EvalContext, z_next=None) -> float:
        """
        Calculate the current through the capacitor using the capacitor equation.
        
        The current is proportional to the rate of change of voltage: I = C * dv/dt.
        Uses the implicit derivative dvdt_branch from the evaluation context.
        
        Args:
            ctx: Evaluation context containing the implicit derivative dvdt_branch and other simulation parameters.
            z_next: Optional state vector for future time step (not used for capacitors).
        
        Returns:
            Current through the capacitor: I = C * dv/dt
        """
        return self.C * ctx.dvdt_branch
    
    def dI_dv(self, ctx: EvalContext, z_next=None) -> float:
        """
        Calculate the derivative of current with respect to voltage.
        
        For implicit Euler integration, this derivative is approximately C/dt.
        This represents the effective conductance of the capacitor at the current time step,
        used by the solver for Newton-Raphson iterations.
        
        Args:
            ctx: Evaluation context containing the time step dt and other simulation parameters.
            z_next: Optional state vector for future time step (not used for capacitors).
        
        Returns:
            Effective conductance: dI/dV ≈ C / dt (with implicit Euler)
        """
        return self.C / ctx.dt
