from dataclasses import dataclass
from .base import BranchComponent, EvalContext

@dataclass
class Resistor(BranchComponent):
    R: float
    def current(self, ctx: EvalContext, z_next=None) -> float:
        """
        Calculate the current through the resistor using Ohm's law.
        
        Args:
            ctx: Evaluation context containing branch voltage and other simulation parameters.
            z_next: Optional state vector for future time step (not used for linear resistors).
        
        Returns:
            Current through the resistor: I = V / R
        """
        return ctx.v_branch / self.R
    
    def dI_dv(self, ctx: EvalContext, z_next=None) -> float:
        """
        Calculate the derivative of current with respect to voltage.
        
        This represents the conductance (G = 1/R) of the resistor, which is constant
        for linear resistors. Used by the solver for Newton-Raphson iterations.
        
        Args:
            ctx: Evaluation context containing branch voltage and other simulation parameters.
            z_next: Optional state vector for future time step (not used for linear resistors).
        
        Returns:
            Conductance of the resistor: dI/dV = 1 / R
        """
        return 1.0 / self.R
