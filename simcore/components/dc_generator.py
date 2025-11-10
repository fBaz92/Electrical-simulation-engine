from dataclasses import dataclass, field
import numpy as np
from .base import BranchComponent, EvalContext

Array = np.ndarray

@dataclass
class DCGenerator(BranchComponent):
    """
    DC generator model. The default internal resistance is set to 1e-9 Ohms to model an ideal generator.
    
    Attributes:
        V_nom: Nominal voltage of the generator (Volts).
        R_internal: Internal resistance of the generator (Ohms). Default is 1e-9 Ohms.
    """
    V_nom: float
    R_internal: float = 1e-9

    # I-V
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the current through the battery using Thevenin equivalent model.
        
        The current is computed as I = (V_nom - V) / R_internal.
        
        Args:
            ctx: Evaluation context containing branch voltage and other simulation parameters.
           
        Returns:
            Current through the generator: I = (V - V_nom) / R_internal
        """
        return (ctx.v_branch - self.V_nom) / self.R_internal


    def dI_dv(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the derivative of current with respect to voltage.
        
        Args:
            ctx: Evaluation context containing branch voltage and other simulation parameters.

        Returns:
            Derivative: dI/dV = 1 / R_internal
        """
        return 1.0 / self.R_internal


