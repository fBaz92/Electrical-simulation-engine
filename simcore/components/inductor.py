from dataclasses import dataclass, field
import numpy as np
from .base import BranchComponent, EvalContext

Array = np.ndarray

@dataclass
class Inductor(BranchComponent):
    """
    Inductor component with inductance L and initial current I_init.
    
    The inductor is modeled using the state variable I_L (inductor current).
    The state equation enforces the relationship:
    I_L_{k+1} - I_L_{k} - (dt/L)*v_{k+1} = 0
    
    Attributes:
        L: Inductance value in Henries (H).
        I_init: Initial current through the inductor in Amperes (A).
    """
    L: float
    I_init: float

    # I-V
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the current through the inductor.
        
        The inductor current is equal to the state variable I_L. If z_next is provided,
        it uses the updated state value; otherwise, it returns the initial current.
        
        Args:
            ctx: Evaluation context containing branch voltage and time information.
            z_next: State vector at next time step [I_L_{k+1}]. If None, uses I_init.
        
        Returns:
            Current through the inductor in Amperes (A).
        """
        i_l = (z_next[0] if z_next is not None else self.I_init)
        return i_l

    def dI_dv(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of current with respect to branch voltage.
        
        This is the Jacobian of the current function w.r.t. branch voltage,
        used by the solver for Newton-Raphson iterations on KCL equations.
        
        For an inductor, the current at a given time step is determined solely
        by the state variable I_L, not directly by the voltage. The voltage affects
        the rate of change of current through the state equation, but the current
        itself has no direct dependence on voltage at the same time step.
        Therefore, dI/dv = 0.
        
        Args:
            ctx: Evaluation context containing branch voltage and time information.
            z_next: State vector at next time step [I_L_{k+1}].
        
        Returns:
            Jacobian matrix: [[0.0]] - derivative of current w.r.t. branch voltage.
        """
        return np.array([[ 0 ]])

    def dI_dz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of current with respect to state variables.
        
        This is the Jacobian of the current function w.r.t. the state vector,
        used by the solver for Newton-Raphson iterations on KCL equations.
        
        For an inductor, the current is directly equal to the state I_L, so
        the derivative is 1.0.
        
        Args:
            ctx: Evaluation context containing branch voltage and time information.
            z_next: State vector at next time step [I_L_{k+1}].
        
        Returns:
            Jacobian matrix: [[1.0]] - derivative of current w.r.t. I_L.
        """
        return np.array([[ 1.0 ]])

    # Stati
    def n_states(self) -> int:
        """
        Return the number of internal state variables.
        
        Returns:
            Number of states: 1 (I_L)
        """
        return 1
    
    def state_names(self) -> list[str]:
        """
        Return the names of the internal state variables.
        
        Returns:
            List of state names: ["I_L"]
        """
        return ["I_L"]
    
    def state_init(self) -> Array:
        """
        Return the initial state vector.
        
        Returns:
            Initial state array: [I_init]
        """
        return np.array([self.I_init], dtype=float)

    
    # I_L_{k+1} - I_L_{k} - (dt/L)*v_{k+1} = 0
    def state_residual(self, ctx: EvalContext, z_next: Array, z_prev: Array) -> Array:
        """
        Calculate the state residual equation for I_L evolution.
        
        The residual enforces the I_L update equation:
        I_L_{k+1} - I_L_{k} - (dt/L)*v_{k+1} = 0
        
        Args:
            ctx: Evaluation context containing time step dt and other simulation parameters.
            z_next: State vector at next time step [I_L_{k+1}].
            z_prev: State vector at previous time step [I_L_{k}].
        
        Returns:
            Residual array: [I_L_{k+1} - I_L_{k} - (dt/L)*v_{k+1}]
        """
        return np.array([ z_next[0] - z_prev[0] - (ctx.dt/self.L) * ctx.v_branch ])

    def dRdz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of state residual with respect to state variables.
        
        This is the Jacobian of the state residual w.r.t. z_next, used by the solver
        for Newton-Raphson iterations on the state equation.
        
        Args:
            ctx: Evaluation context containing time step dt and other simulation parameters.
            z_next: State vector at next time step [I_L_{k+1}].
        
        Returns:
            Jacobian matrix: [[1.0]]
        """
        return np.array([[ 1.0 ]])

    def dRdv(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of state residual with respect to voltage.
        
        This is the Jacobian of the state residual w.r.t. branch voltage, used by the solver
        for Newton-Raphson iterations to couple state and KCL equations.
        
        Args:
            ctx: Evaluation context containing time step dt and other simulation parameters.
            z_next: State vector at next time step [I_L_{k+1}].
        
        Returns:
            Jacobian array: [-(dt/L)]
        """
        return np.array([ -(ctx.dt/self.L) ])
