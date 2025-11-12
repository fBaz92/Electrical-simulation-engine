from dataclasses import dataclass, field
import numpy as np
from .base import BranchComponent, EvalContext

Array = np.ndarray

@dataclass
class LithiumBatteryLUT(BranchComponent):
    """
    A lithium battery model using a lookup table (LUT) for open-circuit voltage.
    
    The battery is modeled as a Thevenin equivalent circuit with internal resistance
    and a state-dependent open-circuit voltage (OCV) that varies with state of charge (SOC).
    The OCV is interpolated from lookup tables as a function of SOC.
    
    Attributes:
        R_internal: Internal resistance of the battery (Ohms).
        Q_Ah: Battery capacity in Ampere-hours.
        soc0: Initial state of charge (0.0 to 1.0).
        soc_pts: Array of SOC points for OCV lookup table interpolation.
        ocv_pts: Array of OCV values corresponding to soc_pts.
        _Qc: Internal capacity in Coulombs (Q_Ah * 3600), computed automatically.
    """
    R_internal: float
    Q_Ah: float
    soc0: float
    soc_pts: Array
    ocv_pts: Array
    _Qc: float = field(init=False)

    def __post_init__(self):
        """
        Initialize the internal capacity in Coulombs.
        
        Converts capacity from Ampere-hours to Coulombs by multiplying by 3600.
        """
        object.__setattr__(self, "_Qc", self.Q_Ah * 3600.0)

    # Helpers
    def ocv(self, soc: float) -> float:
        """
        Get the open-circuit voltage (OCV) for a given state of charge (SOC).
        
        Uses linear interpolation from the lookup table. SOC is clamped to [0.0, 1.0]
        to ensure valid interpolation.
        
        Args:
            soc: State of charge (0.0 to 1.0).
        
        Returns:
            Open-circuit voltage at the given SOC (Volts).
        """
        s = float(np.clip(soc, 0.0, 1.0))
        return float(np.interp(s, self.soc_pts, self.ocv_pts))

    def dOCV_dSOC(self, soc: float, eps: float = 1e-6) -> float:
        """
        Calculate the derivative of OCV with respect to SOC.
        
        Uses finite difference approximation with a small epsilon. This represents
        how the open-circuit voltage changes with state of charge.
        
        Args:
            soc: State of charge at which to evaluate the derivative.
            eps: Small perturbation for finite difference (default: 1e-6).
        
        Returns:
            Derivative dOCV/dSOC (Volts per unit SOC).
        """
        s1 = np.clip(soc - eps, 0, 1); s2 = np.clip(soc + eps, 0, 1)
        return (self.ocv(s2) - self.ocv(s1)) / (float(s2 - s1) + 1e-12)

    # I-V
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the current through the battery using Thevenin equivalent model.
        
        The current is computed as I = (V - OCV(SOC)) / R_internal, where OCV
        depends on the current state of charge. Positive current means discharging
        (current leaving the battery).
        
        Args:
            ctx: Evaluation context containing branch voltage and other simulation parameters.
            z_next: Optional state vector containing SOC at next time step. If None, uses initial SOC.
        
        Returns:
            Current through the battery: I = (V - OCV(SOC)) / R_internal
        """
        soc = (z_next[0] if z_next is not None else self.soc0)
        # Thevenin visto dal nodo: i = (v - OCV)/R
        return (ctx.v_branch - self.ocv(soc)) / self.R_internal

    def dI_dv(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the derivative of current with respect to voltage.
        
        For the Thevenin model, this is simply the inverse of internal resistance,
        representing the conductance. Used by the solver for Newton-Raphson iterations.
        
        Args:
            ctx: Evaluation context containing branch voltage and other simulation parameters.
            z_next: Optional state vector (not used for this derivative).
        
        Returns:
            Derivative: dI/dV = 1 / R_internal
        """
        return 1.0 / self.R_internal

    def dI_dz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of current with respect to state variables.
        
        The derivative w.r.t. SOC accounts for how OCV changes with SOC, which
        affects the current. Used by the solver for Newton-Raphson iterations.
        
        Args:
            ctx: Evaluation context containing branch voltage and other simulation parameters.
            z_next: State vector containing SOC at next time step.
        
        Returns:
            Array of derivatives: [dI/dSOC] = [-dOCV/dSOC / R_internal]
        """
        return np.array([-self.dOCV_dSOC(float(z_next[0])) / self.R_internal])

    # Stati
    def n_states(self) -> int:
        """
        Return the number of internal state variables.
        
        Returns:
            Number of states: 1 (SOC)
        """
        return 1
    
    def state_names(self) -> list[str]:
        """
        Return the names of the internal state variables.
        
        Returns:
            List of state names: ["SOC"]
        """
        return ["SOC"]
    
    def state_init(self) -> Array:
        """
        Return the initial state vector.
        
        Returns:
            Initial state array: [soc0]
        """
        return np.array([self.soc0], dtype=float)

    # SOC_{k+1} - SOC_{k} - (dt/Q)*i_{k+1} = 0   (corrente leaving bus = carica)
    def state_residual(self, ctx: EvalContext, z_next: Array, z_prev: Array) -> Array:
        """
        Calculate the state residual equation for SOC evolution.
        
        The residual enforces the SOC update equation:
        SOC_{k+1} - SOC_{k} - (dt/Q) * i_{k+1} = 0
        
        This represents the charge balance: positive current (discharging) decreases SOC.
        
        Args:
            ctx: Evaluation context containing time step dt and other simulation parameters.
            z_next: State vector at next time step [SOC_{k+1}].
            z_prev: State vector at previous time step [SOC_{k}].
        
        Returns:
            Residual array: [SOC_{k+1} - SOC_{k} - (dt/Q) * i_{k+1}]
        """
        i_next = self.current(ctx, z_next)
        return np.array([ z_next[0] - z_prev[0] - (ctx.dt/self._Qc) * i_next ])

    def dRdz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of state residual with respect to state variables.
        
        This is the Jacobian of the state residual w.r.t. z_next, used by the solver
        for Newton-Raphson iterations on the state equation.
        
        Args:
            ctx: Evaluation context containing time step dt and other simulation parameters.
            z_next: State vector at next time step [SOC_{k+1}].
        
        Returns:
            Jacobian matrix: [[1.0 - (dt/Q) * dI/dSOC]]
        """
        return np.array([[ 1.0 - (ctx.dt/self._Qc) * self.dI_dz(ctx, z_next)[0] ]])

    def dRdv(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of state residual with respect to voltage.
        
        This is the Jacobian of the state residual w.r.t. branch voltage, used by the solver
        for Newton-Raphson iterations to couple state and KCL equations.
        
        Args:
            ctx: Evaluation context containing time step dt and other simulation parameters.
            z_next: State vector at next time step [SOC_{k+1}].
        
        Returns:
            Jacobian array: [-(dt/Q) * dI/dV]
        """
        return np.array([ -(ctx.dt/self._Qc) * self.dI_dv(ctx, z_next) ])
