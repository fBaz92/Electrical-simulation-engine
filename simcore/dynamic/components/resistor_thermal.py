from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .base import BranchComponent, EvalContext

Array = np.ndarray


@dataclass
class ThermalResistor(BranchComponent):
    """
    Resistor with temperature-dependent resistance and RC thermal dynamics.

    Electrical model:
        R(T) = R0 * (1 + alpha * (T - T_ref))
        i = v / R(T)

    Thermal model (single node):
        C_th * dT/dt = P_diss - (T - T_amb) / R_th
        with P_diss = v^2 / R(T)

    The temperature state is integrated implicitly; its history becomes available
    through `state_history("T")`, while the electrical behavior automatically
    reflects the updated resistance at each Newton iteration.
    """
    R0: float                  # resistance at reference temperature [Ohm]
    alpha: float               # temperature coefficient [1/K]
    T_ref: float               # reference temperature for R0 [K]
    C_th: float                # thermal capacitance [J/K]
    R_th: float                # thermal resistance to ambient [K/W]
    T_amb: float               # ambient temperature [K]
    T_init: float = None       # initial resistor temperature [K]
    R_min: float = 1e-9        # floor to keep resistance positive

    def __post_init__(self) -> None:
        """
        Initialize T_init to T_ref if not explicitly provided.
        """
        if self.T_init is None:
            self.T_init = self.T_ref

    # Helpers
    def _dRdT(self) -> float:
        """
        Compute the derivative of resistance with respect to temperature.
        
        Returns:
            dR/dT = R0 * alpha [Ohm/K]
        """
        return self.R0 * self.alpha

    def _resistance(self, T: float) -> float:
        """
        Compute resistance at a given temperature.
        
        Uses the linear temperature coefficient model:
        R(T) = R0 * (1 + alpha * (T - T_ref))
        
        The result is clamped to R_min to ensure positive resistance.
        
        Args:
            T: Temperature in Kelvin.
        
        Returns:
            Resistance in Ohms.
        """
        return max(self.R_min, self.R0 * (1.0 + self.alpha * (T - self.T_ref)))

    # I-V characteristics
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the current through the resistor.
        
        Current is computed using Ohm's law: i = v / R(T)
        where R(T) is the temperature-dependent resistance.
        
        Args:
            ctx: Evaluation context containing branch voltage.
            z_next: State vector at next time step [T]. If None, uses T_init.
        
        Returns:
            Current through the resistor in Amperes.
        """
        T = float(z_next[0]) if z_next is not None else float(self.T_init)
        R_T = self._resistance(T)
        return ctx.v_branch / R_T

    def dI_dv(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the derivative of current with respect to branch voltage.
        
        For a resistor: dI/dv = 1 / R(T)
        
        Args:
            ctx: Evaluation context containing branch voltage.
            z_next: State vector at next time step [T]. If None, uses T_init.
        
        Returns:
            Derivative of current w.r.t. voltage in Amperes per Volt (A/V).
        """
        T = float(z_next[0]) if z_next is not None else float(self.T_init)
        R_T = self._resistance(T)
        return 1.0 / R_T

    def dI_dz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of current with respect to state variables.
        
        The current depends on temperature through the resistance:
        dI/dT = -v * (dR/dT) / R(T)^2
        
        Args:
            ctx: Evaluation context containing branch voltage.
            z_next: State vector at next time step [T].
        
        Returns:
            Jacobian array: [dI/dT]
        """
        T = float(z_next[0])
        R_T = self._resistance(T)
        dRdT = self._dRdT()
        dIdT = -ctx.v_branch * dRdT / (R_T ** 2)
        return np.array([dIdT], dtype=float)

    # State bookkeeping
    def n_states(self) -> int:
        """
        Return the number of internal state variables.
        
        Returns:
            Number of states: 1 (temperature T)
        """
        return 1

    def state_names(self) -> list[str]:
        """
        Return the names of the internal state variables.
        
        Returns:
            List of state names: ["T"]
        """
        return ["T"]

    def state_init(self) -> Array:
        """
        Return the initial state vector.
        
        Returns:
            Initial state array: [T_init]
        """
        return np.array([self.T_init], dtype=float)

    def _power(self, v_branch: float, T: float) -> float:
        """
        Compute the power dissipated in the resistor.
        
        Power is calculated as: P = v^2 / R(T)
        
        Args:
            v_branch: Branch voltage in Volts.
            T: Temperature in Kelvin.
        
        Returns:
            Dissipated power in Watts.
        """
        R_T = self._resistance(T)
        return (v_branch ** 2) / R_T

    # Thermal residual: T_{k+1} - T_k - (dt/C_th)*(P_{k+1} - (T_{k+1}-T_amb)/R_th) = 0
    def state_residual(self, ctx: EvalContext, z_next: Array, z_prev: Array) -> Array:
        """
        Calculate the thermal state residual equation.
        
        The residual enforces the thermal energy balance:
        T_{k+1} - T_k - (dt/C_th) * (P_{k+1} - (T_{k+1} - T_amb) / R_th) = 0
        
        where:
        - P_{k+1} = v^2 / R(T_{k+1}) is the power dissipated at step k+1
        - (T_{k+1} - T_amb) / R_th is the cooling rate to ambient
        
        Args:
            ctx: Evaluation context containing time step dt and branch voltage.
            z_next: State vector at next time step [T_{k+1}].
            z_prev: State vector at previous time step [T_k].
        
        Returns:
            Residual array: [T_{k+1} - T_k - (dt/C_th) * (P_{k+1} - cooling)]
        """
        T_next = float(z_next[0])
        T_prev = float(z_prev[0])
        P_next = self._power(ctx.v_branch, T_next)
        cooling = (T_next - self.T_amb) / self.R_th
        return np.array([
            T_next - T_prev - (ctx.dt / self.C_th) * (P_next - cooling)
        ], dtype=float)

    def dRdz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of state residual with respect to state variables.
        
        This is the Jacobian of the thermal residual w.r.t. temperature, used by
        the Newton-Raphson solver for state equation iterations.
        
        The derivative accounts for:
        - Direct temperature change: 1.0
        - Power dependence on temperature: dP/dT = -v^2 * (dR/dT) / R(T)^2
        - Cooling rate dependence: d(cooling)/dT = 1 / R_th
        
        Args:
            ctx: Evaluation context containing time step dt and branch voltage.
            z_next: State vector at next time step [T_{k+1}].
        
        Returns:
            Jacobian array: [1.0 - (dt/C_th) * (dP/dT - d(cooling)/dT)]
        """
        T_next = float(z_next[0])
        dRdT = self._dRdT()
        R_T = self._resistance(T_next)
        v = ctx.v_branch

        # dP/dT = -v^2 / R(T)^2 * dR/dT
        dP_dT = - (v ** 2) * dRdT / (R_T ** 2)
        coeff = ctx.dt / self.C_th
        dCooling_dT = 1.0 / self.R_th
        return np.array([1.0 - coeff * (dP_dT - dCooling_dT)], dtype=float)

    def dRdv(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of state residual with respect to branch voltage.
        
        This is the Jacobian of the thermal residual w.r.t. branch voltage, used by
        the Newton-Raphson solver to couple state and KCL equations.
        
        The derivative accounts for power dependence on voltage:
        dP/dv = 2 * v / R(T)
        
        Args:
            ctx: Evaluation context containing time step dt and branch voltage.
            z_next: State vector at next time step [T_{k+1}].
        
        Returns:
            Jacobian array: [-(dt/C_th) * dP/dv]
        """
        T_next = float(z_next[0])
        R_T = self._resistance(T_next)
        coeff = ctx.dt / self.C_th
        # dP/dv = 2*v / R(T)
        dP_dv = 2.0 * ctx.v_branch / R_T
        return np.array([-coeff * dP_dv], dtype=float)
