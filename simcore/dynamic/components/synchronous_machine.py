from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from .base import BranchComponent, EvalContext, Array
from ..utils.dq import dq_to_abc_series


def _default_load_torque(t: float, omega: float) -> float:
    """
    Default load torque function (no load).
    
    This function serves as the default value for the `load_torque` parameter
    in SynchronousMachineFOC. It returns zero torque, representing an
    unloaded motor condition.
    
    Args:
        t: Current time (s).
        omega: Mechanical rotor speed (rad/s).
    
    Returns:
        Load torque (N·m). Always returns 0.0.
    """
    return 0.0


def _default_dload_domega(t: float, omega: float) -> float:
    """
    Default derivative of load torque with respect to speed (no speed dependence).
    
    This function serves as the default value for the `dload_domega` parameter
    in SynchronousMachineFOC. It returns zero, indicating that the load torque
    does not depend on speed (used for Jacobian computation).
    
    Args:
        t: Current time (s).
        omega: Mechanical rotor speed (rad/s).
    
    Returns:
        Partial derivative dT_load/domega (N·m·s/rad). Always returns 0.0.
    """
    return 0.0


@dataclass
class SynchronousMachineFOC(BranchComponent):
    """
    Average-value permanent magnet synchronous machine (PMSM) model in dq reference frame.
    
    This component implements a field-oriented control (FOC) model of a synchronous
    motor with electrical, mechanical, and thermal dynamics. The model uses the dq
    (direct-quadrature) transformation to represent the three-phase AC machine in a
    rotating reference frame synchronized with the rotor.
    
    The model includes:
    - Electrical dynamics: stator dq currents
    - Mechanical dynamics: rotor speed and position
    - Thermal dynamics: temperature evolution based on losses
    
    States (5 total):
        - i_d: Direct-axis stator current (A)
        - i_q: Quadrature-axis stator current (A)
        - omega: Mechanical rotor speed (rad/s)
        - theta_e: Electrical angle of the dq reference frame (rad)
        - T: Machine temperature (K)
    
    Voltage Input Modes:
        - If `voltage_cmd` is provided: Direct dq voltage commands (v_d, v_q) are used.
          The branch voltage is ignored and the component acts as a controlled source.
        - If `voltage_cmd` is None: The branch voltage is applied to the q-axis (v_q),
          with v_d = 0. This allows the machine to be connected directly to a voltage
          source in the network.
    
    Electrical Reference Frame:
        The dq reference frame rotates synchronously with the rotor at electrical speed
        omega_e = pole_pairs * omega_mech. This eliminates the need for rotor flux state
        variables (unlike induction machines) since the rotor flux is constant (permanent
        magnet flux psi_f).
    
    Thermal Model:
        The thermal model accounts for:
        - Copper losses: Stator resistive losses (scaled by `copper_loss_factor`)
        - Iron losses: Speed-dependent losses (proportional to omega^2)
        - Cooling: First-order thermal model with thermal resistance R_th and capacitance C_th
    
    Attributes:
        pole_pairs: Number of pole pairs (determines mechanical-to-electrical speed ratio).
        R_s: Stator resistance (Ω).
        L_d: Direct-axis inductance (H).
        L_q: Quadrature-axis inductance (H).
        psi_f: Permanent magnet flux linkage (Wb).
        J: Rotor moment of inertia (kg·m²).
        B: Viscous friction coefficient (N·m·s/rad).
        C_th: Thermal capacitance (J/K).
        R_th: Thermal resistance (K/W).
        T_amb: Ambient temperature (K).
        load_torque: Callable(t, omega) -> torque (N·m). Load torque as function of time
            and mechanical speed. Defaults to zero.
        dload_domega: Callable(t, omega) -> dT_load/domega. Derivative of load torque
            with respect to speed (for Jacobian computation). Defaults to zero.
        voltage_cmd: Optional callable(t) -> (v_d, v_q). If provided, supplies direct
            dq voltage commands. If None, branch voltage is used as v_q.
        iron_loss_coeff: Iron loss coefficient (W·s²/rad²). Losses = coeff * omega².
        copper_loss_factor: Multiplier for copper losses. Typically 3.0 for three-phase
            (accounts for all three phases).
        i_d0: Initial direct-axis current (A).
        i_q0: Initial quadrature-axis current (A).
        omega0: Initial mechanical speed (rad/s).
        theta0: Initial electrical angle (rad).
        temp0: Initial temperature (K). Defaults to 293.15 K (20°C).
        voltage_epsilon: Small voltage threshold (V) to avoid division by zero in
            current computation when branch voltage is near zero.
    
    Example:
        >>> # Create a motor with voltage command control
        >>> motor = SynchronousMachineFOC(
        ...     pole_pairs=4,
        ...     R_s=0.1,
        ...     L_d=0.001, L_q=0.001,
        ...     psi_f=0.1,
        ...     J=0.1, B=0.01,
        ...     C_th=1000.0, R_th=0.1, T_amb=293.15,
        ...     voltage_cmd=lambda t: (0.0, 100.0)  # v_d=0, v_q=100V
        ... )
        >>> 
        >>> # Or connect to network branch voltage
        >>> motor = SynchronousMachineFOC(
        ...     pole_pairs=4,
        ...     R_s=0.1,
        ...     L_d=0.001, L_q=0.001,
        ...     psi_f=0.1,
        ...     J=0.1, B=0.01,
        ...     C_th=1000.0, R_th=0.1, T_amb=293.15,
        ...     voltage_cmd=None  # Use branch voltage
        ... )
    """

    pole_pairs: int
    R_s: float
    L_d: float
    L_q: float
    psi_f: float          # permanent magnet flux linkage
    J: float
    B: float
    C_th: float
    R_th: float
    T_amb: float
    load_torque: Callable[[float, float], float] = _default_load_torque
    dload_domega: Callable[[float, float], float] = _default_dload_domega
    voltage_cmd: Callable[[float], tuple[float, float]] | None = None
    iron_loss_coeff: float = 0.0
    copper_loss_factor: float = 3.0
    i_d0: float = 0.0
    i_q0: float = 0.0
    omega0: float = 0.0
    theta0: float = 0.0
    temp0: float = 293.15
    voltage_epsilon: float = 1e-3

    _state_count: int = field(init=False, default=5)

    # --- Branch interface -------------------------------------------------
    def n_states(self) -> int:
        """
        Return the number of state variables.
        
        Returns:
            Number of states (5).
        """
        return self._state_count

    def state_names(self) -> list[str]:
        """
        Return names of state variables.
        
        Returns:
            List of state names: ["i_d", "i_q", "omega", "theta_e", "T"].
        """
        return ["i_d", "i_q", "omega", "theta_e", "T"]

    def state_init(self) -> Array:
        """
        Return initial state vector.
        
        Returns:
            Initial state vector [i_d0, i_q0, omega0, theta0, temp0].
        """
        return np.array([self.i_d0, self.i_q0, self.omega0, self.theta0, self.temp0], dtype=float)

    # --- Helper accessors -------------------------------------------------
    def dq_trace(self) -> dict[str, Array]:
        """
        Return a dictionary with time stamps and dq/abc current traces.
        
        Converts the dq current states to three-phase abc currents using the
        electrical angle theta_e. Requires a completed simulation (state history
        must be attached via the solver).
        
        Returns:
            Dictionary containing:
            - "t": Time array (s)
            - "i_d": Direct-axis current array (A)
            - "i_q": Quadrature-axis current array (A)
            - "i_a", "i_b", "i_c": Three-phase current arrays (A)
            - "omega": Mechanical speed array (rad/s)
            - "theta_e": Electrical angle array (rad)
            - "temperature": Temperature array (K)
        """
        t, states = self.state_history()
        i_d = states[:, 0]
        i_q = states[:, 1]
        theta = states[:, 3]
        i_a, i_b, i_c = dq_to_abc_series(i_d, i_q, theta)
        return {
            "t": t,
            "i_d": i_d,
            "i_q": i_q,
            "i_a": i_a,
            "i_b": i_b,
            "i_c": i_c,
            "omega": states[:, 2],
            "theta_e": theta,
            "temperature": states[:, 4],
        }

    # --- Electrical helper methods ---------------------------------------
    def _omega_e(self, omega_mech: float) -> float:
        """
        Calculate electrical speed from mechanical speed.
        
        The electrical speed is the mechanical speed multiplied by the number of
        pole pairs, representing the speed of the rotating dq reference frame.
        
        Args:
            omega_mech: Mechanical rotor speed (rad/s).
        
        Returns:
            Electrical speed omega_e (rad/s).
        """
        return self.pole_pairs * omega_mech

    def _voltage_dq(self, ctx: EvalContext) -> tuple[float, float, float, float]:
        """
        Get dq-axis voltages and control flags.
        
        Returns the direct and quadrature axis voltages (v_d, v_q) along with
        flags indicating whether the branch voltage affects the d-axis (dv_d) or
        q-axis (dv_q) for Jacobian computation.
        
        If `voltage_cmd` is provided, returns the commanded voltages with
        dv_d=0, dv_q=0 (branch voltage has no effect).
        Otherwise, returns v_d=0, v_q=branch_voltage with dv_d=0, dv_q=1
        (branch voltage affects q-axis only).
        
        Args:
            ctx: Evaluation context containing branch voltage and time.
        
        Returns:
            Tuple (v_d, v_q, dv_d, dv_q) where:
            - v_d: Direct-axis voltage (V)
            - v_q: Quadrature-axis voltage (V)
            - dv_d: Partial derivative flag for d-axis (0 or 1)
            - dv_q: Partial derivative flag for q-axis (0 or 1)
        """
        if self.voltage_cmd is not None:
            v_d, v_q = self.voltage_cmd(ctx.t_next)
            return float(v_d), float(v_q), 0.0, 0.0
        return 0.0, float(ctx.v_branch), 0.0, 1.0

    def _torque(self, i_d: float, i_q: float) -> float:
        """
        Calculate electromagnetic torque.
        
        The torque is computed using the cross product of flux and current vectors
        in the dq frame. The factor 1.5 accounts for the three-phase to two-phase
        transformation (power invariant). The permanent magnet flux contributes
        to the d-axis flux.
        
        Args:
            i_d: Direct-axis stator current (A).
            i_q: Quadrature-axis stator current (A).
        
        Returns:
            Electromagnetic torque T_e (N·m).
        """
        psi_d = self.L_d * i_d + self.psi_f
        psi_q = self.L_q * i_q
        return 1.5 * self.pole_pairs * (psi_d * i_q - psi_q * i_d)

    # --- Branch I-V -------------------------------------------------------
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate branch current from electrical power.
        
        The branch current is computed as the ratio of airgap power (in dq
        frame) to branch voltage. This allows the component to interface with
        the network solver while internally working with dq quantities.
        
        Airgap power is computed as 1.5 * (v_d * i_d + v_q * i_q), where
        the factor 1.5 accounts for the three-phase to two-phase transformation.
        
        Args:
            ctx: Evaluation context containing branch voltage and time.
            z_next: State vector at next time step. If None, uses initial state.
        
        Returns:
            Branch current (A). Positive current flows from positive to negative terminal.
        """
        if z_next is None:
            z_next = self.state_init()
        i_d, i_q = float(z_next[0]), float(z_next[1])
        v_d, v_q, _, _ = self._voltage_dq(ctx)
        p_airgap = 1.5 * (v_d * i_d + v_q * i_q)
        denom = ctx.v_branch if abs(ctx.v_branch) > self.voltage_epsilon else (
            self.voltage_epsilon if ctx.v_branch >= 0 else -self.voltage_epsilon
        )
        return p_airgap / denom

    def dI_dv(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate partial derivative of branch current with respect to branch voltage.
        
        This derivative is used in the Jacobian matrix for Newton-Raphson solving.
        Since current = power / voltage, the derivative is -power / voltage².
        
        Args:
            ctx: Evaluation context containing branch voltage and time.
            z_next: State vector at next time step. If None, uses initial state.
        
        Returns:
            Partial derivative dI/dV (A/V).
        """
        if z_next is None:
            z_next = self.state_init()
        i_d, i_q = float(z_next[0]), float(z_next[1])
        v_d, v_q, _, _ = self._voltage_dq(ctx)
        p_airgap = 1.5 * (v_d * i_d + v_q * i_q)
        denom = ctx.v_branch if abs(ctx.v_branch) > self.voltage_epsilon else (
            self.voltage_epsilon if ctx.v_branch >= 0 else -self.voltage_epsilon
        )
        return -p_airgap / (denom ** 2)

    def dI_dz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate partial derivatives of branch current with respect to state vector.
        
        Only the dq currents (i_d, i_q) affect the branch current, as the current
        is computed from electrical power which depends only on v_d, v_q, i_d, i_q.
        
        Args:
            ctx: Evaluation context containing branch voltage and time.
            z_next: State vector at next time step [i_d, i_q, omega, theta_e, T].
        
        Returns:
            Array of shape (5,) containing [dI/di_d, dI/di_q, 0, 0, 0] (A/A).
        """
        v_d, v_q, _, _ = self._voltage_dq(ctx)
        denom = ctx.v_branch if abs(ctx.v_branch) > self.voltage_epsilon else (
            self.voltage_epsilon if ctx.v_branch >= 0 else -self.voltage_epsilon
        )
        coeff = 1.5 / denom
        return np.array([coeff * v_d, coeff * v_q, 0.0, 0.0, 0.0], dtype=float)

    # --- State-space ------------------------------------------------------
    def state_residual(self, ctx: EvalContext, z_next: Array, z_prev: Array) -> Array:
        """
        Calculate state equation residuals for implicit integration.
        
        Computes the residual vector R such that R(z_next, z_prev) = 0 at the
        solution. The residuals are formulated using backward Euler discretization
        for the differential equations governing:
        - Stator dq currents (i_d, i_q)
        - Mechanical speed (omega)
        - Electrical angle (theta_e)
        - Temperature (T)
        
        Args:
            ctx: Evaluation context containing branch voltage, time step, and time.
            z_next: State vector at next time step [i_d, i_q, omega, theta_e, T].
            z_prev: State vector at previous time step (same structure as z_next).
        
        Returns:
            Residual vector R of shape (5,) containing [R_id, R_iq, R_omega, R_theta,
            R_temp]. Each residual should be zero at convergence.
        """
        i_d, i_q, omega, theta, temp = [float(x) for x in z_next]
        i_d_prev, i_q_prev, omega_prev, theta_prev, temp_prev = [float(x) for x in z_prev]
        dt = ctx.dt

        v_d, v_q, _, _ = self._voltage_dq(ctx)
        omega_e = self._omega_e(omega)

        R_id = (
            self.R_s * i_d
            + self.L_d * (i_d - i_d_prev) / dt
            - omega_e * self.L_q * i_q
            - v_d
        )
        R_iq = (
            self.R_s * i_q
            + self.L_q * (i_q - i_q_prev) / dt
            + omega_e * (self.L_d * i_d + self.psi_f)
            - v_q
        )

        torque_e = self._torque(i_d, i_q)
        torque_load = self.load_torque(ctx.t_next, omega)
        R_omega = omega - omega_prev - (dt / self.J) * (torque_e - torque_load - self.B * omega)

        R_theta = theta - theta_prev - dt * omega_e

        p_copper = self.copper_loss_factor * self.R_s * (i_d ** 2 + i_q ** 2)
        p_iron = self.iron_loss_coeff * (omega ** 2)
        cooling = (temp - self.T_amb) / self.R_th
        R_temp = temp - temp_prev - (dt / self.C_th) * (p_copper + p_iron - cooling)

        return np.array([R_id, R_iq, R_omega, R_theta, R_temp], dtype=float)

    def dRdz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate Jacobian matrix of state residuals with respect to states.
        
        Computes the 5×5 Jacobian matrix J where J[i,j] = ∂R_i/∂z_j. This matrix
        is used in the Newton-Raphson solver to linearize the state equations
        around the current state estimate.
        
        The Jacobian includes partial derivatives for:
        - Electrical equations (i_d, i_q residuals w.r.t. all states)
        - Mechanical equation (omega residual w.r.t. all states)
        - Electrical angle equation (theta_e residual, mostly diagonal)
        - Thermal equation (temperature residual w.r.t. all states)
        
        Args:
            ctx: Evaluation context containing branch voltage, time step, and time.
            z_next: State vector at next time step [i_d, i_q, omega, theta_e, T].
        
        Returns:
            Jacobian matrix J of shape (5, 5) where J[i,j] = ∂R_i/∂z_j.
        """
        i_d, i_q, omega, _, _ = [float(x) for x in z_next]
        dt = ctx.dt
        omega_e = self._omega_e(omega)

        # Partials for electrical equations
        dR_id_did = self.R_s + self.L_d / dt
        dR_id_diq = -omega_e * self.L_q
        dR_id_domega = -self.pole_pairs * self.L_q * i_q

        dR_iq_did = omega_e * self.L_d
        dR_iq_diq = self.R_s + self.L_q / dt
        dR_iq_domega = self.pole_pairs * (self.L_d * i_d + self.psi_f)

        torque_e = self._torque(i_d, i_q)
        torque_load = self.load_torque(ctx.t_next, omega)
        dload = self.dload_domega(ctx.t_next, omega)
        dT_did = 1.5 * self.pole_pairs * (self.L_d - self.L_q) * i_q
        dT_diq = 1.5 * self.pole_pairs * (self.L_d * i_d + self.psi_f - self.L_q * i_d)

        dR_omega_did = -(ctx.dt / self.J) * dT_did
        dR_omega_diq = -(ctx.dt / self.J) * dT_diq
        dR_omega_domega = 1 + (ctx.dt / self.J) * (self.B + dload)

        dR_theta_domega = -ctx.dt * self.pole_pairs

        # Thermal partials
        dR_temp_did = -(ctx.dt / self.C_th) * (2 * self.copper_loss_factor * self.R_s * i_d)
        dR_temp_diq = -(ctx.dt / self.C_th) * (2 * self.copper_loss_factor * self.R_s * i_q)
        dR_temp_domega = -(ctx.dt / self.C_th) * (2 * self.iron_loss_coeff * omega)
        dR_temp_dtemp = 1 + (ctx.dt / (self.C_th * self.R_th))

        J = np.zeros((5, 5), dtype=float)
        J[0, 0] = dR_id_did
        J[0, 1] = dR_id_diq
        J[0, 2] = dR_id_domega

        J[1, 0] = dR_iq_did
        J[1, 1] = dR_iq_diq
        J[1, 2] = dR_iq_domega

        J[2, 0] = dR_omega_did
        J[2, 1] = dR_omega_diq
        J[2, 2] = dR_omega_domega

        J[3, 2] = dR_theta_domega
        J[3, 3] = 1.0

        J[4, 0] = dR_temp_did
        J[4, 1] = dR_temp_diq
        J[4, 2] = dR_temp_domega
        J[4, 4] = dR_temp_dtemp

        return J

    def dRdv(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate partial derivatives of state residuals with respect to branch voltage.
        
        Only the electrical current residuals (R_id, R_iq) depend on voltage.
        The flags dv_d and dv_q from `_voltage_dq` indicate which voltage components
        affect the residuals (1 if branch voltage affects that axis, 0 otherwise).
        
        Args:
            ctx: Evaluation context containing branch voltage, time step, and time.
            z_next: State vector at next time step [i_d, i_q, omega, theta_e, T].
        
        Returns:
            Array of shape (5,) containing [dR_id/dV, dR_iq/dV, 0, 0, 0].
            The first two elements are -dv_d and -dv_q respectively.
        """
        _, _, dv_d, dv_q = self._voltage_dq(ctx)
        return np.array([-dv_d, -dv_q, 0.0, 0.0, 0.0], dtype=float)
