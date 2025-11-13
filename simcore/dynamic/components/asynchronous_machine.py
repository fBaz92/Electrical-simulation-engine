from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from .base import BranchComponent, EvalContext, Array
from ..utils.dq import dq_to_abc_series


def _default_load(t: float, omega: float) -> float:
    """
    Default load torque function (no load).
    
    This function serves as the default value for the `load_torque` parameter
    in AsynchronousMachineFOC. It returns zero torque, representing an
    unloaded motor condition.
    
    Args:
        t: Current time (s).
        omega: Mechanical rotor speed (rad/s).
    
    Returns:
        Load torque (N·m). Always returns 0.0.
    """
    return 0.0


def _default_dload(t: float, omega: float) -> float:
    """
    Default derivative of load torque with respect to speed (no speed dependence).
    
    This function serves as the default value for the `dload_domega` parameter
    in AsynchronousMachineFOC. It returns zero, indicating that the load torque
    does not depend on speed (used for Jacobian computation).
    
    Args:
        t: Current time (s).
        omega: Mechanical rotor speed (rad/s).
    
    Returns:
        Partial derivative dT_load/domega (N·m·s/rad). Always returns 0.0.
    """
    return 0.0


def _default_electrical_speed(t: float) -> float:
    """
    Default electrical speed function (stationary reference frame).
    
    This function serves as the default value for the `electrical_speed` parameter
    in AsynchronousMachineFOC. It returns zero, representing a stationary (non-rotating)
    dq reference frame.
    
    Args:
        t: Current time (s).
    
    Returns:
        Electrical speed omega_e (rad/s). Always returns 0.0.
    """
    return 0.0


@dataclass
class AsynchronousMachineFOC(BranchComponent):
    """
    Average-value squirrel-cage induction machine model in dq reference frame.
    
    This component implements a field-oriented control (FOC) model of an asynchronous
    (induction) motor with electrical, mechanical, and thermal dynamics. The model uses
    the dq (direct-quadrature) transformation to represent the three-phase AC machine
    in a rotating reference frame, simplifying the analysis and control.
    
    The model includes:
    - Electrical dynamics: stator and rotor currents/fluxes in dq frame
    - Mechanical dynamics: rotor speed and position
    - Thermal dynamics: temperature evolution based on losses
    
    States (7 total):
        - i_d: Direct-axis stator current (A)
        - i_q: Quadrature-axis stator current (A)
        - psi_dr: Direct-axis rotor flux linkage (Wb)
        - psi_qr: Quadrature-axis rotor flux linkage (Wb)
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
        The electrical speed (omega_e) driving the dq reference frame can be:
        - Supplied via `electrical_speed` callable (rad/s) for variable-speed operation
        - Derived from `supply_frequency` (Hz) for fixed-frequency operation
        - Defaults to 0 rad/s if neither is provided
    
    Thermal Model:
        The thermal model accounts for:
        - Copper losses: Stator and rotor resistive losses (scaled by `copper_loss_factor`)
        - Iron losses: Speed-dependent losses (proportional to omega^2)
        - Cooling: First-order thermal model with thermal resistance R_th and capacitance C_th
    
    Attributes:
        pole_pairs: Number of pole pairs (determines mechanical-to-electrical speed ratio).
        R_s: Stator resistance (Ω).
        R_r: Rotor resistance (Ω).
        L_ls: Stator leakage inductance (H).
        L_lr: Rotor leakage inductance (H).
        L_m: Magnetizing inductance (H).
        J: Rotor moment of inertia (kg·m²).
        B: Viscous friction coefficient (N·m·s/rad).
        C_th: Thermal capacitance (J/K).
        R_th: Thermal resistance (K/W).
        T_amb: Ambient temperature (K).
        supply_frequency: Supply frequency in Hz (used if electrical_speed not provided).
        load_torque: Callable(t, omega) -> torque (N·m). Load torque as function of time
            and mechanical speed. Defaults to zero.
        dload_domega: Callable(t, omega) -> dT_load/domega. Derivative of load torque
            with respect to speed (for Jacobian computation). Defaults to zero.
        voltage_cmd: Optional callable(t) -> (v_d, v_q). If provided, supplies direct
            dq voltage commands. If None, branch voltage is used as v_q.
        electrical_speed: Optional callable(t) -> omega_e (rad/s). Electrical speed of
            the dq reference frame. If None, uses supply_frequency.
        iron_loss_coeff: Iron loss coefficient (W·s²/rad²). Losses = coeff * omega².
        copper_loss_factor: Multiplier for copper losses. Typically 3.0 for three-phase
            (accounts for all three phases).
        i_d0: Initial direct-axis current (A).
        i_q0: Initial quadrature-axis current (A).
        psi_dr0: Initial direct-axis rotor flux (Wb).
        psi_qr0: Initial quadrature-axis rotor flux (Wb).
        omega0: Initial mechanical speed (rad/s).
        theta0: Initial electrical angle (rad).
        temp0: Initial temperature (K). Defaults to 293.15 K (20°C).
        voltage_epsilon: Small voltage threshold (V) to avoid division by zero in
            current computation when branch voltage is near zero.
    
    Example:
        >>> # Create a motor with voltage command control
        >>> motor = AsynchronousMachineFOC(
        ...     pole_pairs=2,
        ...     R_s=0.1, R_r=0.15,
        ...     L_ls=0.001, L_lr=0.001, L_m=0.01,
        ...     J=0.1, B=0.01,
        ...     C_th=1000.0, R_th=0.1, T_amb=293.15,
        ...     voltage_cmd=lambda t: (0.0, 100.0)  # v_d=0, v_q=100V
        ... )
        >>> 
        >>> # Or connect to network branch voltage
        >>> motor = AsynchronousMachineFOC(
        ...     pole_pairs=2,
        ...     R_s=0.1, R_r=0.15,
        ...     L_ls=0.001, L_lr=0.001, L_m=0.01,
        ...     J=0.1, B=0.01,
        ...     C_th=1000.0, R_th=0.1, T_amb=293.15,
        ...     voltage_cmd=None,  # Use branch voltage
        ...     supply_frequency=50.0  # 50 Hz operation
        ... )
    """

    pole_pairs: int
    R_s: float
    R_r: float
    L_ls: float          # stator leakage
    L_lr: float          # rotor leakage
    L_m: float           # magnetizing inductance
    J: float
    B: float
    C_th: float
    R_th: float
    T_amb: float
    supply_frequency: float = 0.0   # Hz, used if electrical_speed not provided
    load_torque: Callable[[float, float], float] = _default_load
    dload_domega: Callable[[float, float], float] = _default_dload
    voltage_cmd: Callable[[float], tuple[float, float]] | None = None
    electrical_speed: Callable[[float], float] | None = None
    iron_loss_coeff: float = 0.0
    copper_loss_factor: float = 3.0
    i_d0: float = 0.0
    i_q0: float = 0.0
    psi_dr0: float = 0.0
    psi_qr0: float = 0.0
    omega0: float = 0.0
    theta0: float = 0.0
    temp0: float = 293.15
    voltage_epsilon: float = 1e-3

    _state_count: int = field(init=False, default=7)

    def __post_init__(self) -> None:
        self._omega_sync_default = 2 * np.pi * self.supply_frequency

    # --- Branch interface -------------------------------------------------
    def n_states(self) -> int:
        return self._state_count

    def state_names(self) -> list[str]:
        return ["i_d", "i_q", "psi_dr", "psi_qr", "omega", "theta_e", "T"]

    def state_init(self) -> Array:
        return np.array(
            [self.i_d0, self.i_q0, self.psi_dr0, self.psi_qr0, self.omega0, self.theta0, self.temp0],
            dtype=float,
        )

    # --- Helper traces ----------------------------------------------------
    def dq_trace(self) -> dict[str, Array]:
        t, states = self.state_history()
        i_d = states[:, 0]
        i_q = states[:, 1]
        theta = states[:, 5]
        i_a, i_b, i_c = dq_to_abc_series(i_d, i_q, theta)
        return {
            "t": t,
            "i_d": i_d,
            "i_q": i_q,
            "i_a": i_a,
            "i_b": i_b,
            "i_c": i_c,
            "omega": states[:, 4],
            "theta_e": theta,
            "temperature": states[:, 6],
            "psi_dr": states[:, 2],
            "psi_qr": states[:, 3],
        }

    # --- Helpers ----------------------------------------------------------
    def _omega_e(self, t: float) -> float:
        """
        Get the electrical speed of the dq reference frame.
        
        Returns the electrical speed (rad/s) driving the dq transformation. If
        `electrical_speed` callable is provided, it is evaluated at time t.
        Otherwise, returns the default synchronous speed derived from
        `supply_frequency`.
        
        Args:
            t: Current time (s).
        
        Returns:
            Electrical speed omega_e (rad/s).
        """
        if self.electrical_speed is not None:
            return float(self.electrical_speed(t))
        return self._omega_sync_default

    def _omega_slip(self, t: float, omega_mech: float) -> float:
        """
        Calculate the slip speed (difference between electrical and rotor speeds).
        
        Slip speed is the difference between the electrical speed of the reference
        frame and the electrical speed of the rotor (pole_pairs * omega_mech).
        
        Args:
            t: Current time (s).
            omega_mech: Mechanical rotor speed (rad/s).
        
        Returns:
            Slip speed omega_slip (rad/s).
        """
        return self._omega_e(t) - self.pole_pairs * omega_mech

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

    def _psi_d(self, i_d: float, psi_dr: float) -> float:
        """
        Calculate direct-axis stator flux linkage.
        
        The direct-axis flux is the sum of leakage flux (L_ls * i_d) and the
        rotor flux linkage (psi_dr) coupled through the magnetizing inductance.
        
        Args:
            i_d: Direct-axis stator current (A).
            psi_dr: Direct-axis rotor flux linkage (Wb).
        
        Returns:
            Direct-axis stator flux linkage psi_d (Wb).
        """
        return self.L_ls * i_d + psi_dr

    def _psi_q(self, i_q: float, psi_qr: float) -> float:
        """
        Calculate quadrature-axis stator flux linkage.
        
        The quadrature-axis flux is the sum of leakage flux (L_ls * i_q) and the
        rotor flux linkage (psi_qr) coupled through the magnetizing inductance.
        
        Args:
            i_q: Quadrature-axis stator current (A).
            psi_qr: Quadrature-axis rotor flux linkage (Wb).
        
        Returns:
            Quadrature-axis stator flux linkage psi_q (Wb).
        """
        return self.L_ls * i_q + psi_qr

    def _torque(self, i_d: float, i_q: float, psi_dr: float, psi_qr: float) -> float:
        """
        Calculate electromagnetic torque.
        
        The torque is computed using the cross product of flux and current vectors
        in the dq frame. The factor 1.5 accounts for the three-phase to two-phase
        transformation (power invariant).
        
        Args:
            i_d: Direct-axis stator current (A).
            i_q: Quadrature-axis stator current (A).
            psi_dr: Direct-axis rotor flux linkage (Wb).
            psi_qr: Quadrature-axis rotor flux linkage (Wb).
        
        Returns:
            Electromagnetic torque T_e (N·m).
        """
        psi_d = self._psi_d(i_d, psi_dr)
        psi_q = self._psi_q(i_q, psi_qr)
        return 1.5 * self.pole_pairs * (psi_d * i_q - psi_q * i_d)

    # --- Current interface ------------------------------------------------
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate branch current from electrical power.
        
        The branch current is computed as the ratio of electrical power (in dq
        frame) to branch voltage. This allows the component to interface with
        the network solver while internally working with dq quantities.
        
        Electrical power is computed as 1.5 * (v_d * i_d + v_q * i_q), where
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
        p_e = 1.5 * (v_d * i_d + v_q * i_q)
        denom = ctx.v_branch if abs(ctx.v_branch) > self.voltage_epsilon else (
            self.voltage_epsilon if ctx.v_branch >= 0 else -self.voltage_epsilon
        )
        return p_e / denom

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
        p_e = 1.5 * (v_d * i_d + v_q * i_q)
        denom = ctx.v_branch if abs(ctx.v_branch) > self.voltage_epsilon else (
            self.voltage_epsilon if ctx.v_branch >= 0 else -self.voltage_epsilon
        )
        return -p_e / (denom ** 2)

    def dI_dz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate partial derivatives of branch current with respect to state vector.
        
        Only the dq currents (i_d, i_q) affect the branch current, as the current
        is computed from electrical power which depends only on v_d, v_q, i_d, i_q.
        
        Args:
            ctx: Evaluation context containing branch voltage and time.
            z_next: State vector at next time step [i_d, i_q, psi_dr, psi_qr, omega, theta_e, T].
        
        Returns:
            Array of shape (7,) containing [dI/di_d, dI/di_q, 0, 0, 0, 0, 0] (A/A).
        """
        v_d, v_q, _, _ = self._voltage_dq(ctx)
        denom = ctx.v_branch if abs(ctx.v_branch) > self.voltage_epsilon else (
            self.voltage_epsilon if ctx.v_branch >= 0 else -self.voltage_epsilon
        )
        coeff = 1.5 / denom
        return np.array([coeff * v_d, coeff * v_q, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    # --- State residuals --------------------------------------------------
    def state_residual(self, ctx: EvalContext, z_next: Array, z_prev: Array) -> Array:
        """
        Calculate state equation residuals for implicit integration.
        
        Computes the residual vector R such that R(z_next, z_prev) = 0 at the
        solution. The residuals are formulated using backward Euler discretization
        for the differential equations governing:
        - Stator dq currents (i_d, i_q)
        - Rotor dq flux linkages (psi_dr, psi_qr)
        - Mechanical speed (omega)
        - Electrical angle (theta_e)
        - Temperature (T)
        
        Args:
            ctx: Evaluation context containing branch voltage, time step, and time.
            z_next: State vector at next time step [i_d, i_q, psi_dr, psi_qr, omega, theta_e, T].
            z_prev: State vector at previous time step (same structure as z_next).
        
        Returns:
            Residual vector R of shape (7,) containing [R_id, R_iq, R_psidr, R_psiqr,
            R_omega, R_theta, R_temp]. Each residual should be zero at convergence.
        """
        i_d, i_q, psi_dr, psi_qr, omega, theta, temp = [float(x) for x in z_next]
        i_d_prev, i_q_prev, psi_dr_prev, psi_qr_prev, omega_prev, theta_prev, temp_prev = [
            float(x) for x in z_prev
        ]
        dt = ctx.dt

        v_d, v_q, _, _ = self._voltage_dq(ctx)
        omega_e = self._omega_e(ctx.t_next)
        omega_slip = omega_e - self.pole_pairs * omega

        psi_d = self._psi_d(i_d, psi_dr)
        psi_q = self._psi_q(i_q, psi_qr)

        R_id = (
            self.R_s * i_d
            + self.L_ls * (i_d - i_d_prev) / dt
            + (psi_dr - psi_dr_prev) / dt
            - omega_e * psi_q
            - v_d
        )
        R_iq = (
            self.R_s * i_q
            + self.L_ls * (i_q - i_q_prev) / dt
            + (psi_qr - psi_qr_prev) / dt
            + omega_e * psi_d
            - v_q
        )

        L_r = self.L_m + self.L_lr
        tau_r = L_r / self.R_r
        R_psidr = (
            psi_dr
            - psi_dr_prev
            - dt * (self.L_m / tau_r * i_d - psi_dr / tau_r + omega_slip * psi_qr)
        )
        R_psiqr = (
            psi_qr
            - psi_qr_prev
            - dt * (self.L_m / tau_r * i_q - psi_qr / tau_r - omega_slip * psi_dr)
        )

        torque_e = self._torque(i_d, i_q, psi_dr, psi_qr)
        torque_load = self.load_torque(ctx.t_next, omega)
        R_omega = omega - omega_prev - (dt / self.J) * (torque_e - torque_load - self.B * omega)

        # Electrical angle follows supply frequency
        R_theta = theta - theta_prev - dt * omega_e

        p_copper = self.copper_loss_factor * (
            self.R_s * (i_d ** 2 + i_q ** 2) + self.R_r * ((psi_dr / self.L_m) ** 2 + (psi_qr / self.L_m) ** 2)
        )
        p_iron = self.iron_loss_coeff * (omega ** 2)
        cooling = (temp - self.T_amb) / self.R_th
        R_temp = temp - temp_prev - (dt / self.C_th) * (p_copper + p_iron - cooling)

        return np.array(
            [R_id, R_iq, R_psidr, R_psiqr, R_omega, R_theta, R_temp],
            dtype=float,
        )

    def dRdz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate Jacobian matrix of state residuals with respect to states.
        
        Computes the 7×7 Jacobian matrix J where J[i,j] = ∂R_i/∂z_j. This matrix
        is used in the Newton-Raphson solver to linearize the state equations
        around the current state estimate.
        
        The Jacobian includes partial derivatives for:
        - Electrical equations (i_d, i_q residuals w.r.t. all states)
        - Rotor flux equations (psi_dr, psi_qr residuals w.r.t. all states)
        - Mechanical equation (omega residual w.r.t. all states)
        - Electrical angle equation (theta_e residual, mostly diagonal)
        - Thermal equation (temperature residual w.r.t. all states)
        
        Args:
            ctx: Evaluation context containing branch voltage, time step, and time.
            z_next: State vector at next time step [i_d, i_q, psi_dr, psi_qr, omega, theta_e, T].
        
        Returns:
            Jacobian matrix J of shape (7, 7) where J[i,j] = ∂R_i/∂z_j.
        """
        i_d, i_q, psi_dr, psi_qr, omega, _, temp = [float(x) for x in z_next]
        dt = ctx.dt
        omega_e = self._omega_e(ctx.t_next)
        omega_slip = omega_e - self.pole_pairs * omega

        psi_d = self._psi_d(i_d, psi_dr)
        psi_q = self._psi_q(i_q, psi_qr)

        L_r = self.L_m + self.L_lr
        tau_r = L_r / self.R_r

        J = np.zeros((7, 7), dtype=float)

        # R_id partials
        J[0, 0] = self.R_s + self.L_ls / dt
        J[0, 1] = -omega_e * self.L_ls
        J[0, 2] = 1.0 / dt
        J[0, 3] = -omega_e
        J[0, 4] = -self.pole_pairs * psi_q

        # R_iq partials
        J[1, 0] = omega_e * self.L_ls
        J[1, 1] = self.R_s + self.L_ls / dt
        J[1, 2] = omega_e
        J[1, 3] = 1.0 / dt
        J[1, 4] = self.pole_pairs * psi_d

        # R_psidr
        J[2, 0] = -dt * (self.L_m / tau_r)
        J[2, 2] = 1 + dt / tau_r
        J[2, 3] = -dt * omega_slip
        J[2, 4] = dt * self.pole_pairs * psi_qr

        # R_psiqr
        J[3, 1] = -dt * (self.L_m / tau_r)
        J[3, 2] = dt * omega_slip
        J[3, 3] = 1 + dt / tau_r
        J[3, 4] = -dt * self.pole_pairs * psi_dr

        # Mechanical equation
        torque_e = self._torque(i_d, i_q, psi_dr, psi_qr)
        dload = self.dload_domega(ctx.t_next, omega)
        torque_coeff = 1.5 * self.pole_pairs
        dT_did = -torque_coeff * psi_qr
        dT_diq = torque_coeff * psi_dr
        dT_dpsidr = torque_coeff * i_q
        dT_dpsiqr = -torque_coeff * i_d

        J[4, 0] = -(dt / self.J) * dT_did
        J[4, 1] = -(dt / self.J) * dT_diq
        J[4, 2] = -(dt / self.J) * dT_dpsidr
        J[4, 3] = -(dt / self.J) * dT_dpsiqr
        J[4, 4] = 1 + (dt / self.J) * (self.B + dload)

        J[5, 5] = 1.0  # theta depends only on itself

        # Thermal partials
        dR_temp_did = -(ctx.dt / self.C_th) * (2 * self.copper_loss_factor * self.R_s * i_d)
        dR_temp_diq = -(ctx.dt / self.C_th) * (2 * self.copper_loss_factor * self.R_s * i_q)
        dR_temp_dpsidr = -(ctx.dt / self.C_th) * (
            2 * self.copper_loss_factor * self.R_r * psi_dr / (self.L_m ** 2)
        )
        dR_temp_dpsiqr = -(ctx.dt / self.C_th) * (
            2 * self.copper_loss_factor * self.R_r * psi_qr / (self.L_m ** 2)
        )
        dR_temp_domega = -(ctx.dt / self.C_th) * (2 * self.iron_loss_coeff * omega)
        dR_temp_dtemp = 1 + (ctx.dt / (self.C_th * self.R_th))

        J[6, 0] = dR_temp_did
        J[6, 1] = dR_temp_diq
        J[6, 2] = dR_temp_dpsidr
        J[6, 3] = dR_temp_dpsiqr
        J[6, 4] = dR_temp_domega
        J[6, 6] = dR_temp_dtemp

        return J

    def dRdv(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate partial derivatives of state residuals with respect to branch voltage.
        
        Only the electrical current residuals (R_id, R_iq) depend on voltage.
        The flags dv_d and dv_q from `_voltage_dq` indicate which voltage components
        affect the residuals (1 if branch voltage affects that axis, 0 otherwise).
        
        Args:
            ctx: Evaluation context containing branch voltage, time step, and time.
            z_next: State vector at next time step [i_d, i_q, psi_dr, psi_qr, omega, theta_e, T].
        
        Returns:
            Array of shape (7,) containing [dR_id/dV, dR_iq/dV, 0, 0, 0, 0, 0].
            The first two elements are -dv_d and -dv_q respectively.
        """
        _, _, dv_d, dv_q = self._voltage_dq(ctx)
        return np.array([-dv_d, -dv_q, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
