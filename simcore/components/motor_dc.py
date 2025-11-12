from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional
from .base import BranchComponent, EvalContext

Array = np.ndarray

@dataclass
class MotorDCComponent(BranchComponent):
    """
    DC motor component with coupled electrical, mechanical, and thermal dynamics.
    
    This component models a DC motor as a three-domain system:
    
    1. **Electrical domain**: RL circuit with back-EMF
       - L * di/dt = v - R(T)*i - k_e*omega
       - Resistance R(T) varies with temperature
    
    2. **Mechanical domain**: Rotational dynamics
       - J * domega/dt = k_t*i - tau_load - b*omega
       - Includes optional load torque and viscous friction
    
    3. **Thermal domain**: First-order thermal model
       - C_th * dT/dt = P_cu + P_core - (T - T_amb)/R_th
       - Includes copper losses (P_cu = i^2*R) and optional core losses
    
    The component has three state variables: [i, omega, T] (current, angular velocity,
    temperature). All three domains are solved simultaneously using implicit Euler
    integration, ensuring consistency across electrical, mechanical, and thermal equations.
    
    Attributes:
        R0: Winding resistance at reference temperature T0 [Ohm].
        alpha: Temperature coefficient of resistance [1/K].
        T0: Reference temperature for R0 [K].
        L: Winding inductance [H].
        k_e: Back-EMF constant [V·s/rad]. Relates back-EMF to angular velocity.
        k_t: Torque constant [N·m/A]. Relates electromagnetic torque to current.
        
        J: Rotor moment of inertia [kg·m²].
        b: Viscous friction coefficient [N·m·s/rad].
        
        C_th: Thermal capacitance [J/K].
        R_th: Thermal resistance to ambient [K/W].
        T_amb: Ambient temperature [K].
        
        tau_load: Load torque function tau_load(t, omega) -> float [N·m].
            Default: no load (returns 0.0).
        dtau_domega: Derivative of load torque w.r.t. angular velocity [N·m·s/rad].
            Default: returns 0.0.
        
        P_core: Core losses function P_core(omega, i) -> float [W].
            Default: no core losses (returns 0.0).
        dPcore_di: Derivative of core losses w.r.t. current [W/A].
            Default: returns 0.0.
        dPcore_domega: Derivative of core losses w.r.t. angular velocity [W·s/rad].
            Default: returns 0.0.
        
        i0: Initial current [A]. Default: 0.0.
        omega0: Initial angular velocity [rad/s]. Default: 0.0.
        T0_state: Initial temperature [K]. Default: 293.15 K (20°C).
    """
        # Elettrici (necessari)
    R0: float                 # ohm @ T0
    alpha: float              # 1/K
    T0: float                 # K
    L: float                  # H
    k_e: float                # V·s/rad
    k_t: float                # N·m/A

    # Meccanici (necessari)
    J: float                  # kg·m^2
    b: float                  # N·m·s/rad

    # Termici (necessari)
    C_th: float               # J/K
    R_th: float               # K/W
    T_amb: float              # K

    # Optional load + core losses
    tau_load: callable = lambda t, w: 0.0
    dtau_domega: callable = lambda t, w: 0.0

    P_core: callable = lambda w, i: 0.0
    dPcore_di: callable = lambda w, i: 0.0
    dPcore_domega: callable = lambda w, i: 0.0

    # Stati iniziali
    i0: float = 0.0
    omega0: float = 0.0
    T0_state: float = 293.15
    
    # ---- Corrente di ramo (i_k+1) ----
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the current through the DC motor.
        
        The motor current is a state variable, so it is directly extracted from
        the state vector. The current is the first element of the state vector [i, omega, T].
        
        Args:
            ctx: Evaluation context (not used, current is a state variable).
            z_next: State vector at next time step [i, omega, T]. If None, uses i0.
        
        Returns:
            Motor current in Amperes.
        """
        i = float(z_next[0]) if z_next is not None else self.i0
        return i

    def dI_dv(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        """
        Calculate the derivative of current with respect to branch voltage.
        
        Since the motor current is a state variable (not directly dependent on voltage),
        the derivative is zero. The voltage affects the current through the state
        equation (electrical circuit equation), not through a direct I-V relationship.
        
        Args:
            ctx: Evaluation context (not used).
            z_next: State vector at next time step (not used).
        
        Returns:
            Derivative of current w.r.t. voltage: 0.0
        """
        return 0.0

    def dI_dz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the derivative of current with respect to state variables.
        
        The current depends only on the first state variable (i), so:
        dI/d[i, omega, T] = [1, 0, 0]
        
        Args:
            ctx: Evaluation context (not used).
            z_next: State vector at next time step [i, omega, T].
        
        Returns:
            Jacobian array: [1.0, 0.0, 0.0] - derivative of current w.r.t. [i, omega, T]
        """
        # derivative wrt [i, omega, T] → [1,0,0]
        return np.array([1.0, 0.0, 0.0])

    # ---- Stati ----
    def n_states(self) -> int:
        """
        Return the number of internal state variables.
        
        Returns:
            Number of states: 3 (current i, angular velocity omega, temperature T)
        """
        return 3
    
    def state_names(self) -> list[str]:
        """
        Return the names of the internal state variables.
        
        Returns:
            List of state names: ["i", "omega", "T"]
        """
        return ["i", "omega", "T"]
    
    def state_init(self) -> Array:
        """
        Return the initial state vector.
        
        Returns:
            Initial state array: [i0, omega0, T0_state]
        """
        return np.array([self.i0, self.omega0, self.T0_state], dtype=float)

    # Helpers
    def R_of_T(self, T: float) -> float:
        """
        Compute winding resistance at a given temperature.
        
        Uses the linear temperature coefficient model:
        R(T) = R0 * (1 + alpha * (T - T0))
        
        Args:
            T: Temperature in Kelvin.
        
        Returns:
            Winding resistance in Ohms.
        """
        return self.R0 * (1.0 + self.alpha * (T - self.T0))

    def dR_dT(self) -> float:
        """
        Compute the derivative of resistance with respect to temperature.
        
        Returns:
            dR/dT = R0 * alpha [Ohm/K]
        """
        return self.R0 * self.alpha

    # ---- Residui di stato (Euler implicito) ----
    def state_residual(self, ctx: EvalContext, z_next: Array, z_prev: Array) -> Array:
        """
        Calculate the state residual equations for the DC motor.
        
        Implements three coupled residual equations using implicit Euler integration:
        
        1. Electrical equation:
           Fi = i_{k+1} - i_k - (dt/L) * (v - R(T)*i_{k+1} - k_e*omega_{k+1}) = 0
        
        2. Mechanical equation:
           Fw = omega_{k+1} - omega_k - (dt/J) * (k_t*i_{k+1} - tau_load - b*omega_{k+1}) = 0
        
        3. Thermal equation:
           FT = T_{k+1} - T_k - (dt/C_th) * (P_cu + P_core - (T_{k+1} - T_amb)/R_th) = 0
        
        where:
        - R(T) is the temperature-dependent winding resistance
        - P_cu = i^2 * R(T) is the copper losses
        - P_core is the core losses (function of omega and i)
        - tau_load is the load torque (function of time and omega)
        
        Args:
            ctx: Evaluation context containing time step dt, branch voltage v_branch,
                 and next time t_next.
            z_next: State vector at next time step [i_{k+1}, omega_{k+1}, T_{k+1}].
            z_prev: State vector at previous time step [i_k, omega_k, T_k].
        
        Returns:
            Residual array: [Fi, Fw, FT]
        """
        dt = ctx.dt
        v  = ctx.v_branch

        i_next, w_next, T_next = float(z_next[0]), float(z_next[1]), float(z_next[2])
        i_prev, w_prev, T_prev = float(z_prev[0]), float(z_prev[1]), float(z_prev[2])

        R_T = self.R_of_T(T_next)

        # elettrico
        Fi = i_next - i_prev - (dt/self.L) * ( v - R_T * i_next - self.k_e * w_next )

        # meccanico
        tau_l = self.tau_load(ctx.t_next, w_next)
        Fw = w_next - w_prev - (dt/self.J) * ( self.k_t * i_next - tau_l - self.b * w_next )

        # termico
        Pcu = i_next**2 * R_T
        Pco = self.P_core(w_next, i_next)
        FT = T_next - T_prev - (dt/self.C_th) * ( Pcu + Pco - (T_next - self.T_amb)/self.R_th )

        return np.array([Fi, Fw, FT], dtype=float)

    # ---- Jacobiane dei residui ----
    def dRdz(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the Jacobian of state residuals with respect to state variables.
        
        This is the 3×3 Jacobian matrix of the residual vector [Fi, Fw, FT] with respect
        to the state vector [i, omega, T]. Used by the Newton-Raphson solver for state
        equation iterations.
        
        The Jacobian has the structure:
        [[dFi/di, dFi/domega, dFi/dT],
         [dFw/di, dFw/domega, dFw/dT],
         [dFT/di, dFT/domega, dFT/dT]]
        
        Args:
            ctx: Evaluation context containing time step dt, branch voltage v_branch,
                 and next time t_next.
            z_next: State vector at next time step [i, omega, T].
        
        Returns:
            3×3 Jacobian matrix of residuals w.r.t. states.
        """
        dt = ctx.dt
        i, w, T = float(z_next[0]), float(z_next[1]), float(z_next[2])

        R_T = self.R_of_T(T)
        dRdT = self.dR_dT()

        # dFi/d[i, w, T]
        dFi_di = 1.0 + (dt/self.L) * R_T
        dFi_dw = (dt/self.L) * self.k_e
        dFi_dT = (dt/self.L) * i * dRdT

        # dFw/d[i, w, T]
        dFw_di = -(dt/self.J) * self.k_t
        dFw_dw = 1.0 + (dt/self.J) * ( self.dtau_domega(ctx.t_next, w) + self.b )
        dFw_dT = 0.0

        # dFT/d[i, w, T]
        dFT_di = -(dt/self.C_th) * ( 2.0 * i * R_T + self.dPcore_di(w, i) )
        dFT_dw = -(dt/self.C_th) * ( self.dPcore_domega(w, i) )
        dFT_dT = 1.0 - (dt/self.C_th) * ( i**2 * dRdT - 1.0/self.R_th )

        Jzz = np.array([
            [dFi_di, dFi_dw, dFi_dT],
            [dFw_di, dFw_dw, dFw_dT],
            [dFT_di, dFT_dw, dFT_dT],
        ], dtype=float)
        return Jzz

    def dRdv(self, ctx: EvalContext, z_next: Array) -> Array:
        """
        Calculate the Jacobian of state residuals with respect to branch voltage.
        
        Only the electrical residual (Fi) depends on voltage:
        dFi/dv = -(dt/L)
        
        The mechanical and thermal residuals do not depend directly on voltage,
        so their derivatives are zero.
        
        Args:
            ctx: Evaluation context containing time step dt.
            z_next: State vector at next time step (not used, but required by interface).
        
        Returns:
            Jacobian array: [-(dt/L), 0.0, 0.0] - derivatives of [Fi, Fw, FT] w.r.t. voltage
        """
        # ∂Fi/∂v = -(dt/L); gli altri residui non dipendono da v
        return np.array([-(ctx.dt/self.L), 0.0, 0.0], dtype=float)
