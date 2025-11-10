from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional
from .base import BranchComponent, EvalContext

Array = np.ndarray

@dataclass
class MotorDCComponent(BranchComponent):
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
        i = float(z_next[0]) if z_next is not None else self.i0
        return i

    def dI_dv(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        return 0.0

    def dI_dz(self, ctx: EvalContext, z_next: Array) -> Array:
        # derivative wrt [i, omega, T] → [1,0,0]
        return np.array([1.0, 0.0, 0.0])

    # ---- Stati ----
    def n_states(self) -> int: return 3
    def state_names(self) -> list[str]: return ["i", "omega", "T"]
    def state_init(self) -> Array:
        return np.array([self.i0, self.omega0, self.T0_state], dtype=float)

    # Helpers
    def R_of_T(self, T: float) -> float:
        return self.R0 * (1.0 + self.alpha * (T - self.T0))

    def dR_dT(self) -> float:
        return self.R0 * self.alpha

    # ---- Residui di stato (Euler implicito) ----
    def state_residual(self, ctx: EvalContext, z_next: Array, z_prev: Array) -> Array:
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
        # ∂Fi/∂v = -(dt/L); gli altri residui non dipendono da v
        return np.array([-(ctx.dt/self.L), 0.0, 0.0], dtype=float)
