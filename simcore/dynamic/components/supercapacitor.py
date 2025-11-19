from __future__ import annotations
from dataclasses import dataclass
from .base import CompositeBranchComponent, EvalContext, Array
from .resistor import Resistor
from .capacitor import Capacitor
import numpy as np


@dataclass
class Supercapacitor(CompositeBranchComponent):
    """
    Two-terminal RC network with a resistor in series with a capacitor.

    This demonstrates how to create composite components: internally the branch
    is expanded into the primitive resistor/capacitor pair, yet externally it
    behaves like any other bipole.
    """
    esr10ms: float
    capacitance: float
    esr1s: float
    cells_per_module: int
    modules_per_string: int
    strings: int
    tau_dl: float = 0.03
    
    # Thermal parameters
    R_th: float = 5.0      # K/W
    C_th: float = 100.0    # J/K
    T_amb: float = 25.0    # °C

    def __post_init__(self) -> None:
        super().__init__()

        assert self.esr1s > self.esr10ms, "ESR at 1s must be greater than ESR at 10ms"

        self.branch_esr = (self.esr1s - self.esr10ms) 
        self.branch_esr *= self.cells_per_module * self.modules_per_string / self.strings

        self.branch_capacitance = self.tau_dl / self.branch_esr
        self.branch_capacitance *= self.cells_per_module / self.modules_per_string * self.strings

        self.add_internal_node("mid")
        self.add_internal_node("dl_node")

        # add a parallel branch for the_dl_node
        self.add_branch("dl_r", self.POSITIVE_NODE, "dl_node", Resistor(self.branch_esr))
        self.add_branch("dl_c", self.POSITIVE_NODE, "dl_node", Capacitor(self.branch_capacitance))

        self.add_branch("R", "dl_node", "mid", Resistor(self.esr10ms*self.cells_per_module*self.modules_per_string/self.strings))
        self.add_branch("C", "mid", self.NEGATIVE_NODE, Capacitor(self.capacitance/self.cells_per_module/self.modules_per_string*self.strings))

    # ---- Thermal State Implementation ----
    def n_states(self) -> int:
        return 1

    def state_names(self) -> list[str]:
        return ["Temperature"]

    def state_init(self) -> Array:
        return np.array([self.T_amb])

    def state_residual(self, ctx: EvalContext, z_next: Array, z_prev: Array) -> Array:
        # z[0] is Temperature
        T_next = z_next[0]
        T_prev = z_prev[0]
        dt = ctx.dt

        # Calculate total power dissipated by internal components
        # We use p_last which is updated by the solver in the current iteration
        P_total = 0.0
        for _, _, comp in self._sub_branches.values():
            P_total += getattr(comp, "p_last", 0.0)

        # Thermal equation: C_th * (T_next - T_prev)/dt = P_total - (T_next - T_amb)/R_th
        # Residual: T_next - T_prev - dt/C_th * (P_total - (T_next - T_amb)/R_th)
        
        heat_loss = (T_next - self.T_amb) / self.R_th
        res = T_next - T_prev - (dt / self.C_th) * (P_total - heat_loss)
        return np.array([res])

    def dRdz(self, ctx: EvalContext, z_next: Array) -> Array:
        # dR/dT_next
        # R = T_next - ... + dt/C_th * (T_next - T_amb)/R_th
        # dR/dT = 1 + dt/(C_th * R_th)
        val = 1.0 + ctx.dt / (self.C_th * self.R_th)
        return np.array([[val]])

    def dRdv(self, ctx: EvalContext, z_next: Array) -> Array:
        # We assume weak coupling (dP/dV_ext approx 0 for thermal step)
        return np.zeros(1)

    def dI_dz(self, ctx: EvalContext, z_next: Array) -> Array:
        # No direct current contribution from Temperature to external terminals
        # (The current flows through the internal resistors, which are handled by KCL)
        return np.zeros(1)
