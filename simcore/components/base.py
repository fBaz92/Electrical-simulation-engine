from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class EvalContext:
    """Contesto per valutazione componenti al passo k+1."""
    v_branch: float           # tensione ramo al passo k+1
    dvdt_branch: float        # derivata implicita (v_{k+1}-v_k)/dt
    t_next: float             # tempo k+1
    dt: float                 # passo integrazione

class BranchComponent(ABC):
    """
    Interfaccia componente di ramo (per KCL).
    Convenzione: corrente positiva = 'esce' dal nodo sorgente della colonna di A.
    """

    # ---- I-V nonlineare (richiesta dal solver) ----
    @abstractmethod
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float: ...
    @abstractmethod
    def dI_dv(self, ctx: EvalContext, z_next: Array | None = None) -> float: ...

    # ---- Stato interno opzionale ----
    def n_states(self) -> int: return 0
    def state_names(self) -> list[str]: return []
    def state_init(self) -> Array: return np.empty(0)

    # R(z_{k+1}, v_{k+1}; z_k) = 0
    def state_residual(self, ctx: EvalContext, z_next: Array, z_prev: Array) -> Array:
        return np.empty(0)
    # Jacobiane
    def dRdz(self, ctx: EvalContext, z_next: Array) -> Array:
        return np.empty((0,0))
    def dRdv(self, ctx: EvalContext, z_next: Array) -> Array:
        return np.empty((0,))
    def dI_dz(self, ctx: EvalContext, z_next: Array) -> Array:
        return np.empty((0,))
