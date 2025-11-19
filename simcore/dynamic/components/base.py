from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class EvalContext:
    """Contesto per valutazione componenti al passo k+1."""
    v_branch: float           # tensione ramo al passo k+1
    dvdt_branch: float        # derivata implicita (v_{k+1}-v_k)/dt
    t_next: float             # tempo k+1
    dt: float                 # passo integrazione

@dataclass
class TimeTrace:
    """
    Generic time-series container (states, voltages, etc.).

    Attributes:
        t: Time vector shared by all samples.
        values: Array of samples (shape: len(t) x n_series, or len(t) for single series).
        names: Optional names for each column when values is 2-D.
    """
    t: Array
    values: Array
    names: list[str] | None = None

    def series(self, name: str | None = None) -> tuple[Array, Array]:
        """
        Return the full array or a named column.
        """
        if name is None:
            return self.t, self.values
        if not self.names:
            raise KeyError("This trace does not expose named series.")
        try:
            idx = self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Series '{name}' not found.") from exc
        return self.t, self.values[:, idx]

class BranchComponent(ABC):
    """
    Abstract base class for branch components in electrical circuit simulation.
    
    Branch components represent two-terminal circuit elements (resistors, capacitors,
    inductors, voltage sources, etc.) that connect nodes in the network. They are
    used by the solver to assemble Kirchhoff's Current Law (KCL) equations at each
    node.
    
    Each component must implement the current-voltage relationship and optionally
    can have internal state variables (e.g., capacitor voltage, inductor current)
    that evolve according to differential equations.
    
    Current convention: Positive current flows out of the source node (the node
    corresponding to the positive column in the network's incidence matrix A).
    This means current is positive when it flows from the first node to the second
    node of the branch.
    
    Subclasses must implement:
        - current: Compute branch current as a function of voltage and state
        - dI_dv: Derivative of current with respect to branch voltage
    
    Subclasses may optionally implement:
        - State management methods (n_states, state_names, state_init)
        - State evolution methods (state_residual, dRdz, dRdv, dI_dz)
    """
    # Power tracking (updated by Network.assemble)
    p_last: float = 0.0

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

    # ---- Post-processing helpers ----
    def _attach_state_trace(self, trace: TimeTrace | None) -> None:
        """Bind (or clear) the state-history trace."""
        if trace is None:
            if hasattr(self, "_state_trace"):
                delattr(self, "_state_trace")
        else:
            setattr(self, "_state_trace", trace)

    def _attach_voltage_trace(self, trace: TimeTrace | None) -> None:
        """Bind (or clear) the branch-voltage trace."""
        if trace is None:
            if hasattr(self, "_voltage_trace"):
                delattr(self, "_voltage_trace")
        else:
            setattr(self, "_voltage_trace", trace)

    def state_history(self, state_name: str | None = None) -> tuple[Array, Array]:
        """
        Return the simulated time history for this component's states.
        """
        trace: TimeTrace | None = getattr(self, "_state_trace", None)
        if trace is None:
            raise RuntimeError("No state history attached to this component yet.")
        return trace.series(state_name)

    def voltage_history(self) -> tuple[Array, Array]:
        """
        Return the branch voltage history for this component.
        """
        trace: TimeTrace | None = getattr(self, "_voltage_trace", None)
        if trace is None:
            raise RuntimeError("No voltage history attached to this component yet.")
        return trace.series()


class CompositeBranchComponent(BranchComponent):
    """
    Helper base class for two-terminal components built by composing other branches.

    The composite is described via a virtual subnetwork where the special node names
    `+` and `-` represent the external terminals. Additional internal nodes and
    branches can be registered via `add_internal_node` and `add_branch`. During
    network assembly, composite instances are automatically expanded into the
    underlying primitive branches, so the solver never evaluates them directly.
    """
    POSITIVE_NODE = "+"
    NEGATIVE_NODE = "-"

    def __init__(self) -> None:
        self._internal_nodes: set[str] = set()
        self._sub_branches: Dict[str, Tuple[str, str, BranchComponent]] = {}

    # Composites must be flattened before simulation, so the following methods
    # should never be called. They raise to catch improper usage.
    def current(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        raise RuntimeError("CompositeBranchComponent must be flattened before simulation.")

    def dI_dv(self, ctx: EvalContext, z_next: Array | None = None) -> float:
        raise RuntimeError("CompositeBranchComponent must be flattened before simulation.")

    def add_internal_node(self, name: str) -> None:
        """Register a new internal node for the composite subnetwork."""
        if name in {self.POSITIVE_NODE, self.NEGATIVE_NODE}:
            raise ValueError("Internal node name collides with external terminals.")
        if name in self._internal_nodes:
            raise ValueError(f"Internal node '{name}' already defined.")
        self._internal_nodes.add(name)

    def add_branch(self, name: str, n_from: str, n_to: str, component: BranchComponent) -> None:
        """
        Register a branch of the composite subnetwork.

        Args:
            name: Unique identifier within the composite.
            n_from: Source node name (internal or '+', '-').
            n_to: Destination node name (internal or '+', '-').
            component: BranchComponent instance representing the primitive element.
        """
        if name in self._sub_branches:
            raise ValueError(f"Composite branch '{name}' already exists.")
        for node in (n_from, n_to):
            if node not in self._internal_nodes and node not in {self.POSITIVE_NODE, self.NEGATIVE_NODE}:
                raise ValueError(f"Node '{node}' is not defined for this composite.")
        self._sub_branches[name] = (n_from, n_to, component)

    def _blueprint_nodes(self) -> Tuple[str, ...]:
        return tuple(sorted(self._internal_nodes))

    def _blueprint_branches(self) -> Dict[str, Tuple[str, str, BranchComponent]]:
        return dict(self._sub_branches)
