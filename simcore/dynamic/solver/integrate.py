from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Callable
from ..network.network import Network
from ..components.base import BranchComponent, TimeTrace
from .newton import newton_solve, NewtonConfig

Array = np.ndarray

@dataclass
class SimResult:
    """
    Results from a time-domain circuit simulation.
    
    Contains the time history of node voltages and component states throughout
    the simulation. The data is organized as time series with one value per time step.
    
    Attributes:
        t: Time points array (length N+1).
        v_nodes: Node voltage history, shape (n_nodes, N+1).
            Each column corresponds to a time step, each row to a node.
        z_hist: Component state history, shape (N+1, n_states_total).
            Each row corresponds to a time step, columns are concatenated states
            from all stateful components.
        state_slices: Mapping from branch name to the slice selecting its states.
        state_names: Mapping from branch name to the list of state labels.
        v_branches: Branch voltage history, shape (N+1, n_branches) following network.branch_names.
        branch_index: Mapping branch name -> column index inside v_branches.
    """
    t: Array
    v_nodes: Array     # shape (n_nodes, N+1)
    z_hist: Array      # shape (N+1, n_states_total)
    state_slices: Dict[str, slice]
    state_names: Dict[str, list[str]]
    v_branches: Array
    branch_index: Dict[str, int]

    def component_state(self, branch_name: str, state_name: str | None = None
                        ) -> tuple[Array, Array]:
        """
        Return the time history of the states associated with a branch.
        """
        sl = self.state_slices.get(branch_name)
        if sl is None:
            raise KeyError(f"Component '{branch_name}' has no recorded states.")
        block = self.z_hist[:, sl]
        if state_name is None:
            return self.t, block
        names = self.state_names.get(branch_name, [])
        if not names:
            raise KeyError(f"State names unavailable for component '{branch_name}'.")
        try:
            idx = names.index(state_name)
        except ValueError as exc:
            raise KeyError(
                f"State '{state_name}' not found in component '{branch_name}'."
            ) from exc
        return self.t, block[:, idx]

    def branch_voltage(self, branch_name: str) -> tuple[Array, Array]:
        """
        Return the branch voltage time series for the specified component.
        """
        idx = self.branch_index.get(branch_name)
        if idx is None:
            raise KeyError(f"Branch '{branch_name}' not present in this simulation.")
        return self.t, self.v_branches[:, idx]

    def attach_component_traces(self, components: Dict[str, BranchComponent]) -> None:
        """
        Bind recorded histories directly to component instances.
        """
        for name, comp in components.items():
            sl = self.state_slices.get(name)
            if sl is None:
                comp._attach_state_trace(None)
            else:
                names = self.state_names.get(name, [])
                trace = TimeTrace(t=self.t, values=self.z_hist[:, sl], names=names)
                comp._attach_state_trace(trace)

            idx = self.branch_index.get(name)
            if idx is None:
                comp._attach_voltage_trace(None)
            else:
                v_trace = TimeTrace(t=self.t, values=self.v_branches[:, idx])
                comp._attach_voltage_trace(v_trace)

def run_sim(network: Network, t_stop: float, v0_nodes: Array | None = None,
            newton_cfg: NewtonConfig | None = None) -> SimResult:
    """
    Run a time-domain simulation of an electrical network.
    
    Performs implicit time-stepping using the network's time step (dt). At each
    time step, solves the coupled system of KCL equations and component state
    equations using Newton-Raphson method. Uses continuation (previous solution
    as initial guess) for better convergence.
    
    The simulation solves for both node voltages and component internal states
    simultaneously, ensuring consistency between electrical and state equations.
    
    Args:
        network: Network instance containing topology, components, and time step.
        t_stop: Simulation end time (seconds).
        v0_nodes: Initial node voltages (optional). If None, starts from zero.
        newton_cfg: Newton-Raphson solver configuration (optional).
            If None, uses default NewtonConfig.
    
    Returns:
        SimResult containing:
        - Time points array
        - Node voltage history (n_nodes × N+1)
        - Component state history (N+1 × n_states_total)
    """

    dt = network.dt
    N = int(round(t_stop / dt))
    t = np.linspace(0.0, t_stop, N+1)

    n_nodes = network.A.shape[0]
    v_nodes = np.zeros((n_nodes, N+1))

    # stati
    n_states = network.z0.size
    z_hist = np.zeros((N+1, n_states))
    z_prev = network.z0.copy()
    z_hist[0, :] = z_prev

    if v0_nodes is not None:
        v_nodes[:, 0] = v0_nodes

    if newton_cfg is None:
        newton_cfg = NewtonConfig()

    for k in range(N):
        v_prev = v_nodes[:, k].copy()
        x0 = np.concatenate([v_prev, z_prev])

        def FJ(x):
            nv = n_nodes
            v_next = x[:nv]
            z_next = x[nv:]
            return network.assemble(v_next, v_prev, z_next, z_prev, t[k+1])

        x_sol = newton_solve(FJ, x0, newton_cfg)
        nv = n_nodes
        v_nodes[:, k+1] = x_sol[:nv]
        z_prev = x_sol[nv:]              # aggiorna stato
        z_hist[k+1, :] = z_prev          # <-- salva lo stato al passo k+1

    v_branches = (network.A.T @ v_nodes).T  # shape (N+1, n_branches)
    branch_index = {name: idx for idx, name in enumerate(network.branch_names)}

    result = SimResult(
        t=t,
        v_nodes=v_nodes,
        z_hist=z_hist,
        state_slices=network.state_slice_map,
        state_names=network.state_name_map,
        v_branches=v_branches,
        branch_index=branch_index,
    )
    result.attach_component_traces(network.components)
    return result


def run_sim_with_control(
    network: Network,
    t_stop: float,
    v0_nodes: Array,
    control_callback: Callable[[int, float, Array, Array, Dict[str, BranchComponent]], None],
    newton_cfg: NewtonConfig | None = None
) -> SimResult:
    """
    Run a time-domain simulation with control callback support.
    
    Similar to run_sim, but calls control_callback BEFORE each implicit time step
    resolution. This allows the callback to modify component parameters (e.g., 
    controlled voltage sources) based on the current simulation state.
    
    The control callback is invoked at the beginning of each time step with:
    - Current step index k
    - Current time t_k
    - Previous node voltages v_prev
    - Previous component states z_prev
    - Dictionary of all network components (for modification)
    
    This enables closed-loop control scenarios where component parameters are
    adjusted based on measured voltages or states during the simulation.
    
    Args:
        network: Network instance containing topology, components, and time step.
        t_stop: Simulation end time (seconds).
        v0_nodes: Initial node voltages array (shape: n_nodes).
        control_callback: Callback function called before each time step.
            Signature: callback(k: int, t_k: float, v_prev: Array, z_prev: Array, 
            components: Dict[str, BranchComponent]) -> None
            The callback can modify component parameters in-place via the components dict.
        newton_cfg: Newton-Raphson solver configuration (optional).
            If None, uses default NewtonConfig.
    
    Returns:
        SimResult containing:
        - Time points array
        - Node voltage history (n_nodes × N+1)
        - Component state history (N+1 × n_states_total)
    
    Example:
        def my_controller(k, t, v, z, comps):
            # Update controlled voltage source based on measured voltage
            comps["V_ctrl"].V_setpoint = 5.0 if v[0] < 4.0 else 3.0
        
        result = run_sim_with_control(net, t_stop=10.0, v0_nodes=v0, 
                                     control_callback=my_controller)
    """

    dt = network.dt
    N = int(round(t_stop / dt))
    t = np.linspace(0.0, t_stop, N+1)

    # Numero nodi (equazioni KCL)
    n_nodes = network.A.shape[0]
    v_nodes = np.zeros((n_nodes, N+1))
    v_nodes[:, 0] = v0_nodes.copy()

    # Stati dinamici
    z_prev = network.z0.copy()
    z_hist = np.zeros((N+1, z_prev.size))
    z_hist[0, :] = z_prev

    if newton_cfg is None:
        newton_cfg = NewtonConfig()

    for k in range(N):
        t_k = t[k]
        v_prev = v_nodes[:, k]

        control_callback(k, t_k, v_prev, z_prev, network.components)

        # Risoluzione implicita del passo k+1
        x0 = np.concatenate([v_prev, z_prev])

        def FJ(x):
            v_next = x[:n_nodes]
            z_next = x[n_nodes:]
            return network.assemble(v_next, v_prev, z_next, z_prev, t_k + dt)

        x_sol = newton_solve(FJ, x0, newton_cfg)

        v_next = x_sol[:n_nodes]
        z_next = x_sol[n_nodes:]

        v_nodes[:, k+1] = v_next
        z_prev = z_next.copy()
        z_hist[k+1, :] = z_prev

    v_branches = (network.A.T @ v_nodes).T
    branch_index = {name: idx for idx, name in enumerate(network.branch_names)}

    result = SimResult(
        t=t,
        v_nodes=v_nodes,
        z_hist=z_hist,
        state_slices=network.state_slice_map,
        state_names=network.state_name_map,
        v_branches=v_branches,
        branch_index=branch_index,
    )
    result.attach_component_traces(network.components)
    return result
