from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict
from ..network.network import Network
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
    """
    t: Array
    v_nodes: Array     # shape (n_nodes, N+1)
    z_hist: Array      # shape (N+1, n_states_total)

    def get_state(self, name: str, net: Network) -> Array:
        """
        Extract state history for a specific component by name.
        
        Utility method to retrieve the time history of internal states for a
        named component. Currently returns a placeholder (first state column).
        Should be extended with a proper name-to-slice mapping.
        
        Args:
            name: Name of the component/branch.
            net: Network instance to access component information.
        
        Returns:
            Array of state values over time (placeholder implementation).
        """
        # utility da estendere con mappa nomi->slice
        return self.z_hist[:, 0]  # esempio per batteria singola

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

    return SimResult(t=t, v_nodes=v_nodes, z_hist=z_hist)