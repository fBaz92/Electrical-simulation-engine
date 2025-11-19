from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Callable
from ..network.network import Network
from ..components.base import BranchComponent, TimeTrace, EvalContext
from ..components.capacitor import Capacitor
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
        i_branches: Branch current history, shape (N+1, n_branches).
        branch_index: Mapping branch name -> column index inside v_branches.
    """
    t: Array
    v_nodes: Array     # shape (n_nodes, N+1)
    z_hist: Array      # shape (N+1, n_states_total)
    state_slices: Dict[str, slice]
    state_names: Dict[str, list[str]]
    v_branches: Array
    i_branches: Array
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

def solve_consistent_initial_conditions(network: Network, newton_cfg: NewtonConfig | None = None) -> Array:
    """
    Solve for the consistent initial node voltages (DC Operating Point) at t=0.
    
    This treats:
    - Capacitors as Voltage Sources (fixed at their initial voltage, usually 0).
    - Inductors as Current Sources (fixed at their initial current, usually 0).
    - Resistors and Sources as usual.
    
    This avoids artificial transients when connecting sources to uncharged capacitive networks.
    """
    if newton_cfg is None:
        newton_cfg = NewtonConfig()
        
    n_nodes = network.A.shape[0]
    v_guess = np.zeros(n_nodes)
    
    # We need to solve F(v) = 0 where F is KCL.
    # But components behave differently in DC/Initialization.
    # For now, we only handle Capacitors specially (as voltage sources).
    
    # We can reuse network.assemble BUT we need to trick it or the components.
    # Since we don't want to modify components, we can create a wrapper or 
    # manually assemble the system for initialization.
    # Manual assembly is safer to avoid polluting component logic.
    
    # However, assemble is complex.
    # Let's define a specialized assemble for DC init.
    
    comps = [network.components[name] for name in network.branch_names]
    A = network.A
    
    def FJ_init(v):
        # v is node voltages
        v_branch = A.T @ v
        
        # We need to calculate branch currents and dI/dv
        i_branch = np.zeros(len(comps))
        dI_dvbranch = np.zeros(len(comps)) # Conductance
        
        # For initialization, we assume dv/dt = 0 (DC steady state relative to the instantaneous voltage)
        # OR we treat C as voltage source.
        # If we treat C as voltage source Vc_init (0), then:
        # I_c is unknown? No, KCL solves for node voltages.
        # If C is a voltage source, it fixes the voltage across it.
        # This means it contributes to the Jacobian as a voltage source (G = inf).
        # This is numerically hard.
        
        # Better approach: Model C as a small resistance (short) in series with V_init?
        # Or simply a very large conductance.
        # G_c_init = 1e6 (1 uOhm)
        # I = G_c_init * (V_branch - V_c_init)
        
        G_stiff = 1e9 # 1 nOhm
        
        for bi, comp in enumerate(comps):
            # Check type
            if isinstance(comp, Capacitor):
                # Treat as Voltage Source with series conductance G_stiff
                # V_target = V_init (assume 0 for now or get from somewhere?)
                # Capacitor doesn't store V_init explicitly in a way we can access easily 
                # unless we look at z0.
                # But z0 is global.
                # For this task, we assume uncharged capacitors (V=0) or we need to map z0.
                
                # Let's assume V_c = 0 for uncharged start, which is the common case causing the spike.
                # If we want to support pre-charged, we'd need to map z0.
                
                V_c = 0.0 
                if comp.V_init is not None:
                    V_c = comp.V_init
                
                # I = G * (V_b - V_c)
                i_branch[bi] = G_stiff * (v_branch[bi] - V_c)
                dI_dvbranch[bi] = G_stiff
            else:
                # Regular component (Resistor, Voltage Source, etc.)
                # We evaluate it with dvdt=0
                ctx = EvalContext(v_branch=float(v_branch[bi]),
                                  dvdt_branch=0.0,
                                  t_next=0.0, dt=1.0) # dt doesn't matter for static
                
                # We pass None for states, assuming resistive components don't depend on states for DC I(V)
                # (except maybe thermal? but thermal is slow, so T=T_amb is fine)
                i_branch[bi] = comp.current(ctx, None)
                dI_dvbranch[bi] = comp.dI_dv(ctx, None)
                
        # KCL: A @ i_branch = 0
        F = A @ i_branch
        
        # Jacobian: A @ diag(dI/dv) @ A.T
        J = A @ np.diag(dI_dvbranch) @ A.T
        
        return F, J

    # Solve
    v_sol = newton_solve(FJ_init, v_guess, newton_cfg)
    return v_sol

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

    z_hist[0, :] = z_prev
    
    # Determine initial node voltages
    if v0_nodes is not None:
        v_nodes[:, 0] = v0_nodes
    else:
        # Calculate consistent initial conditions
        v_nodes[:, 0] = solve_consistent_initial_conditions(network, newton_cfg)

    if newton_cfg is None:
        newton_cfg = NewtonConfig()

    # Initialize branch voltages and currents
    n_branches = network.A.shape[1]
    v_branches = np.zeros((N+1, n_branches))
    i_branches = np.zeros((N+1, n_branches))

    # Calculate initial branch voltages and currents (t=0)
    v_branches[0, :] = network.A.T @ v_nodes[:, 0]
    comps = [network.components[name] for name in network.branch_names]
    z_split_list_0 = network._split_z(z_hist[0]) if network.stateful_indices else []
    s_idx_0 = 0
    # For t=0, dvdt is approximated using forward difference
    # This requires v_branches[1] which is not yet computed.
    # For now, we'll set dvdt_b_k to 0 for t=0, or re-evaluate after first step.
    # A more robust approach might be to compute dvdt_b_k for t=0 after the first step.
    # For simplicity, let's assume dvdt_b_k is 0 at t=0 for current calculation.
    # Or, we can compute it after the first step and update i_branches[0].
    # Let's compute it after the first step for t=0.
    # For now, we'll compute i_branches[0] with dvdt_b_k = 0.0
    for bi, comp in enumerate(comps):
        ctx = EvalContext(v_branch=float(v_branches[0, bi]),
                          dvdt_branch=0.0, # Will be updated after first step
                          t_next=float(t[0]), dt=float(dt))
        if comp.n_states() > 0:
            zc = z_split_list_0[s_idx_0]
            i_branches[0, bi] = comp.current(ctx, zc)
            s_idx_0 += 1
        else:
            i_branches[0, bi] = comp.current(ctx, None)


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

        # Calculate branch voltages and currents for the current step k+1
        v_branches[k+1, :] = network.A.T @ v_nodes[:, k+1]
        
        # Calculate dvdt_b_k for current step k+1
        dvdt_b_k_plus_1 = (v_branches[k+1, :] - v_branches[k, :]) / dt
        
        # If this is the first step (k=0), we can now calculate dvdt_b_k for t=0
        # and update i_branches[0]
        if k == 0:
            dvdt_b_k_0 = (v_branches[1, :] - v_branches[0, :]) / dt
            s_idx_0 = 0
            for bi, comp in enumerate(comps):
                ctx = EvalContext(v_branch=float(v_branches[0, bi]),
                                  dvdt_branch=float(dvdt_b_k_0[bi]),
                                  t_next=float(t[0]), dt=float(dt))
                if comp.n_states() > 0:
                    zc = z_split_list_0[s_idx_0]
                    i_branches[0, bi] = comp.current(ctx, zc)
                    s_idx_0 += 1
                else:
                    i_branches[0, bi] = comp.current(ctx, None)

        # Calculate i_branches for k+1
        z_split_list_k_plus_1 = network._split_z(z_hist[k+1]) if network.stateful_indices else []
        s_idx_k_plus_1 = 0
        for bi, comp in enumerate(comps):
            ctx = EvalContext(v_branch=float(v_branches[k+1, bi]),
                              dvdt_branch=float(dvdt_b_k_plus_1[bi]),
                              t_next=float(t[k+1]), dt=float(dt))
            if comp.n_states() > 0:
                zc = z_split_list_k_plus_1[s_idx_k_plus_1]
                i_branches[k+1, bi] = comp.current(ctx, zc)
                s_idx_k_plus_1 += 1
            else:
                i_branches[k+1, bi] = comp.current(ctx, None)

    branch_index = {name: idx for idx, name in enumerate(network.branch_names)}

    result = SimResult(
        t=t,
        v_nodes=v_nodes,
        z_hist=z_hist,
        state_slices=network.state_slice_map,
        state_names=network.state_name_map,
        v_branches=v_branches,
        i_branches=i_branches,
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
    # Numero nodi (equazioni KCL)
    n_nodes = network.A.shape[0]
    v_nodes = np.zeros((n_nodes, N+1))
    
    if v0_nodes is not None:
        v_nodes[:, 0] = v0_nodes.copy()
    else:
        v_nodes[:, 0] = solve_consistent_initial_conditions(network, newton_cfg)

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
    
    # Calcolo correnti (vedi run_sim per dettagli)
    i_branches = np.zeros_like(v_branches)
    comps = [network.components[name] for name in network.branch_names]
    z_split_list = [network._split_z(z_hist[k]) if network.stateful_indices else [] for k in range(N+1)]
    
    for k in range(N+1):
        v_b_k = v_branches[k]
        if k > 0:
            dvdt_b_k = (v_branches[k] - v_branches[k-1]) / dt
        else:
            dvdt_b_k = (v_branches[1] - v_branches[0]) / dt
            
        s_idx = 0
        for bi, comp in enumerate(comps):
            ctx = EvalContext(v_branch=float(v_b_k[bi]),
                              dvdt_branch=float(dvdt_b_k[bi]),
                              t_next=float(t[k]), dt=float(dt))
            if comp.n_states() > 0:
                zc = z_split_list[k][s_idx]
                i_branches[k, bi] = comp.current(ctx, zc)
                s_idx += 1
            else:
                i_branches[k, bi] = comp.current(ctx, None)

    branch_index = {name: idx for idx, name in enumerate(network.branch_names)}

    result = SimResult(
        t=t,
        v_nodes=v_nodes,
        z_hist=z_hist,
        state_slices=network.state_slice_map,
        state_names=network.state_name_map,
        v_branches=v_branches,
        i_branches=i_branches,
        branch_index=branch_index,
    )
    result.attach_component_traces(network.components)
    return result
