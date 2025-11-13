from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from ..components.base import BranchComponent, EvalContext, CompositeBranchComponent
from .graph import NetworkGraph, Node

Array = np.ndarray

@dataclass
class Network:
    """
    Represents an electrical network with components and topology.
    
    The Network class assembles the complete system of equations for circuit simulation,
    combining Kirchhoff's Current Law (KCL) with component I-V relationships and state
    equations. It prepares the residual vector and Jacobian matrix for Newton-Raphson
    solving at each time step.
    
    Attributes:
        graph: NetworkGraph containing the circuit topology (nodes and branches).
        components: Dictionary mapping branch names to BranchComponent instances.
        dt: Time step for numerical integration.
        A: Incidence matrix (computed automatically in __post_init__). The incidence matrix is a matrix that relates the currents flowing through the branches to the voltages at the nodes, encoding the KCL equations.
        node_names: List of non-ground node names in matrix order.
        branch_names: List of branch names in matrix order.
        stateful_indices: List of branch indices that have internal states.
        z0: Initial state vector (concatenated from all stateful components).
        _slices: List of slice objects to extract each component's state from z.
    """
    graph: NetworkGraph
    components: Dict[str, BranchComponent]       # key: branch_name
    dt: float

    # filled at init
    A: Array = field(init=False)
    node_names: List[str] = field(init=False)
    branch_names: List[str] = field(init=False)
    stateful_indices: List[int] = field(init=False)
    z0: Array = field(init=False)
    _slices: List[slice] = field(init=False)
    state_slice_map: Dict[str, slice] = field(init=False, repr=False)
    state_name_map: Dict[str, list[str]] = field(init=False, repr=False)

    def __post_init__(self):
        """
        Initialize network structure after dataclass creation.
        
        Computes the incidence matrix, identifies stateful components, and prepares
        the initial state vector. Also creates slice objects to efficiently extract
        individual component states from the concatenated state vector.
        """
        self._flatten_composites()
        A, node_names, branch_names = self.graph.incidence_matrix()
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "node_names", node_names)
        object.__setattr__(self, "branch_names", branch_names)

        # comps_ordered is a list of components in the order of the branch names.
        comps_ordered = [self.components[name] for name in branch_names]
        # stati
        stateful_indices: List[int] = []
        z_init = []
        state_slice_map: Dict[str, slice] = {}
        state_name_map: Dict[str, list[str]] = {}
        off = 0 # offset is the index of the first state of the current component in the global state vector z.

        # _slices is a list of slice objects to extract each component's state from z.
        # For example, if the network has 3 components with 2 states each, _slices will be [slice(0, 2), slice(2, 4), slice(4, 6)].
        _slices: List[slice] = []
        for i, c in enumerate(comps_ordered):
            if c.n_states() > 0:
                stateful_indices.append(i)
                z_init.append(c.state_init())
                sl = slice(off, off + c.n_states())
                _slices.append(sl)
                state_slice_map[branch_names[i]] = sl
                names = c.state_names()
                if names and len(names) != c.n_states():
                    raise ValueError(
                        f"Component '{branch_names[i]}' declares {c.n_states()} states "
                        f"but state_names() returned {len(names)} entries."
                    )
                if not names:
                    names = [f"{branch_names[i]}[{k}]" for k in range(c.n_states())]
                state_name_map[branch_names[i]] = names
                off += c.n_states()
        object.__setattr__(self, "stateful_indices", stateful_indices) # stateful_indices is a list of the indices of the stateful components in the global state vector z.
        object.__setattr__(self, "z0", np.concatenate(z_init) if z_init else np.empty(0))

        object.__setattr__(self, "_slices", _slices) # _slices is a list of slice objects to extract each component's state from z.
        object.__setattr__(self, "state_slice_map", state_slice_map) # state_slice_map is a dictionary mapping each branch name to the slice object to extract its states from z.
        object.__setattr__(self, "state_name_map", state_name_map) # state_name_map is a dictionary mapping each branch name to the list of state names.

    # helper
    def _split_z(self, z: Array) -> List[Array]:
        """
        Split the concatenated state vector into per-component state arrays.
        
        Uses precomputed slice objects to efficiently extract each stateful component's
        state from the global state vector z.
        
        Args:
            z: Concatenated state vector for all stateful components.
        
        Returns:
            List of state arrays, one per stateful component (in order of stateful_indices).
        """
        return [z[sl] for sl in self._slices]

    def state_slice(self, branch_name: str) -> slice | None:
        """Return the slice inside z corresponding to the given branch, if any."""
        return self.state_slice_map.get(branch_name)

    def state_names_for(self, branch_name: str) -> list[str]:
        """Return the state names associated with a branch."""
        return self.state_name_map.get(branch_name, [])

    def _flatten_composites(self) -> None:
        """
        Expand every CompositeBranchComponent into its primitive subnetwork.

        This allows users to register higher-level components that internally
        consist of multiple branches, while keeping the solver unaware of the
        hierarchy.
        """
        while True:
            target: str | None = None
            for branch_name in list(self.graph.branches.keys()):
                comp = self.components.get(branch_name)
                if isinstance(comp, CompositeBranchComponent):
                    target = branch_name
                    break
            if target is None:
                break
            self._expand_composite_branch(target, self.components[target])

    def _expand_composite_branch(self, branch_name: str, composite: CompositeBranchComponent) -> None:
        """
        Expand a composite branch component into its primitive subnetwork.
        
        This method replaces a single composite branch with multiple primitive branches
        and internal nodes. The composite's external nodes (positive and negative terminals)
        are mapped to the original branch endpoints, while internal nodes are created with
        prefixed names to avoid conflicts.
        
        The expansion process:
        1. Removes the composite branch from the graph and components dictionary
        2. Maps the composite's external nodes (POSITIVE_NODE, NEGATIVE_NODE) to the
           original branch endpoints
        3. Creates new internal nodes for each blueprint node, prefixed with the branch name
        4. Adds new branches for each sub-component in the composite's blueprint
        
        Args:
            branch_name: Name of the composite branch to expand (will be removed).
            composite: The CompositeBranchComponent instance to expand.
        """
        n_from_name, n_to_name = self.graph.branches.pop(branch_name)
        n_from = self.graph.nodes[n_from_name]
        n_to = self.graph.nodes[n_to_name]
        self.components.pop(branch_name, None)

        node_map: dict[str, Node] = {
            composite.POSITIVE_NODE: n_from,
            composite.NEGATIVE_NODE: n_to,
        }

        for internal_name in composite._blueprint_nodes():
            node_obj = Node(name=f"{branch_name}__{internal_name}")
            self.graph.add_node(node_obj)
            node_map[internal_name] = node_obj

        for sub_name, (src, dst, sub_comp) in composite._blueprint_branches().items():
            new_branch = f"{branch_name}__{sub_name}"
            self.graph.add_branch(new_branch, node_map[src], node_map[dst])
            self.components[new_branch] = sub_comp

    def assemble(self, v_next: Array, v_prev: Array, z_next: Array, z_prev: Array, t_next: float
                 ) -> tuple[Array, Array]:
        """
        Assemble the residual vector F and Jacobian matrix J for Newton-Raphson solving.
        
        This method evaluates all components at the current time step and builds the
        complete system of equations:
        - KCL equations: F_nodes = A @ i_branch = 0
        - State equations: F_states = R(z_next, z_prev, v_next) = 0
        
        The Jacobian J has a block structure:
        [[J_vv, J_vz],  where J_vv = dF_nodes/dv, J_vz = dF_nodes/dz,
         [J_zv, J_zz]]   J_zv = dF_states/dv, J_zz = dF_states/dz
        
        Args:
            v_next: Node voltage vector at next time step (k+1).
            v_prev: Node voltage vector at previous time step (k).
            z_next: State vector at next time step (k+1).
            z_prev: State vector at previous time step (k).
            t_next: Time at next step (k+1).
        
        Returns:
            Tuple containing:
            - F: Residual vector [F_nodes; F_states] (should be zero at solution)
            - J: Jacobian matrix with block structure [[J_vv, J_vz], [J_zv, J_zz]]
        """
        A, dt = self.A, self.dt
        comps = [self.components[name] for name in self.branch_names]

        v_branch = A.T @ v_next
        dvdt_branch = A.T @ (v_next - v_prev) / dt

        i_branch = np.zeros(len(comps))
        dI_dvbranch = np.zeros(len(comps))

        z_split = self._split_z(z_next) if self.stateful_indices else []
        z_prev_split = self._split_z(z_prev) if self.stateful_indices else []

        dI_dz_cols: list[Array] = []

        s_idx = 0
        for bi, comp in enumerate(comps):
            ctx = EvalContext(v_branch=float(v_branch[bi]),
                              dvdt_branch=float(dvdt_branch[bi]),
                              t_next=float(t_next), dt=float(dt))
            if comp.n_states() > 0:
                zc = z_split[s_idx]
                i_branch[bi] = comp.current(ctx, zc)
                dI_dvbranch[bi] = comp.dI_dv(ctx, zc)
                dI_dz_cols.append(comp.dI_dz(ctx, zc))
                s_idx += 1
            else:
                i_branch[bi] = comp.current(ctx, None)
                dI_dvbranch[bi] = comp.dI_dv(ctx, None)

        # KCL: F_nodes = A i = 0
        F_nodes = A @ i_branch
        J_vv = A @ np.diag(dI_dvbranch) @ A.T

        # Stati
        if self.stateful_indices:
            R_states = []
            J_zv_rows = []
            J_zz_blocks = []
            s_idx = 0
            for bi, comp in enumerate(comps):
                if comp.n_states() > 0:
                    ctx = EvalContext(v_branch=float(v_branch[bi]),
                                      dvdt_branch=float(dvdt_branch[bi]),
                                      t_next=float(t_next), dt=float(dt))
                    zc_next = z_split[s_idx]
                    zc_prev = z_prev_split[s_idx]
                    r = comp.state_residual(ctx, zc_next, zc_prev)
                    R_states.append(r)
                    J_zz_blocks.append(comp.dRdz(ctx, zc_next))
                    dv = comp.dRdv(ctx, zc_next)      # shape (n_states,)
                    a_col = A[:, bi].reshape(1, -1)   # shape (1, n_nodes)

                    # dv (n_states,) → (n_states,1)
                    dv_col = dv.reshape(-1, 1)        # (n_states, 1)

                    # (n_states,1) @ (1,n_nodes) → (n_states, n_nodes)
                    J_zv_rows.append(dv_col @ a_col)
                    s_idx += 1
            F_st = np.concatenate(R_states) if R_states else np.empty(0)
            J_zv = np.vstack(J_zv_rows) if J_zv_rows else np.zeros((0, A.shape[0]))
            # dF_nodes/dz
            
            # --- costruzione J_vz (dF_nodes/dz) generica per più stati per componente ---
            J_vz_cols = []
            s_cursor = 0
            for bi, comp in enumerate(comps):
                if comp.n_states() > 0:
                    zc_next = z_split[s_cursor]
                    # ricostruisci il contesto per questo ramo
                    ctx = EvalContext(
                        v_branch=float(v_branch[bi]),
                        dvdt_branch=float(dvdt_branch[bi]),
                        t_next=float(t_next),
                        dt=float(dt)
                    )
                    dIdz = comp.dI_dz(ctx, zc_next).ravel()   # shape (m_states_comp,)
                    for d in dIdz:
                        vec = np.zeros(len(comps))
                        vec[bi] = d
                        J_vz_cols.append((A @ vec).reshape(-1,1))
                    s_cursor += 1
            # se nessun componente ha stati, J_vz è (n_nodes x 0)
            J_vz = np.hstack(J_vz_cols) if J_vz_cols else np.zeros((A.shape[0], 0))

            
            J_zz = np.block([[blk for blk in J_zz_blocks]]) if J_zz_blocks else np.zeros((0,0))
        else:
            F_st = np.empty(0)
            J_vz = np.zeros((A.shape[0], 0))
            J_zv = np.zeros((0, A.shape[0]))
            J_zz = np.zeros((0,0))

        F = np.concatenate([F_nodes, F_st])
        J = np.block([[J_vv, J_vz],
                      [J_zv, J_zz]])
        return F, J
