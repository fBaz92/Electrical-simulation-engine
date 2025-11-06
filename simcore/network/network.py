from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from ..components.base import BranchComponent, EvalContext
from .graph import NetworkGraph

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
        A: Incidence matrix (computed automatically in __post_init__).
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

    def _apply_capacitor_initial_conditions(self, v_nodes: np.ndarray):
        """
        If a capacitor declares V_init, apply that as initial voltage difference
        across the branch at t=0 by adjusting the node potentials.
        """
        for branch_name, comp in self.components.items():
            if hasattr(comp, "V_init") and comp.V_init is not None:
                # branch index
                bi = self.branch_names.index(branch_name)
                # find its node endpoints
                n_plus, n_minus = self.graph.branches[bi]  # (Node, Node)

                Vc0 = comp.V_init

                if n_minus.is_ground:
                    v_nodes[n_plus.index] = Vc0
                elif n_plus.is_ground:
                    v_nodes[n_minus.index] = -Vc0
                else:
                    # nessun nodo è ground → imponiamo su n_plus e lasciamo all'utente chiudere il riferimento
                    v_nodes[n_plus.index] = Vc0


    def __post_init__(self):
        """
        Initialize network structure after dataclass creation.
        
        Computes the incidence matrix, identifies stateful components, and prepares
        the initial state vector. Also creates slice objects to efficiently extract
        individual component states from the concatenated state vector.
        """
        A, node_names, branch_names = self.graph.incidence_matrix()
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "node_names", node_names)
        object.__setattr__(self, "branch_names", branch_names)

        # mappa componenti nell'ordine delle colonne
        comps_ordered = [self.components[name] for name in branch_names]
        # stati
        stateful_indices: List[int] = []
        z_init = []
        for i, c in enumerate(comps_ordered):
            if c.n_states() > 0:
                stateful_indices.append(i)
                z_init.append(c.state_init())
        object.__setattr__(self, "stateful_indices", stateful_indices)
        object.__setattr__(self, "z0", np.concatenate(z_init) if z_init else np.empty(0))

        # slice per ciascun componente con stato
        _slices: List[slice] = []
        off = 0
        for i in stateful_indices:
            n = comps_ordered[i].n_states()
            _slices.append(slice(off, off+n))
            off += n
        object.__setattr__(self, "_slices", _slices)

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
                    dv = comp.dRdv(ctx, zc_next)     # shape (1,)
                    a_col = A[:, bi].reshape(-1,1)   # (n_nodes,1)
                    J_zv_rows.append(dv.reshape(1,1) @ a_col.T)
                    s_idx += 1
            F_st = np.concatenate(R_states) if R_states else np.empty(0)
            J_zv = np.vstack(J_zv_rows) if J_zv_rows else np.zeros((0, A.shape[0]))
            # dF_nodes/dz
            J_vz_cols = []
            for s_idx, bi in enumerate(self.stateful_indices):
                vec = np.zeros(len(comps))
                vec[bi] = dI_dz_cols[s_idx][0]
                J_vz_cols.append((A @ vec).reshape(-1,1))
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
