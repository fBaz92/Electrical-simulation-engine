from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List

@dataclass(frozen=True)
class Node:
    """
    Represents a node in the electrical network graph.
    
    A node is a connection point in the circuit. The ground node serves as the
    reference potential (typically 0 V) and is excluded from KCL equations.
    
    Attributes:
        name: Unique identifier for the node.
        is_ground: Whether this node is the ground/reference node (default: False).
    """
    name: str
    is_ground: bool = False

@dataclass
class NetworkGraph:
    """
    Represents the topology of an electrical network as a directed graph.
    
    The graph consists of nodes (connection points) and branches (components
    connecting nodes). The incidence matrix encodes how branches connect nodes,
    which is used for Kirchhoff's Current Law (KCL) equations.
    
    Attributes:
        nodes: Dictionary mapping node names to Node objects.
        branches: Dictionary mapping branch names to (from_node, to_node) tuples.
    """
    nodes: Dict[str, Node] = field(default_factory=dict)
    # branch_name -> (from_node, to_node)
    branches: Dict[str, tuple[str, str]] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """
        Add a node to the network graph.
        
        If a node with the same name already exists, it will be overwritten.
        
        Args:
            node: Node object to add to the graph.
        """
        self.nodes[node.name] = node

    def add_branch(self, name: str, n_from: Node, n_to: Node) -> None:
        """
        Add a branch (component) to the network graph.
        
        The branch connects two nodes with a specific direction (from -> to).
        The nodes are automatically registered if they don't already exist in the graph.
        
        Args:
            name: Unique name for the branch/component.
            n_from: Source node (where current leaves).
            n_to: Destination node (where current enters).
        
        Raises:
            ValueError: If a branch with the given name already exists.
        """
        if name in self.branches:
            raise ValueError(f"Branch {name} already exists")
        # registrazione nodi se non presenti
        self.nodes.setdefault(n_from.name, n_from)
        self.nodes.setdefault(n_to.name, n_to)
        self.branches[name] = (n_from.name, n_to.name)

    def incidence_matrix(self) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Compute the incidence matrix for Kirchhoff's Current Law (KCL).
        
        The incidence matrix A encodes the network topology:
        - Rows correspond to non-ground nodes (one KCL equation per node)
        - Columns correspond to branches/components
        - A[i, j] = +1 if branch j leaves node i (current source)
        - A[i, j] = -1 if branch j enters node i (current sink)
        - A[i, j] = 0 if branch j is not connected to node i
        
        Ground nodes are excluded from the matrix since they serve as the reference.
        
        Returns:
            Tuple containing:
            - A: Incidence matrix (shape: [n_non_ground_nodes, n_branches])
            - node_names: List of non-ground node names in matrix row order
            - branch_names: List of branch names in matrix column order
        """
        # ordina: tutti i nodi non ground come righe
        node_names = [n for n in self.nodes if not self.nodes[n].is_ground]
        branch_names = list(self.branches.keys())
        A = np.zeros((len(node_names), len(branch_names)), dtype=float)
        for j, b in enumerate(branch_names): # cycle through branches
            src, dst = self.branches[b]
            if not self.nodes[src].is_ground:
                A[node_names.index(src), j] += 1.0
            if not self.nodes[dst].is_ground:
                A[node_names.index(dst), j] -= 1.0
        return A, node_names, branch_names
