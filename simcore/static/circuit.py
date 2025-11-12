from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from .components.base import StaticElement, StampData


@dataclass(frozen=True)
class Node:
    name: str
    is_ground: bool = False


@dataclass
class StaticSolution:
    frequency: float | None
    ground: str
    node_order: List[str]
    node_index: Dict[str, int]
    node_voltages: np.ndarray
    aux_values: Dict[str, np.ndarray]
    elements: Dict[str, StaticElement]

    @property
    def omega(self) -> float:
        return 0.0 if (self.frequency is None or self.frequency == 0) else 2 * np.pi * self.frequency

    def node_voltage(self, node: str) -> complex:
        if node == self.ground:
            return 0.0 + 0.0j
        if node not in self.node_index:
            raise KeyError(f"Unknown node '{node}'.")
        return complex(self.node_voltages[self.node_index[node]])

    def branch_voltage(self, element_name: str) -> complex:
        element = self._element(element_name)
        return element.branch_voltage(self)

    def aux_scalar(self, element_name: str) -> complex:
        values = self.aux_values.get(element_name)
        if values is None or values.size == 0:
            raise KeyError(f"No auxiliary variable associated with '{element_name}'.")
        return complex(values[0])

    def branch_current(self, element_name: str) -> complex:
        element = self._element(element_name)
        return complex(element.branch_current(self))

    def branch_current_polar(self, element_name: str) -> tuple[float, float]:
        current = self.branch_current(element_name)
        mag = float(np.abs(current))
        phase = float(np.rad2deg(np.angle(current)))
        return mag, phase

    def branch_power(self, element_name: str) -> tuple[float, float, float]:
        V = self.branch_voltage(element_name)
        I = self.branch_current(element_name)
        S = V * np.conj(I)
        return float(S.real), float(S.imag), float(abs(S))

    def _element(self, name: str) -> StaticElement:
        if name not in self.elements:
            raise KeyError(f"Component '{name}' not present in the circuit.")
        return self.elements[name]


@dataclass
class StaticCircuit:
    """
    Steady-state circuit simulator (DC or single-frequency AC).
    """

    frequency: float | None = None
    ground_name: str = "gnd"
    nodes: Dict[str, Node] = field(default_factory=dict)
    elements: Dict[str, StaticElement] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.add_node(Node(self.ground_name, is_ground=True))

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def add_element(self, element: StaticElement) -> None:
        if element.name in self.elements:
            raise ValueError(f"Element '{element.name}' already exists.")
        self._register_node(element.n_plus)
        self._register_node(element.n_minus)
        self.elements[element.name] = element

    def _register_node(self, name: str) -> None:
        if name in self.nodes:
            return
        is_ground = (name == self.ground_name)
        self.nodes[name] = Node(name=name, is_ground=is_ground)

    def solve(self) -> StaticSolution:
        node_names = [n for n, node in self.nodes.items() if not node.is_ground]
        node_index = {name: idx for idx, name in enumerate(node_names)}

        aux_map: Dict[str, tuple[int, ...]] = {}
        cursor = len(node_names)
        for name, element in self.elements.items():
            n_aux = element.num_aux_vars()
            if n_aux:
                aux_map[name] = tuple(range(cursor, cursor + n_aux))
                cursor += n_aux
            else:
                aux_map[name] = tuple()

        size = cursor
        if size == 0:
            raise RuntimeError("Circuit has no nodes to solve.")
        Y = np.zeros((size, size), dtype=complex)
        b = np.zeros(size, dtype=complex)
        stamp_data = StampData(
            Y=Y,
            b=b,
            node_index=node_index,
            aux_map=aux_map,
            frequency=self.frequency,
            ground=self.ground_name,
        )

        for element in self.elements.values():
            element.stamp(stamp_data)

        solution_vector = np.linalg.solve(Y, b)
        node_voltages = solution_vector[: len(node_names)]
        aux_values: Dict[str, np.ndarray] = {}
        for name, indices in aux_map.items():
            if not indices:
                continue
            aux_values[name] = solution_vector[np.array(indices, dtype=int)]

        return StaticSolution(
            frequency=self.frequency,
            ground=self.ground_name,
            node_order=node_names,
            node_index=node_index,
            node_voltages=node_voltages,
            aux_values=aux_values,
            elements=self.elements,
        )
