from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from simcore.static.circuit import StaticSolution

Array = np.ndarray


@dataclass
class StampData:
    """
    Shared view of the MNA system during stamping.

    Attributes:
        Y:   Admittance matrix (complex, shape n+n_aux).
        b:   Right-hand side vector (currents and sources).
        node_index: Mapping node_name -> row/column index (ground excluded).
        aux_map: Mapping element_name -> tuple of auxiliary indices.
        frequency: Operating frequency (Hz) or None for DC.
        ground: Name of the reference node.
    """
    Y: Array
    b: Array
    node_index: Dict[str, int]
    aux_map: Dict[str, Tuple[int, ...]]
    frequency: float | None
    ground: str

    @property
    def omega(self) -> float:
        return 0.0 if (self.frequency is None or self.frequency == 0) else 2 * np.pi * self.frequency

    def node(self, name: str) -> int | None:
        return self.node_index.get(name)

    def aux(self, element_name: str) -> Tuple[int, ...]:
        return self.aux_map.get(element_name, tuple())


def _add_to_matrix(Y: Array, row: int | None, col: int | None, value: complex) -> None:
    if row is None or col is None:
        return
    Y[row, col] += value


def stamp_series_admittance(data: StampData, n_plus: str, n_minus: str, admittance: complex) -> None:
    if abs(admittance) == 0:
        return
    ip = data.node(n_plus)
    ineg = data.node(n_minus)
    if ip is not None:
        data.Y[ip, ip] += admittance
    if ineg is not None:
        data.Y[ineg, ineg] += admittance
    if ip is not None and ineg is not None:
        data.Y[ip, ineg] -= admittance
        data.Y[ineg, ip] -= admittance


def stamp_current_source(data: StampData, n_plus: str, n_minus: str, current: complex) -> None:
    """
    Positive current flows from n_plus to n_minus.
    """
    ip = data.node(n_plus)
    ineg = data.node(n_minus)
    if ip is not None:
        data.b[ip] -= current
    if ineg is not None:
        data.b[ineg] += current


def stamp_voltage_source(data: StampData, aux_idx: int, n_plus: str, n_minus: str, voltage: complex) -> None:
    ip = data.node(n_plus)
    ineg = data.node(n_minus)
    if ip is not None:
        _add_to_matrix(data.Y, ip, aux_idx, 1.0)
        _add_to_matrix(data.Y, aux_idx, ip, 1.0)
    if ineg is not None:
        _add_to_matrix(data.Y, ineg, aux_idx, -1.0)
        _add_to_matrix(data.Y, aux_idx, ineg, -1.0)
    data.b[aux_idx] += voltage


class StaticElement(ABC):
    """
    Base class for steady-state components stamped into the MNA system.
    """

    def __init__(self, name: str, n_plus: str, n_minus: str) -> None:
        self.name = name
        self.n_plus = n_plus
        self.n_minus = n_minus

    def num_aux_vars(self) -> int:
        return 0

    @abstractmethod
    def stamp(self, data: StampData) -> None:
        """
        Add this element's contribution to the global Y,b system.
        """

    @abstractmethod
    def branch_current(self, solution: StaticSolution) -> complex:
        """
        Return the phasor current flowing from n_plus to n_minus.
        """

    def branch_voltage(self, solution: StaticSolution) -> complex:
        vp = solution.node_voltage(self.n_plus)
        vn = solution.node_voltage(self.n_minus)
        return vp - vn
