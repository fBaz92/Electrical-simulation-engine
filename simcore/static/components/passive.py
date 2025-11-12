from __future__ import annotations
from dataclasses import dataclass
from .base import StaticElement, StampData, stamp_series_admittance


@dataclass
class Resistor(StaticElement):
    name: str
    n_plus: str
    n_minus: str
    resistance: float

    def __post_init__(self) -> None:
        StaticElement.__init__(self, self.name, self.n_plus, self.n_minus)

    def stamp(self, data: StampData) -> None:
        if self.resistance <= 0:
            raise ValueError("Resistor resistance must be positive.")
        y = 1.0 / self.resistance
        stamp_series_admittance(data, self.n_plus, self.n_minus, y)

    def branch_current(self, solution) -> complex:
        return self.branch_voltage(solution) / self.resistance


@dataclass
class Capacitor(StaticElement):
    name: str
    n_plus: str
    n_minus: str
    capacitance: float

    def __post_init__(self) -> None:
        StaticElement.__init__(self, self.name, self.n_plus, self.n_minus)

    def stamp(self, data: StampData) -> None:
        if self.capacitance <= 0:
            raise ValueError("Capacitance must be positive.")
        omega = data.omega
        y = 0j if omega == 0 else 1j * omega * self.capacitance
        stamp_series_admittance(data, self.n_plus, self.n_minus, y)

    def branch_current(self, solution) -> complex:
        omega = solution.omega
        y = 0j if omega == 0 else 1j * omega * self.capacitance
        return y * self.branch_voltage(solution)


@dataclass
class Inductor(StaticElement):
    name: str
    n_plus: str
    n_minus: str
    inductance: float
    dc_resistance: float = 1e-6

    def __post_init__(self) -> None:
        StaticElement.__init__(self, self.name, self.n_plus, self.n_minus)

    def stamp(self, data: StampData) -> None:
        if self.inductance <= 0:
            raise ValueError("Inductance must be positive.")
        omega = data.omega
        if omega == 0:
            # Approximate DC short with finite resistance to keep matrix well conditioned.
            y = 1.0 / max(self.dc_resistance, 1e-9)
        else:
            y = 1.0 / (1j * omega * self.inductance)
        stamp_series_admittance(data, self.n_plus, self.n_minus, y)

    def branch_current(self, solution) -> complex:
        omega = solution.omega
        v = self.branch_voltage(solution)
        if omega == 0:
            return v / max(self.dc_resistance, 1e-9)
        y = 1.0 / (1j * omega * self.inductance)
        return y * v
