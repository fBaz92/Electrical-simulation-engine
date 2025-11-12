from __future__ import annotations
from dataclasses import dataclass
from .base import (
    StaticElement,
    StampData,
    stamp_current_source,
    stamp_voltage_source,
)


@dataclass
class CurrentSource(StaticElement):
    name: str
    n_plus: str
    n_minus: str
    current: complex

    def __post_init__(self) -> None:
        StaticElement.__init__(self, self.name, self.n_plus, self.n_minus)

    def stamp(self, data: StampData) -> None:
        stamp_current_source(data, self.n_plus, self.n_minus, self.current)

    def branch_current(self, solution) -> complex:
        return self.current


@dataclass
class VoltageSource(StaticElement):
    name: str
    n_plus: str
    n_minus: str
    voltage: complex

    def __post_init__(self) -> None:
        StaticElement.__init__(self, self.name, self.n_plus, self.n_minus)

    def num_aux_vars(self) -> int:
        return 1

    def stamp(self, data: StampData) -> None:
        aux = data.aux(self.name)
        if len(aux) != 1:
            raise RuntimeError("VoltageSource requires a single auxiliary variable.")
        stamp_voltage_source(data, aux[0], self.n_plus, self.n_minus, self.voltage)

    def branch_current(self, solution) -> complex:
        return solution.aux_scalar(self.name)


@dataclass
class VoltageControlledCurrentSource(StaticElement):
    name: str
    n_plus: str
    n_minus: str
    ctrl_plus: str
    ctrl_minus: str
    transconductance: complex

    def __post_init__(self) -> None:
        StaticElement.__init__(self, self.name, self.n_plus, self.n_minus)

    def stamp(self, data: StampData) -> None:
        g = self.transconductance
        ip = data.node(self.n_plus)
        ineg = data.node(self.n_minus)
        cp = data.node(self.ctrl_plus)
        cn = data.node(self.ctrl_minus)

        if ip is not None and cp is not None:
            data.Y[ip, cp] += g
        if ip is not None and cn is not None:
            data.Y[ip, cn] -= g
        if ineg is not None and cp is not None:
            data.Y[ineg, cp] -= g
        if ineg is not None and cn is not None:
            data.Y[ineg, cn] += g

    def branch_current(self, solution) -> complex:
        v_ctrl = solution.node_voltage(self.ctrl_plus) - solution.node_voltage(self.ctrl_minus)
        return self.transconductance * v_ctrl


@dataclass
class VoltageControlledVoltageSource(StaticElement):
    name: str
    n_plus: str
    n_minus: str
    ctrl_plus: str
    ctrl_minus: str
    gain: complex

    def __post_init__(self) -> None:
        StaticElement.__init__(self, self.name, self.n_plus, self.n_minus)

    def num_aux_vars(self) -> int:
        return 1

    def stamp(self, data: StampData) -> None:
        aux = data.aux(self.name)
        if len(aux) != 1:
            raise RuntimeError("VCVS requires a single auxiliary variable.")
        k = aux[0]
        ip = data.node(self.n_plus)
        ineg = data.node(self.n_minus)
        cp = data.node(self.ctrl_plus)
        cn = data.node(self.ctrl_minus)

        if ip is not None:
            data.Y[ip, k] += 1.0
            data.Y[k, ip] += 1.0
        if ineg is not None:
            data.Y[ineg, k] -= 1.0
            data.Y[k, ineg] -= 1.0

        if cp is not None:
            data.Y[k, cp] -= self.gain
        if cn is not None:
            data.Y[k, cn] += self.gain

    def branch_current(self, solution) -> complex:
        return solution.aux_scalar(self.name)


@dataclass
class CurrentControlledCurrentSource(StaticElement):
    name: str
    n_plus: str
    n_minus: str
    controlling_source: str
    gain: complex

    def __post_init__(self) -> None:
        StaticElement.__init__(self, self.name, self.n_plus, self.n_minus)

    def stamp(self, data: StampData) -> None:
        ctrl_aux = data.aux(self.controlling_source)
        if len(ctrl_aux) != 1:
            raise RuntimeError("CCCS controller must provide a single current unknown.")
        k = ctrl_aux[0]
        ip = data.node(self.n_plus)
        ineg = data.node(self.n_minus)
        if ip is not None:
            data.Y[ip, k] -= self.gain
        if ineg is not None:
            data.Y[ineg, k] += self.gain

    def branch_current(self, solution) -> complex:
        i_ctrl = solution.aux_scalar(self.controlling_source)
        return self.gain * i_ctrl


@dataclass
class CurrentControlledVoltageSource(StaticElement):
    name: str
    n_plus: str
    n_minus: str
    controlling_source: str
    gain: complex

    def __post_init__(self) -> None:
        StaticElement.__init__(self, self.name, self.n_plus, self.n_minus)

    def num_aux_vars(self) -> int:
        return 1

    def stamp(self, data: StampData) -> None:
        ctrl_aux = data.aux(self.controlling_source)
        if len(ctrl_aux) != 1:
            raise RuntimeError("CCVS controller must provide a single current unknown.")
        ctrl_idx = ctrl_aux[0]
        aux = data.aux(self.name)
        if len(aux) != 1:
            raise RuntimeError("CCVS requires a single auxiliary variable.")
        k = aux[0]
        ip = data.node(self.n_plus)
        ineg = data.node(self.n_minus)

        if ip is not None:
            data.Y[ip, k] += 1.0
            data.Y[k, ip] += 1.0
        if ineg is not None:
            data.Y[ineg, k] -= 1.0
            data.Y[k, ineg] -= 1.0

        data.Y[k, ctrl_idx] -= self.gain

    def branch_current(self, solution) -> complex:
        return solution.aux_scalar(self.name)
