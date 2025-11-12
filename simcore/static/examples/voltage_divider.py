"""
Phasor analysis example (AC voltage divider).

Circuit:
    Vs (230∠0° V) -> R1 (10 Ω) -> node v_out -> parallel of C (47 µF) and Rload (30 Ω) -> ground.

This script solves the circuit at 50 Hz using the static solver, then reports:
- Output voltage magnitude/phase.
- Branch currents for R1 and Rload.
- Active/reactive power dissipated on Rload.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simcore.static.circuit import StaticCircuit
from simcore.static.components.passive import Resistor, Capacitor, Inductor
from simcore.static.components.sources import VoltageSource
from simcore.static.utils import phasor, polar


def main() -> None:
    freq = 50
    circuit = StaticCircuit(frequency=freq)

    circuit.add_element(VoltageSource("Vs", "vin", "gnd", phasor(230.0, 0.0)))
    circuit.add_element(Resistor("R1", "vin", "vout", resistance=10.0))
    circuit.add_element(Capacitor("C1", "vout", "gnd", capacitance=47e-6))
    circuit.add_element(Resistor("Rload", "vout", "gnd", resistance=30.0))
    circuit.add_element(Inductor("L1", "vout", "v2", inductance=10e-3))
    circuit.add_element(Resistor("R2", "v2", "gnd", resistance=2.0))

    solution = circuit.solve()

    v_out = solution.node_voltage("vout")
    mag_v, phase_v = polar(v_out)
    print(f"Vout = {mag_v:.2f} V ∠ {phase_v:.2f}°")

    mag_i_r1, phase_i_r1 = solution.branch_current_polar("R1")
    mag_i_rl, phase_i_rl = solution.branch_current_polar("Rload")
    mag_i_l1, phase_i_l1 = solution.branch_current_polar("L1")
    mag_i_r2, phase_i_r2 = solution.branch_current_polar("R2")
    mag_i_c1, phase_i_c1 = solution.branch_current_polar("C1")
    print(f"I_R1 = {mag_i_r1:.3f} A ∠ {phase_i_r1:.2f}°")
    print(f"I_Rload = {mag_i_rl:.3f} A ∠ {phase_i_rl:.2f}°")
    print(f"I_L1 = {mag_i_l1:.3f} A ∠ {phase_i_l1:.2f}°")
    print(f"I_R2 = {mag_i_r2:.3f} A ∠ {phase_i_r2:.2f}°")
    print(f"I_C1 = {mag_i_c1:.3f} A ∠ {phase_i_c1:.2f}°")

    Pr1, Qr1, Sr1 = solution.branch_power("R1")
    print(f"R1 power: P={Pr1:.2f} W, Q={Qr1:.2f} var, |S|={Sr1:.2f} VA")

    Prl, Qrl, Srl = solution.branch_power("Rload")
    print(f"Rload power: P={Prl:.2f} W, Q={Qrl:.2f} var, |S|={Srl:.2f} VA")

    Pl1, Ql1, Sl1 = solution.branch_power("L1")
    print(f"L1 power: P={Pl1:.2f} W, Q={Ql1:.2f} var, |S|={Sl1:.2f} VA")

    Pr2, Qr2, Sr2 = solution.branch_power("R2")
    print(f"R2 power: P={Pr2:.2f} W, Q={Qr2:.2f} var, |S|={Sr2:.2f} VA")

    Pc1, Qc1, Sc1 = solution.branch_power("C1")
    print(f"C1 power: P={Pc1:.2f} W, Q={Qc1:.2f} var, |S|={Sc1:.2f} VA")

    active_power = {
        "Rload": Prl,
        "L1": Pl1,
        "R2": Pr2,
        "R1": Pr1,
    }
    reactive_power = {
        "C1": Qc1,
        "L1": Ql1,
        "R2": Qr2,
    }

    print(f"The sum of the absorbed active power: {sum(active_power.values()):.2f} W")
    print(f"The sum of the absorbed reactive power: {sum(reactive_power.values()):.2f} var")


if __name__ == "__main__":
    main()
