"""
Thermal RC self-heating example.

We drive a single ThermalResistor with a DC voltage source and observe how the
temperature dynamics and Joule heating interact:

Setup
-----
- 20 V ideal source feeding a 10 Ω resistor at 25 °C.
- The resistor has a positive temperature coefficient (α = 0.004 1/K).
- Thermal network: C_th = 10 J/K, R_th = 2 K/W to an ambient of 298 K.

Expected Behaviour
------------------
- At t = 0 the current is 2 A (40 W). Power raises the resistor temperature.
- As T increases, R(T) grows, so the current gradually falls.
- Steady state is set by \( P_\text{diss} = V^2/R(T) = (T - T_\text{amb})/R_\text{th} \),
  which yields ΔT ≈ 64 K with these parameters, so R ≈ 12.6 Ω and I ≈ 1.59 A.
- The script prints the final temperature/current and, if Matplotlib is
  available, plots both waveforms to highlight the coupled dynamics.
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simcore.components.controlled_voltage_source import ControlledVoltageSource
from simcore.components.resistor_thermal import ThermalResistor
from simcore.network.graph import Node, NetworkGraph
from simcore.network.network import Network
from simcore.solver.integrate import run_sim


def main() -> None:
    dt = 1e-3
    t_stop = 200.0

    graph = NetworkGraph()
    node_hot = Node("hot")
    gnd = Node("gnd", is_ground=True)
    graph.add_branch("Vin", node_hot, gnd)
    graph.add_branch("Rth", node_hot, gnd)

    thermal_res = ThermalResistor(
        R0=10.0,
        alpha=0.004,
        T_ref=298.15,
        C_th=10.0,
        R_th=2.0,
        T_amb=298.15,
        T_init=298.15,
    )

    components = {
        "Vin": ControlledVoltageSource(V=20.0, R_internal=1e-3),
        "Rth": thermal_res,
    }

    net = Network(graph, components, dt=dt)
    result = run_sim(net, t_stop=t_stop)

    t, temps = thermal_res.state_history("T")
    _, v_res = thermal_res.voltage_history()

    # Compute instantaneous resistance and current for reporting.
    r_eff = np.maximum(
        thermal_res.R_min,
        thermal_res.R0 * (1.0 + thermal_res.alpha * (temps - thermal_res.T_ref)),
    )
    i_hist = v_res / r_eff

    print(f"Initial current: {i_hist[0]:.3f} A")
    print(f"Final current:   {i_hist[-1]:.3f} A")
    print(f"Final temp:      {temps[-1]-273.15:.2f} °C "
          f"(ΔT≈{temps[-1]-thermal_res.T_amb:.1f} K)")

    try:
        import matplotlib.pyplot as plt

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

        ax_top.plot(t, i_hist, label="Current [A]", color="tab:blue")
        ax_top.set_ylabel("Current [A]", color="tab:blue")
        ax_top.tick_params(axis="y", labelcolor="tab:blue")
        ax_top.grid(True)

        ax_top_t = ax_top.twinx()
        ax_top_t.plot(t, temps - 273.15, label="Temperature [°C]", color="tab:red")
        ax_top_t.set_ylabel("Temperature [°C]", color="tab:red")
        ax_top_t.tick_params(axis="y", labelcolor="tab:red")

        ax_bottom.plot(t, r_eff, color="tab:green")
        ax_bottom.set_ylabel("Resistance [Ω]")
        ax_bottom.set_xlabel("Time [s]")
        ax_bottom.grid(True)

        fig.suptitle("Self-heating Thermal Resistor")
        fig.tight_layout()
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
