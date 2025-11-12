import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simcore.dynamic.components.controlled_voltage_source import ControlledVoltageSource
from simcore.dynamic.components.rc_branch import SeriesRC
from simcore.dynamic.network.graph import Node, NetworkGraph
from simcore.dynamic.network.network import Network
from simcore.dynamic.solver.integrate import run_sim


def main() -> None:
    dt = 5e-6
    t_stop = 0.1  # 5 ms step response

    vin = Node("vin")
    gnd = Node("gnd", is_ground=True)

    graph = NetworkGraph()
    graph.add_branch("Vstep", vin, gnd)
    graph.add_branch("RC_block", vin, gnd)

    components = {
        "Vstep": ControlledVoltageSource(V=5.0, R_internal=1e-3),
        "RC_block": SeriesRC(R=2e3, C=1e-6),
    }

    net = Network(graph, components, dt=dt)
    result = run_sim(net, t_stop=t_stop)

    # Access the automatically created internal node (resistor-capacitor junction)
    mid_node = "RC_block__mid"
    if mid_node not in net.node_names:
        raise RuntimeError("Expected composite internal node not found. Check naming.")

    vin_idx = net.node_names.index("vin")
    mid_idx = net.node_names.index(mid_node)

    v_in = result.v_nodes[vin_idx]
    v_cap = result.v_nodes[mid_idx]

    print(f"Final capacitor voltage: {v_cap[-1]:.3f} V (target 5 V)")
    print(f"Voltage difference across resistor after {t_stop*1e3:.1f} ms: {v_in[-1] - v_cap[-1]:.6f} V")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(result.t * 1e3, v_in, label="vin")
        plt.plot(result.t * 1e3, v_cap, label="v_cap (RC_block__mid)")
        plt.xlabel("Time [ms]")
        plt.ylabel("Voltage [V]")
        plt.title("Series RC Step Response")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
