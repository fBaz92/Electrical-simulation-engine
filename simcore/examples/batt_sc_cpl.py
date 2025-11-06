import numpy as np
from simcore.components.resistor import Resistor
from simcore.components.capacitor import Capacitor
from simcore.components.battery_lut import LithiumBatteryLUT
from simcore.components.constant_power_load import ConstantPowerLoad
from simcore.network.graph import Node, NetworkGraph
from simcore.network.network import Network
from simcore.solver.integrate import run_sim

def P_const(t: float) -> float:
    return 50.0 if t >= 0.2 else 0.0

def main():
    # nodi
    bus = Node("bus")
    nsc = Node("n_sc")
    gnd = Node("gnd", is_ground=True)

    # topologia
    G = NetworkGraph()
    G.add_branch("CPL",  bus, gnd)
    G.add_branch("Batt", bus, gnd)
    G.add_branch("Rsc",  bus, nsc)
    G.add_branch("C",    nsc, gnd)

    # componenti
    soc_pts = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 1.0])
    ocv_pts = np.array([2.9, 3.0, 3.08, 3.15, 3.18, 3.20])
    comps = {
        "CPL":  ConstantPowerLoad(P_of_t=P_const, vmin=2.0),
        "Batt": LithiumBatteryLUT(R_internal=0.02, Q_Ah=5.0, soc0=0.8,
                                  soc_pts=soc_pts, ocv_pts=ocv_pts),
        "Rsc":  Resistor(0.00018518518518518518),
        "C":    Capacitor(200.0),
    }

    net = Network(G, comps, dt=1e-3)

    # v0: imposta tutti i nodi alla OCV iniziale
    v0 = np.array([float(comps["Batt"].ocv(0.8)), float(comps["Batt"].ocv(0.8))])

    res = run_sim(net, t_stop=1.0, v0_nodes=v0)

    # plot veloce (facoltativo)
    import matplotlib.pyplot as plt
    t = res.t
    plt.plot(t, res.v_nodes[0], label="bus")
    plt.plot(t, res.v_nodes[1], label="n_sc")
    plt.legend(); plt.grid(True); plt.show()

if __name__ == "__main__":
    main()
