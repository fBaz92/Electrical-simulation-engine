"""
Speed-controlled asynchronous machine example.

The script instantiates the AsynchronousMachineFOC branch as the only dynamic
component in the network (the branch is connected between a bus node and
ground). The branch voltage is interpreted as the q-axis command, so we inject
an outer PI controller that modulates the voltage to track a speed reference.

The load torque is speed dependent, mimicking a fan (`T_load = k * omega^2`).

Workflow:
    - Build a NetworkGraph with a single branch (the induction machine).
    - Attach AsynchronousMachineFOC with custom load torque and d/domega hooks.
    - Use `run_sim_with_control` to update the voltage command before every
      implicit step (simple PI regulator acting on mechanical speed error).
    - After the run, call `machine.dq_trace()` to recover dq currents, abc
      currents, speed, rotor fluxes, and temperature, then plot the most
      relevant signals.

Expected behaviour:
    - The PI loop ramps the q-axis voltage to accelerate the rotor up to 150
      rad/s (≈1430 rpm for a 2-pole machine) while compensating the quadratic
      torque load.
    - Induction slip settles at a small steady-state value and the dq currents
      converge.
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simcore.dynamic.components.asynchronous_machine import AsynchronousMachineFOC
from simcore.dynamic.network.graph import Node, NetworkGraph
from simcore.dynamic.network.network import Network
from simcore.dynamic.solver.integrate import run_sim_with_control


def load_torque(_t: float, omega: float) -> float:
    k = 1.5e-4
    return k * omega ** 2


def dload_domega(_t: float, omega: float) -> float:
    k = 1.5e-4
    return 2 * k * omega


class SpeedPI:
    def __init__(self, kp: float, ki: float, vmax: float) -> None:
        self.kp = kp
        self.ki = ki
        self.vmax = vmax
        self.integrator = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def update(self, error: float, dt: float) -> float:
        self.integrator += error * dt
        v = self.kp * error + self.ki * self.integrator
        return float(np.clip(v, -self.vmax, self.vmax))


def main() -> None:
    dt = 1e-4
    t_stop = 0.5

    graph = NetworkGraph()
    bus = Node("bus")
    gnd = Node("gnd", is_ground=True)
    graph.add_branch("IM", bus, gnd)

    machine = AsynchronousMachineFOC(
        pole_pairs=2,
        R_s=0.5,
        R_r=0.3,
        L_ls=2.5e-3,
        L_lr=2.5e-3,
        L_m=60e-3,
        J=0.02,
        B=1e-3,
        C_th=500.0,
        R_th=2.5,
        T_amb=293.15,
        supply_frequency=50.0,
        load_torque=load_torque,
        dload_domega=dload_domega,
    )

    components = {"IM": machine}
    net = Network(graph, components, dt=dt)

    speed_ref = 150.0
    controller = SpeedPI(kp=2.0, ki=200.0, vmax=400.0)

    def control_callback(k, t_k, v_prev, z_prev, comps):
        omega = float(z_prev[4])
        error = speed_ref - omega
        v_q_cmd = controller.update(error, dt)
        comps["IM"].voltage_cmd = lambda _t, vq=v_q_cmd: (0.0, vq)

    result = run_sim_with_control(
        net,
        t_stop=t_stop,
        v0_nodes=np.zeros(net.A.shape[0]),
        control_callback=control_callback,
    )

    trace = machine.dq_trace()

    print(f"Final speed: {trace['omega'][-1]:.1f} rad/s")
    print(f"Final temperature: {trace['temperature'][-1]-273.15:.2f} °C")
    print(f"Final q-axis current: {trace['i_q'][-1]:.2f} A")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        ax[0].plot(trace["t"], trace["omega"], label="Mechanical speed")
        ax[0].axhline(speed_ref, color="k", linestyle="--", label="Reference")
        ax[0].set_ylabel("omega [rad/s]")
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(trace["t"], trace["i_q"], label="i_q")
        ax[1].plot(trace["t"], trace["i_d"], label="i_d")
        ax[1].set_ylabel("Currents [A]")
        ax[1].grid(True)
        ax[1].legend()

        ax[2].plot(trace["t"], trace["temperature"] - 273.15, label="Temp")
        ax[2].set_ylabel("Temperature [°C]")
        ax[2].set_xlabel("Time [s]")
        ax[2].grid(True)
        ax[2].legend()

        fig.tight_layout()
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
