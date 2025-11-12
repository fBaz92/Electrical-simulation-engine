import sys
from pathlib import Path
import numpy as np

# ensure project root on path when running as standalone script
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simcore.components.motor_dc import MotorDCComponent
from simcore.components.controlled_voltage_source import ControlledVoltageSource
from simcore.network.graph import Node, NetworkGraph
from simcore.network.network import Network
from simcore.solver.integrate import run_sim


def main() -> None:
    dt = 1e-4
    t_stop = 0.2  # 200 ms startup

    graph = NetworkGraph()
    bus = Node("bus")
    gnd = Node("gnd", is_ground=True)
    graph.add_branch("Vdc", bus, gnd)
    graph.add_branch("Motor", bus, gnd)

    motor = MotorDCComponent(
        R0=0.5,
        alpha=0.0039,
        T0=298.15,
        L=1e-3,
        k_e=0.08,
        k_t=0.08,
        J=2e-4,
        b=2e-5,
        C_th=40.0,
        R_th=3.0,
        T_amb=298.15,
        tau_load=lambda t, w: 0.02,          # simple constant opposing torque
        dtau_domega=lambda t, w: 0.0,
        P_core=lambda w, i: 0.0,
        dPcore_di=lambda w, i: 0.0,
        dPcore_domega=lambda w, i: 0.0,
        i0=0.0,
        omega0=0.0,
        T0_state=298.15,
    )

    components = {
        "Vdc": ControlledVoltageSource(V=48.0, R_internal=1e-3),
        "Motor": motor,
    }

    net = Network(graph, components, dt=dt)
    result = run_sim(net, t_stop=t_stop)

    # --- new API demo: retrieve state & voltage histories directly from the component ---
    t, motor_states = motor.state_history()
    i_arm = motor_states[:, 0]
    omega = motor_states[:, 1]
    temp = motor_states[:, 2]
    _, v_motor = motor.voltage_history()

    # Calculate torques
    tau_motor = motor.k_t * i_arm  # Electromagnetic torque
    tau_load_vals = np.array([motor.tau_load(t_k, w_k) for t_k, w_k in zip(t, omega)])
    tau_viscous = motor.b * omega  # Viscous friction torque
    tau_resist = tau_load_vals + tau_viscous  # Total resisting torque

    # Same information but via SimResult -> useful when you only have the branch name
    _, omega_from_result = result.component_state("Motor", "omega")
    _, v_motor_from_result = result.branch_voltage("Motor")

    print(f"Final armature current: {i_arm[-1]:.3f} A")
    print(f"Final speed: {omega[-1]:.1f} rad/s")
    print(f"Final temperature: {temp[-1]:.1f} K")
    print(f"Final terminal voltage: {v_motor[-1]:.2f} V")
    print(f"Final motor torque: {tau_motor[-1]:.4f} N·m")
    print(f"Final resisting torque: {tau_resist[-1]:.4f} N·m")
    print(f"omega traces match: {float(abs(omega[-1] - omega_from_result[-1]) < 1e-9)}")
    print(f"voltage traces match: {float(abs(v_motor[-1] - v_motor_from_result[-1]) < 1e-9)}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.subplot(4, 1, 1)
        plt.plot(t, i_arm)
        plt.ylabel("Current [A]")
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(t, omega)
        plt.ylabel("Speed [rad/s]")
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(t, tau_motor, label="Motor torque (k_t * i)")
        plt.plot(t, tau_resist, label="Resisting torque (tau_load + b*omega)")
        plt.ylabel("Torque [N·m]")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(t, temp)
        plt.ylabel("Temp [K]")
        plt.xlabel("Time [s]")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
