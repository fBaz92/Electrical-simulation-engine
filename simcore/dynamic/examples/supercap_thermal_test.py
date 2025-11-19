import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simcore.dynamic.components.controlled_voltage_source import ControlledVoltageSource
from simcore.dynamic.components.supercapacitor import Supercapacitor
from simcore.dynamic.components.resistor import Resistor
from simcore.dynamic.network.graph import Node, NetworkGraph
from simcore.dynamic.network.network import Network
from simcore.dynamic.solver.integrate import run_sim

def main():
    # Simulation parameters
    dt = 0.01  # 10 ms
    t_stop = 10.0 # 10 seconds
    
    # Create nodes
    vin = Node("vin")
    gnd = Node("gnd", is_ground=True)
    
    # Create graph
    graph = NetworkGraph()
    graph.add_branch("Vsrc", vin, gnd)
    graph.add_branch("SC", vin, gnd)
    
    # Create components
    # High current to generate heat: 50V across the supercap
    # Supercap parameters: high ESR for visible heating
    sc = Supercapacitor(
        esr10ms=7e-3,
        capacitance=62,
        esr1s=10e-3,
        cells_per_module=1,
        modules_per_string=1,
        strings=1,
        tau_dl=0.03,
    )
    # Override thermal parameters for faster response in short sim
    sc.C_th = 34000  # 
    sc.R_th = 0.17  # forced air cooling
    
    components = {
        "Vsrc": ControlledVoltageSource(V=162, R_internal=1e-3),
        "SC": sc
    }
    
    # Create network
    net = Network(graph, components, dt=dt)
    
    print(f"Nodes: {net.node_names}")
    print(f"Branches: {net.branch_names}")
    print(f"Initial State z0: {net.z0}")
    print(f"A shape: {net.A.shape}")
    
    # Run simulation
    print("Running simulation...")
    result = run_sim(net, t_stop=t_stop)
    
    print("DEBUG: First 5 time steps:")
    print(f"Time: {result.t[:5]}")
    print(f"V_nodes:\n{result.v_nodes[:, :5]}")
    print(f"I_branches:\n{result.i_branches[:5, :]}")
    print("Simulation complete.")
    
    # Extract results
    t = result.t
    
    # Extract temperature (it's an auxiliary component state)
    # We need to find the slice manually or use the component reference if we had it easily accessible in result
    # But result stores history for all states.
    # The Supercapacitor is an auxiliary component, so its states are in z_hist.
    # We need to know where in z_hist it is.
    
    # Let's inspect the network to find the slice
    # In the new implementation, auxiliary components are appended to z0
    # We can iterate net.auxiliary_components to find our SC and its index
    
    sc_idx = -1
    offset = len(net.stateful_indices) # Start after regular states? No, wait.
    # Let's look at how z0 was built in network.py
    # It iterates regular components first, then auxiliary.
    
    # We can use the component instance to get the trace if we attached it?
    # result.attach_component_traces calls _attach_state_trace on components.
    # But result.attach_component_traces iterates over net.components.
    # Auxiliary components are NOT in net.components anymore!
    
    # We need to manually attach traces for auxiliary components or update result.attach_component_traces
    # For this test, let's just find the index.
    
    # Re-reading network.py logic:
    # z_init = []
    # ... regular components ...
    # ... auxiliary components ...
    # object.__setattr__(self, "z0", np.concatenate(z_init)...)
    
    # So the states are concatenated.
    # We have 0 regular stateful components (Vsrc has 0, Resistors have 0).
    # Wait, SC expands into Resistors and Capacitors.
    # The Capacitors ARE stateful regular components.
    # So z will contain: [Capacitor1_voltage, Capacitor2_voltage, ..., SC_Temperature]
    
    # Let's find the slice for our SC instance
    # We can't easily get it from 'result' by name because SC is not in branch_names.
    
    # However, we have the 'sc' object instance.
    # But 'sc' doesn't know its slice index.
    
    # Let's just plot all states to identify them
    print(f"Total states: {result.z_hist.shape[1]}")
    
    plt.figure(figsize=(10, 6))
    
    plt.figure(figsize=(10, 12)) # Increased figure height for more subplots
    
    # Plot Voltages
    plt.subplot(4, 1, 1)
    plt.plot(t, result.v_nodes[0], label="V_node")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True)
    
    # Plot Current (assuming the first branch current is relevant)
    plt.subplot(4, 1, 2)
    plt.plot(t, result.i_branches[:, 0], label="I_branch")
    plt.ylabel("Current [A]")
    plt.legend()
    plt.grid(True)
    
    # Plot Instantaneous Losses (e.g., from the first branch, if available)
    # This is a placeholder; you might need to calculate or retrieve specific losses
    # For a supercapacitor, resistive losses are often I^2 * ESR
    # Assuming result.power_dissipation exists for total losses, or
    # for a specific component, it would need to be accessed via its index.
    # For this example, let's assume `result.power_dissipation` exists or
    # calculate a simple V*I for a placeholder if no specific loss data is available.
    # If `result.power_dissipation` is not available, a common approximation for resistive loss
    # in the supercapacitor would be `result.i_branches[sc_branch_idx]**2 * sc_esr`.
    # For now, let's plot a generic V*I if no specific loss data is provided by `result`.
    # As a simple example, let's assume the power dissipation of the first branch,
    # which might be available as result.p_branches or needs calculation.
    # If `result.p_branches` exists and contains instantaneous power for each branch:
    # plt.plot(t, result.p_branches[0], label="P_loss")
    # Otherwise, as a generic example:
    plt.subplot(4, 1, 3)
    # This is a generic placeholder. You'll need to replace `result.power_dissipation`
    # with the actual attribute from your simulation results representing instantaneous losses,
    # or calculate it based on component currents and voltages (e.g., I_sc^2 * ESR).
    # If `result.i_branches` and `result.v_branches` (voltage across branch) are available:
    # plt.plot(t, result.i_branches[0] * result.v_branches[0], label="P_loss_branch0")
    # For now, assuming a generic total power dissipation or a specific branch's power is available
    # or needs to be calculated. Let's make a very simple assumption for now:
    try:
        # Assuming a specific attribute for total power dissipation might exist
        plt.plot(t, result.power_dissipation, label="P_loss_total")
    except AttributeError:
        # Fallback if no direct power_dissipation attribute, e.g., for branch 0
        # This calculation might not represent actual losses accurately without component-specific data
        plt.plot(t, result.i_branches[:, 0] * result.v_nodes[0], label="P_loss (V_node0 * I_branch0)")
    plt.ylabel("Power [W]")
    plt.legend()
    plt.grid(True)
    
    # Plot States
    plt.subplot(4, 1, 4)
    for i in range(result.z_hist.shape[1]):
        label = f"State {i}"
        # Try to guess which one is temperature (starts at 25.0)
        if abs(result.z_hist[0, i] - 25.0) < 1e-3:
            label += " (Temperature?)"
        plt.plot(t, result.z_hist[:, i], label=label)
        
    plt.ylabel("State Value")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("supercap_thermal_test.png")
    print("Plot saved to supercap_thermal_test.png")

if __name__ == "__main__":
    main()
