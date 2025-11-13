# Electrical Simulation Engine

This repository hosts two circuit simulators that share a common spirit but work on different time scales:

1. `simcore.dynamic` – a time-domain engine for nonlinear transient analysis driven by Newton–Raphson iterations.
2. `simcore.static` – a steady-state solver for DC and single-frequency AC circuits built on Modified Nodal Analysis (MNA).

Both subsystems ship with reusable components, ready-to-run examples, and helper utilities so you can start analysing circuits quickly and then dive into the source when you need to extend it.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All examples assume you run them from the project root with the virtual environment activated.

---

## Quick Start

### Dynamic (time-domain) examples

```bash
python3 simcore/dynamic/examples/batt_sc_cpl.py            # battery + supercap + CPL load
python3 simcore/dynamic/examples/motor_dc_states.py        # DC motor, states + voltage traces
python3 simcore/dynamic/examples/async_motor_speed_control.py # induction motor with slip & speed loop
python3 simcore/dynamic/examples/series_rc_step.py         # composite RC macro
python3 simcore/dynamic/examples/thermal_resistor_heating.py # self-heating resistor demo
```

Each script prints basic telemetry and (optionally) plots results with Matplotlib.

### Static (phasor) examples

```bash
python3 simcore/static/examples/voltage_divider.py
```

This example solves a 50 Hz voltage divider and reports node voltages, branch currents (magnitude/phase), and active/reactive power.

---

## Using the Dynamic Engine

1. Describe the topology via `simcore.dynamic.network.NetworkGraph`, adding nodes (declare one ground) and branches.
2. Build a `dict` of branch components. Available models live under `simcore.dynamic.components` (resistors, capacitors, inductors, diode, thermal resistor, batteries, composite elements, etc.). Custom models inherit from `BranchComponent`.
3. Instantiate `simcore.dynamic.network.Network(graph, components, dt=...)`.
4. Run a simulation with `simcore.dynamic.solver.integrate.run_sim(network, t_stop, v0_nodes)`.
5. Inspect `SimResult` or call `component.state_history()` / `component.voltage_history()` on the original objects.

Use the notebook at `simcore/dynamic/examples/notebook.ipynb` for a guided tour.

### Dynamic architecture overview

| Layer | Key files | Responsibilities |
|-------|-----------|------------------|
| Topology | `simcore/dynamic/network/graph.py` (`Node`, `NetworkGraph`) | Stores nodes/branches and builds the incidence matrix used for KCL. |
| Component API | `simcore/dynamic/components/base.py` (`BranchComponent`, `EvalContext`) | Defines the contract every branch must satisfy (current, Jacobians, optional states) plus tooling for attaching time traces. |
| Network assembler | `simcore/dynamic/network/network.py` (`Network`) | Expands composite bipoles, orders components, concatenates states, and builds residual/Jacobian blocks for Newton iterations. |
| Solver | `simcore/dynamic/solver/integrate.py`, `simcore/dynamic/solver/newton.py` | Time stepping loop (implicit Euler) and the damped Newton core with optional line search. |
| Examples & control | `simcore/dynamic/examples/*`, `simcore/dynamic/control/*` | Reference setups and external control hooks. |

#### How the dynamic solver works

1. **Topology preprocessing** – `Network.__post_init__` flattens composites, creates the incidence matrix `A`, orders branch names, and concatenates component states into a global `z` vector. Each stateful branch registers a slice and friendly state names.
2. **State handling** – `Network._split_z` and the stored slices let the assembler feed individual state arrays back into each component. After every simulation run, `SimResult.attach_component_traces` pushes per-component time histories (states and branch voltages) into the original objects so users can request `component.state_history("SOC")`, etc.
3. **Per-step assembly** – `Network.assemble` evaluates every component with an `EvalContext` containing branch voltage, implicit dv/dt, time, and dt. It builds:
   - `F_nodes = A @ i_branch` (KCL residual).
   - `F_states` by concatenating each component’s state residuals.
   - Jacobian blocks (`J_vv`, `J_vz`, `J_zv`, `J_zz`) using component derivatives (`dI/dV`, `dI/dz`, `dR/dV`, `dR/dz`).
4. **Newton iterations** – `run_sim` advances time with implicit Euler. For each step it concatenates `[v_next, z_next]`, calls `newton_solve`, and stores the converged solution. The Newton core (`simcore/dynamic/solver/newton.py`) supports damping, line search, and tolerances configured via `NewtonConfig`.
5. **Result handling** – `SimResult` keeps node voltages (`v_nodes`), state histories (`z_hist`), branch voltages (`v_branches`), and metadata. Helper methods expose per-branch currents in polar form, power calculations, and state access by name. Components also gain direct access to their own voltage trace.

Key classes to review:
- `BranchComponent` (`simcore/dynamic/components/base.py`) – implement `current`, `dI_dv`, optional state logic.
- `Network` (`simcore/dynamic/network/network.py`) – orchestrates ordering, slicing, and residual assembly.
- `SimResult` (`simcore/dynamic/solver/integrate.py`) – attaches traces and exposes post-processing utilities.

#### Featured dynamic components

- `simcore.dynamic.components.synchronous_machine.SynchronousMachineFOC`  
  Five-state PMSM model (i_d, i_q, mechanical speed, electrical angle, temperature) with copper/iron losses, load-torque hooks, and helper `dq_trace()` to recover a-b-c currents.
- `simcore.dynamic.components.asynchronous_machine.AsynchronousMachineFOC`  
  Seven-state induction machine (stator currents, rotor fluxes, speed, thermal node) including slip dynamics and dq/abc trace utilities for diagnostics.
- Legacy components such as `Resistor`, `Capacitor`, `Inductor`, `ThermalResistor`, `LithiumBatteryLUT`, and composite macros (`SeriesRC`) remain available under `simcore.dynamic.components`.

---

## Using the Static Engine

1. Create a circuit: `circuit = simcore.static.StaticCircuit(frequency=50.0)` (set `None` for DC).
2. Add elements from `simcore.static.components`, e.g.:
   ```python
   from simcore.static.components.passive import Resistor
   from simcore.static.components.sources import VoltageSource
   circuit.add_element(VoltageSource("Vs", "vin", "gnd", phasor(230, 0)))
   circuit.add_element(Resistor("R1", "vin", "vout", resistance=10))
   ```
3. Solve once: `solution = circuit.solve()`.
4. Query results: `solution.node_voltage("vout")`, `solution.branch_current_polar("R1")`, `solution.branch_power("Rload")`, etc.

### Static architecture overview

| Layer | Key files | Responsibilities |
|-------|-----------|------------------|
| Circuit core | `simcore/static/circuit.py` (`StaticCircuit`, `StaticSolution`) | Manages nodes, allocates auxiliary variables (for voltage sources & controlled elements), builds MNA matrices, and exposes solved voltages/currents/powers. |
| Component API | `simcore/static/components/base.py` (`StaticElement`, `StampData`) | Provides stamping helpers for series admittances, current sources, voltage sources, and gives each element access to node indices and frequency. |
| Passive models | `simcore/static/components/passive.py` | Resistor, capacitor, inductor models translate into admittances (`1/R`, `jωC`, `1/(jωL)` with sensible DC fallbacks). |
| Sources | `simcore/static/components/sources.py` | Independent sources plus voltage/current-controlled versions. Controlled sources stamp entries referencing other nodes or auxiliary unknowns. |
| Utilities | `simcore/static/utils.py` | Helpers for building phasors (`phasor(magnitude, phase_deg)`) and reporting polar values (`polar(complex_value)`). |
| Examples | `simcore/static/examples/*` | Ready-made circuits demonstrating the API. |

#### How the static solver works

1. **Node bookkeeping** – `StaticCircuit` keeps a ground node plus every node encountered while adding elements. Non-ground nodes receive row/column indices for the admittance matrix `Y`.
2. **Auxiliary variables** – Elements such as voltage sources or controlled sources request `num_aux_vars()`. During `solve()` the circuit assigns contiguous indices after the node rows. Each element can retrieve its auxiliary slice via `StampData.aux(...)`.
3. **Stamping** – Each element implements `stamp(self, data: StampData)`:
   - Resistors/capacitors/inductors call `stamp_series_admittance` with their complex admittance (capacitors/inductors derive `jωC` / `1/(jωL)` using `StampData.omega`).
   - Current sources inject into the RHS vector `b`.
   - Voltage sources and controlled sources modify both `Y` and `b`, referencing node indices and auxiliary rows.
4. **Solving** – Once all elements stamp their contribution, `StaticCircuit.solve()` runs a single `np.linalg.solve(Y, b)`. The solution vector is split back into node voltages and auxiliary values per element.
5. **Post-processing** – `StaticSolution` exposes helper methods:
   - `node_voltage(name)` – returns the complex phasor.
   - `branch_current(element_name)` and `branch_current_polar(...)`.
   - `branch_power(element_name)` – computes `S = V * I*` and returns `(P, Q, |S|)`.
   - `branch_voltage(element_name)` for convenience.

Main objects to inspect:
- `StaticElement` & `StampData` (`simcore/static/components/base.py`) – teach you how to add new components or controlled sources.
- `StaticCircuit` (`simcore/static/circuit.py`) – shows the MNA assembly, auxiliary management, and solution storage.
- `StaticSolution` (`simcore/static/circuit.py`) – demonstrates how to query results and derive polar quantities.

---

## Extending the project

- **Add new dynamic components** by inheriting from `BranchComponent`, declaring the number of states, and overriding the residual/Jacobian helpers. If the component is a macro made of simpler parts, derive from `CompositeBranchComponent`.
- **Add new static components** by inheriting from `StaticElement`, overriding `num_aux_vars()` if necessary, and implementing the `stamp`/`branch_current` methods using the helpers in `StampData`.
- **Combine both worlds** by building a static circuit to compute steady-state operating points or load flows and then feeding those values into a dynamic simulation as initial conditions.

The codebase intentionally mirrors textbook formulations, so reading the modules referenced above should give engineers enough insight to trust and extend the solvers. If you plan to contribute new features, please keep the documentation structure in sync so future readers can continue using this README as a learning guide.
