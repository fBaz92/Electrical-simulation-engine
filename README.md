# Electrical Simulation Engine

Motore per la simulazione nel dominio del tempo di reti elettriche con componenti lineari e non lineari. Il progetto combina un modello circuitale basato su matrici di incidenza con un risolutore implicito di tipo Newton-Raphson.

---

## Come si usa

- **Installazione rapida**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Eseguire un esempio**  
  ```bash
  python3 simcore/examples/batt_sc_cpl.py            # battery + supercap + CPL
  python3 simcore/examples/motor_dc_states.py        # DC motor with direct state traces
  python3 simcore/examples/series_rc_step.py         # Series RC composite branch
  python3 simcore/examples/thermal_resistor_heating.py # Self-heating resistor with thermal RC
  ```
  Produce la risposta nel tempo di una batteria con resistenza serie, un condensatore e un carico a potenza costante.
- **Notebook interattivo**  
  Apri `simcore/examples/notebook.ipynb` con Jupyter per esplorare la modellazione in maniera guidata.
- **Simulazione personalizzata**
  1. Definisci nodi e rami con `NetworkGraph`.
  2. Assegna componenti (`Resistor`, `Capacitor`, `ConstantPowerLoad`, `LithiumBatteryLUT`, o componenti custom).
  3. Inizializza la rete `Network(graph, components, dt=...)`.
  4. Esegui `run_sim(network, t_stop=...)`.

---

## Panoramica dell'architettura

- `simcore/network/graph.py`  
  Gestisce nodi (`Node`) e rami direzionati, fornendo la matrice di incidenza \( A \).
- `simcore/network/network.py`  
  Monta il sistema KCL + equazioni di stato, costruisce residuo \( F \) e Jacobiana \( J \).
- `simcore/components/*`  
  Contiene i modelli dei componenti (implementano `BranchComponent`).
- `simcore/solver/integrate.py`  
  Integra nel tempo con schema implicito di Eulero e risoluzione Newton-Raphson.
- `simcore/solver/newton.py`  
  Implementa il risolutore per sistemi non lineari con line-search opzionale.

---

## Formulazione matematica

### Equazioni di rete

Sia \( A \in \mathbb{R}^{n \times m} \) la matrice di incidenza nodo-ramo (nodi senza massa di riferimento, rami in ordine coerente con i componenti). Le tensioni di ramo sono \( v_b = A^{\top} v \) con \( v \) vettore delle tensioni dei nodi. Per ogni ramo \( j \) la corrente è data dal modello del componente: \( i_j = f_j(v_{b,j}, \dot{v}_{b,j}, z_j) \).

Le correnti di nodo rispettano la KCL:
\[
F_{\text{nodes}}(v, z) = A\,i(v, \dot{v}, z) = 0.
\]

### Discretizzazione implicita

Si utilizza Eulero implicito con passo \( \Delta t = \mathrm{dt} \):
\[
\dot{v}_{b} \approx \frac{A^{\top}(v_{k+1} - v_k)}{\Delta t}.
\]
L'insieme di equazioni al passo \( k+1 \) è risolto simultaneamente per \( (v_{k+1}, z_{k+1}) \).

### Equazioni di stato

Ogni componente con stato interno fornisce un residuo \( R_j(z_{k+1}, z_k, v_{b,k+1}) = 0 \). Il vettore globale è \( F_{\text{states}} = [R_1; R_2; \dots] \). Il sistema non lineare completo è
\[
F(x) =
\begin{bmatrix}
F_{\text{nodes}}(v_{k+1}, z_{k+1}) \\[4pt]
F_{\text{states}}(v_{k+1}, z_{k+1})
\end{bmatrix}
= 0,
\quad x = \begin{bmatrix} v_{k+1} \\ z_{k+1} \end{bmatrix}.
\]

### Solver Newton-Raphson

Il file `simcore/solver/newton.py` implementa:
\[
J(x^{(r)}) \, \Delta x = -F(x^{(r)}),
\qquad x^{(r+1)} = x^{(r)} + \alpha \Delta x,
\]
con fattori di regolarizzazione diagonali e ricerca in linea di tipo Armijo per la stabilità globale (`NewtonConfig` consente di regolare tolleranze, iterazioni e damping).

---

## Componenti e relativi modelli matematici

### Resistor (`simcore/components/resistor.py`)

- Legge di Ohm: \( i = \frac{v}{R} \).
- Jacobiana: \( \frac{\partial i}{\partial v} = \frac{1}{R} \).
- Componente puramente statico (nessuno stato interno).

### Thermal Resistor (`simcore/components/resistor_thermal.py`)

- Resistenza dipendente dalla temperatura: \( R(T) = R_0 \left[1 + \alpha (T - T_\text{ref})\right] \).
- Corrente: \( i = \frac{v}{R(T)} \) con derivata \( \partial i / \partial v = 1/R(T) \) e \( \partial i / \partial T = -v\,\alpha R_0 / R(T)^2 \).
- Stato termico: \( C_\text{th} \, \frac{dT}{dt} = \frac{v^2}{R(T)} - \frac{T - T_\text{amb}}{R_\text{th}} \).
- Il residuo viene integrato implicitamente e lo stato è osservabile via `component.state_history("T")`; la resistenza elettrica si aggiorna automaticamente passo dopo passo.

### Capacitor (`simcore/components/capacitor.py`)

- Relazione continuo-temporale: \( i = C \frac{dv}{dt} \).
- Discretizzazione implicita:
  \[
  i_{k+1} = C \frac{v_{k+1} - v_k}{\Delta t}, \qquad
  \frac{\partial i}{\partial v_{k+1}} = \frac{C}{\Delta t}.
  \]
- Condizioni iniziali: in `Network._apply_capacitor_initial_conditions` ogni condensatore può specificare `V_init`. Se omesso, la tensione iniziale deriva dal valore già presente sul nodo (ad es. da `v0_nodes`). Il metodo impone la differenza di potenziale tra i nodi del ramo rispettando i nodi di riferimento.

### Constant Power Load (`simcore/components/constant_power_load.py`)

- Obiettivo: mantenere \( P(t) = V \cdot I \).
- Corrente: \( i = \frac{P(t)}{\max(v, v_{\min})} \).
- Jacobiana (quando \( v > v_{\min} \)):
  \[
  \frac{\partial i}{\partial v} = -\frac{P(t)}{v^2}.
  \]
- Il limite \( v_{\min} \) evita denominatori nulli; sotto soglia la derivata viene posta a zero, pulendo la Jacobiana per tensioni molto basse.

### Lithium Battery LUT (`simcore/components/battery_lut.py`)

Modello di Thevenin:

- Corrente di ramo:
  \[
  i = \frac{v - \text{OCV}(\text{SOC})}{R_\text{internal}}.
  \]
- OCV ricavata da una tabella LUT (`soc_pts`, `ocv_pts`) tramite interpolazione lineare clampata a \([0,1]\).
- Equazione di stato (bilancio di carica):
  \[
  \text{SOC}_{k+1} - \text{SOC}_{k} - \frac{\Delta t}{Q_\text{coulomb}} i_{k+1} = 0.
  \]
- Jacobiane:
  \[
  \frac{\partial i}{\partial v} = \frac{1}{R_\text{internal}},\qquad
  \frac{\partial i}{\partial \text{SOC}} = -\frac{1}{R_\text{internal}}\frac{d\,\text{OCV}}{d\,\text{SOC}},
  \]
  \[
  \frac{\partial R}{\partial z} = 1 - \frac{\Delta t}{Q_\text{coulomb}} \frac{\partial i}{\partial \text{SOC}},
  \qquad
  \frac{\partial R}{\partial v} = -\frac{\Delta t}{Q_\text{coulomb}} \frac{\partial i}{\partial v}.
  \]

Il termine \( Q_\text{coulomb} = Q_\text{Ah} \cdot 3600 \) viene calcolato in `__post_init__`.

---

## Pipeline di simulazione

1. **Configurazione topologica**  
   `NetworkGraph` costruisce \( A \) e gli ordinamenti di nodi/rami.
2. **Creazione `Network`**  
   - Mappa componenti nell'ordine dei rami.  
   - Accoda gli stati iniziali nei vettori `z0`.  
   - Prepara slicing per estrarre gli stati component-wise.  
   - (Se necessario) calcola `V_init` dei condensatori.
3. **Time stepping (`run_sim`)**  
   - Per ciascun passo:  
     a. Usa la soluzione precedente come guess.  
     b. Costruisce residui e Jacobiane tramite `Network.assemble`.  
     c. Chiede a Newton di trovare \( x = [v_{k+1}; z_{k+1}] \).  
     d. Memorizza la storia delle tensioni e degli stati.

Risultato finale: `SimResult` con vettori `t`, `v_nodes` (matrice \( n_\text{nodi} \times (N+1) \)) e `z_hist` (stati concatenati).

---

## Estendere il motore

- **Nuovi componenti**: eredita da `BranchComponent`, implementa `current`, `dI_dv`, opzionalmente stati (`n_states`, `state_init`, `state_residual`, `dRdz`, `dRdv`, `dI_dz`).
- **Opzioni solver**: modifica `NewtonConfig` per cambiare tolleranza, iterazioni massime o disattivare il damping.
- **Analisi avanzate**: usa i dati di `SimResult` per post-processing in NumPy/Pandas o per grafici (`matplotlib`).
- **Componenti compositi**: se vuoi riutilizzare macro a due morsetti (es. filtri RC), eredita da `CompositeBranchComponent`. Registra i nodi interni con `add_internal_node` e i rami primitivi con `add_branch`, usando `+` e `-` per i terminali esterni. `Network` espande automaticamente questi blocchi in rami elementari; l'utente finale li istanzia e li aggiunge al grafo come qualunque altro componente. Il file `simcore/components/composites.py` contiene l'esempio `SeriesRC` (resistenza in serie a un condensatore).
- **Accesso alle grandezze**: dopo una simulazione puoi interrogare direttamente i componenti (`comp.state_history("SOC")`, `comp.voltage_history()`) oppure usare `SimResult.component_state(...)` / `SimResult.branch_voltage(...)` per recuperare le time-series senza passare dagli oggetti originali.

---

## Requisiti

Le dipendenze principali sono in `requirements.txt` (NumPy, Matplotlib, Jupyter e pacchetti correlati). L'uso di un ambiente virtuale dedicato è raccomandato per mantenere coerenti i pacchetti.
