# GCA_simulation

This folder contains the core Generalized Cellular Automaton (GCA) simulation code used in Koopmans, Kay, and Youk (2026).

---

## Files

**`HY_CA_secrete_and_sense_cells.py`** — the CA engine. Implements a generalized cellular automaton of secrete-and-sense cells on a 2D triangular lattice with periodic boundary conditions. Each cell occupies one of four discrete gene-expression states, encoded as a binary pair (gene 1 ON/OFF, gene 2 ON/OFF). At each synchronous timestep, every cell senses the concentrations of two diffusible molecules secreted by its neighbors and updates its state according to the interaction matrix M, threshold matrix K, and diffusion lengths λ. The `CellularLattice` class handles lattice construction, parameter initialization, simulation, trajectory analysis, vortex detection, and visualization.

**`main.py`** — a minimal script that runs one simulation with the parameter values used in the paper and displays a real-time movie of the self-organization dynamics from the initial random configuration until the final pattern forms. The final frame remains on screen until manually closed.

---

## Running the simulation

```bash
python main.py
```

The script will print the final pattern type (static configuration, rectilinear wave, or spiral wave) and the timestep at which the final pattern formed. Because the initial configuration is drawn randomly, the outcome will vary between runs — this variability is the central phenomenon studied in the paper.

---

## Dependencies

- Python ≥ 3.9
- NumPy, SciPy, Matplotlib, pandas, imageio
- A Matplotlib backend that supports interactive display (the script uses `TkAgg` by default; change this in `main.py` if needed for your system)

---

## Reference

Koopmans, Kay, and Youk (2026).

For full model specification, see **Supplementary Note 1** of the paper.
