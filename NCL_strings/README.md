# Non-contractible-loop (NCL) strings

This folder contains the code for the non-contractible-loop strings: how they are extracted,
how they emerge over time, and how well they predict the final pattern.

---

## What an NCL string is, and how it relates to the winding field

The lattice is a torus, so a connected region of same-state cells can close on itself by
wrapping all the way around the lattice. A path that does so cannot be shrunk to a point
without leaving the region, and is called a **non-contractible loop**. An NCL string is the
set of cells forming such a loop.

This is the same topological property that the winding field measures, but recorded
differently. The winding field labels **every** cell with the wrapping of the region it belongs
to, and is therefore dense. An NCL string is the **loop itself**: a thin path of cells, and
therefore sparse. The paper treats the NCL strings as the sparse, late-time instance of the
winding field, not as a different quantity.

For the dense per-cell version, see `../winding_field`.

---

## The pipeline

**Extraction.** `codes/string_mask_extract_v260625.py` finds the loops in each configuration
and records which cells belong to them, aligned row-for-row with the machine-learning splits.
`codes/ncl_string_adapter_v260625.py` and `codes/ncl_string_ensemble_v260625.py` handle the
per-trajectory bookkeeping, The collection step, `ncl_data_collection_script.py`, is in `../analyze_topologicalModes/`.

**Emergence over time.** `codes/ncl_emergence_v260625.py` and its parallel counterpart
`codes/ncl_emergence_parallel_v260625.py` follow individual trajectories and record when
strings appear and when they disappear. This is what shows that strings vanish along
trajectories ending in static configurations or rectilinear waves, but persist along
spiral-wave trajectories.

**Prediction.** `codes/ncl_features_parallel_v260625.py` turns the strings into features.
`codes/ncl_count_predictor_v260625.py` asks how well the number of strings alone predicts the
fate, and `codes/ncl_predictor_v260625.py` uses the fuller description.

---

## The asymmetry these scripts establish

The disappearance of strings is informative and their presence is not, and the reason is worth
stating plainly. An observer watching the automaton forward in time can see that strings have
gone, and that increasingly identifies the static and rectilinear fates. But an observer who
sees strings still present cannot tell whether they will persist indefinitely, which would mean
a spiral wave, or vanish later, which would mean one of the other two fates. Spiral-wave fate
therefore becomes reliably distinguishable only close to pattern formation.

---

## Before you run anything

```bash
pip install numpy scipy scikit-learn matplotlib
```

These scripts import `Elinor_secrete_sense_module.py`, which is in `../winding_field/codes/`.
The loop-finding itself is done by `find_noncontractible_loops` in that module.

Every script has placeholder paths near the top that must be set first:

| placeholder | what it should point to |
|---|---|
| `<PATH_TO_CODE>` | `../winding_field/codes`, so the module can be imported |
| `<PATH_TO_SPLITS>` | the machine-learning splits (see `../ML_along_trajectory/README.md`) |
| `<PATH_TO_OUTPUT>` | where extracted strings and results should be written |

The extracted string data are not included here. They can be regenerated with the extraction
script above.
