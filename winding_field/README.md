# The winding field

This folder contains the code that computes the **winding field**, and the code that tests how
much of the fate-predictive signal that field carries.

---

## What the winding field is

Each cell of the lattice is in one of four states. Group together every set of neighbouring
cells that share the same state; call each such group a **cluster**.

Because the lattice is a torus, a cluster can do something a cluster on a flat sheet cannot: it
can close on itself by wrapping all the way around the lattice. A path through a cluster that
wraps around the torus in this way cannot be shrunk to a point without leaving the cluster, and
is called a **non-contractible loop**.

Every such loop is labelled by two integers `(wx, wy)`, counting how many times it wraps around
the horizontal and vertical directions of the torus, with a sign for direction. A cluster may
contain several loops; the cluster is assigned the componentwise largest absolute value over
them. Every cell then inherits the value of its own cluster. Cells in clusters that do not wrap
get `(0, 0)`.

The result is the winding field: two integers per cell, describing how the region that cell
belongs to wraps the torus.

**The winding field adds no information.** It is computed deterministically from the
configuration alone, so it cannot contain anything the configuration did not already contain.
What it changes is how directly the global topology can be read off.

---

## Where the computation actually happens

Two methods in `codes/Elinor_secrete_sense_module.py` do the real work:

| method | what it returns |
|---|---|
| `label_phase_clusters(frame, phase_value=None)` | a cluster id for every cell |
| `find_noncontractible_loops(frame, phase_value=None)` | for each loop found, its `windings = (wx, wy)` and the cells it passes through |

The step that turns per-loop windings into a per-cell field lives in the calling script, not in
the module. The canonical version is in `codes/winding_field_extract_v260625.py`, in the block
commented `(A) winding field`.

---

## The two scripts that produce the data

**`codes/winding_field_extract_v260625.py`** produces the per-cell winding field for every
configuration in the machine-learning splits, row-for-row aligned with those splits.

```bash
python winding_field_extract_v260625.py --validate
python winding_field_extract_v260625.py --split train --nproc 24
python winding_field_extract_v260625.py --split val
python winding_field_extract_v260625.py --split test
```

Writes `fields_Ttrue_f{f}.npz` containing:

| key | shape | meaning |
|---|---|---|
| `winding` | (N, 2, 196) | channel 0 = wx, channel 1 = wy |
| `vortex` | (N, 2, 196) | channel 0 = +1 cores, channel 1 = -1 cores |

**`codes/winding_order_full_v260625.py`** produces, for each trajectory, how the winding field
grows over time: the fraction of cells with non-zero winding, the number of vortices, and the
spatial autocorrelation (Moran statistic) of each field. Sampling is dense early (every timestep
up to 100) and sparser afterwards, always including the first and last timestep.

---

## Testing what the winding field carries

**`codes/winding_cnn_v260625.py`** trains the same convolutional network on three different
inputs, so their accuracies can be compared directly:

```bash
python winding_cnn_v260625.py --rep winding          # winding field only, 2 channels
python winding_cnn_v260625.py --rep winding_vortex   # winding + vortex, 4 channels
python winding_cnn_v260625.py --rep full             # the raw configuration, 4 channels
```

**`codes/winding_gauge_cnn_v260625.py`** asks how much of the winding field actually matters,
by transforming it before training:

| `--rep` | input |
|---|---|
| `signed` | the field as computed, signs kept |
| `abs` | absolute values, signs discarded |
| `canonical` | signs fixed to a single convention |
| `mask` | one bit per cell: does this cell wrap at all |

---

## Checking that the field is computed correctly

```bash
python codes/check_winding_fields_v260625.py       # shapes and row alignment against the splits
python codes/diag_winding_field_v260625.py         # inspect the field for individual configurations
python codes/verify_static_winding_axis_v260702.py # static fates wrap along a consistent axis
```

---

## Before you run anything

```bash
pip install numpy scipy scikit-learn matplotlib torch
```

Every script has placeholder paths near the top that must be set first:

| placeholder | what it should point to |
|---|---|
| `<PATH_TO_CODE>` | this `codes/` folder, so that the module can be imported |
| `<PATH_TO_SPLITS>` | the machine-learning splits (see `../ML_along_trajectory/README.md`) |
| `<PATH_TO_OUTPUT>` | where extracted fields and results should be written |

The scripts import `Elinor_secrete_sense_module.py` by explicit file path rather than as an
installed package, so `<PATH_TO_CODE>` must be correct even when running from elsewhere.

The extracted fields are not included here. They are several gigabytes, which exceeds what
GitHub is intended to hold, and they can be regenerated with the extraction script above.

---

## Two things that are easy to get wrong

**The winding field and the vortex field are different objects.** The `V_*.npy` arrays stored
inside the machine-learning splits are the **vortex** field: two binary channels marking cells
belonging to +1 and -1 vortex cores. The winding field is not stored there. It exists only in
the output of `winding_field_extract_v260625.py`, under the key `winding`.

**Non-contractible-loop strings are a related but separate object.** The paper treats the NCL
strings as the sparse, late-time instance of the winding field, not as the same quantity. Their
extraction is a different pipeline and is not in this folder.
