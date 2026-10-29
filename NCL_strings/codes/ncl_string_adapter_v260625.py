"""
ncl_string_adapter_v260625.py

Validate that INJECTING a precomputed gridsize-14 trajectory's 4-state history
into a fresh CellularLattice and running Elinor's analysis pipeline reproduces
EXACTLY the NCL-string quantity that her pipeline computes on a freshly simulated
trajectory. Only after this matches do we trust injection on our stored chunks.

Strategy:
  1. Fresh sim (her run_single_CA recipe) at gridsize 14 -> run_model -> analyse_trajectory
     -> record NCL-string-per-frame series via get_percent_of_strings_that_are_NCL.
  2. Take that sim's cell_4state, inject into a fresh object, run the analysis
     pipeline FROM THE cell_4state INJECTION POINT (skip get_4_number_seq(cell_hist)),
     and compute the same series.
  3. Assert identical.

Run from the folder containing Elinor_secrete_sense_module.py.

Usage:
    python ncl_string_adapter_v260625.py --validate         # fresh-vs-inject on 1 sim
    python ncl_string_adapter_v260625.py --validate --n 3   # on a few sims
"""

import argparse
import importlib.util
import numpy as np

MODULE_PATH = "Elinor_secrete_sense_module.py"

# CA parameters (match the paper / her driver)
TMAX = 75000
RCELL = 0.2
A0 = 1.5
LATTICE_TYPE = "triangular"
PERIODIC_BC = [1, 1]
P0 = np.array([0.5, 0.55])
I_MIN = 0.01
DI = 0.05
M_MATRIX = np.array([[1, 1], [-1, 0]])
K_MATRIX = np.array([[3, 8.465876331385257], [11, 4]])
C_MATRIX = np.array([18, 16])
LAMB = np.array([1, 1.2])

GRIDSIZE = 14


def load_module(path=MODULE_PATH):
    spec = importlib.util.spec_from_file_location("esm", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fresh_sim(esm, gridsize, seed=None):
    """Her run_single_CA recipe (without the NCL extraction), returns the analyzed object."""
    if seed is not None:
        np.random.seed(seed)
    num_cells = gridsize ** 2
    t = esm.CellularLattice(gridsize, num_cells, RCELL, A0, TMAX)
    t.init_lattice(gridsize, PERIODIC_BC, A0, RCELL, LATTICE_TYPE)
    t.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)
    t.quiet_warnings = True
    t.init_cell_state(P0, I_MIN, DI)
    t.convert_currentBinaryState_to_4cell_state()
    t.run_model(TMAX)
    t.analyse_trajectory()
    return t


def analyse_from_4state(esm, gridsize, cell_4state):
    """
    Reproduce analyse_trajectory() but inject cell_4state directly instead of
    deriving it from cell_hist. Mirrors the method body from the line after
    get_4_number_seq onward (module source lines ~354-399).
    """
    num_cells = gridsize ** 2
    t = esm.CellularLattice(gridsize, num_cells, RCELL, A0, TMAX)
    t.init_lattice(gridsize, PERIODIC_BC, A0, RCELL, LATTICE_TYPE)
    t.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)
    t.quiet_warnings = True

    # inject
    t.cell_4state = np.asarray(cell_4state)
    t.tmax = t.cell_4state.shape[1]

    # ---- mirror analyse_trajectory() from line 354 (after the cell_hist->4state line) ----
    t.cell_4phase = esm.get_phase_seq_4(t.cell_4state)
    t.state_fractions = esm.compute_4_state_fractions(t)
    t.first_recurrent_state_time = esm.get_first_recurrent_state_time(t)
    t.repeated_time = esm.get_time_that_is_repeated(t)
    t.final_pattern_type = esm.get_final_configuration_type(t)

    if t.number_of_nn == 6:
        t.vortex_cores, t.Hxy, t.Hxy_adjusted = esm.compute_phase_differences(t)
        t.vortex_cores_labeled, t.n_vortex = esm.label_triangular_lattice(t)
        t.xcom, t.ycom = esm.compute_centroid(t)
        t.closest_pos, t.charges, t.xmin_core, t.ymin_core = esm.compute_smallest_movement(t)
        t.idx_annihilation_moments, t.idx_annihilation_moments_core_count = esm.compute_annihilation_moments(t)
    return t


def ncl_string_series(t):
    """The Fig-4 NCL-string-per-frame ratio series, with include_zero_charge=False."""
    return np.asarray(t.get_percent_of_strings_that_are_NCL(include_zero_charge=False), dtype=float)


def summarize(t):
    return dict(
        final_pattern_type=int(t.final_pattern_type),
        first_recurrent_state_time=int(t.first_recurrent_state_time),
        n_frames=int(t.cell_4state.shape[1]),
    )


def validate(esm, n=1):
    all_ok = True
    for k in range(n):
        print(f"\n=== sim {k} ===")
        t_fresh = fresh_sim(esm, GRIDSIZE, seed=1000 + k)
        s_fresh = summarize(t_fresh)
        print("fresh:", s_fresh)
        series_fresh = ncl_string_series(t_fresh)

        # inject the SAME sim's 4-state and re-run
        cell_4state = np.array(t_fresh.cell_4state)
        t_inj = analyse_from_4state(esm, GRIDSIZE, cell_4state)
        s_inj = summarize(t_inj)
        print("inject:", s_inj)
        series_inj = ncl_string_series(t_inj)

        same_meta = (s_fresh == s_inj)
        same_len = (len(series_fresh) == len(series_inj))
        same_series = same_len and np.allclose(series_fresh, series_inj, atol=1e-9, equal_nan=True)
        print(f"  meta match:   {same_meta}")
        print(f"  series len:   fresh={len(series_fresh)} inject={len(series_inj)} match={same_len}")
        print(f"  series match: {same_series}")
        if not same_series and same_len:
            d = np.abs(np.nan_to_num(series_fresh) - np.nan_to_num(series_inj))
            bad = np.where(d > 1e-9)[0]
            print(f"  first mismatching frames: {bad[:10].tolist()}")
        ok = same_meta and same_series
        print("  RESULT:", "PASS" if ok else "FAIL")
        all_ok = all_ok and ok
    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    esm = load_module()
    if args.validate:
        validate(esm, n=args.n)
    else:
        print("Use --validate")


if __name__ == "__main__":
    main()
