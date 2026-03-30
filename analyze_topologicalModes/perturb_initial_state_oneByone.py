"""
For a single randomly initialized CA configuration, perturbs the initial state
one cell at a time — flipping each cell to a randomly chosen different state —
and records the final pattern type and run time for each perturbation.

For each perturbation, the perturbed cell is assigned a new binary state drawn
at random from the three states other than its current state. The CA is then run
from this modified initial configuration and analyzed in the same way as the
original. Outputs allow comparison of how single-cell perturbations redirect
macroscopic pattern formation.

Outputs (written to OUTPUT_perturb_initial_state_oneByone/):
  - Initial and final configuration images (.jpg, .pdf) for the original and
    each perturbed run
  - Binary configuration arrays (.npy) for the original and each perturbed run
  - Final pattern type labels and run times for all perturbed runs
  - Parameter values used
"""

import HY_CA_secrete_and_sense_cells as CA
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import os
import copy
import time


# -----------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------
output_folderName              = 'OUTPUT_perturb_initial_state_oneByone'
initialConfig_inBinary_Array_dir = os.path.join(output_folderName, 'initialConfig_inBinary_Array')
finalConfig_inBinary_Array_dir   = os.path.join(output_folderName, 'finalConfig_inBinary_Array')
finalType_dir                    = os.path.join(output_folderName, 'finalType_Labels')
os.makedirs(initialConfig_inBinary_Array_dir, exist_ok=True)
os.makedirs(finalConfig_inBinary_Array_dir,   exist_ok=True)
os.makedirs(finalType_dir,                    exist_ok=True)


# -----------------------------------------------------------------------
# CA parameters (match those used in the paper, 16x16 lattice)
# -----------------------------------------------------------------------
tmax      = 50000
gridsize  = 16          # 16x16 = 256 cells; intentionally larger than the 14x14 default
num_cells = gridsize ** 2
lat_type  = 'triangular'
periodic_bc = [1, 1]

a0   = 1.5
rcell = 0.2
lamb = np.array([1, 1.2])

M_matrix = np.array([[1,  1],
                     [-1, 0]])
K_matrix = np.array([[3,  10],
                     [11,  4]])
C_matrix = np.array([18, 16])

p0    = np.array([0.5, 0.55])
I_min = 0.01
dI    = 0.05


# -----------------------------------------------------------------------
# Initialize the original (unperturbed) lattice
# -----------------------------------------------------------------------
initial_trajectory = CA.CellularLattice(gridsize, num_cells, rcell, a0, tmax)
initial_trajectory.init_lattice(gridsize, periodic_bc, a0, rcell, lat_type)
initial_trajectory.init_general_parameters(K_matrix, C_matrix, M_matrix, lamb)
initial_trajectory.init_cell_state(p0, I_min, dI)
initial_trajectory.convert_currentBinaryState_to_4cell_state()

# Deep-copy the initialized (pre-run) object for use as the perturbation template
trajectory_before_initialRun = copy.deepcopy(initial_trajectory)

# Store initial configuration before running
initial_config_4state = copy.deepcopy(trajectory_before_initialRun.current_4state)
initial_config_binary = copy.deepcopy(trajectory_before_initialRun.cells)


# -----------------------------------------------------------------------
# Run and analyze the original CA
# -----------------------------------------------------------------------
initial_trajectory.run_model(tmax)
initial_trajectory.analyse_trajectory()

outcomeLabel_for_initialRun = initial_trajectory.final_pattern_type
runTime_for_originalRun     = initial_trajectory.first_recurrent_state_time

print(f'Original run: pattern type = {outcomeLabel_for_initialRun}  '
      f'(1 = static, 2 = spiral wave, 3 = rectilinear wave)')
print(f'Original run: pattern formed at timestep {runTime_for_originalRun}')

# Save original initial configuration (image and binary array)
initial_trajectory.save_single_frame(
    'original_initialConfig', parent_folder=output_folderName,
    jpg_folder='InitialConfig_JPG', pdf_folder='InitialConfig_PDF',
    frame_to_save=0, spin_vect=False
)
np.save(os.path.join(initialConfig_inBinary_Array_dir, 'original_initialConfigBinary.npy'),
        initial_config_binary)

# Save original final configuration (image and binary array)
initial_trajectory.save_single_frame(
    'original_finalConfig', parent_folder=output_folderName,
    jpg_folder='FinalConfig_JPG', pdf_folder='FinalConfig_PDF',
    frame_to_save=initial_trajectory.first_recurrent_state_time - 1, spin_vect=False
)
np.save(os.path.join(finalConfig_inBinary_Array_dir, 'original_finalConfigBinary.npy'),
        initial_trajectory.cell_hist[:, :, initial_trajectory.first_recurrent_state_time - 1])


# -----------------------------------------------------------------------
# Perturbation loop: flip one cell at a time, run, record outcome
# -----------------------------------------------------------------------
num_iterations_for_perturbation = num_cells * 3

outcomeLabels_for_perturbedRuns = np.zeros(num_iterations_for_perturbation)
runTimes_for_perturbedRuns      = np.zeros(num_iterations_for_perturbation)

for i in range(num_iterations_for_perturbation):

    print(f'Perturbation {i+1} / {num_iterations_for_perturbation}')

    copy_trajectory   = copy.deepcopy(trajectory_before_initialRun)
    new_initial_config = copy.deepcopy(initial_config_binary)

    # Choose a random cell and assign it a randomly drawn state different from its current state
    cell_to_perturb = np.random.randint(num_cells)
    new_cellState   = np.random.choice([0, 1], size=2)
    while np.array_equal(new_cellState, new_initial_config[cell_to_perturb]):
        new_cellState = np.random.choice([0, 1], size=2)

    new_initial_config[cell_to_perturb] = new_cellState
    copy_trajectory.cells = copy.deepcopy(new_initial_config)
    copy_trajectory.convert_currentBinaryState_to_4cell_state()

    # Save perturbed initial configuration
    np.save(os.path.join(initialConfig_inBinary_Array_dir, f'initialConfigBinary_Perturbation_{i+1}.npy'),
            copy_trajectory.cells)
    copy_trajectory.save_single_frame(
        f'Perturbed_initialConfig_{i+1}', parent_folder=output_folderName,
        jpg_folder='InitialConfig_JPG', pdf_folder='InitialConfig_PDF',
        frame_to_save=0, spin_vect=False
    )

    # Run and analyze perturbed CA
    copy_trajectory.run_model(tmax)
    copy_trajectory.analyse_trajectory()

    # Save perturbed final configuration
    np.save(os.path.join(finalConfig_inBinary_Array_dir, f'finalConfigBinary_Perturbation_{i+1}.npy'),
            copy_trajectory.cell_hist[:, :, copy_trajectory.first_recurrent_state_time - 1])
    copy_trajectory.save_single_frame(
        f'Perturbed_finalConfig_{i+1}', parent_folder=output_folderName,
        jpg_folder='FinalConfig_JPG', pdf_folder='FinalConfig_PDF',
        frame_to_save=copy_trajectory.first_recurrent_state_time - 1, spin_vect=False
    )

    outcomeLabels_for_perturbedRuns[i] = copy_trajectory.final_pattern_type
    runTimes_for_perturbedRuns[i]      = copy_trajectory.first_recurrent_state_time

    copy_trajectory.clean_data()


# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------
parameterValues = np.array([gridsize, tmax, a0, rcell, lamb[0], lamb[1], p0[0], p0[1], I_min, dI])
np.save(os.path.join(finalType_dir, 'OUTPUT_parameters.npy'),                    parameterValues)
np.save(os.path.join(finalType_dir, 'FinalPattern_labels_for_perturbedRuns.npy'), outcomeLabels_for_perturbedRuns)
np.save(os.path.join(finalType_dir, 'RunTimes_for_perturbedRuns.npy'),            runTimes_for_perturbedRuns)
np.save(os.path.join(finalType_dir, 'RunTime_for_originalCA.npy'),                runTime_for_originalRun)
np.save(os.path.join(finalType_dir, 'FinalPatternLabel_for_originalCA.npy'),      outcomeLabel_for_initialRun)

print(f'\nDone. Completed {num_iterations_for_perturbation} perturbations.')
print(f'Outcome labels: {outcomeLabels_for_perturbedRuns}')
print(f'Run times: {runTimes_for_perturbedRuns}')
