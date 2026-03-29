import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

import HY_CA_secrete_and_sense_cells as CA

# -----------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------
tmax     = 300        # maximum timesteps to simulate; set generously so the full
                      # trajectory until final-pattern formation is always captured
gridsize = 14         # gridsize^2 = total number of cells (14x14 = 196)
lat_type = 'triangular'
periodic_bc = [1, 1]  # periodic boundaries in both x and y

a0   = 1.5
rcell = 0.2
lamb = np.array([1, 1.2])   # diffusion lengths for the two secreted molecules

# Interaction matrix M, threshold matrix K, and secretion rates C.
# These are the parameter values used in the paper.
M_matrix = np.array([[1,  1],
                     [-1, 0]])
K_matrix = np.array([[3,  10],
                     [11,  4]])
C_matrix = np.array([18, 16])

# Initial condition: fraction of ON cells for each gene, and spatial
# correlation target via Moran's I
p0    = np.array([0.5, 0.55])
I_min = 0.01
dI    = 0.05

frame_rate = 5   # frames per second for the real-time display

# -----------------------------------------------------------------------
# Build lattice, set parameters, randomize initial condition
# -----------------------------------------------------------------------
num_cells  = gridsize ** 2
trajectory = CA.CellularLattice(gridsize, num_cells, rcell, a0, tmax)
trajectory.init_lattice(gridsize, periodic_bc, a0, rcell, lat_type)
trajectory.init_general_parameters(K_matrix, C_matrix, M_matrix, lamb)
trajectory.init_cell_state(p0, I_min, dI)

# -----------------------------------------------------------------------
# Run the CA and analyze the resulting trajectory
# -----------------------------------------------------------------------
print('Running simulation...')
t0 = time.time()
trajectory.run_model(tmax)
trajectory.analyse_trajectory()
print(f'Simulation complete in {time.time() - t0:.1f} s')
print(f'Final pattern type: {trajectory.final_pattern_type}  '
      f'(1 = static configuration, 2 = spiral wave, 3 = rectilinear wave)')
print(f'Pattern formed at timestep: {trajectory.first_recurrent_state_time}')

# -----------------------------------------------------------------------
# Display the full trajectory as a real-time movie.
# The movie runs from t=0 to the timestep at which the final pattern
# forms (trajectory.first_recurrent_state_time), then holds the last
# frame on screen until the window is manually closed.
# -----------------------------------------------------------------------
end_frame = trajectory.first_recurrent_state_time

trajectory.show_trajectory(
    folderName_for_frames  = None,   # set to a folder path to save frames to disk
    start_frame            = 0,
    end_frame              = end_frame,
    frame_rate             = frame_rate,
    store_frames           = False,
    spin_vect              = False
)

# Hold the final frame open until the user closes the window
plt.show(block=True)
