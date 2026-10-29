import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

import HY_CA_secrete_and_sense_cells as CA

# -----------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------
tmax     = 60000      # maximum timesteps to simulate. Set generously: static and
                      # rectilinear-wave outcomes usually form within a few hundred
                      # timesteps, but spiral-wave outcomes can take tens of
                      # thousands. See Supplementary fig. 2 for the distribution.
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
K_matrix = np.array([[3,  8.465876331385257],
                     [11,  4]])
C_matrix = np.array([18, 16])

# Initial condition: fraction of ON cells for each gene, and spatial
# correlation target via Moran's I
p0    = np.array([0.5, 0.55])
I_min = 0.01
dI    = 0.05

# Display settings. The simulation may run for tens of thousands of timesteps,
# which is far more than is watchable, so the movie is capped. The topological
# self-organization of interest (vortex formation and annihilation) happens
# within roughly the first hundred timesteps.
frame_rate         = 5     # frames per second for the real-time display
max_frames_to_show = 300   # cap on the number of timesteps displayed

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

pattern_names = {1: 'static configuration', 2: 'spiral wave', 3: 'rectilinear wave'}
ptype = int(trajectory.final_pattern_type)
formation_time = trajectory.first_recurrent_state_time

print(f'Final pattern type: {ptype}  ({pattern_names.get(ptype, "unrecognized")})')

# If the trajectory never became periodic within tmax, the reported time is just
# tmax and is NOT a formation time. Say so, rather than printing a number that
# looks like a result.
if formation_time is None or formation_time >= tmax:
    print(f'WARNING: the final pattern had NOT yet formed after {tmax} timesteps.')
    print(f'         The value above is the simulation limit, not a formation time.')
    print(f'         Increase tmax at the top of this script and run again.')
else:
    print(f'Pattern formed at timestep: {formation_time}')

# -----------------------------------------------------------------------
# Display the trajectory as a real-time movie.
# -----------------------------------------------------------------------
end_frame = int(min(formation_time if formation_time else tmax, max_frames_to_show))

if formation_time and formation_time > max_frames_to_show:
    print(f'Displaying the first {end_frame} of {formation_time} timesteps ' 
          f'(raise max_frames_to_show to see more).')

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
