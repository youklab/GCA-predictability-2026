''''
This module contains the CellularLattice class, which models a cellular automaton on a lattice. The class includes methods for initializing the lattice, running the simulation, analyzing trajectories, and visualizing results.

Most important variables are:

self.cell_hist:  3D array with dimensions (num_cells, 2, tmax) that stores the state of every cell all throughout time, where:
                - num_cells = tot. # of cells
                - '2' = two possible configurations (one for each gene) for each cell
                - 'tmax' = max # of timesteps requested


self.cell_4state:  **contains state of every cell for every timestep as '1', '2', '3', or '4', where:
                - 2D array: one row for each cell, one column for each timestep
                - get_4_number_seq(self.cell_hist) converts the [0/1, 0/1] into '1', '2', '3', or '4' for each cell and timestep.


self.current_4state: **for current timestep**,  1D array of length N: current 4-state for every cell (only for current time).
                - Assigned by calling self.convert_currentBinaryState_to_4cell_state()


self.cells:  **for current timestep**,  num_cells x 2 array, containing the state of every cell now (at the current timestep), where:
                - Each row = a specific cell
                - Two columns: each is either 0 or 1, representing (ON/OFF, ON/OFF) for a cell represented by that row.

'''

import numpy as np
import random
import time
from pylab import *
import time
import copy
import imageio
import pickle
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import os   # for handling file and directory operations
from collections import deque
from itertools import combinations
import cmath



class CellularLattice:
    """
    A class representing a cellular automaton on a lattice. This class models cell interactions,
    computes state changes, and tracks trajectories over time.

    Attributes:
        gridsize (int): Size of the grid.
        N (int): Number of cells in the lattice.
        rcell (float): Cell radius.
        a0 (float): Lattice constant.
        tmax (int): Maximum number of timesteps.
        cell_hist (ndarray): History of cell states over time.
        vortex_cores (ndarray): Locations of vortex cores in the lattice.
        etc.
    """

    def __init__(self, gridsize, num_cells, rcell, a0, tmax, debug=False):
        """
        Initialize the CellularLattice class with grid size, cell count, and lattice parameters.

        Args:
            gridsize (int): The size of the grid.
            num_cells (int): The total number of cells.
            rcell (float): The radius of a cell.
            a0 (float): The lattice constant (spacing).
            tmax (int): The maximum number of timesteps.
            debug (bool): Enable debug print statements (default: False).
        """
        self.relative_distance = []  # Store distances between cells
        self.positions = []  # Store cell positions
        self.gridsize = gridsize  # Grid size
        self.N = num_cells  # Number of cells
        self.rcell = rcell  # Cell radius
        self.a0 = a0  # Lattice constant
        self.debug = debug  # Debug flag for print statements
        self.cell_topology = []  # Topology of cells in the grid
        self.cells = np.zeros((self.N, 2))  # Initialize cell states
        self.periodic_bc = [0, 0]  # Periodic boundary conditions flags
        self.K = np.zeros((2, 2, 2))  # Interaction threshold matrix
        self.M = np.zeros((2, 2, 2))  # Interaction strength matrix
        self.C = np.zeros((2, 1, 2))  # Secretion rate matrix
        self.Coff = np.ones([2, 1, 2])  # Default secretion rates for inactive cells
        self.C0n = []  # Placeholder for other secretion rate data
        self.idx_celltype = []  # Indices for different cell types
        self.idx_nearest_neighbours = []  # Indices for nearest neighbors
        self.lamb = []  # Diffusion lengths
        self.tmax = tmax  # initially this is the maximum number of timesteps. But after run_CA(), this is changed to the number of time steps taken for the CA to end either as static, periodic, or non-repeating configuration.  tmax - 1 is the last frame (because frame starts at 0).
        self.cell_hist = np.zeros((num_cells, 2, tmax), np.int8)  # Store cell history across timesteps
        self.img = []  # Store trajectory images
        self.Fn_a0 = 1  # Normalization factor for lattice constant
        self.double_topology_flag = False  # Flag for topology changes
        self.clock_start = time.time()  # Start time for clocking operations
        self.clock_traject = np.zeros(10)  # Track timepoints during simulation
        self.number_of_nn = 6  # Number of nearest neighbors (default for triangular grid)
        self.cell_4state = []  # Store cell states in 4 possible configurations
        self.current_4state = [] # 1D array of length N: current 4-state for every cell (only for current time). Assigned by calling self.convert_currentBinaryState_to_4cell_state()

        self.cell_4phase = []  # Store phase information for cells
        self.state_fractions = []  # Store state fractions over time
        self.first_recurrent_state_time = []  # Time when the first recurrent state is observed
        self.final_pattern_type = []  # Type of final pattern
        self.vortex_cores = []  # Store vortex core locations
        self.Hxy = []  # XY Hamiltonian for trajectory analysis
        self.Hxy_adjusted = []  # Adjusted Hamiltonian
        self.vortex_cores_labeled = []  # Labeled vortex cores
        self.n_vortex = []  # Number of vortex cores
        self.xcom = []  # X-coordinate of vortex core centroids
        self.ycom = []  # Y-coordinate of vortex core centroids
        self.cell_number_nn = []  # Nearest neighbor indices for cells
        self.xmin_core = []  # Minimum X-coordinate of vortex core
        self.ymin_core = []  # Minimum Y-coordinate of vortex core
        self.closest_pos = []  # Closest positions of vortex cores over time
        self.charges = []  # Charges associated with vortex cores
        self.quiet_warnings = False  # When True, suppress noisy cluster warnings
        self.idx_annihilation_moments = []  # Moments of vortex core annihilation
        self.idx_annihilation_moments_core_count = []  # Core counts at annihilation moments
        self.shells = {}  # Nearest neighbor shells data
        self.repeated_time=[] #the time that is repeated at first reccurent state time 


    def init_lattice(self, gridsize, periodic_bc, a0, rcell, lattice_type, path=''):
        """
        Initialize the lattice with a specified type, either 'triangular' or from an Excel file.

        Args:
            gridsize (int): Grid size.
            periodic_bc (list): List indicating periodic boundary conditions in x and y directions.
            a0 (float): Lattice constant.
            rcell (float): Cell radius.
            type (str): Type of lattice initialization ('triangular', 'excel', 'excel_triangular').
            path (str): Path to Excel file if initializing from an Excel sheet.
        """
        n = gridsize ** 2  # Total number of cells
        self.a0 = a0  # Set lattice constant
        self.gridsize = gridsize  # Set grid size

        # Initialize triangular lattice
        if lattice_type == 'triangular':
            # Generate positions and calculate distances
            [pos, lx, ly] = calculate_triangular_lattice(gridsize)
            dist = calculate_distance(pos, lx, ly, gridsize, periodic_bc)

            # Find nearest neighbors based on distance
            idx_nearest_neighbours = np.round(dist, 1) == 1
            row_coll_idx_nn = np.argwhere(idx_nearest_neighbours == 1)

            # Initialize nearest neighbor structure
            cell_number_nn = np.zeros((n, 7))
            cell_number_nn[:, 0] = np.arange(0, n)
            cell_number_nn[:, 1:] = row_coll_idx_nn[:, 1].reshape(n, 6)

            # Store relevant properties in the object
            self.cell_number_nn = cell_number_nn.astype(int)
            self.idx_nearest_neighbours = idx_nearest_neighbours
            self.relative_distance = dist
            self.N = n
            self.positions = pos
            self.number_of_nn = 6

        # Initialize lattice from Excel sheet
        elif lattice_type == 'excel':
            # Read positions, distances, and states from Excel
            pos, dist, states = initialize_lattice_from_excel(path, periodic_bc)
            n = len(pos[:, 0])  # Number of cells based on Excel data

            # Find nearest neighbors based on distance
            idx_nearest_neighbours = np.round(dist, 1) == 1
            row_coll_idx_nn = np.argwhere(idx_nearest_neighbours == True)

            # Initialize nearest neighbor structure for 4-neighbor grid
            if np.sum(row_coll_idx_nn == 1) == 4 * n:
                cell_number_nn = np.zeros((n, 5))
                cell_number_nn[:, 0] = np.arange(0, n)
                cell_number_nn[:, 1:] = row_coll_idx_nn[:, 1].reshape(n, 4)
            else:
                cell_number_nn = np.zeros((n, 5))
                cell_number_nn[:, 0] = np.arange(0, n)

            # Store relevant properties in the object
            self.number_of_nn = 4
            self.cells = np.squeeze(get_2_number_seq(states[:, np.newaxis]))
            self.idx_nearest_neighbours = idx_nearest_neighbours
            self.cell_number_nn = cell_number_nn.astype(int)
            self.relative_distance = dist
            self.N = n
            self.positions = pos

        # Initialize lattice from an Excel sheet for triangular configuration
        elif lattice_type == 'excel_triangular':
            # Read positions and states from Excel
            pos_temp, dist_temp, states = initialize_lattice_from_excel(path, periodic_bc)
            n_cells = len(pos_temp[:, 0])  # Number of cells from Excel

            # Recalculate grid size based on number of cells
            gridsize = int(np.round(np.sqrt(n_cells)))

            # Generate triangular lattice positions and distances
            [pos, lx, ly] = calculate_triangular_lattice(gridsize)
            dist = calculate_distance(pos, lx, ly, gridsize, periodic_bc)

            # Find nearest neighbors for triangular configuration
            idx_nearest_neighbours = np.round(dist, 1) == 1
            row_coll_idx_nn = np.argwhere(idx_nearest_neighbours == True)

            # Store nearest neighbor structure
            if periodic_bc[0] == 1:
                cell_number_nn = np.zeros((n, 7))
                cell_number_nn[:, 0] = np.arange(0, n)
                cell_number_nn[:, 1:] = row_coll_idx_nn[:, 1].reshape(n, 6)
                self.cell_number_nn = cell_number_nn.astype(int)
            else:
                cell_number_nn = 0
                self.cell_number_nn = cell_number_nn

            # Store relevant properties in the object
            self.number_of_nn = 6
            self.cells = np.squeeze(get_2_number_seq(states[:, np.newaxis]))
            self.idx_nearest_neighbours = idx_nearest_neighbours
            self.relative_distance = dist
            self.N = n
            self.positions = pos
        else:
            # Handle invalid initialization type
            print('Invalid initialization type')

            # Store fallback properties in the object
            self.gridsize = gridsize
            self.a0 = a0
            self.rcell = rcell
            self.periodic_bc = periodic_bc

        # Default topology setup if the topology function is not called
        self.idx_celltype = np.zeros(n) == 0  # Assume all cells are of one type
        self.cell_topology = np.ones((gridsize, gridsize))  # Initialize uniform topology

    def init_lattice_excel(self, path, a0, rcell):
        """
        Initialize the lattice from an Excel file.

        Args:
            path (str): Path to the Excel file.
            a0 (float): Lattice constant.
            rcell (float): Cell radius.
        """
        positions, r, states = initialize_lattice_from_excel(path, a0)  # Get positions, distances, and states
        n = len(positions[:, 0])  # Number of cells

        # Set cell states, distances, and other properties
        self.cells = np.squeeze(get_2_number_seq(states[:, np.newaxis]))
        self.relative_distance = r
        self.N = n
        self.positions = positions
        self.gridsize = np.sqrt(n)
        self.a0 = a0
        self.rcell = rcell

        # Default setup for topology
        self.idx_celltype = np.zeros(n) == 0
        self.cell_topology = np.ones((self.gridsize, self.gridsize))

    def init_topology(self, topology_input):
        """
        Initialize the cell topology based on an input matrix.

        Args:
            topology_input (ndarray): The input matrix representing the topology of the grid.
        """
        gridsize = self.gridsize  # Grid size
        temp = init_topology_mat(np.flip(topology_input), gridsize)  # Initialize the topology matrix
        top_mat = temp  # Store the topology matrix

        # Set the cell topology and cell type indices
        self.cell_topology = top_mat
        self.idx_celltype = top_mat.reshape(self.N, order='F') > 0  # Reshape the matrix to 1D array for easy indexing

    def init_cell_state(self, p0, I_min, dI):
        """
        Initialize the state of each cell based on input parameters.

        Args:
            p0 (ndarray): Fractions of cells in each initial state.
            I_min (float): Minimum value of Moran's I for initial configuration.
            dI (float): Increment of Moran's I.
        """
        dist = self.relative_distance  # Get distances between cells
        N = self.N  # Total number of cells
        init_on = np.round(p0 * N)  # Calculate how many cells should be in the initial ON state
        self.cells = init_I(init_on, self.a0, dist, N, I_min, dI, self.debug)  # Initialize cell states using Moran's I


    def init_general_parameters(self, K_matrix, C_matrix, M_matrix, lamb):
        """
        Initialize the general parameters for cell-cell communication and interaction.

        Args:
            K_matrix (ndarray): Threshold matrix for cell activation.
            C_matrix (ndarray): Secretion rates.
            M_matrix (ndarray): Interaction strengths.
            lamb (ndarray): Diffusion lengths.
        """
        # Ensure the shape of the input matrices is correct and assign them
        self.K, self.double_topology_flag = force_input_matrix_shape(K_matrix, 1)
        M, self.double_topology_flag = force_input_matrix_shape(M_matrix, 1)
        self.Con, self.double_topology_flag = force_input_matrix_shape(C_matrix, 2)

        # Store the interaction strength matrix and diffusion lengths
        self.M = M.astype(int32)
        self.lamb = lamb

    def run_model(self, tmax):
        """
        Run the cellular automaton model for a specified number of timesteps.

        Args:
            tmax (int): Maximum number of timesteps to run the model.
        """
        # Extract parameters for simulation
        dist = self.relative_distance
        Rcell = self.rcell * self.a0
        a0 = self.a0
        lamb = self.lamb
        N = self.N
        idx_celltype = self.idx_celltype

        cells = self.cells
        Coff = self.Coff
        Con = self.Con

        M = self.M
        K = self.K

        # Run the cellular automaton and store the results
        self.cell_hist, self.tmax, self.Y= run_CA(M, K, N, tmax, dist, Rcell, a0, lamb, idx_celltype, Coff, Con, cells)

        # Record the first clock event
        self.clock_traject[1] = np.round(time.time() - self.clock_start, 2)

    def analyse_trajectory(self):
        """
        Analyze the trajectory of the cellular automaton simulation.
        This method calculates various properties of the system such as state fractions,
        recurrent states, vortex cores, and Hamiltonian values.
        """
        # Convert cell history into 4-state sequence representation
        self.cell_4state = get_4_number_seq(self.cell_hist)

        # Convert 4-state sequence into phase sequence representation
        self.cell_4phase = get_phase_seq_4(self.cell_4state)

        # Compute the fractions of cells in each state over time
        self.state_fractions = compute_4_state_fractions(self)

        # Find the first time when a configuration recurs
        self.first_recurrent_state_time = get_first_recurrent_state_time(self)

        #find the time that first reccurent state time repeats
        self.repeated_time = get_time_that_is_repeated(self)

        # Determine the type of the final configuration in the system
        self.final_pattern_type = get_final_configuration_type(self)

        if self.number_of_nn == 6:
            # Clock moment 2
            self.clock_traject[2] = np.round(time.time() - self.clock_start, 2)

            # Compute the phase differences between neighboring cells
            self.vortex_cores, self.Hxy, self.Hxy_adjusted = compute_phase_differences(self)

            # Clock moment 3
            self.clock_traject[3] = np.round(time.time() - self.clock_start, 2)

            # Label vortex cores within the system
            self.vortex_cores_labeled, self.n_vortex = label_triangular_lattice(self)

            # Clock moment 4
            self.clock_traject[4] = np.round(time.time() - self.clock_start, 2)

            # Compute the centroids of vortex cores
            self.xcom, self.ycom = compute_centroid(self)

            # Clock moment 5
            self.clock_traject[5] = np.round(time.time() - self.clock_start, 2)

            # Post-processing: Calculate the smallest movement of vortex cores
            # (compute_smallest_movement returns: closest_pos, charges, CmX, CmY)
            # store charges in self.charges so subsequent analysis uses the populated array
            self.closest_pos, self.charges, self.xmin_core, self.ymin_core = compute_smallest_movement(self)

            # Clock moment 6
            self.clock_traject[6] = np.round(time.time() - self.clock_start, 2)

            # Compute moments of vortex core annihilation
            self.idx_annihilation_moments, self.idx_annihilation_moments_core_count = compute_annihilation_moments(self)

    def clean_data(self):
        """
        Clean up and delete attributes no longer needed after the simulation has been completed.
        """
        del self.C
        del self.C0n
        del self.Coff
        del self.Fn_a0
        del self.Y
        del self.double_topology_flag


    def convert_currentBinaryState_to_4cell_state(self):
        """
        Convert the current binary state of cells to a 4-state representation and store in self.current_4state.
        """
        self.current_4state = get_current_4_number_seq(self.cells, self.N)



    def show_trajectory(self, folderName_for_frames, start_frame, end_frame, frame_rate, store_frames, spin_vect):
        """
        Visualize the trajectory of the system over time.
         I changed how I call show_cells() to pass in the correct slices of the arrays for the selected frames.

        Args:
            start_frame (int): The first frame to display.  (earliest possible frame = 0)
            end_frame (int): The last frame to display.
            frame_rate (int): The frame rate for visualization.
            store_frames (bool): If True, store the frames to disk.
            spin_vect (bool): If True, display spin vectors.
        """

        make_gif = True    # whether to make a movie to display when you run main.py
        # Generate and store images for each frame in the trajectory
        # self.img = show_cells(folderName_for_frames, self.tmax - start_frame, self.cell_hist[:, :, start_frame:end_frame], self.positions,
        #                       self.idx_celltype, frame_rate, spin_vect, self.vortex_cores_labeled, self.charges,
        #                       self.cell_4phase, make_gif, store_frames)

        # --- Debug: Print vortex indices visible to the visualization ---
        test_frame = end_frame//2  # or pick any frame you’re visualizing
        label_map_vis = self.vortex_cores_labeled[:, test_frame]
        unique_vortex_ids = np.unique(label_map_vis[label_map_vis > 0])


        # print('start_frame', start_frame, 'end_frame', end_frame)
        # print('cell_4phase.shape', self.cell_4phase.shape)
        # print('vortex_cores_labeled.shape', self.vortex_cores_labeled.shape)
        #UG] Frame {test_frame}")
        for vid in unique_vortex_ids:
            indices = np.where(label_map_vis == vid)[0]
            print(f"  Vortex ID {vid}: {len(indices)} cells → indices {indices.tolist()}")
        print("----------------------------------------------------------\n")


        self.img = show_cells(folderName_for_frames, self.tmax - start_frame,
                      self.cell_hist[:, :, start_frame:end_frame],
                      self.positions, self.idx_celltype, frame_rate,
                      spin_vect,
                      self.vortex_cores_labeled[:, start_frame:end_frame],
                      self.charges[start_frame:end_frame, :],
                      self.cell_4phase[:, start_frame:end_frame],
                      make_gif, store_frames)


        for vid in unique_vortex_ids:
            indices = np.where(label_map_vis == vid)[0]
            print(f"  Vortex ID {vid}: {len(indices)} cells → indices {indices.tolist()}")
        print("----------------------------------------------------------\n")

    def make_gif(self, frame_start, frame_end, frame_rate, file_name):
        """
        Generate a GIF of the trajectory from the simulation.

        Args:
            frame_start (int): The starting frame for the GIF.
            frame_end (int): The ending frame for the GIF.
            frame_rate (int): Frame rate for the GIF.
            file_name (str): Name of the output GIF file.
        """
        img = self.img  # Get the list of images for the frames
        location = fr'{file_name}.gif'  # Define the file path for the GIF

        # Print start and end frame numbers (for debugging purposes)
        print(frame_start)
        print(frame_end)

        # Save the images as a GIF
        imageio.mimsave(location, img[frame_start:frame_end], fps=frame_rate)

    def show_single_frame(self, folder_for_specificFrame, frame_to_show, store_frames, spin_vect):

        # Ensure directories exist for storing frames if required
        if store_frames:
            frame_dir = folder_for_specificFrame
            jpg_dir = os.path.join(frame_dir, 'Frames_JPG')
            pdf_dir = os.path.join(frame_dir, 'Frames_PDF')
            os.makedirs(jpg_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)


        # Initialize a list to store images for each frame
        img = [None] * 1
        # g_vector = 2  # Gain vector for plotting adjustments
        gain = 0.01

        # Set up the plot background color and layout
        bg_brightness = 240


        # Interactive mode enabled on Matplotlib, which causes the figure to update and close automatically
        plt.ion()


        fig, ax = plt.subplots()
        ax.set_facecolor((bg_brightness / 255, bg_brightness / 255, bg_brightness / 255))
        fig.tight_layout()
        plt.xticks([])  # Remove x-ticks
        plt.yticks([])  # Remove y-ticks
        plt.gca().axes.get_xaxis().set_visible(False)  # Hide x-axis
        plt.gca().axes.get_yaxis().set_visible(False)  # Hide y-axis

        plt.title(f'Time step={frame_to_show + 1}')

        # Set initial state and parameters
        delta = 0.05  # Margin for plot limits
        cells = self.cell_hist[:, :, frame_to_show]  # Initial cell states --- cell_hist is 3D array of the form, (n_cells, n_properties, n_frames)
        gain = 10000 / np.sqrt(len(self.cell_hist[:, 0]))  # Gain for marker size adjustment;  here 'len(...) yields the number of rows of sliced array (= number of cells)


        '''
        # Note about above:
        self.cell_hist[:, 0] means "select all rows (:) from the first dimension (cells) and only the first column (0) from the second dimension (properties)." 
        As a result, this slice returns a 2D array of shape (n_cells, n_frames).
        
        Applying len(): len(self.cell_hist[:, 0]) returns the number of rows in the resulting 2D array, which is equivalent to n_cells.
        
        '''


        # Separate cells by type using the provided index
        idx_celltype1 = self.idx_celltype
        idx_celltype2 = np.invert(self.idx_celltype)

        pos = self.positions
        cell_4phase = self.cell_4phase
        cores_labeled = self.vortex_cores_labeled

        # Initialize color arrays for two cell types
        colours1 = np.asarray([None] * len(pos[idx_celltype1, 0]), dtype=object)
        colours2 = np.asarray([None] * len(pos[idx_celltype2, 0]), dtype=object)
        colours1[:] = 'blue'
        colours2[:] = 'blue'



        # Separate positions by cell type
        pos1 = pos[idx_celltype1, :]
        pos2 = pos[idx_celltype2, :]

        # Plot initial cell states
        if not spin_vect:
            # Scatter plot for two cell types with different markers
            plt_state1_1 = plt.scatter(pos1[:, 0], pos1[:, 1], s=gain, c='blue', marker="o", edgecolors='black')
            plt_state1_2 = plt.scatter(pos2[:, 0], pos2[:, 1], s=gain, c='blue', marker="s", edgecolors='black')
            plt.show()

            # Assign colors based on cell states
            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 0)] = 'blue'
            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 1)] = 'red'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 0)] = 'white'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 1)] = 'black'

            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 0)] = 'blue'
            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 1)] = 'red'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 0)] = 'white'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 1)] = 'black'

            # Update the colors in the plot
            plt_state1_1.set_facecolors(c=colours1.tolist())
            plt_state1_2.set_facecolors(c=colours2.tolist())
        else:
            # Vector field plot for cell phases
            plt.quiver(pos[:, 0], pos[:, 1], np.cos(cell_4phase[:, 0]), np.sin(cell_4phase[:, 0]))
            plt.scatter(pos[:, 0], pos[:, 1], c=cores_labeled[:, 0], edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
            plt.show()

        # Set plot limits based on cell positions
        plt.xlim(0 - delta * 2, np.max(pos[:, 0]) + delta * 2)
        plt.ylim(0 - delta, np.max(pos[:, 1]) + delta)

        fig.canvas.draw()

        if store_frames:
            plt.savefig(os.path.join(jpg_dir, f'step{frame_to_show + 1}.jpg'), format='jpg')  # Save as JPG
            plt.savefig(os.path.join(pdf_dir, f'step{frame_to_show + 1}.pdf'), format='pdf')  # Save as PDF

        # plt.show(block = True)  # Keep the fig window open until you manually close it

    def label_phase_clusters(self, frame, phase_value=None):
        """
        Compute connected-component labels for cells that share the same 4-state at a given global frame.

        Args:
            frame (int): Global frame index to analyze.
            phase_value (int or None): If specified, only clusters of this 4-state value are labeled
                (values are 1..4). If None, clusters are computed separately for each cell value and
                assigned unique labels.

        Returns:
            labels (ndarray): 1D int array of length N. 0 indicates unlabeled (or skipped) cells; positive
                integers indicate cluster id (1..n_clusters).
        """
        if not hasattr(self, 'cell_4state') or self.cell_4state is None or len(self.cell_4state) == 0:
            raise ValueError("cell_4state is empty — run analyse_trajectory() before labeling clusters")

        N = int(self.N)
        if frame < 0 or frame >= self.tmax:
            raise IndexError(f"frame {frame} out of bounds [0, {self.tmax})")

        # Use integer 4-state values for exact equality grouping
        values = self.cell_4state[:, frame].astype(int)

        # neighbor indices are stored in self.cell_number_nn with first column the cell index
        if (not hasattr(self, 'cell_number_nn')) or (self.cell_number_nn is None) or (len(self.cell_number_nn) == 0):
            raise ValueError('cell_number_nn is not initialized — run init_lattice() first')

        nn_cols = self.cell_number_nn[:, 1:1 + int(self.number_of_nn)].astype(int)
        #nn_cols refers to 

        labels = np.zeros(N, dtype=int)  # 0 = unlabeled
        visited = np.zeros(N, dtype=bool)
        next_label = 1

        for i in range(N):
            if visited[i]:
                continue

            # Skip cells that don't match requested phase_value (if provided)
            if (phase_value is not None) and (not np.all(values[i] == phase_value)):

                visited[i] = True
                continue

            # Start BFS/DFS for a new cluster
            stack = [i]
            visited[i] = True
            labels[i] = next_label

            while stack:
                u = stack.pop()
                # iterate neighbors
                for v in nn_cols[u]:
                    # guard against out-of-range or invalid indices
                    if v is None:
                        continue
                    try:
                        v_int = int(v)
                    except Exception:
                        continue
                    if v_int < 0 or v_int >= N:
                        continue
                    if visited[v_int]:
                        continue
                    # Only join cluster when the neighbor has the same 4-state value
                    if np.all(values[v_int] == values[u]):
                        visited[v_int] = True
                        labels[v_int] = next_label
                        stack.append(v_int)
                    else:
                        # mark as visited only if phase_value filtering is active; otherwise leave for other clusters
                        if phase_value is not None:
                            visited[v_int] = True

            next_label += 1

        return labels

    def show_phase_clusters(self, frame, phase_value=None, cmap_name='tab20', save_path=None, marker_size=80, show=True):
        """
        Visualize connected clusters of cells that share the same 4-state at the given frame.

        Args:
            frame (int): Global frame index to show.
            phase_value (int or None): If specified, only clusters with this 4-state are shown.
            cmap_name (str): Matplotlib colormap name used to color different clusters.
            save_path (str or None): If provided, save the figure to this path (png). If None, show interactively.
            marker_size (float): Marker size for scatter.

        Returns:
            labels (ndarray): The cluster labels returned from `label_phase_clusters`.
        """
        labels = self.label_phase_clusters(frame, phase_value=phase_value)

        pos = np.asarray(self.positions)
        # Ensure pos is Nx2 in case positions were stored flattened
        if pos.ndim == 1:
            if pos.size == 2 * self.N:
                pos = pos.reshape((self.N, 2))
            else:
                # try to interpret a list-of-tuples-like structure
                try:
                    pos = np.array(list(self.positions))
                except Exception:
                    raise ValueError('Unexpected positions shape; expected (N,2)')
        # ensure pos is Nx2 (some saved snapshots store flattened or lists)
        if pos.ndim == 1:
            # try to reshape if length matches 2*N
            if pos.size == 2 * self.N:
                pos = pos.reshape((self.N, 2))
            else:
                raise ValueError('Unexpected positions shape; expected (N,2)')
        if pos.shape[0] != self.N:
            raise ValueError('positions array length does not match N')

        # draw unlabeled (0) as light gray, labeled clusters via colormap
        cmap = plt.get_cmap(cmap_name)
        unique_labels = np.unique(labels)

        fig, ax = plt.subplots()
        ax.set_facecolor((240 / 255, 240 / 255, 240 / 255))
        plt.xticks([])
        plt.yticks([])

        # Draw the quiver using the same simple call as `show_cells` so visuals match exactly.
        try:
            # Use the raw phase array so this quiver matches the behavior in show_cells (spin_vect=True)
            cell_4phase = np.asarray(self.cell_4phase)
            if cell_4phase.shape[0] == self.N and frame >= 0 and frame < cell_4phase.shape[1]:
                ph_raw = cell_4phase[:, frame]
                plt.quiver(pos[:, 0], pos[:, 1], np.cos(ph_raw), np.sin(ph_raw))
        except Exception:
            # If phase data isn't available or shaped unexpectedly, skip quiver silently.
            pass

        # plot unlabeled background cells first (draw on top of quiver)
        mask_unlabeled = labels == 0
        if np.any(mask_unlabeled):
            plt.scatter(pos[mask_unlabeled, 0], pos[mask_unlabeled, 1], c='lightgray', edgecolors='black', s=marker_size, zorder=3)

        # plot each cluster with a different color
        label_list = [l for l in unique_labels if l != 0]
        for i, lab in enumerate(label_list):
            mask = labels == lab
            color = cmap(i % cmap.N)
            # when passing a single color, don't pass cmap to avoid matplotlib warnings
            plt.scatter(pos[mask, 0], pos[mask, 1], c=[color], edgecolors='black', s=marker_size, zorder=4)

        plt.title(f'Phase clusters frame {frame}' + (f' (phase={phase_value})' if phase_value is not None else ''))

        # Save and/or show according to arguments. If show=False but save_path is provided, only save.
        if save_path:
            plt.savefig(save_path, dpi=200)

        if show:
            plt.show()
        #else:
            # Close figure to avoid opening interactive window when not requested
            #plt.close(fig)

        return labels

    def get_shortest_cluster_connections(self, frame,  include_zero_charge, count_mode='cells'):
        """
        For a single global frame index, compute the shortest cluster-path connection between every
        pair of vortices present in that frame.

        Returns:
            connections: dict mapping (v1_id, v2_id) -> dict with keys:
                - 'distance': int or None (depending on count_mode)
                - 'path': list of cell indices (if available; otherwise None)
                - 'state': the 4-state id (1..4) of the cluster that provided the shortest path
                - 'frame': the input frame

        Notes:
            - This searches clusters for each 4-state (1..4) and keeps the shortest connection across states.
            - Uses the helper shortest_paths_between_vortices_in_cluster which prefers trajectory.cell_number_nn
              when present.
                        - Implementation note: `shortest_paths_between_vortices_in_cluster` finds at most one path
                            (the shortest) for a given vortex-pair inside a single contiguous cluster. Therefore
                            this function computes per-cluster shortest candidates and then keeps the globally
                            shortest candidate per vortex-pair across all clusters/states.
        """
        #print(f"DEBUG get_shortest_cluster_connections(): count_mode={count_mode!r}")

        if count_mode not in ('edges','cells','intermediate'):
            print(f"⚠️ Invalid count_mode '{count_mode}', resetting to 'cells', get shortest cluster")
            count_mode = 'cells'

        if not hasattr(self, 'cell_4state') or self.cell_4state is None or len(self.cell_4state) == 0:
            raise ValueError('call analyse_trajectory() before requesting connections')

        frame_states = self.cell_4state[:, frame]
        # frame_vortices should be a length-N array giving vortex id for each cell at this frame.
        # try several fallbacks for compatibility with variations in how vortex arrays are stored
        frame_vortices = None
        try:
            if hasattr(self, 'vortex_cores') and isinstance(self.vortex_cores, np.ndarray):
                # expected shape (N, tmax)
                if self.vortex_cores.ndim == 2 and frame < self.vortex_cores.shape[1]:
                    frame_vortices = self.vortex_cores[:, frame]
        except Exception:
            frame_vortices = None

        if frame_vortices is None:
            try:
                if hasattr(self, 'vortex_cores_labeled') and isinstance(self.vortex_cores_labeled, np.ndarray):
                    frame_vortices = self.vortex_cores_labeled[:, frame]
            except Exception:
                frame_vortices = None

        if frame_vortices is None:
            # final fallback: no per-cell vortex map available — use zeros
            frame_vortices = np.zeros(int(self.N), dtype=int)

        # collect candidate vortex ids
        vortex_ids = np.unique(self.vortex_cores_labeled[:, frame])
        vortex_ids = vortex_ids[vortex_ids > 0]


        # --- ADDED 10/15: Handle vortex charges and optional zero-charge filtering ---
        vortex_charges = {}
        #the ultimate goal of this block is to make a charge dictionary to use easily later
        if hasattr(self, 'charges') and self.charges is not None:
            charges_arr = np.array(self.charges)
            if charges_arr.ndim == 2 and frame < charges_arr.shape[0]:
                frame_charges = charges_arr[int(frame), 2:]
                for vid in vortex_ids:
                    idx = int(vid) - 1  # because vortex IDs start at 1
                    if 0 <= idx < frame_charges.shape[0]:
                        vortex_charges[int(vid)] = frame_charges[idx]
            if not include_zero_charge:
                charged_vortex_ids = [vid for vid in vortex_ids if vortex_charges.get(vid, 0) != 0]
                vortex_ids = np.array(charged_vortex_ids, dtype=int)


        results = {}

        # for each state, compute cluster labels and evaluate distances
        for state in [1, 2, 3, 4]:
            # label clusters of this state
            try:
                cluster_labels = self.label_phase_clusters(frame, phase_value=state)
            except Exception:
                # fallback to older labeler
                mask = (frame_states == state).astype(int).reshape(-1, 1)
                labeled, _ = label_elements_triangular_lattice(mask, self.relative_distance, 0, 1)
                cluster_labels = labeled[:, 0]

            for cid in np.unique(cluster_labels):
                if cid == 0:
                    continue
                cluster_cells = np.where(cluster_labels == cid)[0]
                # expand to include vortex core CELLS whose neighbours touch the cluster
                # Use the per-cell vortex map when available to robustly find core cell indices
                expanded = set(cluster_cells.tolist())
                if hasattr(self, 'vortex_cores_labeled') and isinstance(self.vortex_cores_labeled, np.ndarray):
                    vortex_map = self.vortex_cores_labeled[:, frame]
                else:
                    vortex_map = frame_vortices

                unique_vids = np.unique(vortex_map)
                for vid in unique_vids:
                    if vid == 0:
                        continue
                    # get all cell indices belonging to this vortex id
                    core_cells = np.where(vortex_map == vid)[0]
                    # collect neighbours of these core cells and test intersection with cluster
                    nn_union = []
                    for core in core_cells:
                        try:
                            nn = self.cell_number_nn[int(core), 1:1 + int(self.number_of_nn)].astype(int)
                        except Exception:
                            nn = np.where(np.round(self.relative_distance[int(core), :], 1) == 1)[0]
                        if nn is not None and len(nn) > 0:
                            nn_union.extend([int(x) for x in nn])

                    if np.intersect1d(np.unique(np.array(nn_union, dtype=int)), cluster_cells).size > 0:
                        # add the actual core cell indices to the expanded cluster
                        for c in core_cells:
                            expanded.add(int(c))

                expanded_arr = np.array(list(expanded), dtype=int)

                # compute shortest distances inside this expanded cluster
                dists = shortest_paths_between_vortices_in_cluster(expanded_arr, self.vortex_cores_labeled[:, frame], trajectory=self, return_paths=True, count_mode=count_mode, frame=frame, include_zero_charge=include_zero_charge)
                # dists maps (v1,v2) -> (distance, path)
                for key, val in dists.items():
                    if val is None:
                        continue
                    if return_paths := (isinstance(val, tuple) and len(val) == 2):
                        dist_val, path = val
                    else:
                        dist_val = val
                        path = None


                    #ADDED 10/15----------------------------------------------------------------
                    v1, v2 = key
                    if not include_zero_charge:
                        q1 = vortex_charges.get(int(v1), None)
                        q2 = vortex_charges.get(int(v2), None)
                        if q1 is None or q2 is None or q1 == 0 or q2 == 0 or np.sign(q1) == np.sign(q2):
                            continue  # skip same-sign or zero-charge pairs
                    # end ADDED 10/15----------------------------------------------------------------


                    # keep shortest across states/clusters
                    existing = results.get(key)
                    if existing is None or (dist_val is not None and (existing['distance'] is None or dist_val < existing['distance'])):
                        results[key] = {'distance': dist_val, 'path': path, 'state': state, 'frame': frame}

        return results
    


    def dedupe_candidates(self, all_candidates):
        deduped = {}

        for pair, cands in all_candidates.items():
            seen = {}
            for c in cands:
                # path identity: order-independent cell set
                path_key = tuple(sorted(c['path'])) if c['path'] is not None else None

                best = seen.get(path_key)
                if best is None or c['distance'] < best['distance']:
                    seen[path_key] = c

            deduped[pair] = list(seen.values())

        return deduped




    def get_all_cluster_connections(self, frame,  include_zero_charge, count_mode='cells', return_all=False):
        """
        Like `get_shortest_cluster_connections` but returns every candidate string found between
        all vortex pairs (from every cluster/state). By default this returns the single shortest
        string per pair (same format as `get_shortest_cluster_connections`). If `return_all=True`
        the function returns a tuple `(best_results, all_candidates)` where `all_candidates` is
        a dict mapping (v1,v2) -> list of candidate dicts with keys: 'distance','path','state','cluster'.

                Clarification:
                        - The internal helper `shortest_paths_between_vortices_in_cluster` computes the shortest
                            path (if any) for a given vortex-pair inside a single contiguous cluster. That means
                            for each cluster this helper yields at most one candidate path per vortex-pair.
                        - `get_all_cluster_connections` calls that helper for every cluster (and for each state),
                            aggregates the per-cluster candidates into `all_candidates` (so a vortex-pair may have
                            multiple candidates coming from different clusters or states), and builds `best_results`
                            by selecting the globally shortest candidate per vortex-pair across all clusters/states.

        Args:
            frame (int): global frame index
            count_mode (str): 'edges'|'cells'|'intermediate'
            include_zero_charge (bool): whether to include zero-charge vortices
            return_all (bool): if True, also return all candidate strings per pair

        Returns:
            best_results or (best_results, all_candidates)
        """

        # print(f"[DEBUG CANT] unique(self.vortex_cores_labeled[:, {frame}]) →",
        # np.unique(self.vortex_cores_labeled[:, frame]))
        # print(f"[DEBUG GACC] np.sum(vortex_map==1) = {np.sum(self.vortex_cores_labeled[:, frame]==1)}")
        # print(f"[DEBUGCGAC] np.sum(vortex_map==2) = {np.sum(self.vortex_cores_labeled[:, frame]==2)}")
        # print(f"[DEBUGGACT] np.sum(vortex_map==3) = {np.sum(self.vortex_cores_labeled[:, frame]==3)}")
        # print(f"[DEBUGCGAC] np.sum(vortex_map==4) = {np.sum(self.vortex_cores_labeled[:, frame]==4)}")
        if count_mode not in ('edges','cells','intermediate'):
            print(f"⚠️ Invalid count_mode '{count_mode}', resetting to 'cells', get cluster connections")
            count_mode = 'cells'
        # Ensure analysis has been run
        if not hasattr(self, 'cell_4state') or self.cell_4state is None or len(self.cell_4state) == 0:
            raise ValueError('call analyse_trajectory() before requesting connections')

        frame_states = self.cell_4state[:, frame]

        # try to build per-cell vortex map (fallbacks as in get_shortest_cluster_connections)
        frame_vortices = None
        try:
            if hasattr(self, 'vortex_cores') and isinstance(self.vortex_cores, np.ndarray):
                if self.vortex_cores.ndim == 2 and frame < self.vortex_cores.shape[1]:
                    frame_vortices = self.vortex_cores[:, frame]
        except Exception:
            frame_vortices = None

        if frame_vortices is None:
            try:
                if hasattr(self, 'vortex_cores_labeled') and isinstance(self.vortex_cores_labeled, np.ndarray):
                    frame_vortices = self.vortex_cores_labeled[:, frame]
            except Exception:
                frame_vortices = None

        if frame_vortices is None:
            frame_vortices = np.zeros(int(self.N), dtype=int)

        best_results = {}
        all_candidates = {}

        # start ADDED 10/15-------------------------------------------------------------------------

        vortex_map = self.vortex_cores_labeled[:, frame]
        vortex_ids = np.unique(vortex_map[vortex_map > 0])
        vortex_charges = {}

        if hasattr(self, 'charges') and self.charges is not None:
            charges_arr = np.array(self.charges)
            if charges_arr.ndim == 2 and frame < charges_arr.shape[0]:
                frame_charges = charges_arr[int(frame), 2:]
                for vid in vortex_ids:
                    #vortex ids start at 1 so we need to subtract one so its starts at 0
                    idx = int(vid) - 1
                    if 0 <= idx < frame_charges.shape[0]:
                        vortex_charges[int(vid)] = frame_charges[idx]
                        #now we have a dictionary of charges

            if not include_zero_charge:
                charged_vortex_ids = [vid for vid in vortex_ids if vortex_charges.get(vid, 0) != 0]
                vortex_ids = np.array(charged_vortex_ids, dtype=int)
        # start ADDED 10/15-------------------------------------------------------------------------

        # iterate over states and clusters similarly to get_shortest_cluster_connections
        for state in [1, 2, 3, 4]:
            try:
                cluster_labels = self.label_phase_clusters(frame, phase_value=state)
            except Exception:
                mask = (frame_states == state).astype(int).reshape(-1, 1)
                labeled, _ = label_elements_triangular_lattice(mask, self.relative_distance, 0, 1)
                cluster_labels = labeled[:, 0]

            for cid in np.unique(cluster_labels):
                if cid == 0:
                    continue
                cluster_cells = np.where(cluster_labels == cid)[0]

                # expand to include vortex core CELLS whose neighbours touch the cluster
                expanded = set(cluster_cells.tolist())
                if hasattr(self, 'vortex_cores_labeled') and isinstance(self.vortex_cores_labeled, np.ndarray):
                    vortex_map = self.vortex_cores_labeled[:, frame]
                    # Respect the object's quiet_warnings flag to suppress debug prints when desired
                    if not getattr(self, 'quiet_warnings', False):
                        print(f"\n[DEBUG] Frame {frame} vortex indices from get_all_cluster_connections:")
                        vortex_map = np.array(vortex_map)
                        unique_vids = np.unique(vortex_map[vortex_map > 0])
                        for vid in unique_vids:
                            cell_ids = np.where(vortex_map == vid)[0]
                            print(f"  Vortex {vid}: {len(cell_ids)} cells → {cell_ids.tolist()}")
                        #print("----------------------------------------------------------")
                        # --- DEBUG: show vortex map contents ---
                        unique_vals = np.unique(vortex_map[vortex_map > 0])
                        #print(f"[DEBUG GET ALL CLUSTER CONNECT] Frame {frame}: vortex_map has vortex IDs {unique_vals}")
                        for vid in unique_vals:
                            cells = np.where(vortex_map == vid)[0]
                            print(f"  vortex {vid}: {cells.tolist()}")
                        print("----------------------------------------------------------")
                    else:
                        vortex_map = np.array(vortex_map)
                else:
                    vortex_map = frame_vortices

                unique_vids = np.unique(vortex_map)
                for vid in unique_vids:
                    if vid == 0:
                        continue
                    core_cells = np.where(vortex_map == vid)[0]
                    nn_union = []
                    for core in core_cells:
                        try:
                            nn = self.cell_number_nn[int(core), 1:1 + int(self.number_of_nn)].astype(int)
                        except Exception:
                            nn = np.where(np.round(self.relative_distance[int(core), :], 1) == 1)[0]
                        if nn is not None and len(nn) > 0:
                            nn_union.extend([int(x) for x in nn])

                    if np.intersect1d(np.unique(np.array(nn_union, dtype=int)), cluster_cells).size > 0:
                        for c in core_cells:
                            expanded.add(int(c))

                expanded_arr = np.array(list(expanded), dtype=int)

                # get candidate distances/paths inside this expanded cluster
                dists = shortest_paths_between_vortices_in_cluster(expanded_arr, self.vortex_cores_labeled[:, frame], trajectory=self, return_paths=True, count_mode=count_mode, frame=frame, include_zero_charge=include_zero_charge)

                # collect candidates
                for key, val in dists.items():
                    if val is None:
                        continue
                    if isinstance(val, tuple) and len(val) == 2:
                        dist_val, path = val
                    else:
                        dist_val = val
                        path = None

                    #start ADDED 10/15--------------------------------------------------------------
                    v1, v2 = key  # ⬅️ added: explicitly unpack the pair
                    if not include_zero_charge:
                        q1 = vortex_charges.get(int(v1),None) #default None if cant find charge info (should never happen)
                        q2 = vortex_charges.get(int(v2),None)
                        if q1 is None or q2 is None or q1 == 0 or q2 == 0 or np.sign(q1) == np.sign(q2):
                            continue
                    #end ADDED 10/15--------------------------------------------------------------

                    # prepare candidate record
                    cand = {'pair': (v1, v2),  'distance': dist_val, 'path': path, 'state': state, 'cluster': int(cid), 'frame': frame}
                    pair_key = tuple(sorted((v1, v2)))
                    all_candidates.setdefault(pair_key, []).append(cand)
                    # update best_results if this candidate is better
                    existing = best_results.get(key)
                    if existing is None:
                        best_results[key] = {'distance': dist_val, 'path': path, 'state': state, 'cluster': int(cid), 'frame': frame}
                    else:
                        # if either distance is None treat as worse; else keep smaller
                        exd = existing.get('distance')
                        if dist_val is not None and (exd is None or dist_val < exd):
                            best_results[key] = {'distance': dist_val, 'path': path, 'state': state, 'cluster': int(cid), 'frame': frame}

        if return_all:
            all_candidates = self.dedupe_candidates(all_candidates)
            return best_results, all_candidates
        return best_results
    

    
    def track_shortest_cluster_lengths(self, start_frame, end_frame, include_zero_charge, count_mode='edges'):
        """
        Track shortest cluster connection lengths between vortex pairs over a frame range.

        Args:
            start_frame (int): inclusive start (global frame index)
            end_frame (int): exclusive end (global frame index)
            count_mode (str): 'edges'|'cells'|'intermediate' same as shortest_paths_between_vortices_in_cluster

        Returns:
            time_series: dict mapping (v1_id, v2_id) -> dict with keys:
                - 'frames': list of frame indices
                - 'distances': list of distances (int or None) aligned with 'frames'
                - 'paths': list of paths (or None) aligned with 'frames'

        This function makes it easy to plot the string length between two vortices over time.
        """

        if count_mode not in ('edges','cells','intermediate'):
            print(f"⚠️ Invalid count_mode '{count_mode}', resetting to 'cells', shortest_cluster_length")
            count_mode = 'cells'

        time_series = {}

        for fn in range(start_frame, end_frame):
            conns = self.get_shortest_cluster_connections(fn, count_mode=count_mode, include_zero_charge=include_zero_charge)
            for pair, info in conns.items():
                rec = time_series.get(pair)
                if rec is None:
                    rec = {'frames': [], 'distances': [], 'paths': []}
                    time_series[pair] = rec
                rec['frames'].append(fn)
                rec['distances'].append(info.get('distance'))
                rec['paths'].append(info.get('path'))

        return time_series

    def find_last_two_vortices(self,include_zero_charge, reference_frame=None, lookback=50):
        """
        Return a tuple (v1, v2) of vortex labels to track.

        Behavior:
        - If `require_charged=True`, only return pairs where both vortices have nonzero charge.
        - If fewer than 2 (charged) vortices exist at the reference frame, search backward up to `lookback` frames.
        - If no suitable pair is found, returns None.

        Args:
            reference_frame (int or None): Frame to inspect. Defaults to `self.first_recurrent_state_time`.
            lookback (int): Number of frames to search backward if fewer than 2 vortices are found.
            include_zero_charge (bool): Whether to restrict to charged vortices only.

        Returns:
            (v1, v2) tuple or None
        """
        if reference_frame is None:
            reference_frame = int(self.first_recurrent_state_time )
            #I changed to first reccurent state time rather than recurent state time -1 because maybe state time-1 could still have 2 vortices
            #I might change this later

        # ensure frame in range
        reference_frame = max(0, min(reference_frame, int(self.tmax - 1)))
        reference_frame=int(reference_frame)

        # search backward until we find >=2 vortex labels or we hit lookback limit
        frames_checked = 0
        for fn in range(reference_frame, max(-1, reference_frame - lookback) , -1):
            #go from the reference frame backwards to find the last frame with at least 2 vortices
            labels = np.unique(self.vortex_cores_labeled[:, fn]) if hasattr(self, 'vortex_cores_labeled') and self.vortex_cores_labeled is not None else np.array([])
            # access vortex labels at this frame
            labels = labels[labels > 0]
            #only keep vortex labels > 0
            #confirm that this corresponds to the vortex labels at this frame


           
            if labels.size >= 2:

                if not include_zero_charge and hasattr(self, 'charges') and self.charges is not None:
                # Keep only vortices with nonzero charge at this frame
                    frame_charges = self.charges[fn, 2:]  # skip first two summary cols
                    charged_labels = []
                    for lbl in labels:
                        if lbl - 1 < len(frame_charges):
                            if frame_charges[lbl - 1] != 0:
                                charged_labels.append(lbl)
                    labels = np.array(charged_labels)



                # --- ADDED 10/15: build charge dictionary for opposite-sign filtering ---
                vortex_charges = {}   # --- created charge dictionary which is easy to use later
                if hasattr(self, 'charges') and self.charges is not None:   
                    charges_arr = np.array(self.charges)                    
                    if charges_arr.ndim == 2 and fn < charges_arr.shape[0]: 
                        frame_charges = charges_arr[int(fn), 2:]            
                        for lbl in labels:                                  
                            idx = int(lbl) - 1                              
                            if 0 <= idx < len(frame_charges):               
                                vortex_charges[int(lbl)] = frame_charges[idx]  
                # --- END ADDED BLOCK ---

                if labels.size >= 2:
                     # --- ADDED 10/15: same-sign exclusion logic ---
                    if not include_zero_charge:                          
                        found_pair = None                                  
                        for i, v1 in enumerate(labels):                     
                            for v2 in labels[i + 1:]:                       
                                q1 = vortex_charges.get(int(v1), None)      
                                q2 = vortex_charges.get(int(v2), None)     
                                if q1 is None or q2 is None:                
                                    continue                               
                                if q1 == 0 or q2 == 0:                   
                                    continue                                
                                if np.sign(q1) == np.sign(q2):             
                                    continue                               
                                found_pair = (int(v1), int(v2))          
                                break                                     
                            if found_pair:                             
                                break                                 
                        if found_pair:                                  
                            return found_pair                       
                        else:                                              
                            continue                                        
                    # --- END ADDED BLOCK ---
                    labels_sorted = np.sort(labels)
                    v1 = int(labels_sorted[-2])
                    v2 = int(labels_sorted[-1])
                    return (v1, v2)

                # choose two largest numeric labels to keep identity stable
                #when a vortex annihilates, all the labels > that vortex get decremented by 1
                #so picking the largest two labels means we are less likely to pick a vortex that will annihilate soon
                
            frames_checked += 1

        return None

    def track_pair_over_time(self,  include_zero_charge, pair=None, start_frame=0, end_frame=None, count_mode='cells'):
        """
        Track the string length between a fixed pair of vortex labels across frames.

        Args:
            pair (tuple or None): (v1_label, v2_label). If None, `find_last_two_vortices` will be used
                with reference_frame=end_frame-1 to pick a pair.
            start_frame (int): inclusive start frame
            end_frame (int): exclusive end frame. Defaults to self.tmax if not provided.
            count_mode (str): 'cells'|'edges'|'intermediate'
            include_zero_charge (bool): whether to include vortices with zero charge in analysis

        Returns:
            dict with keys: 'pair' (tuple), 'frames' (list), 'distances' (list), 'paths' (list)
            distances/paths are aligned with frames; if no connection present for a frame, distance/path is None.
        """

        if count_mode not in ('cells', 'edges', 'intermediate'):
            print(f"⚠️ Warning: invalid count_mode='{count_mode}' passed, resetting to 'cells',track pair over time")
            count_mode = 'cells'
        if end_frame is None:
            end_frame = int(self.first_recurrent_state_time)
            #changed to first reccurent state time because tmax is arbitrary and might include frames after vortices annihilate

        if pair is None:
            # pick pair from last available frame in range
            ref = end_frame - 1
            pair = self.find_last_two_vortices(include_zero_charge=include_zero_charge, reference_frame=ref)
            if pair is None:
                raise ValueError('Could not determine a pair of last two vortices to track')

        v1, v2 = int(pair[0]), int(pair[1])

        frames = list(range(int(start_frame), int(end_frame)))
        # just generate a list of frames to analyze
        distances = []
        paths = []

        for fn in frames:
            # get all connections for this frame but keep pair fixed
            conns = self.get_shortest_cluster_connections(fn, count_mode=count_mode, include_zero_charge=include_zero_charge)
            # conns is a dict mapping (v1_id, v2_id) -> dict with keys: 'distance', 'path', 'state', 'frame' for all pairs
            # we only want the fixed pair (v1, v2) or (v2, v1)
            info = None
            if conns and (v1, v2) in conns:
                info = conns.get((v1, v2))
            elif conns and (v2, v1) in conns:
                info = conns.get((v2, v1))

            if info is None:
                distances.append(None)
                paths.append(None)
            else:
                distances.append(info.get('distance'))
                paths.append(info.get('path'))

        return {'pair': (v1, v2), 'frames': frames, 'distances': distances, 'paths': paths}

    def show_quiver_with_string(self, frame, include_zero_charge, pair=None, count_mode='cells', save_path=None, show=True, arrow_scale=1.0, marker_size=80):
        """
        Draw a quiver plot for `frame` but only color the vortex cores and the shortest connecting string between them.

        Args:
            frame (int): global frame index
            pair (tuple or None): (v1_id, v2_id). If None, the function will pick the shortest available pair at this frame.
            count_mode (str): passed to shortest-path routine ('edges'|'cells'|'intermediate')
            save_path (str or None): path to save PNG. If None and show=True, displays interactively.
            show (bool): whether to call plt.show(); if False and save_path provided, figure will be closed after saving.
            arrow_scale (float): multiplier applied to quiver scale to control arrow length
            marker_size (float): size for highlighted markers

        Returns:
            info dict for the chosen pair (same structure returned by get_shortest_cluster_connections) or None if no connection.
        """
        # Ensure trajectory analysis has been run
        if not hasattr(self, 'cell_4phase') or self.cell_4phase is None:
            raise ValueError('call analyse_trajectory() before plotting')

        # Get connections for this frame
        conns = self.get_shortest_cluster_connections(frame, count_mode=count_mode, include_zero_charge=include_zero_charge)
        #conns is a dict mapping (v1_id, v2_id) -> dict with keys: 'distance', 'path', 'state', 'frame'
        if not conns:
            # If no connections exist, draw the exact same layers as the connected case
            # (base quiver, white circular scatter for all cells, and colored vortex cores),
            # but do not draw any colored path/string overlays.
            if not self.quiet_warnings:
                print('No connections found at this frame — saving base quiver with cores')

            pos = np.asarray(self.positions)

            fig, ax = plt.subplots()
            # Match background and layout style used by show_cells
            bg_brightness = 240
            ax.set_facecolor((bg_brightness / 255, bg_brightness / 255, bg_brightness / 255))
            fig.tight_layout()
            plt.xticks([])
            plt.yticks([])
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            # draw light quiver for the whole field using raw phases (same as connected branch)
            try:
                cell_4phase = np.asarray(self.cell_4phase)
                if cell_4phase.shape[0] == self.N and frame >= 0 and frame < cell_4phase.shape[1]:
                    #if phase data is available and shaped as expected
                    ph_raw = cell_4phase[:, frame]
                    # draw quiver using raw phases so appearance matches show_cells
                    U = np.cos(ph_raw)
                    V = np.sin(ph_raw)
                    plt.quiver(pos[:, 0], pos[:, 1], U, V)

            except Exception:
                # If phase data isn't available or shaped unexpectedly, skip drawing
                pass

            # Draw canonical scatter overlay (white circles with black edge)
            plt.scatter(pos[:, 0], pos[:, 1], c='white', edgecolors='black', s=marker_size, zorder=5)

            # Draw colored circle markers for vortex cores (no path markers)
            unique_labels = np.unique(self.vortex_cores_labeled[:, frame]) if hasattr(self, 'vortex_cores_labeled') and self.vortex_cores_labeled is not None else []
            #the above line handles the case where vortex_cores_labeled is not yet computed
            cmap = plt.get_cmap('tab10')
            for idx, lab in enumerate(unique_labels):
                if lab == 0:
                    #if label is 0, skip (0 means no core)
                    continue
                cells_lab = np.where(self.vortex_cores_labeled[:, frame] == lab)[0]# get indices of cells in this core

                if cells_lab.size > 0:
                    # draw core cells with distinct color
                    color = cmap(idx % cmap.N)
                    # when passing a single color, don't pass cmap to avoid matplotlib warnings
                    plt.scatter(pos[cells_lab, 0], pos[cells_lab, 1], c=[color], edgecolors='black', s=marker_size * 1.5, zorder=8)
                    # scatter showing core cells with larger size

            plt.title(f'Frame {frame} - no connections')
            if save_path:
                plt.savefig(save_path, dpi=200)
            if show:
                plt.show()
            #else:
                #plt.close(fig)

            return None

        # Choose pair
        if pair is None:
            #if I didnt pick a specific pair,
            # pick pair with smallest non-None distance
            best = None
            bestd = None
            for k, v in conns.items():
                #k is (v1_id, v2_id), v is dict with 'distance', 'path', etc
                d = v.get('distance')
                # d is the distance for this pair
                if d is None:
                    continue
                if bestd is None or d < bestd:
                    bestd = d
                    best = k
            if best is None:
                # nothing with a numeric distance; pick arbitrary
                pair = next(iter(conns.keys()))
            else:
                pair = best

        info = conns.get(pair)
        if info is None:
            print(f'No connection information for pair {pair} at frame {frame}')
            return None

        # Prepare plot
        pos = np.asarray(self.positions)
        ph = self.cell_4phase[:, frame]

        fig, ax = plt.subplots()
        # Match background and layout style used by show_cells
        bg_brightness = 240
        ax.set_facecolor((bg_brightness / 255, bg_brightness / 255, bg_brightness / 255))
        fig.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        # draw light quiver for the whole field using the same simple call as `show_cells` so visuals match exactly
        cell_4phase = np.asarray(self.cell_4phase)
        if not (cell_4phase.shape[0] == self.N and frame >= 0 and frame < cell_4phase.shape[1]):
            # If phase data isn't available or shaped unexpectedly, skip drawing
            if save_path:
                plt.savefig(save_path, dpi=200)
            if show:
                plt.show()
            #else:
                #plt.close(fig)
            return None

        ph_raw = cell_4phase[:, frame]
        U = np.cos(ph_raw)
        V = np.sin(ph_raw)

        # Base quiver (no color override) — draw first so appearance matches show_cells
        base_q = plt.quiver(pos[:, 0], pos[:, 1], U, V)

        # Draw canonical scatter overlay (same visual as other quiver plots)
        # Use a fixed fill color with black edges to match show_cells style
        plt.scatter(pos[:, 0], pos[:, 1], c='white', edgecolors='black', s=marker_size, zorder=5)

        # If there are any pairs to draw, create an overlay quiver that uses the same
        # scale/angles as the base quiver but with per-arrow colors (transparent for
        # non-path cells, colored for path/core cells). This guarantees exact alignment
        # with the base quiver while only changing color where requested.
        pairs_to_draw = list(conns.keys()) if pair is None else [pair]
        cmap = plt.get_cmap('tab10')

        # default transparent RGBA
        transparent = (0, 0, 0, 0)
        colors_list = [transparent] * len(pos)

        # collect all highlighted indices (cores + path cells) and assign colors
        for idx, pr in enumerate(pairs_to_draw):
            info_pr = conns.get(pr)
            if info_pr is None:
                continue
            # path color and core color chosen to be distinct (core uses next cmap index)
            path_color = cmap(idx % cmap.N)
            core_color = cmap((idx + 1) % cmap.N)
            path = info_pr.get('path')
            v1, v2 = pr
            cells_v1 = np.where(self.vortex_cores_labeled[:, frame] == v1)[0]
            cells_v2 = np.where(self.vortex_cores_labeled[:, frame] == v2)[0]
            # assign core colors first (so they override path if overlap)
            for c in cells_v1.tolist() + cells_v2.tolist():
                colors_list[int(c)] = core_color
            if path is not None:
                for c in np.array(path, dtype=int).tolist():
                    # only set path color if not already a core (so core keeps its distinct color)
                    if colors_list[int(c)] == transparent:
                        colors_list[int(c)] = path_color

        # Try to extract base quiver scaling parameters to match overlay exactly
        try:
            base_scale = getattr(base_q, 'scale', None)
            base_angles = getattr(base_q, 'angles', 'uv')
            base_scale_units = getattr(base_q, 'scale_units', None)
        except Exception:
            base_scale = None
            base_angles = 'uv'
            base_scale_units = None

        # draw overlay quiver with per-arrow colors (transparent for non-highlighted cells)
        try:
            if base_scale is not None and base_scale_units is not None:
                plt.quiver(pos[:, 0], pos[:, 1], U, V, color=colors_list, angles=base_angles, scale=base_scale, scale_units=base_scale_units, zorder=7)
            elif base_scale is not None:
                plt.quiver(pos[:, 0], pos[:, 1], U, V, color=colors_list, angles=base_angles, scale=base_scale, zorder=7)
            else:
                plt.quiver(pos[:, 0], pos[:, 1], U, V, color=colors_list, angles=base_angles, zorder=7)
        except Exception:
            # fallback to simple overlay quiver if matching parameters fail
            plt.quiver(pos[:, 0], pos[:, 1], U, V, color=colors_list, zorder=7)

        # Finally, draw colored circle markers for vortex cores and path cells so that
        # the only visual difference is color (shapes remain identical)
        for idx, pr in enumerate(pairs_to_draw):
            info_pr = conns.get(pr)
            if info_pr is None:
                continue
            path_color = cmap(idx % cmap.N)
            core_color = cmap((idx + 1) % cmap.N)
            path = info_pr.get('path')
            v1, v2 = pr
            cells_v1 = np.where(self.vortex_cores_labeled[:, frame] == v1)[0]
            cells_v2 = np.where(self.vortex_cores_labeled[:, frame] == v2)[0]
            if cells_v1.size > 0:
                plt.scatter(pos[cells_v1, 0], pos[cells_v1, 1], c=[core_color], edgecolors='black', s=marker_size, zorder=8)
            if cells_v2.size > 0:
                plt.scatter(pos[cells_v2, 0], pos[cells_v2, 1], c=[core_color], edgecolors='black', s=marker_size, zorder=8)
            if path is not None:
                path = np.array(path, dtype=int)
                plt.scatter(pos[path, 0], pos[path, 1], c=[path_color], edgecolors='black', s=marker_size, zorder=8)

        # Title summarizing number of pairs
        plt.title(f'Frame {frame} - pairs drawn: {len(pairs_to_draw)}')

        if save_path:
            plt.savefig(save_path, dpi=200)

        if show:
            plt.show()
        #else:
            #plt.close(fig)

        # If a single pair was requested, return its info; else return the full conns dict
        return conns if pair is None else conns.get(pair)

    def save_single_frame(self, fileName, parent_folder, jpg_folder, pdf_folder, frame_to_save, spin_vect):

        # Ensure directories exist for storing frames if required
        jpg_dir = os.path.join(parent_folder, jpg_folder)
        pdf_dir = os.path.join(parent_folder, pdf_folder)
        os.makedirs(jpg_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        # Initialize a list to store images for each frame
        img = [None] * 1
        # g_vector = 2  # Gain vector for plotting adjustments
        gain = 0.01

        # Set up the plot background color and layout
        bg_brightness = 240

        # Interactive mode enabled on Matplotlib, which causes the figure to update and close automatically
        # plt.ion()

        fig, ax = plt.subplots()
        ax.set_facecolor((bg_brightness / 255, bg_brightness / 255, bg_brightness / 255))
        fig.tight_layout()
        plt.xticks([])  # Remove x-ticks
        plt.yticks([])  # Remove y-ticks
        plt.gca().axes.get_xaxis().set_visible(False)  # Hide x-axis
        plt.gca().axes.get_yaxis().set_visible(False)  # Hide y-axis

        plt.title(f'Time step={frame_to_save + 1}')

        # Set initial state and parameters
        delta = 0.05  # Margin for plot limits
        cells = self.cell_hist[:, :,
                frame_to_save]  # Initial cell states --- cell_hist is 3D array of the form, (n_cells, n_properties, n_frames)
        gain = 10000 / np.sqrt(len(self.cell_hist[:,
                                   0]))  # Gain for marker size adjustment;  here 'len(...) yields the number of rows of sliced array (= number of cells)

        '''
        # Note about above:
        self.cell_hist[:, 0] means "select all rows (:) from the first dimension (cells) and only the first column (0) from the second dimension (properties)." 
        As a result, this slice returns a 2D array of shape (n_cells, n_frames).

        Applying len(): len(self.cell_hist[:, 0]) returns the number of rows in the resulting 2D array, which is equivalent to n_cells.

        '''

        # Separate cells by type using the provided index
        idx_celltype1 = self.idx_celltype
        idx_celltype2 = np.invert(self.idx_celltype)

        pos = self.positions
        cell_4phase = self.cell_4phase
        cores_labeled = self.vortex_cores_labeled

        # Initialize color arrays for two cell types
        colours1 = np.asarray([None] * len(pos[idx_celltype1, 0]), dtype=object)
        colours2 = np.asarray([None] * len(pos[idx_celltype2, 0]), dtype=object)
        colours1[:] = 'blue'
        colours2[:] = 'blue'

        # Separate positions by cell type
        pos1 = pos[idx_celltype1, :]
        pos2 = pos[idx_celltype2, :]

        # Plot initial cell states
        if not spin_vect:
            # Scatter plot for two cell types with different markers
            plt_state1_1 = plt.scatter(pos1[:, 0], pos1[:, 1], s=gain, c='blue', marker="o", edgecolors='black')
            plt_state1_2 = plt.scatter(pos2[:, 0], pos2[:, 1], s=gain, c='blue', marker="s", edgecolors='black')
            # plt.show()

            # Assign colors based on cell states
            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 0)] = 'blue'
            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 1)] = 'red'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 0)] = 'white'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 1)] = 'black'

            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 0)] = 'blue'
            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 1)] = 'red'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 0)] = 'white'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 1)] = 'black'

            # Update the colors in the plot
            plt_state1_1.set_facecolors(c=colours1.tolist())
            plt_state1_2.set_facecolors(c=colours2.tolist())
        else:
            # Vector field plot for cell phases
            plt.quiver(pos[:, 0], pos[:, 1], np.cos(cell_4phase[:, 0]), np.sin(cell_4phase[:, 0]))
            plt.scatter(pos[:, 0], pos[:, 1], c=cores_labeled[:, 0], edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
            # plt.show()

        # Set plot limits based on cell positions
        plt.xlim(0 - delta * 2, np.max(pos[:, 0]) + delta * 2)
        plt.ylim(0 - delta, np.max(pos[:, 1]) + delta)

        fig.canvas.draw()

        plt.savefig(os.path.join(jpg_dir, f'{fileName}.jpg'), format='jpg')  # Save as JPG
        plt.savefig(os.path.join(pdf_dir, f'{fileName}.pdf'), format='pdf')  # Save as PDF

        # plt.show(block = True)  # Keep the fig window open until you manually close it

    def find_noncontractible_loops(self, frame, phase_value=None, return_zero_winding=False):
        """
        Find non-contractible loops in the phase clusters at a given frame. No consideration 
        of vortices

                Returns a list of dictionaries, each with keys:
                    - 'nodes': ordered list of cell indices forming the loop (simple cycle)
                    - 'windings': tuple (wx, wy) giving integer winding numbers around the torus
                    - 'wrap_count': integer = max(abs(wx), abs(wy)) how many times it wraps
                    - 'length': number of cells in the loop

                By default the function returns only non-contractible loops (those with non-zero windings).
                If `return_zero_winding=True` it returns a tuple (all_cycles, non_contractible_cycles).
        """
        N = int(self.N)
        # cluster labels for requested phase (or all phases if None)
        cluster_labels = self.label_phase_clusters(frame, phase_value=phase_value)

        # positions and lattice extents (used for torus wrap detection)
        pos = np.asarray(self.positions)
        # If positions are flattened, try to reshape; if missing or malformed, try to
        # reconstruct the lattice positions from the known gridsize.
        try:
            if pos.ndim == 1 and pos.size == 2 * N:
                pos = pos.reshape((N, 2))
        except Exception:
            # leave pos as-is and handle below
            pass

        # If positions are missing or not Nx2, attempt to rebuild using gridsize
        if pos.size == 0 or pos.ndim != 2 or pos.shape[1] != 2:
            try:
                # calculate_triangular_lattice returns (pos, lx, ly)
                pos_calc, lx_calc, ly_calc = calculate_triangular_lattice(self.gridsize)
                pos = np.asarray(pos_calc)
                lx = lx_calc
                ly = ly_calc
            except Exception:
                # As a final fallback, try to coerce a list-of-tuples into an array
                try:
                    pos = np.array(list(self.positions))
                except Exception:
                    raise ValueError('Unexpected positions shape; expected (N,2)')
        else:
            # compute lattice extents from positions
            if self.number_of_nn == 6:
                lx = np.max(pos[:, 0]) + pos[0, 0] / 2.0
                ly = np.max(pos[:, 1]) + pos[0, 1] / 2.0
                #says the length of the lattice in x and y and accounts
                #for the triangular nature of it by adding a half
                #lattice spacing in x and y
            else:
                lx = np.max(pos[:, 0]) + pos[0, 0]
                ly = np.max(pos[:, 1]) + pos[0, 1]

        # neighbor table (NxK) — prefer explicit neighbor indices
        try:
            nn_table = self.cell_number_nn[:, 1:1 + int(self.number_of_nn)].astype(int)
            # every row is a list of nearest neighbors of that cell
        except Exception:
            # fallback to scanning relative_distance
            rel = getattr(self, 'relative_distance', None)
            if rel is None:
                raise ValueError('No neighbor information available')
            nn_table = None

        def neighbors(i):
            #return the neighbor indices of cell i
            if nn_table is not None:
                return [int(x) for x in nn_table[int(i)] if int(x) >= 0]
            else:
                return list(np.where(np.round(rel[int(i), :], 1) == 1)[0])

        results = []
        #only the non contractible loops
        all_cycles = []
        #all loops 

        # helper to canonicalize cycles to avoid duplicates (rotate + reverse)
        def canonical_cycle(cycle):
            L = len(cycle)
            if L == 0:
                return ()
            seqs = []
            for k in range(L):
                seqs.append(tuple(cycle[k:] + cycle[:k]))
                #conputes all rotations of that cycle
            rev = list(reversed(cycle))
            for k in range(L):
                seqs.append(tuple(rev[k:] + rev[:k]))
                #reverses it and then computes all rotations of the reverse cycle 
            return min(seqs)
        #returns just one cycle (arbitrary but its the one that smallest in dictionary order)
        

        seen_cycles = set()

        # iterate over connected components (clusters)
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == 0:
                continue
            comp = np.where(cluster_labels == cluster_id)[0].tolist()
            #list of indices in the cluster
            if len(comp) < 3:
                continue
            #skip if the cluster has less than three elements because it cant form a cycle
            comp_set = set(comp)
            #comp is the list of indices of a given cluster

            # build adjacency restricted to this component
            adj = {c: [nb for nb in neighbors(c) if nb in comp_set] for c in comp}
            #maps each node c in the cluster to a list of its nearest neighbors t
            # that are also in the cluster

            # DFS to build spanning tree and detect back-edges
            parent = {}
            #easier if you think of it like a tree
            depth = {}
            #discovery time of each visitted node
            visited = set()
            #the set of all the nodes we already visitted
            time = 0
            #every node is labelled with the "time" we reached it, this helps with 
            #knowing who is parent of who

            def dfs(u):
                #this is a depth first search algorithm
                nonlocal time
                #allows us to change time value
                visited.add(u)
                #record that we saw the node u
                time += 1
                depth[u] = time
                #record the time at which we visitted u
                for v in adj.get(u, []):
                    #for every cell in the set of cells of the same state adjecent to u
                    if v not in visited:
                        parent[v] = u
                        #if we have not already visitted v, we know u is v's parent
                        dfs(v)
                        #now perform the algorithm on v
                        #this is a recursive dfs algorithm
                    else:
                        # found an edge to an already visited node; if it's an ancestor this is a back-edge
                        if depth.get(v, 0) < depth.get(u, 0):
                            # reconstruct simple cycle from u back to v via parent links
                            p = [u]
                            w = u
                            while w != v and w in parent:
                                w = parent[w]
                                p.append(w)
                            # p is [u,...,v]; reverse to start at v
                            p.reverse()
                            # canonicalize
                            can = canonical_cycle(p)
                            #put the path through the canonicalize function to get only one ordering
                            if can in seen_cycles:
                                continue
                            #ignore if we have already seen it
                            seen_cycles.add(can)

                            # compute winding numbers using positions and periodic wrapping
                            cycle_nodes = list(p)
                            L = len(cycle_nodes)
                            total_dx = 0.0
                            #we define ncls by looking at their displacemnt in a and y
                            total_dy = 0.0
                            for i in range(L):
                                #for ever node in the parent map
                                a = cycle_nodes[i]
                                b = cycle_nodes[(i + 1) % L]
                                #the nect one after i considering boundary conditions 
                                x1, y1 = pos[int(a)]
                                x2, y2 = pos[int(b)]
                                dx = float(x2) - float(x1)
                                dy = float(y2) - float(y1)
                                #calculate th displacement over those 2 nodes 
                                # wrap-correct dx
                                if dx > lx / 2.0:
                                    dx -= lx
                                elif dx < -lx / 2.0:
                                    dx += lx
                                # wrap-correct dy
                                if dy > ly / 2.0:
                                    dy -= ly
                                elif dy < -ly / 2.0:
                                    dy += ly
                                total_dx += dx
                                total_dy += dy
                                #in the end, this returns the total amount that a loop moves in the x and y direction

                            # integer winding numbers
                            wx = int(round(total_dx / lx)) if lx != 0 else 0
                            wy = int(round(total_dy / ly)) if ly != 0 else 0
                            #calculates the winding number to be the total number or lattice lengths a given loop travels 
                            #in either the x or y direction

                            cycle_record = {'nodes': cycle_nodes, 'windings': (wx, wy), 'length': len(cycle_nodes)}
                            all_cycles.append(cycle_record)
                            if wx != 0 or wy != 0:
                                cycle_record['wrap_count'] = max(abs(wx), abs(wy))
                                results.append(cycle_record)
                                #cycles only get added in this condition block, therefor they only get added if a back end 
                                #is found so only closed cycles can get added here
                                #after it is confirmed that it is a closed cycle, 
                                #the code then looks to see if it has a winding number greater than 1 
                                #if it is than it is ncl

            # run DFS from each unvisited node in component
            for node in comp:
                #remember comp is the list of indices in a given cluster 
                if node not in visited:
                    parent[node] = None
                    dfs(node)
                    #perform a dfs on node 

        if return_zero_winding:
            return all_cycles, results
        return results
    #results is a list of dictionaries, one for each loop


    def find_intersection_ncl_and_shortest_string(self, frame, include_zero_charge, pair=None, phase_value=None, return_all=False):
        """Looks at all the nodes in all noncontractible loops, looks at all the nodes in all strings between vortices 
        (this definition changes depending on the truth value of include_zero charge)      
        """
        if return_all:
            all_loops, ncls = self.find_noncontractible_loops(frame, phase_value, return_zero_winding=True)
        else:
            ncls = self.find_noncontractible_loops(frame, phase_value, return_zero_winding=False)
        if pair is None:
            ncl_and_strings=[]
        # --- Normalize get_all_cluster_connections output ---
            result = self.get_all_cluster_connections(frame, include_zero_charge, return_all=return_all)
            if isinstance(result, tuple):
                shortest_strings, all_strings = result
            else:
                shortest_strings, all_strings = result, None

        # --- Choose which connections to use ---
            strings = all_strings if return_all else shortest_strings

            seen = set() #ADDED 10/15

            for loop in ncls:
                # if isinstance(loop, dict):
                #     loop_nodes = set(loop.get('nodes', []))
                # elif isinstance(loop, list):
                #     # assume it's a list of node indices
                #     loop_nodes = set(loop)
                # else:
                #     # unknown type → skip
                #     continue
                loop_nodes = set(loop['nodes'])
                
                # for pair, info in strings.items(): #pair=(v1,v2) , info={'distance}:..., 'path':...}
                #     candidates = info if isinstance(info, list) else [info]
                #     for cand in candidates:
                #         if not isinstance(cand, dict):
                #             continue
                #         path_nodes= set(cand.get('path') , [])
                #         shared= loop_nodes & path_nodes
                #         if shared:
                #             ncl_and_strings.append({
                #                 'loop': loop,
                #                 'string_pair': pair,
                #                 'string_path': cand.get('path'),
                #                 'shared_nodes': list(shared)
                #                 })

                for pair, infos in strings.items():  # infos is a list of dicts
                    if not isinstance(infos, list):
                        infos = [infos]  # wrap single dict as list
                    v1, v2 = pair#ADDED 10/15
                    pair = tuple(sorted((v1, v2))) #ADDED 10/15
                    for info in infos:
                        string_path = tuple(sorted(info.get("path", [])))  # SORT FOR CONSISTENT HASH
                        path_nodes = set(string_path)  # now info is a dict, path is a list of ints
                        shared = loop_nodes & path_nodes 
                        if shared:
                            key = (pair, string_path)
                            if key in seen:
                                continue
                            seen.add(key)
                            
                            ncl_and_strings.append({
                                'loop': loop,
                                'string_pair': pair,
                                'string_path': info.get('path'),
                                'shared_nodes': list(shared)
                            })

        else:
            ncl_and_strings=[]
            v1, v2 = int(pair[0]), int(pair[1])
            #this is the vortex pair we are looking at
            conns = self.get_shortest_cluster_connections(frame,include_zero_charge, count_mode='cells')
            # conns is a dict mapping (v1_id, v2_id) -> dict with keys: 'distance', 'path', 'state', 'frame' for all pairs
            # we only want the fixed pair (v1, v2) or (v2, v1)
            info = None
            if conns and (v1, v2) in conns:
                info = conns.get((v1, v2))
            elif conns and (v2, v1) in conns:
                info = conns.get((v2, v1))
            #get the info on just those pairs

            if info is None:
                ncl_and_strings.append({
                        'loop': None,
                        'string_pair': pair,
                        'string_path': None,
                        'shared_nodes': []
                    })
            else:
                pairpath = info.get('path', [])
                #the path between that pair at that timestep


                for loop in ncls:
                    loop_nodes=set(loop.get('nodes', []))
                    #the set of all nodes in the non contractibe loop
                    path_nodes = set(pairpath or [])
                    #the set of all the nodes in the shortest string between the vortices

                    shared = loop_nodes & path_nodes
                    #the set that are in both 


                    if shared:
                            ncl_and_strings.append({
                                'loop': loop,
                                'string_pair': pair,
                                'string_path': info.get('path'),
                                'shared_nodes': list(shared)
                                })

                    
            

        return ncl_and_strings
    


    def exists_contractible_string(self, frame, include_zero_charge, pair=None, phase_value=None, count_mode='cells'):
        """
        Determine whether there exists a continuous same-phase string (cluster path) that connects
        two charged vortices and is contractible (i.e., does NOT intersect any non-contractible loop).

        "for every vortex pair, is there at least one contractible string that connects them?"

        Behavior:
            - If `pair` is None, the function searches all found candidate strings between any pairs
              and returns True as soon as it finds one candidate whose path shares no nodes with
              any non-contractible loop at `frame`.
            - If `pair` is provided (tuple (v1,v2)), only candidates for that pair are tested.

        Args:
            frame (int): global frame index to inspect.
            include_zero_charge (bool): whether to include zero-charge vortices when building candidates.
            pair (tuple or None): optional specific vortex label pair (v1, v2) to test.
            phase_value (int or None): restrict candidate strings to clusters of this 4-state value.
            count_mode (str): passed to get_all_cluster_connections (kept for API compatibility).

        Returns:
            dict: {
                'exists': bool,
                'checked_pair': (v1, v2) or None,
                'contractible_candidates': list of candidate dicts (each: {'pair','path','distance','state','cluster'})
            }

        Notes:
            - A candidate is considered contractible if its `path` (list of cell indices) does NOT
              intersect any node that belongs to any non-contractible loop (as returned by
              `find_noncontractible_loops`). If no non-contractible loops exist at the frame,
              any candidate path is considered contractible.
        """
        # get non-contractible loops (NCLs) at this frame
        ncls = self.find_noncontractible_loops(frame, phase_value=phase_value, return_zero_winding=False)
        # build a set of all nodes that belong to any NCL
        ncl_nodes = set()
        #ncl_nodes is the set of all cells that are a part of a ncl
        for loop in ncls:
            nodes = loop.get('nodes', [])
            ncl_nodes.update(int(x) for x in nodes)

        # get all candidate strings between vortex pairs
        try:
            best_results, all_candidates = self.get_all_cluster_connections(frame, include_zero_charge, count_mode=count_mode, return_all=True)
            #this returns all candidate strings
        except Exception:
            # If older signature, fall back to non-return_all call
            all_candidates = self.get_all_cluster_connections(frame, include_zero_charge, count_mode=count_mode)
            # normalize to dict of lists
            all_candidates = {k: [v] for k, v in all_candidates.items()}

        contractible_candidates = []

        # helper to test a single candidate record
        def _is_candidate_contractible(path):
            if path is None:
                return False
            if len(ncl_nodes) == 0:
                # no non-contractible loops -> candidate is contractible
                return True
            # check intersection
            path_set = set(int(x) for x in path)
            return len(path_set & ncl_nodes) == 0
            #this is a bool, true if path_set and ncl_nodes have no intersection
            #this helper function tests each individual string to see if it intersects
            #  with a non contractible string

        # normalize pair input
        target_pairs = None
        if pair is not None: 
            try:
                v1, v2 = int(pair[0]), int(pair[1])
                target_pairs = {(v1, v2), (v2, v1)}
            except Exception:
                raise ValueError('pair must be a 2-tuple of vortex labels')

        # iterate over all candidates
        for pkey, cand_list in all_candidates.items():
            # pkey is (v1,v2)
            if target_pairs is not None and pkey not in target_pairs:
                # try reversed key as some callers store keys in either order
                if (pkey[1], pkey[0]) not in target_pairs:
                    continue

            # cand_list is a list of candidate dicts
            for cand in cand_list:
                # cand is expected to have 'path' and other metadata
                path = None
                if isinstance(cand, dict):
                    path = cand.get('path')
                elif isinstance(cand, tuple) and len(cand) == 2:
                    # older return (distance, path)
                    path = cand[1]
                else:
                    path = cand

                if _is_candidate_contractible(path):
                    # build normalized candidate record
                    rec = {
                        'pair': pkey,
                        'path': path,
                        'distance': cand.get('distance') if isinstance(cand, dict) else (cand[0] if isinstance(cand, tuple) else None),
                        'state': cand.get('state') if isinstance(cand, dict) else None,
                        'cluster': cand.get('cluster') if isinstance(cand, dict) else None,
                        'frame': frame
                    }
                    contractible_candidates.append(rec)

                    # If the user asked about a specific pair, we can return immediately
                    if target_pairs is not None:
                        return {'exists': True, 'checked_pair': (v1, v2), 'contractible_candidates': contractible_candidates}

        exists = len(contractible_candidates) > 0
        return {'exists': exists, 'checked_pair': (v1, v2) if pair is not None else None, 'contractible_candidates': contractible_candidates}

    def determine_end_frame(self, include_zero_charge):
        "this is a method that I am writing in to make my data collection less computationally"
        "expensive. End frame is just the final frame that we look at when we are trying to study vortex"
        "annihilation. For the spiral case, it is the first reccurent time. for the static and wave "
        "configurations, its the last timestep before the vortexes annihilate. If there are no vortexes"
        "then it returns end time= 0"
        end_frame = 0
        case_type = "unknown"
        pair = None
        # defensive copies / coercions: these attributes may be lists, arrays, or scalars
        finalPattern_Label = getattr(self, 'final_pattern_type', None)
        # if finalPattern_Label is an array-like, try take scalar
        try:
            if isinstance(finalPattern_Label, (list, tuple, np.ndarray)) and len(finalPattern_Label) > 0:
                finalPattern_Label = finalPattern_Label[0]
        except Exception:
            pass
        computationTime = getattr(self, 'first_recurrent_state_time', None)
        try:
            if isinstance(computationTime, (list, tuple, np.ndarray)) and len(computationTime) > 0:
                computationTime = computationTime[0]
        except Exception:
            pass
        annihilation_frames = np.array(getattr(self, 'idx_annihilation_moments', []))
        try:
            try:
                pair = self.find_last_two_vortices(include_zero_charge=include_zero_charge, lookback=300)
            except Exception:
                pair = None

            if pair is None:
                try:
                    if hasattr(self, 'charges') and self.charges is not None:
                        charges_arr = np.array(self.charges)
                        # try to shape into (frames x vortices) if possible
                        if charges_arr.ndim == 1 and len(charges_arr) > 0 and isinstance(charges_arr[0], (list, tuple, np.ndarray)):
                            try:
                                charges_arr = np.vstack(charges_arr)
                            except Exception:
                                charges_arr = charges_arr.reshape((1, -1))
                        if charges_arr.ndim == 2 and charges_arr.shape[1] >= 3:
                            n_vortex = np.sum(charges_arr[:, 2:] != 0, axis=1)
                            frames_with_vortices = np.where(n_vortex > 0)[0]
                            last_with_vortices = int(frames_with_vortices[-1]) if len(frames_with_vortices) > 0 else None
                        else:
                            last_with_vortices = None
                    else:
                        last_with_vortices = None
                except Exception:
                    last_with_vortices = None


            # --- CASE 1: Spiral (persistent pair) ---
            # Spiral (persistent pair) — computationTime may be list or scalar
            if finalPattern_Label == 2:
                try:
                    end_frame = int(computationTime)
                except Exception:
                    # fallback to last_with_vortices if available
                    end_frame = int(last_with_vortices) if last_with_vortices is not None else 0
                case_type = "spiral"


                # --- CASE 2: Annihilating or disappearing pair ---
            elif annihilation_frames.size > 0:
                try:
                    last_ann = np.ravel(annihilation_frames)[-1]
                    end_frame = int(last_ann - 1)
                except Exception:
                    end_frame = int(min(self.tmax - 1, 0))
                case_type = "annihilating pair"
            else:
                try:
                    charges_arr = np.array(self.charges)
                    if charges_arr.ndim == 1 and len(charges_arr) > 0 and isinstance(charges_arr[0], (list, tuple, np.ndarray)):
                        try:
                            charges_arr = np.vstack(charges_arr)
                        except Exception:
                            charges_arr = charges_arr.reshape((1, -1))
                    if charges_arr.ndim == 2 and charges_arr.shape[1] >= 3:
                        n_vortex = np.sum(charges_arr[:, 2:] != 0, axis=1)
                    else:
                        n_vortex = np.array([])
                except Exception:
                    n_vortex = np.array([])
                frames_with_2 = np.where(n_vortex == 2)[0]
                if len(frames_with_2) > 0:
                    try:
                        end_frame = int(frames_with_2[-1])
                    except Exception:
                        end_frame = 0
                    case_type = "disappearing pair"
        except Exception as e:
            print("determine_end_frame has exception", e)
        try:
            end_frame = int(min(int(end_frame), int(self.tmax - 1)))
        except Exception:
            try:
                end_frame = int(self.tmax - 1)
            except Exception:
                end_frame = 0
        return pair, end_frame



    def get_percent_of_strings_that_are_NCL(self, include_zero_charge, return_all=False, return_details=False):
        """This answers the question, what percent off all strings between all vortices are NCLs, so if there
        is 2 vortices and there is one connection that is an ncl and one that isnt, the percent will be 50 percent
        to answer the questions "what percent of all vortex pairs cant be connected without an ncl", go to 
        percentage_of_vortices_with_contractible_connections and invert it
        
        
        returns a list with a percent for every frame"""
        pair, end_frame= self.determine_end_frame(include_zero_charge)
        percentage_ncl_perframe=[]
        detailed_results = [] # for if i want to look at the details 
        for frame in range (end_frame):


            # 🟢 10/15 ADDED: Extract vortex IDs and their charges for this frame
            vortex_ids = []
            vortex_charges = {}
            try:
                if hasattr(self, 'n_vortex'):
                    n_vortex_frame = int(self.n_vortex[frame])
                    vortex_ids = np.arange(1, n_vortex_frame + 1)
                if hasattr(self, 'charges') and self.charges is not None:
                    charges_arr = np.array(self.charges)
                    if charges_arr.ndim == 2 and frame < charges_arr.shape[0]:
                        frame_charges = charges_arr[int(frame), 2:]
                        for vid in vortex_ids:
                            idx = int(vid) - 1
                            if 0 <= idx < len(frame_charges):
                                vortex_charges[int(vid)] = frame_charges[idx]
            except Exception:
                pass
            # END 🟢 10/15 ADD

            # 🟢 10/15 ADDED: Helper to check if a pair is valid (+1/-1 only if include_zero_charge=False)
            def _pair_is_valid(v1, v2):
                if include_zero_charge:
                    return True
                q1 = vortex_charges.get(v1, None)
                q2 = vortex_charges.get(v2, None)
                if q1 is None or q2 is None:
                    return False
                if q1 == 0  or q2 == 0:
                    return False
                if np.sign(q1) == np.sign(q2):
                    return False  # skip same-charge pair
                return True
            # END 🟢 10/15 ADD`



            # get all cluster connections
            shortest_strings, all_strings = self.get_all_cluster_connections(
                frame, include_zero_charge=True, count_mode="cells", return_all=True
            )

            non_unique_strings = []  # reset per frame
            connected_pairs = []     # 🟢 10/15 ADDED: track which vortex pairs are connected

            if isinstance(all_strings, dict):
                for pair_key, candidates in all_strings.items():

                    if pair_key[0] > pair_key[1]:
                        continue

                    # 🟢 10/15 ADDED: Apply +1/-1 filtering
                    try:
                        v1, v2 = pair_key
                        if not _pair_is_valid(v1, v2):
                            continue  # skip invalid pair
                    except Exception:
                        continue
                    # END 🟢 10/15 ADD
                    
                    for cand in candidates:
                        
                        path_tuple = ()
                        if isinstance(cand, dict):
                            # old behavior: get 'path' key
                            #print('step 1a')
                            
                            path_tuple = tuple(cand.get('path', []))
                        elif isinstance(cand, (list, np.ndarray)):
                            # treat the list/array directly as the path
                            path_tuple = tuple(cand)
                            #print('step 1b')
                        else:
                            # unknown type: ignore
                            path_tuple = ()
                            #print('step 1c')
                        
                        if path_tuple:
                            path_tuple = tuple(sorted(path_tuple))
                            non_unique_strings.append(path_tuple)
                            connected_pairs.append(pair_key)  # 🟢 10/15 ADDED: record this valid connection

            unique_strings = {p for p in non_unique_strings if p}
            total_strings = len(unique_strings)
            # intersection
            
            intersections = self.find_intersection_ncl_and_shortest_string(frame, include_zero_charge=include_zero_charge, return_all=True)


            # 🟢 10/15: Store all intersection info for this frame
            intersection_details = []
            for item in intersections:
                if not isinstance(item, dict):
                    continue
                intersection_details.append({
                    "string_path": item.get("string_path", []),
                    "ncl_path": item.get("ncl_path", []),
                    "string_pair": item.get("string_pair"),
                    "distance": item.get("distance"),
                    "cluster": item.get("cluster"),
                    "state": item.get("state"),
                    "frame": frame
                })



            #print(f"Iteration {iteration}, Frame {frame_idx}: intersections type = {type(intersections)}")
            #ncl_strings = set()
            ncl_strings = {tuple(sorted(item.get('string_path', []))) for item in intersections if item.get('string_path')}

            # for item in intersections:
            #     path_tuple = tuple(item.get('string_path') or [])
            #     if path_tuple:
            #         ncl_strings.add(path_tuple)

            ncl_count = len(ncl_strings)
            percentage_ncl = 100 * ncl_count / total_strings if total_strings > 0 else 0.0
            percentage_ncl_perframe.append(percentage_ncl)


             # 🟢 10/15 ADDED: Collect per-frame details if requested
            if return_details:
                detailed_results.append({
                    "frame": frame,
                    "vortex_ids": list(vortex_ids),
                    "vortex_charges": vortex_charges.copy(),
                    "connected_pairs": list(set(connected_pairs)),
                    "ncl_count": ncl_count,
                    "total_strings": total_strings,
                    "intersections": intersection_details,  # 🟢 10/15 new field: full metadata
                    "percentage_ncl": percentage_ncl
                })
            # END 🟢 10/15 ADD


        # 🟢 10/15 ADDED: Conditional return logic
        if return_details:
            return detailed_results
        return percentage_ncl_perframe
        # END 🟢 10/15 ADD
        
    def final_pair_on_strings(self, include_zero_charge):
        "returns a list with a 1 or 0 for each frame  that says if at each frame, the final"
        "pair is connected by any same color cluster"
        pair, end_frame=self.determine_end_frame(include_zero_charge)
        final_pair_on_any_string = []
        track_data = self.track_pair_over_time(
            include_zero_charge=True, pair=pair, start_frame=0,
            end_frame=end_frame+1, count_mode='cells'
        )
        frames = track_data.get('frames', [])
        #print("frames", frames)
        paths = track_data.get('paths', [None]*len(frames))
        for frame_idx, frame in enumerate(frames):
            path_at_frame = paths[frame_idx]
            if path_at_frame is None:
                final_pair_on_any_string.append(0)
            elif isinstance(path_at_frame, (list, np.ndarray)):
                final_pair_on_any_string.append(1 if len(path_at_frame) > 0 else 0)
            else:
                final_pair_on_any_string.append(1 if bool(path_at_frame) else 0)
        return final_pair_on_any_string
    

    def final_pair_on_ncl(self, include_zero_charge):
        "returns a list with a 1 or 0 for each frame  that says if at each frame, the final"
        "pair is connected by any same color cluster that is non_contractible due"
        "to the boundary conditions "
        pair, end_frame=self.determine_end_frame(include_zero_charge)
        final_pair_on_ncl=[]
        track_data = self.track_pair_over_time(
            include_zero_charge=True, pair=pair, start_frame=0,
            end_frame=end_frame + 1, count_mode='cells'
        )
        frames = track_data.get('frames', [])
        for frame_idx, frame in enumerate(frames):
            nclstringfinalpair = self.find_intersection_ncl_and_shortest_string(
                frame, True, track_data['pair']
            )
            has_ncl_string = (
                isinstance(nclstringfinalpair, np.ndarray)
                and np.any(nclstringfinalpair)
            ) or (
                hasattr(nclstringfinalpair, "__len__")
                and len(nclstringfinalpair) > 0
            )
            final_pair_on_ncl.append(1 if has_ncl_string else 0)

        return final_pair_on_ncl
    
    def final_pair_on_contractible_loop(self, include_zero_charge):
        final_pair_on_contractible_loop = []
        pair, end_frame=self.determine_end_frame(include_zero_charge)
        track_data = self.track_pair_over_time(
            include_zero_charge=True, pair=pair, start_frame=0,
            end_frame=end_frame + 1, count_mode='cells'
        )
        frames = track_data.get('frames', [])
        for frame_idx, frame in enumerate(frames):
            contractible_result = self.exists_contractible_string(
                frame, include_zero_charge=True, pair=track_data['pair']
            )
            final_pair_on_contractible_loop.append(1 if contractible_result.get('exists') else 0)
        return final_pair_on_contractible_loop
    

    def percentage_vortices_with_contractible_connections(self, include_zero_charge, return_details=False):
        "returns a time series with the percentages of vortices that are connected without "
        "touching a ncl for each time step"
        """asks question "what percent of all vortex pairs can be connected in a same state string that
        does not touch an ncl"""
        finalpair, end_frame=self.determine_end_frame(include_zero_charge)
        n_vortex = self.n_vortex
        #print(n_vortex)
        #result = self.exists_contractible_string(end_frame//2, include_zero_charge=include_zero_charge)
        
        contractible_percentages = []

        # 🟢 10/15 added: store one dictionary per frame if return_details=True
        frame_details_list = []

        # Include the returned end_frame itself (determine_end_frame returns the last frame index to consider)
        frames = list(range(int(end_frame) + 1)) if end_frame is not None else []

        for frame in frames:
            # Robustly obtain number of vortices for this frame
            try:
                n_vortex_frame = int(self.n_vortex[frame])
            except Exception:
                # fallback: try to infer from labeled vortex map if available
                try:
                    if hasattr(self, 'vortex_cores_labeled') and isinstance(self.vortex_cores_labeled, np.ndarray) and frame < self.vortex_cores_labeled.shape[1]:
                        vmap = self.vortex_cores_labeled[:, frame]
                        n_vortex_frame = int(np.max(vmap)) if vmap.size > 0 else 0
                    else:
                        n_vortex_frame = 0
                except Exception:
                    n_vortex_frame = 0

        # ---------------------------------------------------------------
        #  🟢 10/15: DETERMINE VORTEX IDS TO CONSIDER (WITH CHARGE LOGIC)
        # ---------------------------------------------------------------
            vortex_ids = np.arange(1, n_vortex_frame + 1)
            #all possible ids for vortices in that frame
            vortex_charges = {}  # 🟢 10/15 added dictionary to store charge mapping
            try:
                if hasattr(self, 'charges') and self.charges is not None:
                    charges_arr = np.array(self.charges)
                    if charges_arr.ndim == 2 and frame < charges_arr.shape[0]:
                        frame_charges = charges_arr[int(frame), 2:]
                        for vid in vortex_ids:
                            idx = int(vid)-1
                            #go back to starting at 0 for index comparison
                            if 0 <= idx <len(frame_charges):
                                vortex_charges[int(vid)] = frame_charges[idx]
            except Exception:
                pass
            
            if include_zero_charge == False:
                charged_vortex_ids = [i for i, q in vortex_charges.items() if q != 0]
                #all the indices for nonzero vortices
                vortex_ids = np.array(charged_vortex_ids, dtype=int)
            else: 
                vortex_ids = np.arange(1,n_vortex_frame+1)
                #all possible vortex_ids


        # ---------------------------------------------------------------
        # 🟢 10/15: BUILD PAIRS WITH NEW FILTERING RULE
        # --------------------------------------------------------------
            vortex_pairs = []
            for (v1, v2) in combinations(vortex_ids, 2):
                if not include_zero_charge:
                    q1 = vortex_charges.get(v1, None)
                    q2 = vortex_charges.get(v2, None)
                    if q1 is None or q2 is None:
                        continue #skip undefined charges
                    if np.sign(q1)==np.sign(q2):
                        continue 
                    #skip same charge pair
                
                vortex_pairs.append((v1,v2))
        # ---------------------------------------------------------------
        #  🟢 10/15: COMPUTE FRACTION CONNECTED
        # ---------------------------------------------------------------
            connected_count = 0
            connected_pairs=[]
            for pair in vortex_pairs:
                result = self.exists_contractible_string(frame, include_zero_charge=include_zero_charge, pair=pair)
                # i think this will work even before i fix exists_contracitble_string bc I am specifying the pair
                if result.get('exists', False):
                    connected_count +=1 
                    connected_pairs.append(pair)


            total_pairs = len(vortex_pairs)
            percentage= 100.0*connected_count / total_pairs if total_pairs > 0 else np.nan

            contractible_percentages.append(percentage)



                # 🟢 10/15: if return_details=True, store all data for this frame
            if return_details:
                frame_details_list.append({
                    "frame": int(frame),
                    "vortex_ids": vortex_ids.tolist(),
                    "vortex_charges": {int(k): float(v) for k, v in vortex_charges.items()},
                    "connected_pairs": connected_pairs,
                    "percentage": float(percentage)
        })


        if return_details:
            return frame_details_list
        else:
            return contractible_percentages
            




            

        #     # Determine vortex ids to consider (respect include_zero_charge)
        #     if include_zero_charge == False and hasattr(self, 'charges') and self.charges is not None:
        #         try:
        #             frame_charges = np.array(self.charges)[frame, 2:]
        #             charged_vortex_ids = [i + 1 for i, charge in enumerate(frame_charges) if charge != 0]
        #             vortex_ids = np.array(charged_vortex_ids, dtype=int)
        #         except Exception:
        #             # if charges malformed, fall back to all vortices
        #             vortex_ids = np.arange(1, n_vortex_frame + 1)
        #     else:
        #         vortex_ids = np.arange(1, n_vortex_frame + 1)

        #     vortex_pairs = list(combinations(vortex_ids, 2))

        #     connected_count = 0
        #     for pair in vortex_pairs:
        #         result = self.exists_contractible_string(frame, include_zero_charge=include_zero_charge, pair=pair)
        #         if result.get('exists', False):
        #             connected_count += 1

        #     total_pairs = len(vortex_pairs)
        #     percentage = 100.0 * connected_count / total_pairs if total_pairs > 0 else np.nan
        #     contractible_percentages.append(percentage)

        # return contractible_percentages
    

    def color_agnostic_connection_timeseries(self, include_zero_charge, pair=None, return_details=False, verbose=False):

        """answers the question, what fraction of all vortex pairs can be connected without touching an ncl
       regardless of color (can be a multistate path)"""
        # ------------------------------------------------------------
        # 1️⃣  Determine frame range to analyze
        # ------------------------------------------------------------
        _, end_frame = self.determine_end_frame(include_zero_charge)
        #frame=end_frame//2
        # print(f"[DEBUG CANT] unique(self.vortex_cores_labeled[:, {end_frame//2}]) →",
        # np.unique(self.vortex_cores_labeled[:, frame]))
        # print(f"[DEBUG CANT] np.sum(vortex_map==1) = {np.sum(self.vortex_cores_labeled[:, frame]==1)}")
        # print(f"[DEBUGCANT] np.sum(vortex_map==2) = {np.sum(self.vortex_cores_labeled[:, frame]==2)}")
        # print(f"[DEBUGCANT] np.sum(vortex_map==3) = {np.sum(self.vortex_cores_labeled[:, frame]==3)}")
        # print(f"[DEBUGCANT] np.sum(vortex_map==4) = {np.sum(self.vortex_cores_labeled[:, frame]==4)}")
        if pair is not None:
            # Determine frame range for this tracked pair
            _, end_frame = self.determine_end_frame(include_zero_charge)
            #get information on the vortex ids for each timestep
            track_data = self.track_pair_over_time(
                include_zero_charge=include_zero_charge,
                pair=pair,
                start_frame=0,
                end_frame=end_frame + 1,
                count_mode='cells'
            )
            frames = track_data.get('frames', [])
            tracked_pair = track_data.get('pair', pair)
        else:
            #find end frame
            _, end_frame = self.determine_end_frame(include_zero_charge)
            frames = list(range(end_frame + 1))
            tracked_pair = None
        results = []  # this will hold the output (either floats or 1/0s — integers where relevant)
        # Use a dict keyed by frame so callers can reliably index by global frame
        connected_sets = {} if return_details else None  # only build if requested

        # ------------------------------------------------------------
        # 2️⃣ Precompute static adjacency (only once)
        # ------------------------------------------------------------
        if not hasattr(self, "_adjacency_full"): #first check that this attribute doesnt already exist
            N=int(self.N)
            nn_matrix = self. cell_number_nn[:, 1:1+int(self.number_of_nn)].astype(int)
            # Extract only the neighbor indices from the full nearest-neighbor table.
                # self.cell_number_nn is an (N x (1 + number_of_nn)) array where:
                #   - column 0  → the cell's own index (i)
                #   - columns 1..number_of_nn → indices of the cell's nearest neighbors
                # The slice [:, 1:1+int(self.number_of_nn)] drops the first column and
                # keeps exactly the neighbor columns, giving an (N x number_of_nn) array.
                # We then cast to int so all neighbor indices are integers (no floats).
            self._adjacency_full = {i: [int(j) for j in nn_matrix[i] if 0 <= j < N] for i in range(N)}



        #----------------------------------------------------------
        #loop over frames 
        #-----------------------------------------------------------

        for frame in frames:
        #try:
            #---------------------------------------------------------
            #get the vortex labelling for each frame
            #each cell in the lattice has a vortex label (0 if not a vortex)
            #---------------------------------------------------------
            frame = int(frame)
            # build per-cell vortex map with the same fallbacks as other routines
            vortex_map = None
            try:
                if hasattr(self, 'vortex_cores') and isinstance(self.vortex_cores, np.ndarray):
                    if self.vortex_cores.ndim == 2 and frame < self.vortex_cores.shape[1]:
                        vortex_map = self.vortex_cores[:, frame]
            except Exception:
                vortex_map = None

            if vortex_map is None:
                try:
                    if hasattr(self, 'vortex_cores_labeled') and isinstance(self.vortex_cores_labeled, np.ndarray):
                        vortex_map = self.vortex_cores_labeled[:, frame]
                except Exception:
                    vortex_map = None

            if vortex_map is None:
                # final fallback: zeros
                vortex_map = np.zeros(int(self.N), dtype=int)

            
            # If vortex_map is binary (0/1) coming from `vortex_cores`, prefer
            # the labeled map when available, or convert binary cores -> labeled
            try:
                unique_vals = np.unique(vortex_map)
                # if only 0 and 1 present, it is a binary mask
                if set(unique_vals.tolist()) <= {0, 1}:
                    # prefer precomputed labeled map if present
                    if hasattr(self, 'vortex_cores_labeled') and isinstance(self.vortex_cores_labeled, np.ndarray) and self.vortex_cores_labeled.shape[1] > frame:
                        vortex_map = self.vortex_cores_labeled[:, frame]
                    else:
                        # try to compute labeled map from binary cores
                        try:
                            # ensure vortex_cores is a numpy array available to the labeling function
                            if not hasattr(self, 'vortex_cores') or not isinstance(self.vortex_cores, np.ndarray):
                                self.vortex_cores = np.array(vortex_map)
                            labeled_map, _ = label_triangular_lattice(self)
                            vortex_map = np.array(labeled_map)[:, frame]
                            # cache labeled map
                            self.vortex_cores_labeled = np.array(labeled_map)
                        except Exception:
                            # if labeling fails, keep the binary map (graceful fallback)
                            pass
            except Exception:
                pass

            # Always start clean for each frame
            vortex_ids = np.unique(vortex_map[vortex_map > 0])
            vortex_ids = np.array(np.sort(vortex_ids), dtype=int)  # stable sorted ordering
            unique_vals = vortex_ids
            if verbose and not getattr(self, 'quiet_warnings', False):
                print(f"[DEBUG Color agnostic] Frame {frame}: vortex_map has vortex IDs {unique_vals}")
                for vid in unique_vals:
                    cells = np.where(vortex_map == vid)[0]
                    print(f"  vortex {vid}: {cells.tolist()}")

        # ------------------------------------------------------------
        # 2️⃣ If zero-charge vortices should be excluded
        # ------------------------------------------------------------
            vortex_charges = {}
            if include_zero_charge == False and hasattr(self, 'charges') and self.charges is not None:
                try:
                    # Coerce charges into a numpy 2D array if possible
                    charges_arr = np.array(self.charges)
                    if charges_arr.ndim == 1:
                        # If it's a 1D array of per-frame rows (lists), try to stack
                        if len(charges_arr) > 0 and isinstance(charges_arr[0], (list, tuple, np.ndarray)):
                            try:
                                charges_arr = np.vstack(charges_arr)
                            except Exception:
                                charges_arr = charges_arr.reshape((1, -1))
                        else:
                            # Can't interpret as per-frame 2D; be conservative and skip filtering
                            raise ValueError('charges not in expected 2D per-frame shape')
                    # Now do safe indexing
                    if charges_arr.ndim == 2 and frame < charges_arr.shape[0]:
                        frame_charges = charges_arr[int(frame), 2:]
                    else:
                        raise IndexError('frame index out of range for charges array')

                    charged_vortex_ids = []
                    for vid in vortex_ids:
                        idx = int(vid) - 1
                        if 0 <= idx < frame_charges.shape[0]:
                            if frame_charges[idx] != 0:
                                charged_vortex_ids.append(int(vid))
                                # 🟢 ADDED: record charge for later
                                vortex_charges[int(vid)] = frame_charges[idx]
                    vortex_ids = np.array(charged_vortex_ids, dtype=int)
                except Exception as e:
                    # If charges are missing or malformed, quietly skip filtering and keep all vortex_ids
                    if verbose:
                        print(f"ℹ️ charges unavailable or malformed for frame {frame}; skipping zero-charge filtering")
            else:
                # 🟢 ADDED Whole block: still record charge info if available
                if hasattr(self, 'charges') and self.charges is not None:
                    try:
                        charges_arr = np.array(self.charges)
                        if charges_arr.ndim == 2 and frame < charges_arr.shape[0]:
                            frame_charges = charges_arr[int(frame), 2:]
                            for vid in vortex_ids:
                                idx = int(vid) - 1
                                if 0 <= idx < frame_charges.shape[0]:
                                    vortex_charges[int(vid)] = frame_charges[idx]
                    except Exception:
                        pass

            n_vortices=len(vortex_ids)
            if n_vortices<2:
                results.append(0.0)
                # no connections in that frame
                if return_details:
                    connected_sets[frame] = {'connected_vortices': set(), "connected_pairs": []}
                continue
            # ------------------------------------------------------------
            # 3️⃣  Collect all nodes in non-contractible loops
            # ------------------------------------------------------------
            # noncontractible_loops = self.find_noncontractible_loops(frame, return_all=False)
            # ncl_nodes = set()
            # for loop in noncontractible_loops:
            #     ncl_nodes.update(loop['nodes'])

            if not hasattr(self, "_ncl_cache"):
                self._ncl_cache = {}
            if frame not in self._ncl_cache:
                # store whatever the function returns (list or empty)
                try:
                    self._ncl_cache[frame] = self.find_noncontractible_loops(frame, return_zero_winding=False)
                except TypeError:
                    # older signature fallback
                    self._ncl_cache[frame] = self.find_noncontractible_loops(frame)

            noncontractible_loops = self._ncl_cache.get(frame) or []

            # build set of nodes belonging to any NCL; if none, leave empty (do not skip frame)
            ncl_nodes = set()
            for loop in noncontractible_loops:
                try:
                    ncl_nodes.update(loop.get('nodes', []))
                except Exception:
                    pass

            # ------------------------------------------------------------
            # 4️⃣  Build adjacency excluding NCL nodes
            # ------------------------------------------------------------


            N=int(self.N)
            adjacency = {i: [v for v in self._adjacency_full[i] if v not in ncl_nodes]
                            for i in self._adjacency_full if i not in ncl_nodes}
            # a dictionary where each cell index i (except ncl_nodes) us a key to a list of nearest neigbbors
            

            # --------------------------------------------------------
            # (d) Label connected components (each "island" of reachable cells)
            # --------------------------------------------------------
            component_label = np.zeros(int(self.N), dtype=int)
            component_id = 1
            for node in adjacency.keys():
                if component_label[node] != 0:
                    #it has already been labeled
                    continue
                q = deque([node])
                #deque means double ended queue
                #like a list that is optimized for popping from both ends
                #create a queue that initially just contains starting node
                while q:
                    #runs until there are no nodes left to explore
                    u = q.popleft()
                    #takes the first element off of q and now u equals that element
                    if component_label[u] != 0:
                        #if the node that we have just popped off and are looking at is not 0
                        #it has been labeled
                        continue
                    component_label[u] = component_id
                    #if the component that we are holding has not been labeled
                    #label it with the current component id
                    for v in adjacency[u]:
                        if component_label[v]==0:
                            #if v (nearest neighbor of u) has not been labeled
                            q.append(v) #add it to the list
                component_id += 1
                            
                # --------------------------------------------------------
                # (e) Map each vortex → which component(s) it lives in
                # --------------------------------------------------------
            # end for node in adjacency.keys()  (component labeling complete)

            vortex_component = {}  # empty dictionary
            for vid in vortex_ids:
                cell_indices = np.where(vortex_map == vid)[0]
                vortex_component[vid] = np.unique(component_label[cell_indices])

            # --------------------------------------------------------
            # (f) Pair analysis (run once per frame)
            # --------------------------------------------------------
            connected_pairs = []
            #n_total_pairs = n_vortices * (n_vortices - 1) / 2
            n_total_pairs = 0 #ADDED

            # Case A — single vortex pair check
            if pair is not None:
                v1, v2 = tracked_pair
                if v1 not in vortex_component or v2 not in vortex_component:
                    connected = False
                else:
                    shared = np.intersect1d(vortex_component[v1], vortex_component[v2])
                    connected = shared.size > 0
                results.append(1 if connected else 0)
                if return_details:
                    # store result keyed by the global frame
                    connected_sets[frame] = {"connected": connected}

            # Case B — all-vortex analysis (percentage)
            else:
                for i, v1 in enumerate(vortex_ids):
                    for v2 in vortex_ids[i + 1:]:
                        #if np.intersect1d(vortex_component[v1], vortex_component[v2]).size > 0:
                            #connected_pairs.append((v1, v2))
                        # 🟢 ADDED: only apply opposite-charge restriction if include_zero_charge=False
                        if not include_zero_charge:
                            q1 = vortex_charges.get(int(v1), None)
                            q2 = vortex_charges.get(int(v2), None)
                            if q1 is None or q2 is None:
                                continue
                            if np.sign(q1) == np.sign(q2):
                                continue  # skip same-sign pair

                        else:
                            # 🟢 ADDED: INCLUDE ALL PAIRS WHEN include_zero_charge=True
                            # even if charge info missing or zero
                            q1 = vortex_charges.get(int(v1), None)
                            q2 = vortex_charges.get(int(v2), None)
                        n_total_pairs += 1

                        #check if they are connected
                        if np.intersect1d(vortex_component[v1], vortex_component[v2]).size > 0:
                            connected_pairs.append((v1, v2))

                n_connected_pairs = len(connected_pairs)
                percent = 100.0 * n_connected_pairs / n_total_pairs if n_total_pairs > 0 else 0.0
                results.append(percent)

                if return_details:
                    vortex_cell_indices = {
                        vid: np.where(self.vortex_cores_labeled[:, frame] == vid)[0].tolist()
                        for vid in vortex_ids}

                    # emit one connected_sets entry per global frame
                    connected_sets[frame] = {
                        "connected_pairs": connected_pairs,
                        "n_connected_pairs": n_connected_pairs,
                        "n_total_pairs": n_total_pairs,
                        "fraction_connected_pairs": n_connected_pairs / n_total_pairs if n_total_pairs > 0 else 0.0,
                        "vortex_cell_indices": vortex_cell_indices
                    }

            # except Exception as e:
            #     print(f"⚠️ Frame {frame}: error in connectivity_without_ncl() → {e}")
            #     results.append(0.0)
            #     if return_details:
            #         if pair is None:
            #             connected_sets.append({"connected_pairs": [], "n_connected_pairs": 0, "n_total_pairs": 0})
            #         else:
            #             connected_sets.append({"connected": False})

        return (results, connected_sets) if return_details else results
    


    def compute_shells(self):

        nn_matrix = self. cell_number_nn[:, 1:1+int(self.number_of_nn)].astype(int)
        N=int(self.N)
        shells = {d: [] for d in range(1, int(self.gridsize//2+1))}

        # Preconvert neighbor lists to Python sets for fast lookup
        nn_sets = [set(row) for row in nn_matrix]


        for i in range(N):
            seen = {i}
            current_shell = {i}

            for d in range (1, int(self.gridsize//2+1)):
                if d>1:
                    seen.update(last_shell)
                next_shell = set()
                for node in current_shell:
                    next_shell.update(nn_sets[node])

                next_shell -= seen

                #record all pairs at this distance 
                for j in next_shell:
                    shells[d].append((i,int(j)))

                #prepare for next layer
                seen.update(current_shell)
                current_shell = next_shell
                last_shell = current_shell

                if not current_shell:
                    break

        self.shells = shells


    def unique_pairs_at_fixed_distance(self, distance):
        """This function adapts the output from compute_shells to only return pairs
        at a specified distance and only unique pairs (i,e., (i,j) but not (j,i))
        distance: integer distance (shell number) NEVER PUT IN ZERO"""
        
        self.compute_shells()
        all_pairs = self.shells.get(distance, [])
        unique_pairs=set()
        for i, j in all_pairs:
            if i<j:
                unique_pairs.add((i,j))
        return list(unique_pairs)
    

    def correlation_function(self, frame, distance):
        """computes the correlation function of a specific frame and specified distance, 
        using the definition C(r) = SUM(s_j - <s>)(s_i - <s>) / N_r"""
        
        states = self.cell_4state[:, frame]
        N= int(self.N)
        mean_state = np.mean(states)
        pairs= self.unique_pairs_at_fixed_distance(distance)
        N_r=len(pairs)
        
        if N_r ==0:
            return 0.0
        correlation_sum = 0.0
        for i,j in pairs:
            correlation_sum += (states[i]-mean_state)*(states[j]-mean_state)

        C_r=correlation_sum / N_r
        
        return C_r
    


    def correlation_function_complex_plane(self, frame, distance):
        """computes the correlation function of a specific frame and specified distance, 
        using the definition C(r) = SUM(s_j - <s>)complex conjugate(s_i - <s>) / N_r
        
        
        
        ALERT!! this is not properly normalized nor does it really work for the 4 state systen"""
        
        states = self.cell_4state[:, frame]
        states = np.exp(1j * np.pi/2 * states)
        N= int(self.N)
        mean_state = np.mean(states)
        pairs= self.unique_pairs_at_fixed_distance(distance)
        N_r=len(pairs)
        
        if N_r ==0:
            return 0.0
        correlation_sum = 0.0
        for i,j in pairs:
            what_i_conjugate=states[j]-mean_state
            correlation_sum += (states[i]-mean_state)*what_i_conjugate.conjugate()

        #correlation_sum_magnitude=(correlation_sum.real**2 + correlation_sum.imag**2)**0.5

        C_r=np.abs(correlation_sum / N_r)
        
        return C_r
    
    def spacial_correlation_complex_plane(self, frame, distance):
        """computes the correlation function of a specific frame and specified distance, 
        using the definition C(r) = SUM(s_j - <s>)complex conjugate(s_i - <s>) / N_r"""
        
        states = self.cell_4state[:, frame]
        states = np.exp(1j * np.pi/2 * states)
        N= int(self.N)
        mean_state = np.mean(states)
        pairs= self.unique_pairs_at_fixed_distance(distance)
        N_r=len(pairs)
        
        if N_r ==0:
            return 0.0
        correlation_sum = 0.0
        for i,j in pairs:
            correlation_sum += ((states[i]*states[j].conjugate())-(mean_state*mean_state.conjugate()))

        #correlation_sum_magnitude=(correlation_sum.real**2 + correlation_sum.imag**2)**0.5

        #C_r=np.abs(correlation_sum / N_r)
        C_r=correlation_sum / N_r
        
        return C_r
    
    def cor_of_r (self, frame, complexexp=False):
        """computes the correlation fucntion for all distances at a given frame"""
        max_distance = int(self.gridsize//2)
        corrs = {d: 0.0 for d in range(1,max_distance+1)}
        if complexexp==True:
            for d in range(1, max_distance+1):
                C_r = self.spacial_correlation_complex_plane(frame, d)
                corrs[d] = C_r

        else:

            for d in range(1, max_distance+1):
                C_r = self.spacial_correlation_complex_plane(frame, d)
                corrs[d] = C_r
        return corrs




def calculate_triangular_lattice(gridsize):
    """
    Calculates the positions of a triangular lattice.

    Args:
        gridsize (int): The number of cells along one dimension of the grid. Total number of cells will be gridsize^2.

    Returns:
        tuple: A tuple containing:
            - pos (ndarray): A 2D array with the x and y positions of each cell in the triangular lattice.
            - lx (float): The length of the lattice in the x-direction.
            - ly (float): The length of the lattice in the y-direction.
    """
    # Set the unit lattice length in the x-direction
    lx = 1

    # Calculate the spacing between points in the x and y directions
    delx = lx / gridsize  # Distance between points in the x-direction
    dely = np.sqrt(3) / 2 * delx  # Distance between points in the y-direction, accounting for triangular packing
    ly = dely * gridsize  # Total length in the y-direction

    # Generate evenly spaced points for the grid indices
    x = np.linspace(0, gridsize - 1, gridsize)

    # Generate meshgrid for the lattice points (x, y coordinates)
    xm, ym = np.meshgrid(x, x)

    # Adjust x-coordinates for triangular arrangement (offset every other row by half the x spacing)
    x = (xm + np.mod(ym, 2) / 2) * delx

    # y-coordinates based on row index
    y = ym * dely

    # Combine the x and y coordinates into a single 2D array of positions
    pos = np.column_stack((x.flatten('F'), y.flatten('F')))  # Flatten the arrays column-wise and stack them

    return pos, lx, ly


def calculate_distance(pos, lx, ly, gz, periodic_bc):
    """
    Calculates the pairwise distance between points on a lattice, accounting for periodic boundary conditions.

    Args:
        pos (ndarray): A 2D array of positions for each point on the lattice.
        lx (float): The length of the lattice in the x-direction.
        ly (float): The length of the lattice in the y-direction.
        gz (int): Grid size, used to normalize the distances.
        periodic_bc (list): A list of two integers indicating whether periodic boundary conditions are applied
                            in the x and y directions, respectively.

    Returns:
        ndarray: A 2D array containing the distances between all pairs of points on the lattice.
    """
    # Create meshgrid of all pairwise x and y distances between positions
    [x1, x2] = np.meshgrid(pos[:, 0], pos[:, 0])  # x1 and x2 are grids of the x-coordinates of the points
    [y1, y2] = np.meshgrid(pos[:, 1], pos[:, 1])  # y1 and y2 are grids of the y-coordinates of the points

    # Calculate the absolute differences between x and y coordinates
    dx = np.mod(abs(x1 - x2), lx)  # Modulo to handle periodic boundary in x-direction
    dy = np.mod(abs(y1 - y2), ly)  # Modulo to handle periodic boundary in y-direction

    # Apply periodic boundary conditions in the x-direction if specified
    if periodic_bc[0] == 1:
        dx[dx > (lx - dx)] = lx - dx[dx > (lx - dx)]  # Adjust for periodic wraparound in the x-direction

    # Apply periodic boundary conditions in the y-direction if specified
    if periodic_bc[1] == 1:
        dy[dy > (ly - dy)] = ly - dy[dy > (ly - dy)]  # Adjust for periodic wraparound in the y-direction

    # Calculate the Euclidean distance between all pairs of points
    dist = (dx ** 2 + dy ** 2) ** 0.5  # Pythagorean theorem to calculate distance

    # Normalize the distances by the grid size
    dist = dist / (lx / gz)

    return dist


def init_I(init_on, a0, dist, N, I_min, dI, debug=False):
    """
    Initializes the cell states based on the initial configuration and adjusts the Moran's I value
    until it is within the specified range.

    Args:
        init_on (array-like): Initial number of 'on' cells for each of the two states.
        a0 (float): Lattice constant for scaling.
        dist (ndarray): Pairwise distance matrix between cells.
        N (int): Number of cells in the lattice.
        I_min (float): Minimum threshold for Moran's I.
        dI (float): The allowed deviation from I_min for Moran's I.

    Returns:
        cells (ndarray): An N x 2 array representing the final state of cells after initialization.
    """
    # Initialize all cells to zero
    cells = np.zeros((N, 2))

    # Loop over the two states to initialize cells
    for idx in range(2):
        # Set the first 'init_on' number of cells to 1 for each state
        cells[:int(init_on[idx]), idx] = 1

        # Randomly permute the cells in each state to shuffle their positions
        cells[:, idx] = np.random.permutation(cells[:, idx])

        # Set the maximum number of iterations for the adjustment process
        maxsteps = 5000

        # Calculate Moran's I and theta for the initial configuration of cells
        I, theta = calculate_moranI(cells[:, idx], a0 * dist)

        # Determine whether the cells are in a valid state for Moran's I adjustment
        if sum(cells[:, idx]) == N or sum(cells[:, idx]) == 0 or sum(cells[:, idx]) == 1 or sum(cells[:, idx]) == N - 1:
            check = False  # Stop adjustment if all cells are identical or if the state is fully determined
        else:
            check = True  # Proceed with adjustment

        # Set a small epsilon value for distance thresholding
        eps = 1e-5

        # Calculate unique distance values from the distance matrix
        dist_vec = calculate_unique_distances(dist, eps)
        if debug:
            print(f"\n=== DEBUG init_I (idx={idx}) ===")
            print(f"Unique distances (dist_vec): {dist_vec}")
            print(f"eps: {eps}")
            print(f"dist matrix shape: {dist.shape}")
            print(f"dist min: {np.min(dist)}, max: {np.max(dist)}")
            print(f"dist contains NaN: {np.any(np.isnan(dist))}, Inf: {np.any(np.isinf(dist))}")
        
        dist1 = dist_vec[1]
        if debug:
            print(f"dist1 (second unique distance): {dist1}")
            print(f"dist1 + eps: {dist1 + eps}")
            print(f"dist1 - eps: {dist1 - eps}")

        # Identify first nearest neighbors based on distance
        temp1 = dist1 + eps > dist
        temp2 = dist > dist1 - eps
        first_nei_idx = temp1 * temp2  # Boolean mask for first neighbors
        if debug:
            print(f"first_nei_idx number of True values: {np.sum(first_nei_idx)}")

        # Initialize a matrix indicating the first neighbors
        first_nei = np.zeros_like(first_nei_idx)
        first_nei[first_nei_idx] = 1  # Set first neighbor cells to 1
        if debug:
            print(f"first_nei min: {np.min(first_nei)}, max: {np.max(first_nei)}, sum: {np.sum(first_nei)}")
            print(f"first_nei contains NaN: {np.any(np.isnan(first_nei))}, Inf: {np.any(np.isinf(first_nei))}")

        # Determine whether to increase or decrease Moran's I
        increase = (I < I_min)
        t = 0  # Iteration counter

        # Calculate the upper limit for Moran's I
        I_max = I_min + dI

        # Iterate until Moran's I is within the specified range or the max number of steps is reached
        while (I < I_min or I > I_max) and t < maxsteps and check:
            t += 1  # Increment the iteration count

            # Create a copy of the current cell state
            cells_new = cells[:, idx]

            # Calculate the number of 'on' neighbors for each cell
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                nei_ON_temp = np.matmul(first_nei, cells_new)
                nan_count = np.isnan(nei_ON_temp).sum()
                inf_count = np.isinf(nei_ON_temp).sum()
                if debug and (nan_count > 0 or inf_count > 0):
                    print(f"  ⚠️  nan_to_num NEEDED: NaN={nan_count}, Inf={inf_count}")
                nei_ON = np.nan_to_num(nei_ON_temp, nan=0.0, posinf=0.0, neginf=0.0)
            if debug:
                print(f"  Iteration {t}: cells_new shape: {cells_new.shape}, sum: {np.sum(cells_new)}, dtype: {cells_new.dtype}")
                print(f"  cells_new contains NaN: {np.any(np.isnan(cells_new))}, Inf: {np.any(np.isinf(cells_new))}")
                print(f"  nei_ON after matmul shape: {nei_ON.shape}, min: {np.min(nei_ON)}, max: {np.max(nei_ON)}")
                print(f"  nei_ON contains NaN: {np.any(np.isnan(nei_ON))}, Inf: {np.any(np.isinf(nei_ON))}")
            nei_ON_1 = nei_ON[cells_new > 0]  # Only consider cells that are 'on'
            cond1 = 1  # Initialize condition for updating cells

            # Decide which cells to update based on the current state and Moran's I
            if increase:
                # If increasing Moran's I, find cells with fewer than 3 'on' neighbors
                idx_temp = res = np.argwhere(nei_ON_1 < 3)
                if idx_temp.size == 0:
                    # If no such cells, choose cells with the minimum number of 'on' neighbors
                    idx_temp = np.argwhere(nei_ON_1 == np.min(nei_ON_1))
                    cond1 = 0  # Set condition to 0 to prevent further increase
            else:
                # If decreasing Moran's I, find cells with more than 3 'on' neighbors
                idx_temp = res = np.argwhere(nei_ON_1 > 3)
                if idx_temp.size == 0:
                    # If no such cells, choose cells with the maximum number of 'on' neighbors
                    idx_temp = np.argwhere(nei_ON_1 == np.max(nei_ON_1))
                    cond1 = 0.5  # Set condition to prevent excessive decrease

            # Randomly choose an 'on' cell to switch off
            idx_ON = np.random.choice(idx_temp.ravel())
            allON = np.argwhere(cells_new == 1)
            idx_1 = allON[:int(idx_ON)]

            # Calculate the number of 'on' neighbors for cells that are 'off'
            nei_ON_0 = nei_ON[cells_new < 1]
            cond2 = 1  # Initialize condition for switching 'off' cells

            # Decide which 'off' cells to update based on the current state
            if increase:
                # If increasing Moran's I, find 'off' cells with more than 3 'on' neighbors
                idx_temp = res = np.argwhere(nei_ON_0 > 3)
                if idx_temp.size == 0:
                    # If no such cells, choose 'off' cells with the minimum number of 'on' neighbors
                    idx_temp = np.argwhere(nei_ON_0 == np.min(nei_ON_0))
                    cond2 = 0  # Set condition to 0 to prevent further switching
            else:
                # If decreasing Moran's I, find 'off' cells with fewer than 3 'on' neighbors
                idx_temp = res = np.argwhere(nei_ON_0 < 3)
                if idx_temp.size == 0:
                    # If no such cells, choose 'off' cells with the maximum number of 'on' neighbors
                    idx_temp = np.argwhere(nei_ON_0 == np.max(nei_ON_0))
                    cond2 = 0  # Set condition to prevent further switching

            # Randomly choose an 'off' cell to switch on
            idx_OFF = np.random.choice(idx_temp.ravel())
            allOFF = np.argwhere(cells_new == 0)
            idx_0 = allOFF[:int(idx_OFF)]

            # Perform the cell state update based on the increase/decrease flag
            if increase:
                idx_inc = 1  # Define how many cells to switch
                cells_new[idx_0[-idx_inc:]] = abs(cells_new[idx_0[-idx_inc:]] - 1)  # Switch 'off' cells 'on'
                cells_new[idx_1[-idx_inc:]] = abs(cells_new[idx_1[-idx_inc:]] - 1)  # Switch 'on' cells 'off'
            else:
                cells_new[idx_0[-1:]] = abs(cells_new[idx_0[-1:]] - 1)  # Switch 'off' cells 'on'
                cells_new[idx_1[-1:]] = abs(cells_new[idx_1[-1:]] - 1)  # Switch 'on' cells 'off'

            # Recalculate Moran's I for the updated configuration
            I_new, theta = calculate_moranI(cells_new, a0 * dist)

            # Update the cell configuration based on the new Moran's I
            if cond1 == 1 and cond2 == 1:
                cells[:, idx] = cells_new  # Accept the new configuration
                I = I_new  # Update Moran's I
            elif increase and (I_new >= I):
                cells[:, idx] = cells_new  # Accept if increasing I and the new I is higher
                I = I_new  # Update Moran's I
            elif not increase and I_new <= I:
                cells[:, idx] = cells_new  # Accept if decreasing I and the new I is lower
                I = I_new  # Update Moran's I

            # Determine whether to continue increasing Moran's I
            increase = (I < I_min)

    return cells  # Return the initialized cell configuration


def calculate_moranI(cells, dist):
    """
    Calculates Moran's I for the current cell configuration, which measures spatial autocorrelation.

    Args:
        cells (ndarray): Array representing the state of cells.
        dist (ndarray): Pairwise distance matrix between cells.

    Returns:
        tuple: Moran's I and the calculated spatial autocorrelation (theta).
    """
    # Convert cell states to +/- 1 for Moran's I calculation
    cells_pm = 2 * cells - 1

    # Calculate the mean value of the cells
    cell_mean = np.mean(cells_pm)

    # Create meshgrids for cell pairwise products
    cells_matx, cells_maty = np.meshgrid(cells_pm, cells_pm)

    # Calculate the distance-based weights for Moran's I
    idx = dist > 0  # Ignore self-pairs (distance 0)

    # Create a weight matrix based on inverse distances
    M = np.zeros_like(dist)
    M[idx] = (np.exp(-dist[idx]) / dist[idx])  # Weighting by distance decay

    # Sum of the weights
    w_summed = np.sum(np.exp(-dist[idx]) / dist[idx])

    # Calculate spatial autocorrelation (theta) for Moran's I
    theta = np.sum(M * cells_matx * cells_maty) / w_summed

    # Calculate the numerator for Moran's I
    temp = np.sum(M * (cells_matx - cell_mean) * (cells_maty - cell_mean))

    # Compute the variance of the cells
    if temp != 0:
        cells_var = np.var(cells_pm, axis=0)
        I = np.sum(temp) / w_summed / cells_var  # Calculate Moran's I
    else:
        I = 0  # Set Moran's I to 0 if there is no variance

    return I, theta  # Return Moran's I and theta


def calculate_unique_distances(dist, eps):
    """
    Calculates unique distances from the distance matrix with a specified precision.

    Args:
        dist (ndarray): Pairwise distance matrix between cells.
        eps (float): Small epsilon value for rounding precision.

    Returns:
        ndarray: Array of unique rounded distances.
    """
    # Calculate the precision based on the epsilon value
    round_value = int(abs(np.log(eps) / np.log(10)))

    # Round the distances to the specified precision and return unique values
    distance_value = np.unique(np.round(dist, round_value))

    return distance_value


def force_input_matrix_shape(M_in, lattice_type):
    """
    Ensures that the input matrix has the correct shape based on the type specified. The function handles
    matrices with one or two layers and adjusts them accordingly for input processing.

    Args:
        M_in (ndarray): The input matrix.
        type (int): Type 1 or 2 indicating the desired matrix shape transformation.

    Returns:
        tuple: The reshaped matrix and a flag indicating if a double topology is present.
    """
    # Get the shape of the input matrix
    sz = M_in.shape
    double_topology_flag = False  # Flag to indicate if the matrix has double topology

    # Reshape the input matrix based on the specified type
    if lattice_type == 1:
        # Ensure a (2, 2, 2) matrix output for type 1
        M_out = np.zeros((2, 2, 2))

        # Handle input matrices with different dimensions
        if len(sz) == 3 and sz[2] == 1: #if 3 dimensional and the third dimension has length of 1, add the same metrix on top
            # Single layer: replicate across both layers
            M_out[0, :, :] = M_in
            M_out[1, :, :] = M_in
        elif len(sz) == 2: #if 2 dimensional  add the same metrix on top
            # 2D matrix: replicate across both layers
            M_out[0, :, :] = M_in
            M_out[1, :, :] = M_in
        elif len(sz) == 3 and sz[2] == 2: #if three dimensional and the third dimension has length 2 that is good
            #in this case we have a double topology idk what that means 
            # Double layer matrix
            M_out = M_in
            double_topology_flag = True  # Set flag for double topology
        else:
            # Handle incorrect input size
            print('Incorrect input size')
    elif lattice_type == 2: 
        # Ensure a (2, 1, 2) matrix output for type 2
        M_out = np.zeros((2, 1, 2))
        if len(sz) == 3 and sz[2] == 1: #if 3 dimensional and the third dimension has length of 1,
            # Transpose and replicate for single layer
            M_out[0, :, :] = M_in.T
            M_out[1, :, :] = M_in.T
        elif len(sz) == 1:
            # Handle 1D matrix
            M_out[0, :, :] = M_in
            M_out[1, :, :] = M_in
        elif len(sz) == 3 and sz[2] == 2:
            # Double layer matrix
            M_out = M_in
        else:
            # Handle incorrect input size
            print('Incorrect input size')

    else:
        # Handle incorrect type
        M_out = []
        print('Incorrect input size')

    return M_out, double_topology_flag  # Return the reshaped matrix and flag



# Running the cellular automaton
def run_CA(M, K, N, tmax, dist, Rcell, a0, lamb, idx_celltype, Coff, Con, cells):
    """
    Runs the cellular automaton model for a maximum number of timesteps (tmax). It simulates cell interactions
    using given matrices for inter-cell communication, diffusion, and cell states.

    Args:
        M (ndarray): Interaction matrix for the cellular automaton.
        K (ndarray): Threshold matrix for cell states.
        N (int): Number of cells in the system.
        tmax (int): Maximum number of timesteps to run the simulation.
        dist (ndarray): Pairwise distance matrix between cells.
        Rcell (float): Cell interaction radius.
        a0 (float): Lattice constant for distance scaling.
        lamb (ndarray): Diffusion lengths for each molecule.
        idx_celltype (ndarray): Boolean array indicating the type of each cell.
        Coff (ndarray): Basal secretion rate matrix for inactive cells.
        Con (ndarray): Secretion rate matrix for active cells.
        cells (ndarray): Initial states of cells.

    Returns:
        cells_hist (ndarray): History of cell states over time.
        t (int): Final timestep of the simulation.
        Y (ndarray): Interaction strength between cells over time.
    """
    # Transpose the interaction matrix and threshold matrix for easier matrix operations later
    M_int = np.transpose(M)
    K = np.transpose(K)

    # Initialize the current timestep to 0
    t = 0

    # Initialize the history of cell states to store the state of each cell at each timestep
    cells_hist = np.zeros((N, 2, tmax), np.int8)

    # Create a boolean mask to indicate which cells have non-zero distances from others
    idx = dist > 0

    # Initialize the interaction strength matrix (M) and state update matrix (Y)
    M = np.ones((N, N, 2))
    Y = np.ones((N, 2, tmax))

    # Calculate the interaction strengths between cells based on their distances and diffusion lengths
    for k in range(2):
        # Apply the interaction strength formula based on sinh, distance, and diffusion length
        M[idx, k] = np.sinh(Rcell / lamb[k]) / (a0 * dist[idx] / lamb[k]) * np.exp((Rcell - a0 * dist[idx]) / lamb[k])

    # Set a flag to detect if a periodic state has been reached (early termination)
    period = False

    # Main loop to simulate the cellular automaton over time
    while t < tmax and not period:
        # Record the current cell states in the history array
        cells_hist[:, :, t] = cells

        # Initialize a loop to go through different cell types (active and inactive)
        idx_loop = idx_celltype

        # Temporary arrays to store the state updates and interaction results
        out = np.zeros((N, 4))
        C0 = np.zeros((N, 2))
        cells_out = np.zeros((N, 2))

        # Calculate secretion rates based on current cell states (active or inactive)
        C0[idx_loop, :] = Coff[0, :, :] + (Con[0, :, :] - Coff[0, :, :]) * cells[idx_loop, :]
        C0[np.invert(idx_loop), :] = Coff[1, :, :] + (Con[1, :, :] - Coff[1, :, :]) * cells[np.invert(idx_loop), :]

        # Calculate the interaction strengths (Y) for each molecule and update cell states
        for k in range(2):
            # Matrix multiplication to calculate interaction effects on cells
            Y[:, k, t] = np.matmul(np.squeeze(M[:, :, k]), C0[:, k])

        # Iterate through each molecule to calculate cell state changes based on interaction thresholds
        for j in range(2):
            # Calculate the state of cells based on interaction strengths and thresholds
            out[idx_loop, 0] = ((Y[idx_loop, 0, t] - np.squeeze(K[0, 0, j])) * np.squeeze(M_int[0, 0, j]) > 0) + (
                    1 - abs(M_int[0, 0, j]))
            out[idx_loop, 1] = ((Y[idx_loop, 1, t] - np.squeeze(K[1, 0, j])) * np.squeeze(M_int[1, 0, j]) > 0) + (
                    1 - abs(M_int[1, 0, j]))
            out[idx_loop, 2] = ((Y[idx_loop, 0, t] - np.squeeze(K[0, 1, j])) * np.squeeze(M_int[0, 1, j]) > 0) + (
                    1 - abs(M_int[0, 1, j]))
            out[idx_loop, 3] = ((Y[idx_loop, 1, t] - np.squeeze(K[1, 1, j])) * np.squeeze(M_int[1, 1, j]) > 0) + (
                    1 - abs(M_int[1, 1, j]))

            # Update the cell states based on interaction outcomes for each molecule
            cells_out[idx_loop, 0] = np.squeeze(out[idx_loop, 0]) * np.squeeze(out[idx_loop, 1])
            cells_out[idx_loop, 1] = np.squeeze(out[idx_loop, 2]) * np.squeeze(out[idx_loop, 3])

            # Invert the cell type to process the other cell type in the next iteration
            idx_loop = np.invert(idx_loop)

        # Move to the next timestep and update the cells for the next iteration
        t += 1
        cells = cells_out

    # Return the history of cell states, the final timestep, and the interaction strength matrix
    return cells_hist, t, Y


def calculate_rectangular_lattice(gridsize):
    lx = 1

    delx = lx / gridsize
    dely = np.sqrt(3) / 2 * delx
    ly = dely * gridsize

    x = np.linspace(0, gridsize - 1, gridsize)
    xm, ym = np.meshgrid(x, x)

    x = (xm + np.mod(ym, 2) / 2) * delx
    y = ym * dely

    pos = np.column_stack((x.flatten('F'), y.flatten('F')))

    return pos, lx, ly









def compute_annihilation_moments(self):
    n_cores = self.n_vortex
    n_vortex = np.sum(self.charges != 0, axis=1)

    delta_n_vortex = np.diff(n_cores, axis=0) == -1
    delta_n_charge = np.diff(n_vortex, axis=0) == -2

    idx_annihilation_moments = np.argwhere(delta_n_vortex * delta_n_charge * (n_cores[:-1] == n_vortex[:-1]))
    idx_annihilation_moments_core_count = n_vortex[idx_annihilation_moments]

    return idx_annihilation_moments, idx_annihilation_moments_core_count


def compute_smallest_movement(self):
    t_end = self.tmax
    t_start = 0

    charges = np.zeros((t_end - t_start, 2 + int(np.max(self.vortex_cores_labeled))))
    # Initialize arrays to store charges and center of mass positions
    #the array is of size (number of time steps, number of vortex cores + 2)
    #the +2 is to account for the fact that the vortex cores are labeled starting from 1, and we want to have an index for 0 as well
    #the first two columns are for the total charge and the total number of vortex cores
    #the rest of the columns are for the individual vortex cores
    #the number of colums is determined by the maximum label in the vortex_cores_labeled array
    CmX = np.zeros((t_end - t_start, 2 + int(np.max(self.vortex_cores_labeled))))
    #CmX and CmY store the x and y coordinates of the center of mass of each vortex core
    CmY = np.zeros((t_end - t_start, 2 + int(np.max(self.vortex_cores_labeled))))

    if self.number_of_nn == 6:
        #ie if triangular lattice
        lx = np.max(self.positions[:, 0]) + self.positions[0, 0]/2
        # lx and ly are the lengths of the lattice in the x and y directions
        ly = np.max(self.positions[:, 1]) + self.positions[0, 1]/2
    else:
        lx = np.max(self.positions[:, 0]) + self.positions[0, 0]
        ly = np.max(self.positions[:, 1]) + self.positions[0, 1]

    closest_pos = np.zeros((self.tmax, 20)) * np.nan
    #closest_pos will store the index of the closest vortex core in the previous time step for each vortex core in the current time step
    #the array is of size (number of time steps, 20)
    #the 20 is an arbitrary number that should be larger than the maximum number of vortex
    for p in range(t_end - t_start):
        # Set frame number
        fn = p + t_start

        # Set frame and contours of frame
        hist_run = self.cell_4phase[:, fn]
        c_run = self.vortex_cores_labeled[:, fn]
        # c_run is the labeled vortex cores for the current frame
        # hist_run is the phase of the cells for the current frame

        # Run through all labels
        idx_all = np.unique(self.vortex_cores_labeled[:, fn], axis=0)

        for idx_label in idx_all[1:]:
            #for each unique label in the vortex_cores_labeled array
            # Skip if label is zero (background)
            x_hat, y_hat, c_r, m_x_r, m_y_r = get_contour_pos_relative_2_center(c_run, hist_run, idx_label, self)
            # Get positions of contour points relative to center of mass
            # x_hat and y_hat are the x and y coordinates of the contour points relative to the center of mass
            # c_r is the phase of the contour points
            # m_x_r and m_y_r are the x and y coordinates of the center of mass of the vortex core
            #contour points are the points that make up the boundary of the vortex core

            theta_r = np.arctan2(y_hat, x_hat) + np.pi
            #put in polar coordinates
            #theta_r is the angle of the contour points relative to the center of mass
            #arctan2 returns values between -pi and pi, so we add pi to shift the range to 0 to 2pi
            idx_sort = np.argsort(theta_r)
            #sort the contour points in order of increasing angle relative to cm

            cwise = c_r[idx_sort]
            #sort the phase of the contour points in order of increasing angle
            #get ready for counter clockwise walk
            cwise_wrap = np.zeros((1, len(cwise[0, :]) + 1))
            #wrap the phase array to account for the periodicity of the phase
            cwise_wrap[0, :-1] = cwise
            cwise_wrap[0, len(cwise[0, :])] = cwise[0, 0]

            temp = np.diff(cwise_wrap)
            #calculate the difference in phase between adjacent contour points
            #this will give us an array of values between -3 and 3, where each value corresponds to a phase difference of -3pi/2, -pi, -pi/2, 0, pi/2, pi, 3pi/2
            #we can then use this array to calculate the total charge of the vortex core
            temp[temp / np.pi == -3 / 2] = 0.5 * np.pi
            temp[temp / np.pi == 3 / 2] = -0.5 * np.pi

            charges[p, idx_label +1] = np.sum(temp) / np.pi
            #CHANGED TO +1
            #calculate the total charge of the vortex core
            #the charge is the sum of the phase differences divided by pi
            #store the charge in the charges array
            CmX[p, idx_label + 1] = m_x_r
             #CHANGED TO +1
            #store the x coordinate of the center of mass in the CmX array
            CmY[p, idx_label + 1] = m_y_r
             #CHANGED TO +1
            #store the y coordinate of the center of mass in the CmY array

        # if p > 0:
        #     tempX = CmX[p, :]
        #     tempY = CmY[p, :]
        #     tempX_min = CmX[p - 1, :]
        #     tempY_min = CmY[p - 1, :]
        #
        #     idx_true = charges[p-1, :] != 0
        #
        #     dx = (tempX[idx_true, np.newaxis].T - tempX_min[idx_true, np.newaxis])
        #     dy = (tempY[idx_true, np.newaxis].T - tempY_min[idx_true, np.newaxis])
        #
        #     if self.periodic_bc[0] == 1:
        #         dx = (dx + lx / 2) % lx - lx / 2
        #
        #     if self.periodic_bc[1] == 1:
        #         dy = (dy + ly / 2) % ly - ly / 2
        #
        #     dist = (dx ** 2 + dy ** 2) ** 0.5
        #
        #     if dist.size != 0:
        #         closest_pos[p, :len(dist[0, :])] = np.argmin(dist, axis=0)

        #this section has been commented out since i inhertited the code. I am not sure why it is 
        #commented out but I have other ways of doing what this does 
        #so i leave it commented (elinor)

    return closest_pos, charges, CmX, CmY



def interchange_numbers(input_array, v1, v2):
    idx_v1 = input_array == v1
    idx_v2 = input_array == v2

    input_array[idx_v1] = v2
    input_array[idx_v2] = v1

    return input_array


def compute_phase_differences(self):
    # using integer numbers to speed up computations
    C4 = self.cell_4state.astype(np.int8)
    C4 = interchange_numbers(C4, 4, 3)

    temp = C4[self.cell_number_nn.ravel(), :]

    frame_values = temp.reshape(self.N, self.number_of_nn + 1, self.tmax)

    temp = frame_values[:, 0, :]
    delta = frame_values[:, 1:] - temp[:, np.newaxis, :]

    # Determine cos of the phase differences
    cos_nn = lut_Hxy(delta)
    xy_adjusted_nn = lut_Hxy_adjusted(delta)

    # Assign integer value to vortex cores
    vortex_cores = (np.squeeze(np.sum(abs(delta) == 2, axis=1)) > 0) * 1

    # Compute Hamiltonians from data
    Hxy = -np.squeeze(np.sum(cos_nn, axis=1)) * 1 / (self.N * self.number_of_nn)
    Hxy_adjusted = -1 / 2 - 2 * np.squeeze(np.sum(xy_adjusted_nn, axis=1)) * 1 / (self.N * self.number_of_nn)

    return vortex_cores, np.sum(Hxy, axis=0), np.sum(Hxy_adjusted, axis=0)


def lut_Hxy(values):
    # mimics the cosine for {-pi*3/2, -pi, -pi/2, 0, pi/2, pi, pi*3/2}

    values = abs(values)
    idx0 = values == 0
    idx1 = values == 1
    idx2 = values == 2
    idx3 = values == 3

    values[idx0] = 1
    values[idx1] = 0
    values[idx2] = -1
    values[idx3] = 0

    return values


def lut_Hxy_adjusted(values):
    # mimics the Xy Hamiltonain with corrected term for {-pi*3/2, -pi, -pi/2, 0, pi/2, pi, pi*3/2}

    values = abs(values)
    idx0 = values == 0
    idx1 = values == 1
    idx2 = values == 2
    idx3 = values == 3

    values[idx0] = 0
    values[idx1] = 0
    values[idx2] = -1
    values[idx3] = 0

    return values


def get_final_configuration_type(self):
    # Check if the last time step has only one unique state across all cells
    # This condition checks if all cells are in the same state at the last time step.
    # NOTE: '-1' refers to the last column (i.e., final timestep)
    if len(np.unique(self.cell_4state[:, -1])) == 1:
        final_pattern_type = 1  # Pattern is static: All cells have converged to the same, single state.

    # Check if the fraction of cells in state 2 is constant after the first recurrent state time
    # This condition checks if the system has reached a dynamic equilibrium or periodic state where the fractions
    # of cells in state 2 do not change after the first recurrent state time.
    elif len(np.unique(np.diff(self.state_fractions[1, self.first_recurrent_state_time:]))) == 1:
        final_pattern_type = 3  # Pattern is periodic: A dynamic steady-state is achieved.

    # If neither of the above conditions is met, the system is still evolving and has not settled into
    # a static or periodic state. This could indicate a more complex or chaotic dynamic pattern.
    else:
        final_pattern_type = 2  # Pattern is non-periodic or dynamic: No convergence to a simple pattern.

    # A static outcome is defined by its dynamics: the configuration maps to
    # itself under a single update, so the attractor has period 1. Spatial
    # uniformity, tested first above, is the usual signature of this and is
    # sufficient but not necessary, since a configuration can be frozen
    # without being spatially uniform. A frozen configuration also has
    # unchanging state fractions, which the rectilinear test cannot
    # distinguish from a rigidly translating wave, so period 1 is tested
    # directly here.
    #
    # Determinism makes this test exact: if two consecutive configurations
    # are identical then the configuration is a fixed point at all later
    # times.
    if final_pattern_type != 1 and np.shape(self.cell_4state)[1] >= 2:
        if np.array_equal(self.cell_4state[:, -1], self.cell_4state[:, -2]):
            final_pattern_type = 1

    return final_pattern_type


def get_first_recurrent_state_time(self):
    cell4 = self.cell_4state

    unique_configurations = np.unique(cell4, axis=1)
    sz = unique_configurations.shape
    p_time = sz[1]

    return p_time

def get_time_that_is_repeated(self, vectorize=False):
    if self.first_recurrent_state_time >= self.cell_4state.shape[1]:
        return None
    

    if vectorize:
        first_reccurent_time=self.first_recurrent_state_time
        first_reccurent_state = self.cell_4state[:,first_reccurent_time]
        matches = np.all(self.cell_4state == first_reccurent_state[:, None], axis=0)

        # Ignore the reference time itself if needed
        matches[first_reccurent_time] = False

        # First timestep where all cells match
        repeated_times = np.flatnonzero(matches)
        return repeated_times[0] if repeated_times.size > 0 else None
    
    if not vectorize:
        #iteratatively check all the cells states, this is much slower but doesnt create a temporary array
        #so it is more memory efficient

        first_reccurent_time=self.first_recurrent_state_time
        first_reccurent_state = self.cell_4state[:,first_reccurent_time]
        # for t in range(self.cell_4state.shape[1]):
        for t in range(first_reccurent_time-1, -1, -1): #Start Stop Step
            if t == first_reccurent_time:
                continue
            if np.array_equal(self.cell_4state[:, t], first_reccurent_state):
                return t

    return None
        




def compute_4_state_fractions(self):
    self.cell_4state[0]
    state_fractions = np.zeros((4, self.tmax))
    state_fractions[0, :] = np.sum(self.cell_4state == 1, axis=0) / self.N
    state_fractions[1, :] = np.sum(self.cell_4state == 2, axis=0) / self.N
    state_fractions[2, :] = np.sum(self.cell_4state == 3, axis=0) / self.N
    state_fractions[3, :] = np.sum(self.cell_4state == 4, axis=0) / self.N

    return state_fractions


def compute_red_cell_count(self):
    """Return the number of red (state 2) cells at every timestep.

    Returns:
        red_counts (ndarray): 1D array of length tmax, where each entry is
            the number of cells in state 2 at that timestep.
    """
    red_counts = np.sum(self.cell_4state == 2, axis=0)
    return red_counts


def label_triangular_lattice(self):
    tmax = self.tmax
    vortex_cores = copy.deepcopy(self.vortex_cores)
    #vortex_cores is a list that was created in compute_phase_differences. It is just a binary 
    #mask of whether each cell is part of a set of cells that have a pi phase difference to its neighbors
    #this function returns a labeled version of that array, where each set of connected cells is given a unique integer label

    # Apply two-pass algorithm writen for a triangular lattice
    for frame_number in range(0, tmax):
        # phase 1: first labeling
        labels = np.where(vortex_cores[:, frame_number] == 1)
        #any cells that is part of a pi phase difference set
        #np.where returns a tuple of arrays, one for each dimension of the input array.
        #in this case, it returns a tuple with one array containing the indices of the cells
        for i in range(len(labels[0][:])):
            current_cell = vortex_cores[labels[0][i], frame_number]
            #label[0] just means the first dimension of the tuple, which contains the indices of the cells
            #i is the index of the current cell in the list of cells that are part of
            #index over all cells that are part of a pi phase difference set

            if current_cell == 1:
                #if the current cell is part of a pi phase difference set
                neighbours_idx = np.where(np.round(self.relative_distance[labels[0][i], :], 1) == 1)
                #find the indices of all neighboring cells that are exactly 1 unit away
                neighbours_values = vortex_cores[neighbours_idx, frame_number]
                #get the values of those neighboring cells in the vortex_cores array
                temp = neighbours_values[neighbours_values > 1]
                #to have label of 1 means that the cell is part of a pi phase difference set but not yet labeled
                #so we only care about neighbors that have already been labeled (i.e., have a value greater than 1)

                if not any(temp):
                    min_neighbour_value = 1
                    #if none of the neighbors have been labeled yet, set min_neighbour_value to 1
                    #this mean unlabeled
                else:
                    min_neighbour_value = min(temp)
                    #otherwise, set min_neighbour_value to the minimum label value among the labeled neighbors

                if min_neighbour_value == 1:
        
                    vortex_cores[labels[0][i], frame_number] = np.max(vortex_cores[:, frame_number]) + 1
                    #if min_neighbour_value is 1, it means none of the neighbors have been labeled yet
                    #so we assign a new label to the current cell, which is one greater than the current maximum label in the frame
                else:
                    vortex_cores[labels[0][i], frame_number] = min_neighbour_value
                    #if min_neighbour_value is greater than 1, it means at least one neighbor has been labeled
                    #so we assign the minimum label value among the labeled neighbors to the current cell

        # phase 2: assign minimum to all connected and labeled areas
        for i in range(len(labels[0][:])):
            neighbours_idx = np.where(np.round(self.relative_distance[labels[0][i], :], 1) <= 1)
            #find the indices of all neighboring cells that are exactly 1 unit away or less
            neighbours_values = vortex_cores[neighbours_idx, frame_number]
            neighbours_unique = np.unique(neighbours_values[neighbours_values > 0])
            if len(neighbours_unique) > 1:
                for j in range(len(neighbours_unique)):
                    vortex_cores[vortex_cores[:, frame_number] == neighbours_unique[j], frame_number] = \
                        np.min(neighbours_unique)
                    #if the current cell has multiple labeled neighbors, we assign the minimum label value among those neighbors to all cells with those labels

        # phase 3: Make sure no index is missed (e.g. [1 2 4 5] --> [1 2 3 4])
        # more elegant: np.unique(a, return_inverse=True)[1].reshape(a.shape)

        for i in range(np.max(vortex_cores[:, frame_number])):
            if not any(vortex_cores[:, frame_number] == (i + 0)):
                vortex_cores[vortex_cores[:, frame_number] >= (i + 0), frame_number] -= 1

    vortex_cores_labeled = vortex_cores
    n_vortex = np.max(vortex_cores, axis=0)

    return vortex_cores_labeled, n_vortex


def label_elements_triangular_lattice(input_images, relative_distance, t_start, t_end):
    #the arguments of this function are:
    #input_images: a 2D numpy array where each column represents a time frame and each row represents a cell
    #the values in the array are either 0 (background) or 1 (part of an element to be labeled)
    #relative_distance: a 2D numpy array representing the pairwise distances between cells
    #t_start: the starting time frame (inclusive)
    #t_end: the ending time frame (exclusive)
    # Apply two-pass algorithm writen for a triangular lattice
    for frame_number in range(t_start, t_end):
        # phase 1: first labeling
        labels = np.where(input_images[:, frame_number] == 1)
        #any cells that is part of an element to be labeled
        #np.where returns a tuple of arrays, one for each dimension of the input array.
        for i in range(len(labels[0][:])):
            #index over all cells that are part of an element to be labeled


            current_cell = input_images[labels[0][i], frame_number]

            if current_cell == 1:
                #if the current cell is part of an element to be labeled 
                neighbours_idx = np.where(np.round(relative_distance[labels[0][i], :], 1) == 1)
                neighbours_values = input_images[neighbours_idx, frame_number]
                temp = neighbours_values[neighbours_values > 1]
                #to have label of 1 means that the cell is part of an element to be labeled but not yet labeled
                #so we only care about neighbors that have already been labeled (i.e., have a value greater than 1)

                if not any(temp):
                    min_neighbour_value = 1
                else:
                    min_neighbour_value = min(temp)

                if min_neighbour_value == 1:
                    input_images[labels[0][i], frame_number] = np.max(input_images[:, frame_number]) + 1
                else:
                    input_images[labels[0][i], frame_number] = min_neighbour_value

        # phase 2: assign minimum to all connected and labeled areas
        for i in range(len(labels[0][:])):
            neighbours_idx = np.where(np.round(relative_distance[labels[0][i], :], 1) == 1)
            neighbours_values = input_images[neighbours_idx, frame_number]
            neighbours_unique = np.unique(neighbours_values[neighbours_values > 0])
            if len(neighbours_unique) > 1:
                for j in range(len(neighbours_unique)):
                    input_images[input_images[:, frame_number] == neighbours_unique[j], frame_number] = \
                        np.min(neighbours_unique)

        # phase 3: Make sure no index is missed (e.g. [1 2 4 5] --> [1 2 3 4])
        # more elegant: np.unique(a, return_inverse=True)[1].reshape(a.shape)

        for i in range(np.max(input_images[:, frame_number])):
            if not any(input_images[:, frame_number] == (i + 0)):
                input_images[input_images[:, frame_number] >= (i + 0), frame_number] -= 1

    images_labeled = input_images[:, t_start:t_end]
    number_of_elements = np.max(input_images, axis=0)

    #this function does the same thing as label_triangular_lattice but for arbitrary binary images
    #it returns a labeled version of the input images, where each connected component is given a unique integer label
    #it also returns the number of connected components in each frame

    return images_labeled, number_of_elements


def compute_wrap_coefficients(trajectory, fn_start, fn_end):

    periodic_neighbours = np.round(trajectory.relative_distance, 1) <= 1

    # compute relative distance again but now with non-periodic bc
    [x1, x2] = np.meshgrid(trajectory.positions[:, 0], trajectory.positions[:, 0])
    [y1, y2] = np.meshgrid(trajectory.positions[:, 1], trajectory.positions[:, 1])

    dx = x1 - x2
    dy = y1 - y2

    r = (dx ** 2 + dy ** 2) ** 0.5
    r = r / r[0, 1]

    non_periodic_neighbours = np.round(r, 1) <= 1
    trajectory.relative_distance = r

    edge_cells = ~non_periodic_neighbours * periodic_neighbours

    y_top = trajectory.positions[:, 1] == np.max(trajectory.positions[:, 1])
    x_top = np.logical_or(trajectory.positions[:, 0] == trajectory.positions[-1, 0],
                          trajectory.positions[:, 0] == trajectory.positions[-2, 0])

    x_top_pos = np.squeeze(np.array(np.where(x_top)))
    y_top_pos = np.squeeze(np.array(np.where(y_top)))

    wrap_all = np.zeros((fn_end-fn_start, 2))

    for j in range(fn_end-fn_start):
        fn = fn_start + j

        frame = trajectory.cell_4state[:, fn]
        frame_not_4_state = (frame != 4) * 1

        trajectory.vortex_cores = frame_not_4_state[:, np.newaxis]

        trajectory.tmax = 1
        label_values, nn = label_triangular_lattice(trajectory)

        x_wrap = 0
        c = 0

        while x_wrap == 0 | c < trajectory.gridsize:
            x_wrap = any(label_values[x_top_pos[c]] == label_values[edge_cells[x_top_pos[c], :]])
            c += 1

        wrap_all[j, 0] = x_wrap

        y_wrap = 0
        c = 0

        while y_wrap == 0 | c < trajectory.gridsize:
            y_wrap = any(label_values[y_top_pos[c]] == label_values[edge_cells[y_top_pos[c], :]])
            c += 1

        wrap_all[j, 1] = y_wrap

    return wrap_all





def init_topology_mat(topology_mat_in, gz):
    Sx = len(topology_mat_in)
    Sx_scale = np.floor(gz / Sx)
    Sx_res = int(gz - Sx * Sx_scale)
    Sx_vector = [None] * Sx

    Sy = len(topology_mat_in[0])
    Sy_scale = np.floor(gz / Sy)
    Sy_res = int(gz - Sy * Sy_scale)
    Sy_vector = [None] * Sy

    for i in range(Sx):
        if Sx_res != 0:
            Sx_vector[i] = (Sx_scale + 1)
            Sx_res -= 1
        else:
            Sx_vector[i] = Sx_scale

    for i in range(Sy):
        if Sy_res != 0:
            Sy_vector[i] = Sy_scale + 1
            Sy_res -= 1
        else:
            Sy_vector[i] = Sy_scale

    temp = np.repeat(topology_mat_in, Sx_vector, axis=0)
    Topology_mat = np.repeat(temp, Sy_vector, axis=1)

    return Topology_mat



def show_cells(folderName_for_frames, trange, cells_hist, pos, idx_celltype, frame_rate, spin_vect, cores_labeled, charges, cell_4phase, make_gif, store_frames):
    """

I changed all tmax to trange to avoid confusion with the tmax variable in main.py

    This function visualizes the evolution of a cellular automaton over time. It displays the states of cells at each time step using either color-coded scatter plots or vector fields to represent different cell states and phases. The function can create animations and optionally save frames as individual images or a GIF.

    Arguments:
    - trange: Maximum number of time steps for visualization. (= T - initial frame, where T = 'tmax' defined in main.py)
    - cells_hist: A 3D array storing the history of cell states over time.
    - pos: Positions of cells in the lattice.
    - idx_celltype: Boolean array indicating different cell types.
    - frame_rate: The frame rate for animation.
    - spin_vect: Boolean to indicate whether to display vector fields.
    - cores_labeled, charges, cell_4phase: Arrays containing additional cell state information.
    - make_gif: Boolean to indicate if a GIF should be created.
    - store_frames: Boolean to indicate if individual frames should be saved.

    Returns:
    - img: A list of images (frames) representing the evolution of the cellular automaton.
    """

    # Initialize a list to store images for each frame
    img = [None] * trange
    g_vector = 2  # Gain vector for plotting adjustments


    # Set up the plot background color and layout
    bg_brightness = 240

    # Interactive mode enabled on Matplotlib, which causes the figure to update and close automatically (must be enabled for the movie)
    plt.ion()

    fig, ax = plt.subplots()
    ax.set_facecolor((bg_brightness / 255, bg_brightness / 255, bg_brightness / 255))
    fig.tight_layout()
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.gca().axes.get_xaxis().set_visible(False)  # Hide x-axis
    plt.gca().axes.get_yaxis().set_visible(False)  # Hide y-axis

    # Set initial state and parameters
    delta = 0.05  # Margin for plot limits
    cells = cells_hist[:, :, 0]  # Initial cell states
    gain = 10000 / np.sqrt(len(cells_hist[:, 0]))  # Gain for marker size adjustment

    # Separate cells by type using the provided index
    idx_celltype1 = idx_celltype
    idx_celltype2 = np.invert(idx_celltype)

    # Initialize color arrays for two cell types
    colours1 = np.asarray([None] * len(pos[idx_celltype1, 0]), dtype=object)
    colours2 = np.asarray([None] * len(pos[idx_celltype2, 0]), dtype=object)
    colours1[:] = 'blue'
    colours2[:] = 'blue'

    # Separate positions by cell type
    pos1 = pos[idx_celltype1, :]
    pos2 = pos[idx_celltype2, :]

    # Plot initial cell states
    if not spin_vect:
        # Scatter plot for two cell types with different markers
        plt_state1_1 = plt.scatter(pos1[:, 0], pos1[:, 1], s=gain, c='blue', marker="o", edgecolors='black')
        plt_state1_2 = plt.scatter(pos2[:, 0], pos2[:, 1], s=gain, c='blue', marker="s", edgecolors='black')
        plt.show()

        # Assign colors based on cell states
        colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 0)] = 'blue'
        colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 1)] = 'red'
        colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 0)] = 'white'
        colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 1)] = 'black'

        colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 0)] = 'blue'
        colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 1)] = 'red'
        colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 0)] = 'white'
        colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 1)] = 'black'

        # Update the colors in the plot
        plt_state1_1.set_facecolors(c=colours1.tolist())
        plt_state1_2.set_facecolors(c=colours2.tolist())
    else:
        # Vector field plot for cell phases
        plt.quiver(pos[:, 0], pos[:, 1], np.cos(cell_4phase[:, 0]), np.sin(cell_4phase[:, 0]))
        plt.scatter(pos[:, 0], pos[:, 1], c=cores_labeled[:, 0], edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
        # Color cells based on their core charges
        plt.show()

    # Set plot limits based on cell positions
    plt.xlim(0 - delta * 2, np.max(pos[:, 0]) + delta * 2)
    plt.ylim(0 - delta, np.max(pos[:, 1]) + delta)

    fig.canvas.draw()
    plt.show()

    # Initialize GIF creation if required
    if make_gif:
        fig.canvas.draw()
        temp = np.asarray(fig.canvas.buffer_rgba())  # gives you RGBA buffer
        temp = temp.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        temp = temp[:, :, :3]   # keep only RGB
        img[0] = temp

    # Ensure directories exist for storing frames if required
    if store_frames:
        frame_dir = folderName_for_frames
        jpg_dir = os.path.join(frame_dir, 'Frames_JPG')
        pdf_dir = os.path.join(frame_dir, 'Frames_PDF')
        os.makedirs(jpg_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

    # Main loop for visualizing each time step
    trange = len(cells_hist[0, 0, :])
    for t in range(0, trange):
        t1 = time.time()
       
        plt.title('Time step=%i' % (t + 1))
        

        if not spin_vect:
            # Update cell states
            cells = cells_hist[:, :, t]

            # Recalculate colors for cells based on new states
            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 0)] = 'blue'
            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 1)] = 'red'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 0)] = 'white'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 1)] = 'black'
            plt_state1_1.set_facecolors(c=colours1.tolist())

            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 0)] = 'blue'
            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 1)] = 'red'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 0)] = 'white'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 1)] = 'black'
            plt_state1_2.set_facecolors(c=colours2.tolist())
            fig.canvas.draw()
            plt.show()
        else:
            # Clear and redraw for vector field visualization
            plt.clf()
            plt.quiver(pos[:, 0], pos[:, 1], np.cos(cell_4phase[:, t]), np.sin(cell_4phase[:, t]))
            plt.scatter(pos[:, 0], pos[:, 1], c='white', edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
            for p in np.unique(cores_labeled[:, t]):
                if p > 0:
                    temp_label = cores_labeled[:, t] == p
                    col_idx = p + 1 if (p + 1) < charges.shape[1] else None
                    charge_val = charges[t, col_idx] if col_idx is not None else 0.0
                    if charge_val < 0:
                         #CHANGED TO +1
                        plt.scatter(pos[temp_label, 0], pos[temp_label, 1], c='orange', edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
                    elif charge_val > 0:
                         #CHANGED TO +1
                        plt.scatter(pos[temp_label, 0], pos[temp_label, 1], c='blue', edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
                    else:
                        plt.scatter(pos[temp_label, 0], pos[temp_label, 1], c='purple', edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
                    plt.title('T=' + str(t))
        fig.canvas.flush_events()
        plt.show()
        t2 = time.time()
        delta_time = t2 - t1

        # Store frames for GIF or video creation
        if make_gif:
            fig.canvas.draw()
            temp = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            temp = temp.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (H, W, 4)
            temp = temp[:, :, :3]  # keep only RGB
            img[t] = temp
        if store_frames:
            plt.savefig(os.path.join(jpg_dir, f'step{t}.jpg'), format='jpg')  # Save as JPG
            plt.savefig(os.path.join(pdf_dir, f'step{t}.pdf'), format='pdf')  # Save as PDF

        # Ensure consistent frame rate for animation
        if delta_time < 1 / frame_rate:
            time.sleep(1 / frame_rate - delta_time)

    # plt.show(block = True)   # Keep the figure window open after the last frame

    return img




def copy_CA(CA_in):
    CA_out = copy.deepcopy(CA_in)
    return CA_out


def add_defects(CA_in, mode, *argv):
    CA_defect = copy.deepcopy(CA_in)

    if mode == 'random':
        randy = int(np.round(random.uniform(0, len(CA_defect.cell_hist[:, 0]) - 1)))
        randx = int(np.round(random.uniform(0, 1)))
        CA_defect.cell_hist[randy, randx] = abs(CA_defect.cell_hist[randy, randx] - 1)
    elif mode == 'fixed':
        idx_fixed = argv
        CA_defect.cell_hist[idx_fixed[0], idx_fixed[1]] = abs(CA_defect.cell_hist[idx_fixed[0], idx_fixed[1]] - 1)
    else:
        print('No changes have been made, select a correct mode type')

    return CA_defect


def save_trajectory(file, file_path, file_name):
    path = fr'\\NAS\homes\Lars\MEP_Data\{file_path}\{file_name}.pkl'

    with open(path, 'wb') as output:
        pickle.dump(file, output, pickle.HIGHEST_PROTOCOL)


def load_trajectory(path):
    # path = fr'C:\Users\larsk\Documents\1. TU delft\1. Vakken\MEP\Python\MEP\{file_path}\{file_name}.pkl'

    with open(path, 'rb') as input:
        output = pickle.load(input)

    return output


def get_4_number_seq(cell_hist):
    """
    Converts a 2-state cell history (represented by two binary arrays) into a 4-state format.

    This function transforms the state of each cell, which is represented by two binary values
    (e.g., [0, 0], [0, 1], [1, 0], [1, 1]), into a single integer representing one of four possible states:
    - 1: State [0, 0]
    - 2: State [0, 1]
    - 3: State [1, 0]
    - 4: State [1, 1]

    Args:
        cell_hist (ndarray): A 3D array of shape (num_cells, 2, num_timesteps), where each element is either 0 or 1,
                             representing the binary state of two attributes for each cell over time.

    Returns:
        cell_hist_out (ndarray): A 2D array of shape (num_cells, num_timesteps), where each element is an integer
                                 (1, 2, 3, or 4), representing one of the four possible states for each cell at each timestep.
    """

    # Initialize an output array with zeros, having the same number of cells and timesteps as the input.
    # This array will store the transformed 4-state values.
    cell_hist_out = np.zeros((len(cell_hist[:, 0, 0]), len(cell_hist[0, 0, :])))

    # Convert the binary state [0, 0] into the 4-state representation (1).
    cell_hist_out[(cell_hist[:, 0, :] == 0) * (cell_hist[:, 1, :] == 0)] = 1

    # Convert the binary state [0, 1] into the 4-state representation (2).
    cell_hist_out[(cell_hist[:, 0, :] == 0) * (cell_hist[:, 1, :] == 1)] = 2

    # Convert the binary state [1, 0] into the 4-state representation (3).
    cell_hist_out[(cell_hist[:, 0, :] == 1) * (cell_hist[:, 1, :] == 0)] = 3

    # Convert the binary state [1, 1] into the 4-state representation (4).
    cell_hist_out[(cell_hist[:, 0, :] == 1) * (cell_hist[:, 1, :] == 1)] = 4

    # Return the transformed 4-state history of the cells.
    return cell_hist_out



def get_current_4_number_seq(cell_current_BinaryState, num_cells):
    """
    Convert the current binary state of cells into a 4-state representation.

    Args:
        cell_current_BinaryState (ndarray): A 2D array where each row represents a cell's binary state.
        num_cells (int): The total number of cells.

    Returns:
        cell_4_state_out (ndarray): A 1D array representing the 4-state configuration of each cell.
    """

    # Initialize output (1D array)
    cell_4_state_out = np.zeros(num_cells)

    # Convert the binary state [0, 0] into the 4-state representation (1). Blue
    cell_4_state_out[(cell_current_BinaryState[:, 0] == 0) * (cell_current_BinaryState[:, 1] == 0)] = 1

    # Convert the binary state [0, 1] into the 4-state representation (2). Red
    cell_4_state_out[(cell_current_BinaryState[:, 0] == 0) * (cell_current_BinaryState[:, 1] == 1)] = 2

    # Convert the binary state [1, 0] into the 4-state representation (3). white
    cell_4_state_out[(cell_current_BinaryState[:, 0] == 1) * (cell_current_BinaryState[:, 1] == 0)] = 3

    # Convert the binary state [1, 1] into the 4-state representation (4). black
    cell_4_state_out[(cell_current_BinaryState[:, 0] == 1) * (cell_current_BinaryState[:, 1] == 1)] = 4

    return cell_4_state_out



def get_phase_seq_4(cell_hist):
    cell_hist_out = np.zeros((len(cell_hist[:, 1]), len(cell_hist[1, :])))

    cell_hist_out[cell_hist == 1] = 0
    cell_hist_out[cell_hist == 2] = 1 / 2 * np.pi
    cell_hist_out[cell_hist == 4] = 1 * np.pi
    cell_hist_out[cell_hist == 3] = 3 / 2 * np.pi

    return cell_hist_out


def get_2_number_seq(cell_hist):
    cell_hist_out = np.zeros((len(cell_hist[:, 0]), len(cell_hist[0, :]), 2))

    cell_hist_out[cell_hist == 1, 0] = 0
    cell_hist_out[cell_hist == 1, 1] = 0

    cell_hist_out[cell_hist == 2, 0] = 0
    cell_hist_out[cell_hist == 2, 1] = 1

    cell_hist_out[cell_hist == 3, 0] = 1
    cell_hist_out[cell_hist == 3, 1] = 0

    cell_hist_out[cell_hist == 4, 0] = 1
    cell_hist_out[cell_hist == 4, 1] = 1

    return cell_hist_out


def get_probability_matrix(CA1, start, stop):
    histo = np.zeros((len(CA1.cell_hist[:, 1, 1]), 1 + len(CA1.cell_hist[1, 1, start:stop])))
    histo[:, :-1] = get_4_number_seq(CA1.cell_hist[:, :, start:stop])

    idx1 = np.where(histo == 1)
    idx2 = np.where(histo == 2)
    idx3 = np.where(histo == 3)
    idx4 = np.where(histo == 4)

    idx1_valnext = [idx1[0], idx1[1] + 1]
    idx2_valnext = [idx2[0], idx2[1] + 1]
    idx3_valnext = [idx3[0], idx3[1] + 1]
    idx4_valnext = [idx4[0], idx4[1] + 1]

    valnext1 = histo[idx1_valnext[0], idx1_valnext[1]]
    valnext2 = histo[idx2_valnext[0], idx2_valnext[1]]
    valnext3 = histo[idx3_valnext[0], idx3_valnext[1]]
    valnext4 = histo[idx4_valnext[0], idx4_valnext[1]]

    bins = np.zeros((4, 4))

    for c in range(4):
        bins[0, c] = np.sum(valnext1 == c + 1)
        bins[1, c] = np.sum(valnext2 == c + 1)
        bins[2, c] = np.sum(valnext3 == c + 1)
        bins[3, c] = np.sum(valnext4 == c + 1)

    norm = np.sum(bins, axis=1)
    idx_nonz = norm != 0
    bins_norm = np.zeros((4, 4))
    bins_norm[idx_nonz] = np.round(bins[idx_nonz] * 1 / np.tile(norm[idx_nonz], [4, 1]).transpose(), 7)

    return bins_norm


def get_cell_neighbours(CA1, cell_number):
    neighbours_idx = np.where(np.round(CA1.relative_distance[cell_number, :], 1) == 1)
    cell4 = get_4_number_seq(CA1.cell_hist)
    neighbours = cell4[neighbours_idx, :]

    return neighbours


def get_average_cell_state_value(CA1):
    cell4 = get_4_number_seq(CA1.cell_hist)
    idx_states = np.zeros((4, len(cell4[0, :])))

    for i in range(4):
        idx_states[i, :] = np.sum(cell4 == i + 1, axis=0) / len(cell4[:, 1])

    return idx_states


def get_stationary_dist(P, N):
    state = np.array([[1.0, 0.0, 0.0, 0.0]])
    for x in range(N):
        state = np.dot(state, P)

    return state


def get_unique_distances(CA1):
    """
    Function to compute unique distances from a relative distance matrix and count the frequency
    of those distances. It also creates a boolean index matrix for each unique distance.

    Parameters:
    CA1 : An object that contains a relative_distance attribute, assumed to be a 2D numpy array.
          CA1.relative_distance represents a matrix of distances between cells or other entities.

    Returns:
    D_unique : A 2xN numpy array where the first row contains unique distances, and the second row contains
               the frequency of those unique distances.
    idx : A boolean 3D array that has the same dimensions as CA1.relative_distance, extended along the third
          axis to mark the positions of each unique distance.
    """

    # Step 1: Round the relative distances to 2 decimal places for uniqueness and simplicity.
    # D will now be a 2D array where each element is the rounded relative distance.
    D = np.round(CA1.relative_distance, 2)

    # Step 2: Find the unique distances in the rounded distance matrix `D`.
    # `np.unique(D)` returns a sorted array of the unique values in `D`.
    D_unique = np.unique(D)

    # Step 3: Prepare a 2xN matrix to store the unique distances and their frequency counts.
    # The first row will store the unique distances, and the second row will store the counts.
    D_unique = np.zeros((2, len(D_unique)))

    # Populate the first row of `D_unique` with the unique distance values.
    D_unique[0, :] = np.unique(D)

    # Step 4: Count the occurrences of each unique distance.
    # For each unique distance, count how many times it appears in the original distance matrix `D`.
    for c in range(len(D_unique[0, :])):  # Iterate through all unique distances.
        # Create a boolean mask that is True wherever the current unique distance is found in `D`.
        idx = D[0, :] == D_unique[0, c]

        # Sum the boolean mask to count the occurrences of the current unique distance.
        # Store the count in the second row of `D_unique`.
        D_unique[1, c] = np.sum(idx)

    # Step 5: Prepare a boolean index array `idx` to mark the positions of each unique distance in the matrix.
    # This will be a 3D boolean array where the first two dimensions match the dimensions of `D`,
    # and the third dimension represents each unique distance.
    idx = np.zeros((len(D[:, 0]), len(D[0, :]), len(D_unique[0, :])), dtype=bool)

    # Step 6: For each unique distance, create a boolean mask over the entire matrix `D`
    # and store it in the corresponding slice of the 3D `idx` array.
    for i in range(len(D_unique[0, :])):  # Iterate over each unique distance.
        # Create a boolean mask that marks the positions where the current unique distance is found in `D`.
        idx[:, :, i] = D == D_unique[0, i]

    # Step 7: Return the unique distances and the boolean index array.
    # `D_unique` contains the unique distances and their counts.
    # `idx` contains a 3D boolean array marking the positions of each unique distance.
    return D_unique, idx


def get_AT_Hamiltonian(CA1, J):
    # Define Ising planes
    plane1 = copy.deepcopy(np.squeeze(CA1.cell_hist[:, 0, :]))
    plane2 = copy.deepcopy(np.squeeze(CA1.cell_hist[:, 1, :]))

    # Set change 0 spin state to -1
    plane1[plane1 == 0] = -1
    plane2[plane2 == 0] = -1

    # Calculate all dot products
    temp11 = np.swapaxes(np.repeat(plane1[:, :, np.newaxis], CA1.gridsize ** 2, axis=2), 1, 2)
    temp12 = np.swapaxes(temp11, 0, 1)

    temp21 = np.swapaxes(np.repeat(plane2[:, :, np.newaxis], CA1.gridsize ** 2, axis=2), 1, 2)
    temp22 = np.swapaxes(temp21, 0, 1)

    # Find neighbouring cells
    idx = np.round(CA1.relative_distance, 1) == 1
    neighbours = np.repeat(idx[:, :, np.newaxis], CA1.tmax, axis=2)

    # Calculate single elements of Hamiltonian
    H1 = np.sum(np.round(np.squeeze(np.sum(temp12 * temp11 * neighbours, axis=0))), axis=0)
    H2 = np.sum(np.round(np.squeeze(np.sum(temp22 * temp21 * neighbours, axis=0))), axis=0)
    H3 = np.sum(np.round(np.squeeze(np.sum(temp11 * temp12 * temp21 * temp22 * neighbours, axis=0))), axis=0)

    H4 = 0  # np.sum(plane1, axis=0)
    H5 = 0  # np.sum(plane2, axis=0)

    H = J[0] * H1 + J[1] * H2 + J[2] * H3 + J[3] * H4 + J[4] * H5

    H_all = np.zeros((5, len(H)))
    H_all[0, :] = H1
    H_all[1, :] = H2
    H_all[2, :] = H3
    H_all[3, :] = H4
    H_all[4, :] = H5

    return H, H_all


def get_AT_Hamiltonian_hist(hist_in, relative_distance, J, dist_val):
    N = len(hist_in[:, 0])
    tmax = len(hist_in[0, :])

    ''' 
    idx1 = hist_in == 1
    idx2 = hist_in == 2
    idx3 = hist_in == 3
    idx4 = hist_in == 4

    hist_in[idx1] = 4
    hist_in[idx2] = 1
    hist_in[idx3] = 2
    hist_in[idx4] = 3
    '''

    hist_temp = get_2_number_seq(hist_in)

    # Define Ising planes
    plane1 = copy.deepcopy((hist_temp[:, :, 0]))
    plane2 = copy.deepcopy((hist_temp[:, :, 1]))

    # Set change 0 spin state to -1
    plane1[plane1 == 0] = -1
    plane2[plane2 == 0] = -1

    # Calculate all dot products
    temp11 = np.swapaxes(np.repeat(plane1[:, :, np.newaxis], N, axis=2), 1, 2)
    temp12 = np.swapaxes(temp11, 0, 1)

    temp21 = np.swapaxes(np.repeat(plane2[:, :, np.newaxis], N, axis=2), 1, 2)
    temp22 = np.swapaxes(temp21, 0, 1)

    # Find neighbouring cells
    idx = np.round(relative_distance, 1) == dist_val
    neighbours = np.repeat(idx[:, :, np.newaxis], tmax, axis=2)

    # Calculate single elements of Hamiltonian
    H1 = J[0] * np.round(np.squeeze(np.sum(temp12 * temp11 * neighbours, axis=0)))
    H2 = J[1] * np.round(np.squeeze(np.sum(temp22 * temp21 * neighbours, axis=0)))
    H3 = J[2] * np.round(np.squeeze(np.sum(temp11 * temp12 * temp21 * temp22 * neighbours, axis=0)))
    H4 = J[3] * np.round(
        np.squeeze(np.sum((temp22 + temp21) * neighbours + (temp12 * temp11 + 1) * neighbours, axis=0)))

    H = np.sum(H1 + H2 + H3 + H4, axis=0)

    H_all = H1 + H2 + H3 + H4

    return H, H_all


def get_connected_areas(h, relative_distance):
    # Give size of the region to involve
    temp1 = np.round(relative_distance, 1) > 0
    temp2 = np.round(relative_distance, 1) <= 1
    temp = temp1 * temp2

    # Format True/False cells by repeating in time axis direction
    idx_neighbours = np.repeat(temp[:, :, np.newaxis], len(h[0, :]), axis=2)

    # Define matrix to store cluster labels and define background (h=-6 as 0)
    clusters = np.zeros_like(h)
    clusters[h != -6] = 1

    # Empty matrix to store cluster labels
    label_c = np.ones((len(h[0, :])))

    # Integer value of the number of neighbours
    nn = np.sum(temp[:, 0], axis=0)

    for j in range(3):
        for k in range(len(h[:, 0])):
            # get non zero pixel values
            idx1 = clusters[k, :] > 0

            # get values of the neighbouring cells
            idx_n = np.swapaxes(clusters[idx_neighbours[:, k, :].squeeze()].reshape(nn, len(h[0, :])), 0, 1)

            # find out if it belongs to new region or old
            idx_n[idx_n < 2] = 1
            idx_n_max = np.max(idx_n, axis=1)

            idx_old_region = idx_n_max > 1
            clusters[k, idx1 * idx_old_region] = idx_n_max[idx1 * idx_old_region]

            idx_new_region = idx_n_max == 1

            label_c[idx1 * idx_new_region] += 1
            clusters[k, idx1 * idx_new_region] = label_c[idx1 * idx_new_region]

    nclusters = np.zeros((len(h[0, :]), 1))
    for j in range(len(clusters[0, :])):
        nclusters[j] = len(np.unique(clusters[:, j], axis=0)) - 1

    return nclusters, clusters


def show_h_and_clusters(t0, nclusters, h, hist, positions, gz):
    fig, ax = plt.subplots()
    plt.subplot(3, 2, 1)
    plt.plot(nclusters)
    plt.title('Clusters')
    plt.xlabel('Time')
    plt.ylabel('Count')

    plt.subplot(3, 4, 3)
    plt.plot(np.sum(h, axis=0))
    plt.title('Total Hamiltonian')
    plt.ylabel('H')
    plt.xlabel('Time')

    plt.subplot(3, 4, 4)
    plt.plot(np.sum((hist[:, :] == 2) / gz ** 2, axis=0))
    plt.title('2-state cells')
    plt.xlabel('Time')
    plt.ylabel('Fraction')

    num_tsteps = 4

    cNorm = colors.Normalize(vmin=-6, vmax=6)  # normalise the colormap
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap='hot')  # map numbers to colors

    for t in range(num_tsteps):
        plt.subplot(3, num_tsteps, num_tsteps + 1 + t)
        plt.scatter(positions[:, 0], positions[:, 1], c=scalarMap.to_rgba(h[:, t0 + t]), edgecolors='black',
                    cmap=plt.get_cmap('hot'), s=40)
        plt.title('T=%i' % (t + t0))

    for t in range(num_tsteps):
        plt.subplot(3, num_tsteps, num_tsteps * 2 + 1 + t)
        plt.scatter(positions[:, 0], positions[:, 1], c=hist[:, t0 + t], cmap=plt.get_cmap('hot'),
                    edgecolors='black', s=40)
        plt.title('T=%i' % (t + t0))


def calculate_H_for_single(plane1, plane2, J, CA1):
    # Calculate all dot products
    temp11 = np.swapaxes(np.repeat(plane1[:, :, np.newaxis], CA1.gridsize ** 2, axis=2), 1, 2)
    temp12 = np.swapaxes(temp11, 0, 1)

    temp21 = np.swapaxes(np.repeat(plane2[:, :, np.newaxis], CA1.gridsize ** 2, axis=2), 1, 2)
    temp22 = np.swapaxes(temp21, 0, 1)

    # Find neighbouring cells
    idx = np.round(CA1.relative_distance, 1) == 1
    neighbours = np.repeat(idx[:, :, np.newaxis], 1, axis=2)

    # Calculate single elements of Hamiltonian
    H1 = J[0] * np.round(np.squeeze(np.sum(temp12 * temp11 * neighbours, axis=1)))
    H2 = J[1] * np.round(np.squeeze(np.sum(temp22 * temp21 * neighbours, axis=1)))
    H3 = J[2] * np.round(np.squeeze(np.sum(temp11 * temp12 * temp21 * temp22 * neighbours, axis=1)))

    H = (H1 + H2 + H3)

    return np.sum(H)


def compute_centroid(self):
    # number of cores to trace
    f_max = np.max(self.vortex_cores_labeled)

    x_com = np.ones((self.tmax, f_max))
    x_com[x_com == 1] = np.nan

    y_com = np.ones((self.tmax, f_max))
    y_com[y_com == 1] = np.nan

    # Set positions from CA object
    xx = self.positions[:, 0]
    yy = self.positions[:, 1]

    for j in range(f_max):
        # find indices of specific area
        c = self.vortex_cores_labeled == j + 1

        # compute center of mass
        thetaX = xx / np.max(xx) * np.pi * 2
        thetaY = yy / np.max(yy) * np.pi * 2

        alphaX = np.cos(thetaX)
        alphaY = np.cos(thetaY)

        betaX = np.sin(thetaX)
        betaY = np.sin(thetaY)

        count = np.sum(c, axis=0)
        count[count == 0] = 1

        # Safely compute matmul with error handling to avoid divide by zero and overflow
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                betaX_c = np.matmul(betaX, c)
                alphaX_c = np.matmul(alphaX, c)
                betaY_c = np.matmul(betaY, c)
                alphaY_c = np.matmul(alphaY, c)
                
                # Check if nan_to_num will be needed
                nan_count = np.isnan(betaX_c).sum() + np.isnan(alphaX_c).sum() + np.isnan(betaY_c).sum() + np.isnan(alphaY_c).sum()
                inf_count = np.isinf(betaX_c).sum() + np.isinf(alphaX_c).sum() + np.isinf(betaY_c).sum() + np.isinf(alphaY_c).sum()
                if self.debug and (nan_count > 0 or inf_count > 0):
                    print(f"  ⚠️  nan_to_num NEEDED in centroid: NaN={nan_count}, Inf={inf_count}")
                
                # Replace any NaN or Inf values with 0 before division
                betaX_c = np.nan_to_num(betaX_c, nan=0.0, posinf=0.0, neginf=0.0)
                alphaX_c = np.nan_to_num(alphaX_c, nan=0.0, posinf=0.0, neginf=0.0)
                betaY_c = np.nan_to_num(betaY_c, nan=0.0, posinf=0.0, neginf=0.0)
                alphaY_c = np.nan_to_num(alphaY_c, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.debug:
            print(f"\n=== DEBUG compute_centroid (j={j}) ===")
            print(f"c (vortex core mask) sum: {np.sum(c)}, shape: {c.shape}")
            print(f"count: {count}, type: {type(count)}")
            print(f"Before division:")
            print(f"  betaX_c min: {np.min(betaX_c)}, max: {np.max(betaX_c)}, NaN: {np.any(np.isnan(betaX_c))}, Inf: {np.any(np.isinf(betaX_c))}")
            print(f"  alphaX_c min: {np.min(alphaX_c)}, max: {np.max(alphaX_c)}, NaN: {np.any(np.isnan(alphaX_c))}, Inf: {np.any(np.isinf(alphaX_c))}")
            print(f"  betaY_c min: {np.min(betaY_c)}, max: {np.max(betaY_c)}, NaN: {np.any(np.isnan(betaY_c))}, Inf: {np.any(np.isinf(betaY_c))}")
            print(f"  alphaY_c min: {np.min(alphaY_c)}, max: {np.max(alphaY_c)}, NaN: {np.any(np.isnan(alphaY_c))}, Inf: {np.any(np.isinf(alphaY_c))}")
            
            thetaX_m = np.arctan2(-betaX_c / count, -alphaX_c / count) + np.pi
            thetaY_m = np.arctan2(-betaY_c / count, -alphaY_c / count) + np.pi
            
            print(f"thetaX_m contains NaN: {np.any(np.isnan(thetaX_m))}, Inf: {np.any(np.isinf(thetaX_m))}")
            print(f"thetaY_m contains NaN: {np.any(np.isnan(thetaY_m))}, Inf: {np.any(np.isinf(thetaY_m))}")
        else:
            thetaX_m = np.arctan2(-betaX_c / count, -alphaX_c / count) + np.pi
            thetaY_m = np.arctan2(-betaY_c / count, -alphaY_c / count) + np.pi

        x_com[:, j] = thetaX_m / (2 * np.pi) * np.max(xx)
        y_com[:, j] = thetaY_m / (2 * np.pi) * np.max(yy)

    return x_com, y_com


def initialize_lattice_from_excel(path, periodic_bc):  # Read input file
    position_matrix = pd.read_excel(path)

    # Format input data to remove nan padding
    position_matrix = position_matrix.to_numpy()
    position_matrix = position_matrix[~np.isnan(position_matrix).all(axis=1)]
    position_matrix = position_matrix[:, ~np.all(np.isnan(position_matrix), axis=0)]

    # Compute lattice coordinates
    xx, yy = np.meshgrid(np.arange(0, position_matrix.shape[1]), np.arange(0, position_matrix.shape[0]))

    # Define positions of each cell
    position_matrix[np.isnan(position_matrix)] = 0

    state = position_matrix[position_matrix > 0]

    x = (xx + 1) * (position_matrix > 0)
    y = (yy + 1) * (position_matrix > 0)

    positions = np.zeros((np.sum(x > 0), 2))
    positions[:, 0] = (x[x > 0] - 1)
    positions[:, 1] = (y[y > 0] - 1)

    # Flip y-axis to get correct configuration
    positions[:, 1] = np.abs(positions[:, 1] - np.max(positions[:, 1]))

    # Compute the relative distances and take periodic bc into account
    lx = np.max(positions[:, 0]) + 1
    ly = np.max(positions[:, 1]) + 1

    [x1, x2] = np.meshgrid(positions[:, 0], positions[:, 0])
    [y1, y2] = np.meshgrid(positions[:, 1], positions[:, 1])

    dx = np.mod(abs(x1 - x2), lx)
    dy = np.mod(abs(y1 - y2), ly)

    if periodic_bc[0] == 1:
        dx[dx > (lx - dx)] = lx - dx[dx > (lx - dx)]

    if periodic_bc[1] == 1:
        dy[dy > (ly - dy)] = ly - dy[dy > (ly - dy)]

    r = (dx ** 2 + dy ** 2) ** 0.5

    return positions, r, state


def get_contour_of_vortex_core(c, idx_label, relative_distance):
    hp_in = c == idx_label
    hp = c == idx_label
    idx = np.where(hp_in)

    # calculate neighbouring values and set them all to true
    neighbours_idx = np.where(np.round(relative_distance[idx[0], :], 1) == 1)
    hp[neighbours_idx[1]] = True

    hp_in = hp_in * 1
    hp = hp * 1

    return hp - hp_in


def get_area_contour(c, idx_label, relative_distance):
    #c is the vortex field which contains the labels of the vortices
    #idx_label is the specific vortex label we want to find the contour for
    #relative_distance is the matrix of distances between cells

    hp_in = c == idx_label
    # hp_in is a boolean array where True indicates the cells that belong to the vortex with label idx_label

    hp = c == idx_label
    idx = np.where(hp_in)
    # idx contains the indices of the cells that belong to the vortex with label idx_label

    # calculate neighbouring values and set tem all to true
    neighbours_idx = np.where(np.round(relative_distance[idx[0], :], 1) == 1)
    hp[neighbours_idx[1]] = True
    # hp is updated to include all neighboring cells of the vortex

    hp_in = hp_in * 1
    # we do this because we want to convert the boolean array to an integer array (0s and 1s)
    hp = hp * 1
#return the contour by subtracting the original vortex area from the expanded area
#so basically the contour is the area that was added by including the neighbours
    return hp - hp_in



def get_contour_pos_relative_2_center(c_run, hist_run, idx_label, CA1):
    # compute values around vortex
    ring = get_area_contour(c_run, idx_label, CA1.relative_distance)
    #ring is the neighbouring area around the vortex with label idx_label

    idx_ring = np.where(ring == 1)
    #idx_ring contains the indices of the cells that are part of the contour (ring) around the vortex
    x_r = CA1.positions[idx_ring, 0]
    #CA1 is the cellular automaton object which contains the positions of the cells
    #x_r is the x-coordinates of the cells that are part of the contour around the vortex
    y_r = CA1.positions[idx_ring, 1]
    c_r = hist_run[idx_ring]
    #c_r is the state of the cells that are part of the contour around the vortex

    xx = CA1.positions[:, 0]
    # this is the x-coordinates of all the cells in the cellular automaton
    yy = CA1.positions[:, 1]

    lx = np.max(CA1.positions[:, 0]) + CA1.positions[1, 0]

    ly = np.max(CA1.positions[:, 1]) + CA1.positions[1, 1]

    ring_bin = (ring == 1) * 1
    # ring_bin is a binary array where 1 indicates the cells that are part of the contour around the vortex

    m_x_r, m_y_r = compute_center_of_mass(xx, yy, ring_bin)
    # m_x_r and m_y_r are the coordinates of the center of mass of the contour cells

    # Apply centroid algorithm
    y_hat = (y_r - m_y_r + ly / 2) % ly - ly / 2
    x_hat = (x_r - m_x_r + lx / 2) % lx - lx / 2
    # x_hat and y_hat are the coordinates of the contour cells relative to the center of mass of the contour


    return x_hat, y_hat, c_r, m_x_r, m_y_r


def get_vorticity(x_hat, y_hat, c_r):
    f_clock = np.zeros((4, 1))
    f_anti = np.zeros((4, 1))

    theta_r = np.arctan2(y_hat, x_hat) + np.pi
    idx_sort = np.argsort(theta_r)

    cwise = c_r[idx_sort]
    cwise_wrap = np.zeros((1, len(cwise[0, :]) + 1))
    cwise_wrap[0, :-1] = cwise
    cwise_wrap[0, len(cwise[0, :])] = cwise[0, 0]

    temp = cwise_wrap
    temp[temp == 4] = 5
    temp[temp == 3] = 4
    temp[temp == 5] = 3

    for j in range(4):
        temp = np.where(cwise == 1 + j)
        nn = cwise_wrap[0, temp[1] + 1]

        if j == 0 and len(nn) > 0:
            f_clock[j, 0] = (np.sum(nn == 1) + np.sum(nn == 2)) / len(nn)
            f_anti[j, 0] = (np.sum(nn == 1) + np.sum(nn == 3)) / len(nn)
        elif j == 1 and len(nn) > 0:
            f_clock[j, 0] = (np.sum(nn == 2) + np.sum(nn == 4)) / len(nn)
            f_anti[j, 0] = (np.sum(nn == 2) + np.sum(nn == 1)) / len(nn)
        elif j == 2 and len(nn) > 0:
            f_clock[j, 0] = (np.sum(nn == 3) + np.sum(nn == 1)) / len(nn)
            f_anti[j, 0] = (np.sum(nn == 3) + np.sum(nn == 4)) / len(nn)
        elif j == 3 and len(nn) > 0:
            f_clock[j, 0] = (np.sum(nn == 4) + np.sum(nn == 3)) / len(nn)
            f_anti[j, 0] = (np.sum(nn == 4) + np.sum(nn == 2)) / len(nn)

    return np.mean(f_clock, axis=0), np.mean(f_anti, axis=0)


def get_number_of_vortices(CA1, t_start, t_end, n_vortex, vortex_field, cell4):
    f_clock_tot = np.zeros((len(n_vortex), 1))
    f_anti_tot = np.zeros((len(n_vortex), 1))

    m_x_r_all = np.zeros((t_end - t_start, 2 + int(np.max(vortex_field)))) * np.nan
    m_y_r_all = np.zeros((t_end - t_start, 2 + int(np.max(vortex_field)))) * np.nan

    f_anti = np.zeros((t_end - t_start, 2 + int(np.max(vortex_field))))
    f_clock = np.zeros((t_end - t_start, 2 + int(np.max(vortex_field))))

    for p in range(t_end - t_start):
        # Set frame number
        fn = p + t_start

        # 4 states, h field and vortex field for frame
        hist_run = cell4[:, fn]
        c_run = vortex_field[:, fn]

        for idx_label in range(1, 1 + int(np.max(vortex_field[:, fn], axis=0))):
            x_hat, y_hat, c_r, m_x_r, m_y_r = get_contour_pos_relative_2_center(c_run, hist_run, idx_label, CA1)
            # print(np.sum(diff(c_r)))
            if np.any(x_hat > 0):
                m_x_r_all[p, idx_label] = m_x_r
                m_y_r_all[p, idx_label] = m_y_r
                f_clock[p, idx_label], f_anti[p, idx_label] = get_vorticity(x_hat, y_hat, c_r)

                if f_clock[p, idx_label] == 1 or f_anti[p, idx_label] == 1:
                    f_clock_tot[p, 0] += (f_clock[p, idx_label] >= 1) * 1
                    f_anti_tot[p, 0] += (f_anti[p, idx_label] >= 1) * 1
                else:  # post processing in case no integer vorticity is found in het contour
                    f_clock_tot[p, 0] += (f_clock[p, idx_label] >= 5 / 6) * 1
                    f_anti_tot[p, 0] += (f_anti[p, idx_label] >= 5 / 6) * 1

    return f_clock, f_anti, f_clock_tot, f_anti_tot, m_x_r_all, m_y_r_all


def compute_center_of_mass(xx, yy, idx_cells):
    # compute center of mass with periodic BC
    # See: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.151.8565&rep=rep1&type=pdf
    # 'Calculating Center of Mass in an Unbounded 2D Environment, L. Bai and D. Breen, journal of graphics tools'

    thetaX = xx / np.max(xx) * np.pi * 2
    thetaY = yy / np.max(yy) * np.pi * 2

    alphaX = np.cos(thetaX)
    alphaY = np.cos(thetaY)

    betaX = np.sin(thetaX)
    betaY = np.sin(thetaY)

    count = np.sum(idx_cells)

    if count > 0:
        thetaX_m = np.arctan2(-np.matmul(betaX, idx_cells) / count, -np.matmul(alphaX, idx_cells) / count) + np.pi
        thetaY_m = np.arctan2(-np.matmul(betaY, idx_cells) / count, -np.matmul(alphaY, idx_cells) / count) + np.pi
    else:
        thetaX_m = 0
        thetaY_m = 0

    x_com = thetaX_m / (2 * np.pi) * np.max(xx)
    y_com = thetaY_m / (2 * np.pi) * np.max(yy)

    return x_com, y_com


def apply_periodic_convolution(frame, kernel):
    r = sc.signal.convolve2d(frame, kernel, boundary='wrap', mode='same')
    r[r <= -np.pi] = r[r <= -np.pi] + 2 * np.pi
    r[np.pi <= r] = r[np.pi <= r] - 2 * np.pi

    return r


def get_contour_kernel(Nkernel):
    tot_num_kernels = 4*(Nkernel-1)

    K = np.zeros((Nkernel, Nkernel, tot_num_kernels))
    top_vect = np.zeros((1, Nkernel))
    top_vect[0, 0] = 1
    top_vect[0, 1] = -1

    # fill top row
    for i in range(Nkernel-1):
        K[0, i:(i+2), i] = np.array([-1, 1])

    # fill right column
    for i in range(Nkernel-1):
        K[i:(i+2), -1, i+Nkernel-1] = np.array([-1, 1])

    # fill bottom row
    for i in range(Nkernel-1):
        K[-1, i:(i+2), i+2*(Nkernel-1)] = np.array([1, -1])

    # fill left column
    for i in range(Nkernel-1):
        K[i:(i+2), 0, i+3*(Nkernel-1)] = np.array([1, -1])

        return K


#### Hamiltonian functions
def show_spin_states_of_configuration(hist_in, h_in, pos, t):
    U = np.zeros_like(hist_in, dtype=np.double)
    V = np.zeros_like(hist_in, dtype=np.double)

    U[hist_in == 1] = np.cos(0)
    V[hist_in == 1] = np.sin(0)
    U[hist_in == 2] = np.cos(np.pi / 2)
    V[hist_in == 2] = np.sin(np.pi / 2)
    U[hist_in == 3] = np.cos(np.pi * 3 / 2)
    V[hist_in == 3] = np.sin(np.pi * 3 / 2)
    U[hist_in == 4] = np.cos(np.pi)
    V[hist_in == 4] = np.sin(np.pi)

    g = 2
    fig, ax = plt.subplots()
    ax.quiver(pos[:, 0], pos[:, 1], U * g, V * g)

    ax.scatter(pos[:, 0], pos[:, 1], c=h_in, edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
    plt.title('Time step: ' + str(t))
    plt.show()

    return ax


def show_hamiltonians(input):
    plt.figure()
    plt.plot(input.Hxy_adjusted, label='Adjust Hamiltonian')
    plt.plot(input.Hxy_classic, label='Classic XY Hamiltonian')
    plt.xlabel('Time')
    plt.ylabel('H')
    plt.title('Normalized Hamiltonian for trajectory')
    plt.legend()






def shortest_paths_between_vortices_in_cluster(cluster_cells, vortex_labels, include_zero_charge, relative_distance=None,
                                                return_paths=False, count_mode='edges', trajectory=None,
                                                frame=None,  verbose=False):
    """
    Compute shortest-path distances (within a cluster) between each pair of distinct vortex IDs
    that are present inside that cluster.

    Args:
        cluster_cells: 1D array-like of integer cell indices that form the cluster.
        vortex_labels: 1D array (length N) giving vortex ID per cell (0 means "no vortex").
        relative_distance: NxN array of pairwise distances (same as CA.relative_distance).
                           Neighbours are where np.round(relative_distance[i,:],1) == 1
        return_paths: if True, return also the path (list of cell indices) for each pair found.
        count_mode: what the distance value should mean:
            'edges'      -> number of edges between endpoint cells (default)
            'cells'      -> number of cells in path including endpoints (edges + 1)
            'intermediate' -> number of cells strictly between endpoints (edges - 1, min 0)

    Returns:
        dict mapping (v1_id, v2_id) -> distance (or (distance, path) if return_paths True).
        Pairs where no connection exists within the cluster map to None.
    """
    cluster_cells = np.asarray(cluster_cells, dtype=int)
    if cluster_cells.size == 0:
        return {}

    # set for fast membership test
    cluster_set = set(cluster_cells.tolist())

    # vortex IDs present inside this cluster (exclude 0)
    vortex_ids = np.unique(vortex_labels[cluster_cells])
    vortex_ids = vortex_ids[vortex_ids > 0]

    # Optionally exclude vortices that have zero charge at this frame (default behavior)
    if (not include_zero_charge) and (trajectory is not None) and (frame is not None):
        try:
            # trajectory.charges expected shape: (tmax, n_vortices+2)
            filtered = []
            for vid in vortex_ids:
                try:
                    ch = trajectory.charges[int(frame), int(vid) + 1]
                    #CHANGED TO +1
                except Exception:
                    ch = None
                # keep vortex if charge is None (unknown) or non-zero; drop if explicitly zero
                if ch is None or (hasattr(ch, '__len__') and len(ch) == 0):
                    filtered.append(vid)
                else:
                    try:
                        if float(ch) != 0.0:
                            filtered.append(vid)
                    except Exception:
                        filtered.append(vid)
            vortex_ids = np.array(filtered, dtype=int)
        except Exception:
            # if any error occurs while checking charges, fall back to not filtering
            pass

    # need at least two different vortices to measure a connection
    #if len(vortex_ids) < 2:
        # Respect a `quiet_warnings` flag on the provided trajectory (if present)
        #try:
           # if trajectory is None or not getattr(trajectory, 'quiet_warnings', False):
               # print("Warning: less than two vortices in cluster, skipping.")
        #except Exception:
            #print("Warning: less than two vortices in cluster, skipping.")
        #return {}

    results = {}

    # helper to get neighbours on the lattice (returns np.array of indices)
    def neighbors(i):
        # Prefer trajectory.cell_number_nn when provided (fast, explicit neighbor indices)
        if trajectory is not None and hasattr(trajectory, 'cell_number_nn') and trajectory.cell_number_nn is not None and len(trajectory.cell_number_nn) > 0:
            try:
                nn = trajectory.cell_number_nn[int(i), 1:1 + int(trajectory.number_of_nn)].astype(int)
                return nn
            except Exception:
                pass

        # Fallback to computing neighbors via relative_distance if available
        if relative_distance is not None:
            return np.where(np.round(relative_distance[int(i), :], 1) == 1)[0]

        raise ValueError('No neighbor information available: provide `relative_distance` or `trajectory` with `cell_number_nn`.')

    # iterate over distinct unordered pairs of vortex IDs
    for ai in range(len(vortex_ids)):
        for bi in range(ai + 1, len(vortex_ids)):
            v1 = int(vortex_ids[ai])
            v2 = int(vortex_ids[bi])

            # Instead of using vortex-core cells themselves as path endpoints, use the
            # cluster cells that are nearest-neighbours of each vortex core. This allows
            # strings that connect the neighbourhoods of the two vortices even if the
            # path cells are not the same phase as the vortex cores.
            # Collect v1_front_cells and v2_front_cells = cluster cells that neighbour the core.
            # consider ALL core cells for each vortex when building front-sets
            # (do not require the core cell itself to lie inside this cluster)
            core_v1_cells = [int(c) for c in np.where(vortex_labels == v1)[0]]
            core_v2_cells_all = [int(c) for c in np.where(vortex_labels == v2)[0]]

            # helper to get neighbor indices (same as below)
            def _neighbors_local(i):
                if trajectory is not None and hasattr(trajectory, 'cell_number_nn') and trajectory.cell_number_nn is not None and len(trajectory.cell_number_nn) > 0:
                    try:
                        return trajectory.cell_number_nn[int(i), 1:1 + int(trajectory.number_of_nn)].astype(int)
                    except Exception:
                        pass
                if relative_distance is not None:
                    return np.where(np.round(relative_distance[int(i), :], 1) == 1)[0]
                raise ValueError('No neighbor information available: provide `relative_distance` or `trajectory` with `cell_number_nn`.')

            v1_front = set()
            for core in core_v1_cells:
                for nb in _neighbors_local(core):
                    nb = int(nb)
                    if nb in cluster_set and nb not in core_v1_cells:
                        v1_front.add(nb)

            v2_front = set()
            for core in core_v2_cells_all:
                for nb in _neighbors_local(core):
                    nb = int(nb)
                    if nb in cluster_set and nb not in core_v2_cells_all:
                        v2_front.add(nb)

            v1_cells = list(v1_front)
            v2_cells_set = set(v2_front)

            # Quick check: if there is any shared nearest-neighbour (a cell that is
            # adjacent to both vortex cores), treat that single cell as the string
            # with length 1 and return immediately.
            shared = v1_front.intersection(v2_front)
            if len(shared) > 0:
                chosen = int(next(iter(shared)))
                if return_paths:
                    results[(v1, v2)] = (1, [chosen])
                else:
                    results[(v1, v2)] = 1
                continue

            # if either vortex has no neighbouring cluster-front cells, no string can connect them
            if len(v1_cells) == 0 or len(v2_cells_set) == 0:
                results[(v1, v2)] = None
                continue

            # BFS: multi-source from all v1_cells at once
            q = deque()
            visited = set()
            parent = {}  # to reconstruct path if requested

            for s in v1_cells:
                q.append((s, 0))   # (node_index, distance_in_edges_from_any_v1_source)
                visited.add(s)
                parent[s] = None

            found_distance = None
            found_target = None

            while q and found_distance is None:
                node, dist = q.popleft()
                # if we've reached a front-cell of the second vortex, stop
                #a front cell is a cluster cell that neighbours the vortex core
                if node in v2_cells_set:
                    found_distance = dist+1
                    found_target = node
                    break

                for nb in neighbors(node):
                    # for each unvisited neighbour that is part of the cluster, add to queue    
                    nb = int(nb)
    
                    if nb in cluster_set and nb not in visited:
                        # add to queue
                        visited.add(nb)
                        parent[nb] = node
                        q.append((nb, dist + 1))

            if found_distance is None:
                # no connection detected inside this cluster
                if verbose:
                    print(f"DEBUG: no path in this cluster between vortices {v1} and {v2}")
                    print(f"  core_v1_cells={core_v1_cells}")
                    print(f"  core_v2_cells_all={core_v2_cells_all}")
                    print(f"  v1_front={sorted(list(v1_front))}")
                    print(f"  v2_front={sorted(list(v2_front))}")
                    print(f"  cluster_set size={len(cluster_set)} sample(cluster_cells)={list(cluster_cells)[:20]}")
                results[(v1, v2)] = None
            else:
                # When reporting sizes, exclude vortex-core cells from the count.
                # found_distance currently counts edges between front-cells. This is
                # the preferred interpretation for 'edges'. For 'cells' and 'intermediate'
                # translate accordingly but do NOT add back core cells.
                if count_mode == 'edges':
                    value = int(found_distance)
                elif count_mode == 'cells':
                    # number of front-cells in path equals edges + 1
                    value = int(found_distance + 1)
                elif count_mode == 'intermediate':
                    value = max(0, int(found_distance - 1))
                else:
                    raise ValueError("count_mode must be 'edges','cells' or 'intermediate'")
                    count_mode = 'cells'

                if return_paths:
                    # reconstruct path from a v1 front-cell to found_target (a v2 front-cell)
                    path = []
                    n = found_target
                    while n is not None:
                        path.append(int(n))
                        n = parent.get(n)
                    path.reverse()
                    # Ensure returned path does NOT include vortex-core cell indices; the
                    # front-cells are cluster cells neighbouring the cores.
                    # (If upstream callers expect core indices, they still exist in vortex_labels.)
                    results[(v1, v2)] = (value, path)
                else:
                    results[(v1, v2)] = value

    return results




def get_vortex_connection_lengths(trajectory, t_start, t_end):
    """
    For each frame, compute shortest connections between vortices
    through clusters of ANY state (1–4). The shortest connection wins.
    """
    all_results = {}

    for fn in range(t_start, t_end):
        frame_states = trajectory.cell_4state[:, fn]
        frame_vortices = trajectory.vortex_cores[:, fn]

        frame_results = {}
        cores = np.unique(trajectory.vortex_cores_labeled[:, fn])
        print(f"Frame {fn}, cores detected: {cores}")

        # loop over all possible states (1–4). Use the trajectory.label_phase_clusters helper
        for state in [1, 2, 3, 4]:
            # get per-frame cluster labels for this state only
            try:
                cluster_labels = trajectory.label_phase_clusters(fn, phase_value=state)
            except Exception:
                # fallback: compute via mask and the older labeler
                mask = (frame_states == state).astype(int).reshape(-1, 1)
                labeled, _ = label_elements_triangular_lattice(mask, trajectory.relative_distance, 0, 1)
                cluster_labels = labeled[:, 0]

            if np.all(cluster_labels == 0):
                continue

            for cluster_id in np.unique(cluster_labels):
                if cluster_id == 0:
                    continue
                cluster_cells = np.where(cluster_labels == cluster_id)[0]

                # expand cluster to include vortices whose neighbors touch the cluster
                expanded_cluster = set(cluster_cells.tolist())
                for v in np.unique(frame_vortices):
                    if v == 0:
                        continue
                    if int(v) not in expanded_cluster:
                        # use trajectory.cell_number_nn when possible
                        try:
                            nn = trajectory.cell_number_nn[int(v), 1:1 + int(trajectory.number_of_nn)].astype(int)
                        except Exception:
                            nn = np.where(np.round(trajectory.relative_distance[int(v), :], 1) == 1)[0]

                        if np.intersect1d(nn, cluster_cells).size > 0:
                            expanded_cluster.add(int(v))

                print(f"  State {state}, cluster {cluster_id}, size {len(cluster_cells)} -> expanded to {len(expanded_cluster)}")
                cluster_cells = np.array(list(expanded_cluster), dtype=int)

                dists = shortest_paths_between_vortices_in_cluster(
                            cluster_cells,
                            frame_vortices,
                            relative_distance=None,
                            return_paths=False,
                            count_mode='cells',
                            trajectory=trajectory
                        )

                # now merge: keep shortest across all states
                for key, dist in dists.items():
                    if key not in frame_results or dist < frame_results[key][0]:
                        # store both the dist and which state made it
                        frame_results[key] = (dist, state)

        all_results[fn] = frame_results

    return all_results


