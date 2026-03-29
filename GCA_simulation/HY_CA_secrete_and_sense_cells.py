'''
Generalized Cellular Automaton (GCA) of secrete-and-sense cells on a triangular lattice.

Developed by Lars Koopmans and Hyun Youk (Youk laboratory).

Each cell occupies one of four discrete gene-expression states (1, 2, 3, or 4),
encoded as a binary pair (gene1 ON/OFF, gene2 ON/OFF). Cells update synchronously
based on the concentrations of secreted molecules they sense from neighbors,
governed by interaction matrix M, threshold matrix K, and diffusion lengths lambda.

Key instance variables
----------------------
self.cells : ndarray, shape (N, 2)
    Initial binary cell states at t = 0. Not modified by run_model() or run_CA();
    always holds the initial configuration.

self.cell_hist : ndarray, shape (N, 2, tmax), dtype int8
    Full binary state history. cell_hist[:, :, t] gives the state of all cells
    at timestep t.

self.cell_4state : ndarray, shape (N, tmax)
    4-state encoding of cell_hist, where:
        state 1 = (gene1=0, gene2=0)
        state 2 = (gene1=0, gene2=1)
        state 3 = (gene1=1, gene2=0)
        state 4 = (gene1=1, gene2=1)
    Populated by analyse_trajectory().

self.cell_4phase : ndarray, shape (N, tmax)
    Phase angle (radians) corresponding to each 4-state value:
        state 1 -> 0
        state 2 -> pi/2
        state 3 -> 3*pi/2
        state 4 -> pi
    Populated by analyse_trajectory().

self.current_4state : ndarray, shape (N,)
    4-state encoding of the current binary state self.cells (single timestep).
    Populated only when convert_currentBinaryState_to_4cell_state() is called;
    not updated automatically during the simulation.
'''

import numpy as np
import random
import time
from pylab import *
import copy
import imageio
import pickle
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import os


class CellularLattice:
    """
    Represents a generalized cellular automaton (GCA) of secrete-and-sense cells
    on a lattice. Handles lattice construction, parameter initialization, simulation,
    trajectory analysis, vortex detection, and visualization.

    Attributes
    ----------
    gridsize : int
        Number of cells along one side of the square grid.
    N : int
        Total number of cells (= gridsize^2).
    rcell : float
        Cell radius, used to compute interaction strengths.
    a0 : float
        Lattice constant (nearest-neighbor spacing).
    tmax : int
        Maximum number of timesteps requested.
    cell_hist : ndarray, shape (N, 2, tmax), dtype int8
        Binary state history of all cells over time.
    vortex_cores : ndarray
        Boolean array marking cells that belong to a vortex core at each timestep.
    vortex_cores_labeled : ndarray
        Integer-labeled version of vortex_cores (each connected core gets a unique label).
    charges : ndarray
        Topological charge (+1 or -1) assigned to each labeled vortex core.
    """

    def __init__(self, gridsize, num_cells, rcell, a0, tmax):
        """
        Initialize the CellularLattice class with grid size, cell count, and lattice parameters.

        Args:
            gridsize (int): The size of the grid.
            num_cells (int): The total number of cells.
            rcell (float): The radius of a cell.
            a0 (float): The lattice constant (spacing).
            tmax (int): The maximum number of timesteps.
        """
        self.runTime = 0
        self.relative_distance = []
        self.positions = []
        self.gridsize = gridsize
        self.N = num_cells
        self.rcell = rcell
        self.a0 = a0
        self.cell_topology = []
        self.cells = np.zeros((self.N, 2))  # Initial cell states; not modified by run_model() or run_CA()
        self.periodic_bc = [0, 0]
        self.K = np.zeros((2, 2, 2))        # Activation threshold matrix
        self.M = np.zeros((2, 2, 2))        # Interaction strength matrix
        self.C = np.zeros((2, 1, 2))        # Secretion rate matrix
        self.Coff = np.ones([2, 1, 2])      # Basal secretion rates for inactive cells
        self.C0n = []
        self.idx_celltype = []
        self.idx_nearest_neighbours = []
        self.lamb = []                       # Diffusion lengths
        self.tmax = tmax
        self.cell_hist = np.zeros((num_cells, 2, tmax), np.int8)
        self.img = []
        self.Fn_a0 = 1
        self.double_topology_flag = False
        self.clock_start = time.time()
        self.clock_traject = np.zeros(10)
        self.number_of_nn = 6               # Default: triangular lattice has 6 nearest neighbors
        self.cell_4state = []
        self.current_4state = []            # 4-state encoding of self.cells; set by convert_currentBinaryState_to_4cell_state()
        self.cell_4phase = []
        self.state_fractions = []
        self.first_recurrent_state_time = []
        self.final_pattern_type = []
        self.vortex_cores = []
        self.vortex_cores_labeled = []
        self.n_vortex = []
        self.xcom = []
        self.ycom = []
        self.cell_number_nn = []
        self.xmin_core = []
        self.ymin_core = []
        self.closest_pos = []
        self.charges = []
        self.idx_annihilation_moments = []
        self.idx_annihilation_moments_core_count = []


    def init_lattice(self, gridsize, periodic_bc, a0, rcell, type, path=''):
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
        n = gridsize ** 2
        self.a0 = a0
        self.gridsize = gridsize

        if type == 'triangular':
            [pos, lx, ly] = calculate_triangular_lattice(gridsize)
            dist = calculate_distance(pos, lx, ly, gridsize, periodic_bc)

            idx_nearest_neighbours = np.round(dist, 1) == 1
            row_coll_idx_nn = np.argwhere(idx_nearest_neighbours == 1)

            cell_number_nn = np.zeros((n, 7))
            cell_number_nn[:, 0] = np.arange(0, n)
            cell_number_nn[:, 1:] = row_coll_idx_nn[:, 1].reshape(n, 6)

            self.cell_number_nn = cell_number_nn.astype(int)
            self.idx_nearest_neighbours = idx_nearest_neighbours
            self.relative_distance = dist
            self.N = n
            self.positions = pos
            self.number_of_nn = 6

        elif type == 'excel':
            pos, dist, states = initialize_lattice_from_excel(path, periodic_bc)
            n = len(pos[:, 0])

            idx_nearest_neighbours = np.round(dist, 1) == 1
            row_coll_idx_nn = np.argwhere(idx_nearest_neighbours == True)

            if np.sum(row_coll_idx_nn == 1) == 4 * n:
                cell_number_nn = np.zeros((n, 5))
                cell_number_nn[:, 0] = np.arange(0, n)
                cell_number_nn[:, 1:] = row_coll_idx_nn[:, 1].reshape(n, 4)
            else:
                cell_number_nn = np.zeros((n, 5))
                cell_number_nn[:, 0] = np.arange(0, n)

            self.number_of_nn = 4
            self.cells = np.squeeze(get_2_number_seq(states[:, np.newaxis]))
            self.idx_nearest_neighbours = idx_nearest_neighbours
            self.cell_number_nn = cell_number_nn.astype(int)
            self.relative_distance = dist
            self.N = n
            self.positions = pos

        elif type == 'excel_triangular':
            pos_temp, dist_temp, states = initialize_lattice_from_excel(path, periodic_bc)
            n_cells = len(pos_temp[:, 0])
            gridsize = int(np.round(np.sqrt(n_cells)))

            [pos, lx, ly] = calculate_triangular_lattice(gridsize)
            dist = calculate_distance(pos, lx, ly, gridsize, periodic_bc)

            idx_nearest_neighbours = np.round(dist, 1) == 1
            row_coll_idx_nn = np.argwhere(idx_nearest_neighbours == True)

            if periodic_bc[0] == 1:
                cell_number_nn = np.zeros((n, 7))
                cell_number_nn[:, 0] = np.arange(0, n)
                cell_number_nn[:, 1:] = row_coll_idx_nn[:, 1].reshape(n, 6)
                self.cell_number_nn = cell_number_nn.astype(int)
            else:
                cell_number_nn = 0
                self.cell_number_nn = cell_number_nn

            self.number_of_nn = 6
            self.cells = np.squeeze(get_2_number_seq(states[:, np.newaxis]))
            self.idx_nearest_neighbours = idx_nearest_neighbours
            self.relative_distance = dist
            self.N = n
            self.positions = pos
        else:
            print('Invalid initialization type')
            self.gridsize = gridsize
            self.a0 = a0
            self.rcell = rcell
            self.periodic_bc = periodic_bc

        # Default topology: all cells are of a single type
        self.idx_celltype = np.zeros(n) == 0
        self.cell_topology = np.ones((gridsize, gridsize))

    def init_lattice_excel(self, path, a0, rcell):
        """
        Initialize the lattice from an Excel file.

        Args:
            path (str): Path to the Excel file.
            a0 (float): Lattice constant.
            rcell (float): Cell radius.
        """
        positions, r, states = initialize_lattice_from_excel(path, a0)
        n = len(positions[:, 0])

        self.cells = np.squeeze(get_2_number_seq(states[:, np.newaxis]))
        self.relative_distance = r
        self.N = n
        self.positions = positions
        self.gridsize = np.sqrt(n)
        self.a0 = a0
        self.rcell = rcell

        self.idx_celltype = np.zeros(n) == 0
        self.cell_topology = np.ones((self.gridsize, self.gridsize))

    def init_topology(self, topology_input):
        """
        Initialize the cell topology based on an input matrix.

        Args:
            topology_input (ndarray): The input matrix representing the topology of the grid.
        """
        gridsize = self.gridsize
        top_mat = init_topology_mat(np.flip(topology_input), gridsize)
        self.cell_topology = top_mat
        self.idx_celltype = top_mat.reshape(self.N, order='F') > 0

    def init_cell_state(self, p0, I_min, dI):
        """
        Initialize the state of each cell based on input parameters.

        Args:
            p0 (ndarray): Fractions of cells in each initial state.
            I_min (float): Minimum value of Moran's I for initial configuration.
            dI (float): Increment of Moran's I.
        """
        dist = self.relative_distance
        N = self.N
        init_on = np.round(p0 * N)
        self.cells = init_I(init_on, self.a0, dist, N, I_min, dI)


    def init_general_parameters(self, K_matrix, C_matrix, M_matrix, lamb):
        """
        Initialize the general parameters for cell-cell communication and interaction.

        Args:
            K_matrix (ndarray): Threshold matrix for cell activation.
            C_matrix (ndarray): Secretion rates.
            M_matrix (ndarray): Interaction strengths.
            lamb (ndarray): Diffusion lengths.
        """
        # Reshape input matrices to the internal (2,2,2) or (2,1,2) format used by run_CA
        self.K, self.double_topology_flag = force_input_matrix_shape(K_matrix, 1)
        M, self.double_topology_flag = force_input_matrix_shape(M_matrix, 1)
        self.Con, self.double_topology_flag = force_input_matrix_shape(C_matrix, 2)

        self.M = M.astype(int32)
        self.lamb = lamb

    def run_model(self, tmax):
        """
        Run the cellular automaton model for a specified number of timesteps.

        Args:
            tmax (int): Maximum number of timesteps to run the model.
        """
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

        self.cell_hist, self.tmax, self.Y = run_CA(M, K, N, tmax, dist, Rcell, a0, lamb, idx_celltype, Coff, Con, cells)

        self.clock_traject[1] = np.round(time.time() - self.clock_start, 2)

    def analyse_trajectory(self):
        """
        Analyze the trajectory of the cellular automaton simulation.
        This method calculates various properties of the system such as state fractions,
        recurrent states, and vortex cores.
        """
        self.cell_4state = get_4_number_seq(self.cell_hist)
        self.cell_4phase = get_phase_seq_4(self.cell_4state)
        self.state_fractions = compute_4_state_fractions(self)
        self.first_recurrent_state_time = get_first_recurrent_state_time(self)
        self.final_pattern_type = get_final_configuration_type(self)

        if self.number_of_nn == 6:
            self.clock_traject[2] = np.round(time.time() - self.clock_start, 2)
            self.vortex_cores = compute_phase_differences(self)

            self.clock_traject[3] = np.round(time.time() - self.clock_start, 2)
            self.vortex_cores_labeled, self.n_vortex = label_triangular_lattice(self)

            self.clock_traject[4] = np.round(time.time() - self.clock_start, 2)
            self.xcom, self.ycom = compute_centroid(self)

            self.clock_traject[5] = np.round(time.time() - self.clock_start, 2)
            self.closest_pos, self.charges, self.xmin_core, self.ymin_core = compute_smallest_movement(self)

            self.clock_traject[6] = np.round(time.time() - self.clock_start, 2)
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

        Args:
            start_frame (int): The first frame to display.  (earliest possible frame = 0)
            end_frame (int): The last frame to display.
            frame_rate (int): The frame rate for visualization.
            store_frames (bool): If True, store the frames to disk.
            spin_vect (bool): If True, display spin vectors.
        """

        make_gif = True
        self.img = show_cells(folderName_for_frames, self.tmax - start_frame, self.cell_hist[:, :, start_frame:end_frame], self.positions,
                              self.idx_celltype, frame_rate, spin_vect, self.vortex_cores_labeled, self.charges,
                              self.cell_4phase, make_gif, store_frames)

    def make_gif(self, frame_start, frame_end, frame_rate, file_name):
        """
        Generate a GIF of the trajectory from the simulation.

        Args:
            frame_start (int): The starting frame for the GIF.
            frame_end (int): The ending frame for the GIF.
            frame_rate (int): Frame rate for the GIF.
            file_name (str): Name of the output GIF file (without extension).
        """
        img = self.img
        location = fr'{file_name}.gif'
        imageio.mimsave(location, img[frame_start:frame_end], fps=frame_rate)

    def show_single_frame(self, folder_for_specificFrame, frame_to_show, store_frames, spin_vect):

        if store_frames:
            frame_dir = folder_for_specificFrame
            jpg_dir = os.path.join(frame_dir, 'Frames_JPG')
            pdf_dir = os.path.join(frame_dir, 'Frames_PDF')
            os.makedirs(jpg_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)

        gain = 0.01
        bg_brightness = 240

        plt.ion()

        fig, ax = plt.subplots()
        ax.set_facecolor((bg_brightness / 255, bg_brightness / 255, bg_brightness / 255))
        fig.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        plt.title(f'Time step={frame_to_show + 1}')

        delta = 0.05
        cells = self.cell_hist[:, :, frame_to_show]
        gain = 10000 / np.sqrt(self.N)  # scale marker size to lattice size

        idx_celltype1 = self.idx_celltype
        idx_celltype2 = np.invert(self.idx_celltype)

        pos = self.positions
        cell_4phase = self.cell_4phase
        cores_labeled = self.vortex_cores_labeled

        colours1 = np.asarray([None] * len(pos[idx_celltype1, 0]), dtype=object)
        colours2 = np.asarray([None] * len(pos[idx_celltype2, 0]), dtype=object)
        colours1[:] = 'blue'
        colours2[:] = 'blue'

        pos1 = pos[idx_celltype1, :]
        pos2 = pos[idx_celltype2, :]

        if not spin_vect:
            plt_state1_1 = plt.scatter(pos1[:, 0], pos1[:, 1], s=gain, c='blue', marker="o", edgecolors='black')
            plt_state1_2 = plt.scatter(pos2[:, 0], pos2[:, 1], s=gain, c='blue', marker="s", edgecolors='black')
            plt.show()

            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 0)] = 'blue'
            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 1)] = 'red'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 0)] = 'white'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 1)] = 'black'

            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 0)] = 'blue'
            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 1)] = 'red'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 0)] = 'white'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 1)] = 'black'

            plt_state1_1.set_facecolors(c=colours1.tolist())
            plt_state1_2.set_facecolors(c=colours2.tolist())
        else:
            plt.quiver(pos[:, 0], pos[:, 1], np.cos(cell_4phase[:, 0]), np.sin(cell_4phase[:, 0]))
            plt.scatter(pos[:, 0], pos[:, 1], c=cores_labeled[:, 0], edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
            plt.show()

        plt.xlim(0 - delta * 2, np.max(pos[:, 0]) + delta * 2)
        plt.ylim(0 - delta, np.max(pos[:, 1]) + delta)

        fig.canvas.draw()

        if store_frames:
            plt.savefig(os.path.join(jpg_dir, f'step{frame_to_show + 1}.jpg'), format='jpg')
            plt.savefig(os.path.join(pdf_dir, f'step{frame_to_show + 1}.pdf'), format='pdf')

    def save_single_frame(self, fileName, parent_folder, jpg_folder, pdf_folder, frame_to_save, spin_vect):

        jpg_dir = os.path.join(parent_folder, jpg_folder)
        pdf_dir = os.path.join(parent_folder, pdf_folder)
        os.makedirs(jpg_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        bg_brightness = 240

        fig, ax = plt.subplots()
        ax.set_facecolor((bg_brightness / 255, bg_brightness / 255, bg_brightness / 255))
        fig.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        plt.title(f'Time step={frame_to_save + 1}')

        delta = 0.05
        cells = self.cell_hist[:, :, frame_to_save]
        gain = 10000 / np.sqrt(self.N)  # scale marker size to lattice size

        idx_celltype1 = self.idx_celltype
        idx_celltype2 = np.invert(self.idx_celltype)

        pos = self.positions
        cell_4phase = self.cell_4phase
        cores_labeled = self.vortex_cores_labeled

        colours1 = np.asarray([None] * len(pos[idx_celltype1, 0]), dtype=object)
        colours2 = np.asarray([None] * len(pos[idx_celltype2, 0]), dtype=object)
        colours1[:] = 'blue'
        colours2[:] = 'blue'

        pos1 = pos[idx_celltype1, :]
        pos2 = pos[idx_celltype2, :]

        if not spin_vect:
            plt_state1_1 = plt.scatter(pos1[:, 0], pos1[:, 1], s=gain, c='blue', marker="o", edgecolors='black')
            plt_state1_2 = plt.scatter(pos2[:, 0], pos2[:, 1], s=gain, c='blue', marker="s", edgecolors='black')

            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 0)] = 'blue'
            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 1)] = 'red'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 0)] = 'white'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 1)] = 'black'

            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 0)] = 'blue'
            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 1)] = 'red'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 0)] = 'white'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 1)] = 'black'

            plt_state1_1.set_facecolors(c=colours1.tolist())
            plt_state1_2.set_facecolors(c=colours2.tolist())
        else:
            plt.quiver(pos[:, 0], pos[:, 1], np.cos(cell_4phase[:, 0]), np.sin(cell_4phase[:, 0]))
            plt.scatter(pos[:, 0], pos[:, 1], c=cores_labeled[:, 0], edgecolors='black', cmap=plt.get_cmap('hot'), s=80)

        plt.xlim(0 - delta * 2, np.max(pos[:, 0]) + delta * 2)
        plt.ylim(0 - delta, np.max(pos[:, 1]) + delta)

        fig.canvas.draw()

        plt.savefig(os.path.join(jpg_dir, f'{fileName}.jpg'), format='jpg')
        plt.savefig(os.path.join(pdf_dir, f'{fileName}.pdf'), format='pdf')






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
    lx = 1
    delx = lx / gridsize
    dely = np.sqrt(3) / 2 * delx   # row spacing for triangular packing
    ly = dely * gridsize

    x = np.linspace(0, gridsize - 1, gridsize)
    xm, ym = np.meshgrid(x, x)

    # Offset every other row by half the column spacing to produce triangular geometry
    x = (xm + np.mod(ym, 2) / 2) * delx
    y = ym * dely

    pos = np.column_stack((x.flatten('F'), y.flatten('F')))

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
    [x1, x2] = np.meshgrid(pos[:, 0], pos[:, 0])
    [y1, y2] = np.meshgrid(pos[:, 1], pos[:, 1])

    dx = np.mod(abs(x1 - x2), lx)
    dy = np.mod(abs(y1 - y2), ly)

    if periodic_bc[0] == 1:
        dx[dx > (lx - dx)] = lx - dx[dx > (lx - dx)]

    if periodic_bc[1] == 1:
        dy[dy > (ly - dy)] = ly - dy[dy > (ly - dy)]

    dist = (dx ** 2 + dy ** 2) ** 0.5
    dist = dist / (lx / gz)   # normalize so that nearest-neighbor distance = 1

    return dist


def init_I(init_on, a0, dist, N, I_min, dI):
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
    cells = np.zeros((N, 2))

    for idx in range(2):
        cells[:int(init_on[idx]), idx] = 1
        cells[:, idx] = np.random.permutation(cells[:, idx])

        maxsteps = 5000
        I, theta = calculate_moranI(cells[:, idx], a0 * dist)

        # Skip adjustment if the configuration is degenerate (all ON or all OFF)
        if sum(cells[:, idx]) == N or sum(cells[:, idx]) == 0 or sum(cells[:, idx]) == 1 or sum(cells[:, idx]) == N - 1:
            check = False
        else:
            check = True

        eps = 1e-5
        dist_vec = calculate_unique_distances(dist, eps)
        dist1 = dist_vec[1]

        temp1 = dist1 + eps > dist
        temp2 = dist > dist1 - eps
        first_nei_idx = temp1 * temp2
        first_nei = np.zeros_like(first_nei_idx)
        first_nei[first_nei_idx] = 1

        increase = (I < I_min)
        t = 0
        I_max = I_min + dI

        while (I < I_min or I > I_max) and t < maxsteps and check:
            t += 1

            cells_new = cells[:, idx]
            nei_ON = np.matmul(first_nei, cells_new)
            nei_ON_1 = nei_ON[cells_new > 0]
            cond1 = 1

            if increase:
                idx_temp = res = np.argwhere(nei_ON_1 < 3)
                if idx_temp.size == 0:
                    idx_temp = np.argwhere(nei_ON_1 == np.min(nei_ON_1))
                    cond1 = 0
            else:
                idx_temp = res = np.argwhere(nei_ON_1 > 3)
                if idx_temp.size == 0:
                    idx_temp = np.argwhere(nei_ON_1 == np.max(nei_ON_1))
                    cond1 = 0.5

            idx_ON = np.random.choice(idx_temp.ravel())
            allON = np.argwhere(cells_new == 1)
            idx_1 = allON[:int(idx_ON)]

            nei_ON_0 = nei_ON[cells_new < 1]
            cond2 = 1

            if increase:
                idx_temp = res = np.argwhere(nei_ON_0 > 3)
                if idx_temp.size == 0:
                    idx_temp = np.argwhere(nei_ON_0 == np.min(nei_ON_0))
                    cond2 = 0
            else:
                idx_temp = res = np.argwhere(nei_ON_0 < 3)
                if idx_temp.size == 0:
                    idx_temp = np.argwhere(nei_ON_0 == np.max(nei_ON_0))
                    cond2 = 0

            idx_OFF = np.random.choice(idx_temp.ravel())
            allOFF = np.argwhere(cells_new == 0)
            idx_0 = allOFF[:int(idx_OFF)]

            if increase:
                idx_inc = 1
                cells_new[idx_0[-idx_inc:]] = abs(cells_new[idx_0[-idx_inc:]] - 1)
                cells_new[idx_1[-idx_inc:]] = abs(cells_new[idx_1[-idx_inc:]] - 1)
            else:
                cells_new[idx_0[-1:]] = abs(cells_new[idx_0[-1:]] - 1)
                cells_new[idx_1[-1:]] = abs(cells_new[idx_1[-1:]] - 1)

            I_new, theta = calculate_moranI(cells_new, a0 * dist)

            if cond1 == 1 and cond2 == 1:
                cells[:, idx] = cells_new
                I = I_new
            elif increase and (I_new >= I):
                cells[:, idx] = cells_new
                I = I_new
            elif not increase and I_new <= I:
                cells[:, idx] = cells_new
                I = I_new

            increase = (I < I_min)

    return cells


def calculate_moranI(cells, dist):
    """
    Calculates Moran's I for the current cell configuration, which measures spatial autocorrelation.

    Args:
        cells (ndarray): Array representing the state of cells.
        dist (ndarray): Pairwise distance matrix between cells.

    Returns:
        tuple: Moran's I and the calculated spatial autocorrelation (theta).
    """
    cells_pm = 2 * cells - 1
    cell_mean = np.mean(cells_pm)

    cells_matx, cells_maty = np.meshgrid(cells_pm, cells_pm)

    idx = dist > 0
    M = np.zeros_like(dist)
    M[idx] = (np.exp(-dist[idx]) / dist[idx])   # exponential distance-decay weights
    w_summed = np.sum(np.exp(-dist[idx]) / dist[idx])

    theta = np.sum(M * cells_matx * cells_maty) / w_summed

    temp = np.sum(M * (cells_matx - cell_mean) * (cells_maty - cell_mean))

    if temp != 0:
        cells_var = np.var(cells_pm, axis=0)
        I = np.sum(temp) / w_summed / cells_var
    else:
        I = 0

    return I, theta


def calculate_unique_distances(dist, eps):
    """
    Calculates unique distances from the distance matrix with a specified precision.

    Args:
        dist (ndarray): Pairwise distance matrix between cells.
        eps (float): Small epsilon value for rounding precision.

    Returns:
        ndarray: Array of unique rounded distances.
    """
    round_value = int(abs(np.log(eps) / np.log(10)))
    distance_value = np.unique(np.round(dist, round_value))
    return distance_value


def force_input_matrix_shape(M_in, type):
    """
    Ensures that the input matrix has the correct shape based on the type specified. The function handles
    matrices with one or two layers and adjusts them accordingly for input processing.

    Args:
        M_in (ndarray): The input matrix.
        type (int): Type 1 or 2 indicating the desired matrix shape transformation.

    Returns:
        tuple: The reshaped matrix and a flag indicating if a double topology is present.
    """
    sz = M_in.shape
    double_topology_flag = False

    if type == 1:
        M_out = np.zeros((2, 2, 2))
        if len(sz) == 3 and sz[2] == 1:
            M_out[0, :, :] = M_in
            M_out[1, :, :] = M_in
        elif len(sz) == 2:
            M_out[0, :, :] = M_in
            M_out[1, :, :] = M_in
        elif len(sz) == 3 and sz[2] == 2:
            M_out = M_in
            double_topology_flag = True
        else:
            print('Incorrect input size')
    elif type == 2:
        M_out = np.zeros((2, 1, 2))
        if len(sz) == 3 and sz[2] == 1:
            M_out[0, :, :] = M_in.T
            M_out[1, :, :] = M_in.T
        elif len(sz) == 1:
            M_out[0, :, :] = M_in
            M_out[1, :, :] = M_in
        elif len(sz) == 3 and sz[2] == 2:
            M_out = M_in
        else:
            print('Incorrect input size')
    else:
        M_out = []
        print('Incorrect input size')

    return M_out, double_topology_flag



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
    M_int = np.transpose(M)
    K = np.transpose(K)

    t = 0
    cells_hist = np.zeros((N, 2, tmax), np.int8)

    idx = dist > 0

    # Precompute pairwise interaction strengths using the Green's function kernel
    M = np.ones((N, N, 2))
    Y = np.ones((N, 2, tmax))

    for k in range(2):
        M[idx, k] = np.sinh(Rcell) / (a0 * dist[idx] / lamb[k]) * np.exp((Rcell - a0 * dist[idx]) / lamb[k])

    period = False

    while t < tmax and not period:
        cells_hist[:, :, t] = cells

        idx_loop = idx_celltype

        out = np.zeros((N, 4))
        C0 = np.zeros((N, 2))
        cells_out = np.zeros((N, 2))

        # Secretion rates depend on each cell's current ON/OFF state
        C0[idx_loop, :] = Coff[0, :, :] + (Con[0, :, :] - Coff[0, :, :]) * cells[idx_loop, :]
        C0[np.invert(idx_loop), :] = Coff[1, :, :] + (Con[1, :, :] - Coff[1, :, :]) * cells[np.invert(idx_loop), :]

        for k in range(2):
            Y[:, k, t] = np.matmul(np.squeeze(M[:, :, k]), C0[:, k])

        for j in range(2):
            out[idx_loop, 0] = ((Y[idx_loop, 0, t] - np.squeeze(K[0, 0, j])) * np.squeeze(M_int[0, 0, j]) > 0) + (
                    1 - abs(M_int[0, 0, j]))
            out[idx_loop, 1] = ((Y[idx_loop, 1, t] - np.squeeze(K[1, 0, j])) * np.squeeze(M_int[1, 0, j]) > 0) + (
                    1 - abs(M_int[1, 0, j]))
            out[idx_loop, 2] = ((Y[idx_loop, 0, t] - np.squeeze(K[0, 1, j])) * np.squeeze(M_int[0, 1, j]) > 0) + (
                    1 - abs(M_int[0, 1, j]))
            out[idx_loop, 3] = ((Y[idx_loop, 1, t] - np.squeeze(K[1, 1, j])) * np.squeeze(M_int[1, 1, j]) > 0) + (
                    1 - abs(M_int[1, 1, j]))

            cells_out[idx_loop, 0] = np.squeeze(out[idx_loop, 0]) * np.squeeze(out[idx_loop, 1])
            cells_out[idx_loop, 1] = np.squeeze(out[idx_loop, 2]) * np.squeeze(out[idx_loop, 3])

            idx_loop = np.invert(idx_loop)

        t += 1
        # cells_out is a local variable; self.cells (the initial configuration) is never overwritten
        cells = cells_out

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
    CmX = np.zeros((t_end - t_start, 2 + int(np.max(self.vortex_cores_labeled))))
    CmY = np.zeros((t_end - t_start, 2 + int(np.max(self.vortex_cores_labeled))))

    if self.number_of_nn == 6:
        lx = np.max(self.positions[:, 0]) + self.positions[0, 0]/2
        ly = np.max(self.positions[:, 1]) + self.positions[0, 1]/2
    else:
        lx = np.max(self.positions[:, 0]) + self.positions[0, 0]
        ly = np.max(self.positions[:, 1]) + self.positions[0, 1]

    closest_pos = np.zeros((self.tmax, 20)) * np.nan
    for p in range(t_end - t_start):
        # Set frame number
        fn = p + t_start

        # Set frame and contours of frame
        hist_run = self.cell_4phase[:, fn]
        c_run = self.vortex_cores_labeled[:, fn]

        # Run through all labels
        idx_all = np.unique(self.vortex_cores_labeled[:, fn], axis=0)

        for idx_label in idx_all[1:]:
            x_hat, y_hat, c_r, m_x_r, m_y_r = get_contour_pos_relative_2_center(c_run, hist_run, idx_label, self)

            theta_r = np.arctan2(y_hat, x_hat) + np.pi
            idx_sort = np.argsort(theta_r)

            cwise = c_r[idx_sort]
            cwise_wrap = np.zeros((1, len(cwise[0, :]) + 1))
            cwise_wrap[0, :-1] = cwise
            cwise_wrap[0, len(cwise[0, :])] = cwise[0, 0]

            temp = np.diff(cwise_wrap)
            temp[temp / np.pi == -3 / 2] = 0.5 * np.pi
            temp[temp / np.pi == 3 / 2] = -0.5 * np.pi

            charges[p, idx_label - 1] = np.sum(temp) / np.pi
            CmX[p, idx_label - 1] = m_x_r
            CmY[p, idx_label - 1] = m_y_r

    return closest_pos, charges, CmX, CmY


def interchange_numbers(input_array, v1, v2):
    idx_v1 = input_array == v1
    idx_v2 = input_array == v2

    input_array[idx_v1] = v2
    input_array[idx_v2] = v1

    return input_array


def compute_phase_differences(self):
    # Integer arithmetic on 4-state values for speed; swap states 3 and 4
    # so that the ordering matches the phase sequence 0, pi/2, pi, 3pi/2
    C4 = self.cell_4state.astype(np.int8)
    C4 = interchange_numbers(C4, 4, 3)

    temp = C4[self.cell_number_nn.ravel(), :]
    frame_values = temp.reshape(self.N, self.number_of_nn + 1, self.tmax)

    temp = frame_values[:, 0, :]
    delta = frame_values[:, 1:] - temp[:, np.newaxis, :]

    # A cell is a vortex core if any neighboring phase difference equals 2 (i.e., pi)
    vortex_cores = (np.squeeze(np.sum(abs(delta) == 2, axis=1)) > 0) * 1

    return vortex_cores




def get_final_configuration_type(self):
    # All cells in the same state at the final timestep -> static configuration
    if len(np.unique(self.cell_4state[:, -1])) == 1:
        final_pattern_type = 1

    # State-fraction increments are constant after the first recurrence -> periodic (wave) pattern
    elif len(np.unique(np.diff(self.state_fractions[1, self.first_recurrent_state_time:]))) == 1:
        final_pattern_type = 3

    else:
        final_pattern_type = 2

    return final_pattern_type


def get_first_recurrent_state_time(self):
    # Count distinct whole-lattice configurations across time;
    # the number of unique configurations equals the timestep at which the first recurrence occurs.
    cell4 = self.cell_4state
    unique_configurations = np.unique(cell4, axis=1)
    p_time = unique_configurations.shape[1]
    return p_time


def compute_4_state_fractions(self):
    state_fractions = np.zeros((4, self.tmax))
    state_fractions[0, :] = np.sum(self.cell_4state == 1, axis=0) / self.N
    state_fractions[1, :] = np.sum(self.cell_4state == 2, axis=0) / self.N
    state_fractions[2, :] = np.sum(self.cell_4state == 3, axis=0) / self.N
    state_fractions[3, :] = np.sum(self.cell_4state == 4, axis=0) / self.N

    return state_fractions



def label_triangular_lattice(self):
    tmax = self.tmax
    vortex_cores = copy.deepcopy(self.vortex_cores)

    # Apply two-pass algorithm writen for a triangular lattice
    for frame_number in range(0, tmax):
        # phase 1: first labeling
        labels = np.where(vortex_cores[:, frame_number] == 1)
        for i in range(len(labels[0][:])):
            current_cell = vortex_cores[labels[0][i], frame_number]

            if current_cell == 1:
                neighbours_idx = np.where(np.round(self.relative_distance[labels[0][i], :], 1) == 1)
                neighbours_values = vortex_cores[neighbours_idx, frame_number]
                temp = neighbours_values[neighbours_values > 1]

                if not any(temp):
                    min_neighbour_value = 1
                else:
                    min_neighbour_value = min(temp)

                if min_neighbour_value == 1:
                    vortex_cores[labels[0][i], frame_number] = np.max(vortex_cores[:, frame_number]) + 1
                else:
                    vortex_cores[labels[0][i], frame_number] = min_neighbour_value

        # phase 2: assign minimum to all connected and labeled areas
        for i in range(len(labels[0][:])):
            neighbours_idx = np.where(np.round(self.relative_distance[labels[0][i], :], 1) <= 1)
            neighbours_values = vortex_cores[neighbours_idx, frame_number]
            neighbours_unique = np.unique(neighbours_values[neighbours_values > 0])
            if len(neighbours_unique) > 1:
                for j in range(len(neighbours_unique)):
                    vortex_cores[vortex_cores[:, frame_number] == neighbours_unique[j], frame_number] = \
                        np.min(neighbours_unique)

        # phase 3: Make sure no index is missed (e.g. [1 2 4 5] --> [1 2 3 4])
        # more elegant: np.unique(a, return_inverse=True)[1].reshape(a.shape)

        for i in range(np.max(vortex_cores[:, frame_number])):
            if not any(vortex_cores[:, frame_number] == (i + 0)):
                vortex_cores[vortex_cores[:, frame_number] >= (i + 0), frame_number] -= 1

    vortex_cores_labeled = vortex_cores
    n_vortex = np.max(vortex_cores, axis=0)

    return vortex_cores_labeled, n_vortex


def label_elements_triangular_lattice(input_images, relative_distance, t_start, t_end):

    # Apply two-pass algorithm writen for a triangular lattice
    for frame_number in range(t_start, t_end):
        # phase 1: first labeling
        labels = np.where(input_images[:, frame_number] == 1)
        for i in range(len(labels[0][:])):
            current_cell = input_images[labels[0][i], frame_number]

            if current_cell == 1:
                neighbours_idx = np.where(np.round(relative_distance[labels[0][i], :], 1) == 1)
                neighbours_values = input_images[neighbours_idx, frame_number]
                temp = neighbours_values[neighbours_values > 1]

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



def show_cells(folderName_for_frames, tmax, cells_hist, pos, idx_celltype, frame_rate, spin_vect, cores_labeled, charges, cell_4phase, make_gif, store_frames):
    """
    This function visualizes the evolution of a cellular automaton over time. It displays the states of cells at each time step using either color-coded scatter plots or vector fields to represent different cell states and phases. The function can create animations and optionally save frames as individual images or a GIF.

    Arguments:
    - tmax: Maximum number of time steps for visualization. (= T - initial frame, where T = 'tmax' defined in main.py)
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
    img = [None] * tmax
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
        plt.show()

    # Set plot limits based on cell positions
    plt.xlim(0 - delta * 2, np.max(pos[:, 0]) + delta * 2)
    plt.ylim(0 - delta, np.max(pos[:, 1]) + delta)

    fig.canvas.draw()
    plt.show()

    # Initialize GIF creation if required
    if make_gif:
        buf = fig.canvas.buffer_rgba()
        temp = np.asarray(buf, dtype=np.uint8)
        temp = temp[:, :, :3]  # convert RGBA to RGB
        img[0] = temp.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Ensure directories exist for storing frames if required
    if store_frames:
        frame_dir = folderName_for_frames
        jpg_dir = os.path.join(frame_dir, 'Frames_JPG')
        pdf_dir = os.path.join(frame_dir, 'Frames_PDF')
        os.makedirs(jpg_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

    # Main loop for visualizing each time step
    tmax = len(cells_hist[0, 0, :])
    for t in range(0, tmax):
        t1 = time.time()
        print(t)
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
                    if charges[t, p - 1] < 0:
                        plt.scatter(pos[temp_label, 0], pos[temp_label, 1], c='orange', edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
                    elif charges[t, p - 1] > 0:
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
            buf = fig.canvas.buffer_rgba()
            temp = np.asarray(buf, dtype=np.uint8)
            temp = temp[:, :, :3]  # convert RGBA to RGB
            img[t] = temp.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if store_frames:
            plt.savefig(os.path.join(jpg_dir, f'step{t}.jpg'), format='jpg')  # Save as JPG
            plt.savefig(os.path.join(pdf_dir, f'step{t}.pdf'), format='pdf')  # Save as PDF

        # Ensure consistent frame rate for animation
        if delta_time < 1 / frame_rate:
            time.sleep(1 / frame_rate - delta_time)

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


def save_trajectory(file, file_path):
    """Save a CellularLattice trajectory object to disk as a pickle file."""
    with open(file_path, 'wb') as output:
        pickle.dump(file, output, pickle.HIGHEST_PROTOCOL)


def load_trajectory(path):
    """Load a CellularLattice trajectory object from a pickle file."""
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

    # Convert the binary state [0, 0] into the 4-state representation (1).
    cell_4_state_out[(cell_current_BinaryState[:, 0] == 0) * (cell_current_BinaryState[:, 1] == 0)] = 1

    # Convert the binary state [0, 1] into the 4-state representation (2).
    cell_4_state_out[(cell_current_BinaryState[:, 0] == 0) * (cell_current_BinaryState[:, 1] == 1)] = 2

    # Convert the binary state [1, 0] into the 4-state representation (3).
    cell_4_state_out[(cell_current_BinaryState[:, 0] == 1) * (cell_current_BinaryState[:, 1] == 0)] = 3

    # Convert the binary state [1, 1] into the 4-state representation (4).
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
    Compute unique pairwise distances from the relative distance matrix and build
    a boolean index array marking each unique distance.

    Parameters
    ----------
    CA1 : CellularLattice
        Object containing CA1.relative_distance, a 2D pairwise distance array.

    Returns
    -------
    D_unique : ndarray, shape (2, K)
        Row 0: sorted unique distances. Row 1: occurrence count of each distance.
    idx : ndarray, shape (N, N, K), dtype bool
        idx[:, :, i] is True wherever the i-th unique distance appears in D.
    """
    D = np.round(CA1.relative_distance, 2)
    D_unique_vals = np.unique(D)

    D_unique = np.zeros((2, len(D_unique_vals)))
    D_unique[0, :] = D_unique_vals

    for c in range(len(D_unique[0, :])):
        mask = D[0, :] == D_unique[0, c]
        D_unique[1, c] = np.sum(mask)

    idx = np.zeros((len(D[:, 0]), len(D[0, :]), len(D_unique[0, :])), dtype=bool)
    for i in range(len(D_unique[0, :])):
        idx[:, :, i] = D == D_unique[0, i]

    return D_unique, idx




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

        thetaX_m = np.arctan2(-np.matmul(betaX, c) / count, -np.matmul(alphaX, c) / count) + np.pi
        thetaY_m = np.arctan2(-np.matmul(betaY, c) / count, -np.matmul(alphaY, c) / count) + np.pi

        x_com[:, j] = thetaX_m / (2 * np.pi) * np.max(xx)
        y_com[:, j] = thetaY_m / (2 * np.pi) * np.max(yy)

    return x_com, y_com




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
    hp_in = c == idx_label
    hp = c == idx_label
    idx = np.where(hp_in)

    # calculate neighbouring values and set tem all to true
    neighbours_idx = np.where(np.round(relative_distance[idx[0], :], 1) == 1)
    hp[neighbours_idx[1]] = True

    hp_in = hp_in * 1
    hp = hp * 1

    return hp - hp_in


def get_contour_pos_relative_2_center(c_run, hist_run, idx_label, CA1):
    # compute values around vortex
    ring = get_area_contour(c_run, idx_label, CA1.relative_distance)

    idx_ring = np.where(ring == 1)
    x_r = CA1.positions[idx_ring, 0]
    y_r = CA1.positions[idx_ring, 1]
    c_r = hist_run[idx_ring]

    xx = CA1.positions[:, 0]
    yy = CA1.positions[:, 1]

    lx = np.max(CA1.positions[:, 0]) + CA1.positions[1, 0]
    ly = np.max(CA1.positions[:, 1]) + CA1.positions[1, 1]

    ring_bin = (ring == 1) * 1

    m_x_r, m_y_r = compute_center_of_mass(xx, yy, ring_bin)

    # Apply centroid algorithm
    y_hat = (y_r - m_y_r + ly / 2) % ly - ly / 2
    x_hat = (x_r - m_x_r + lx / 2) % lx - lx / 2

    return x_hat, y_hat, c_r, m_x_r, m_y_r


def compute_core_charges_and_type_counts(trajectory):
    """
    Compute per-core integer charges (q in {-1,0,+1}) for each labeled core at each timestep,
    and return per-timestep counts: n_pos, n_neg, n_zero, n_total, net_charge.
    """
    if trajectory.number_of_nn != 6:
        raise ValueError("compute_core_charges_and_type_counts assumes triangular lattice (6 nn).")

    c_labels = trajectory.vortex_cores_labeled   # (N, tmax)
    phases   = trajectory.cell_4phase           # (N, tmax) in radians
    tmax     = trajectory.tmax

    max_label = int(np.max(c_labels))
    charges_int = np.zeros((tmax, max_label + 1), dtype=np.int8)  # col 0 unused

    for t in range(tmax):
        c_run = c_labels[:, t]
        hist_run = phases[:, t]

        labels_here = np.unique(c_run)
        labels_here = labels_here[labels_here > 0]
        if labels_here.size == 0:
            continue

        for lab in labels_here:
            x_hat, y_hat, c_r, _, _ = get_contour_pos_relative_2_center(c_run, hist_run, lab, trajectory)

            # ---- CRITICAL: force everything to 1D ----
            x_hat = np.asarray(x_hat).ravel()
            y_hat = np.asarray(y_hat).ravel()
            c_r   = np.asarray(c_r).ravel()

            if c_r.size == 0:
                charges_int[t, lab] = 0
                continue

            theta_r = np.arctan2(y_hat, x_hat) + np.pi
            idx_sort = np.argsort(theta_r)

            cwise = c_r[idx_sort].astype(float, copy=False)  # 1D

            # 1D wrap
            cwise_wrap = np.empty(cwise.size + 1, dtype=float)
            cwise_wrap[:-1] = cwise
            cwise_wrap[-1]  = cwise[0]

            temp = np.diff(cwise_wrap)

            # unwrap phase jumps across the branch cut, consistent with compute_smallest_movement
            temp[temp / np.pi == -3 / 2] = 0.5 * np.pi
            temp[temp / np.pi ==  3 / 2] = -0.5 * np.pi

            q_raw = np.sum(temp) / np.pi      # typically ±2 for ±1 vortex
            q_int = int(np.round(q_raw / 2))  # map to ±1

            if q_int > 1:  q_int = 1
            if q_int < -1: q_int = -1
            charges_int[t, lab] = q_int

    # Per-timestep counts
    n_pos  = np.zeros(tmax, dtype=np.int32)
    n_neg  = np.zeros(tmax, dtype=np.int32)
    n_zero = np.zeros(tmax, dtype=np.int32)
    n_tot  = np.zeros(tmax, dtype=np.int32)
    net    = np.zeros(tmax, dtype=np.int32)

    for t in range(tmax):
        labels_here = np.unique(c_labels[:, t])
        labels_here = labels_here[labels_here > 0]
        if labels_here.size == 0:
            continue
        q = charges_int[t, labels_here]
        n_pos[t]  = int(np.sum(q ==  1))
        n_neg[t]  = int(np.sum(q == -1))
        n_zero[t] = int(np.sum(q ==  0))
        n_tot[t]  = int(labels_here.size)
        net[t]    = int(n_pos[t] - n_neg[t])

    return charges_int, n_pos, n_neg, n_zero, n_tot, net


def detect_one_step_total_blips(n_total, t_blip_start=150):
    """
    Detect one-timestep +1 blips in n_total(t), only for t >= t_blip_start,
    requiring a return to baseline in the next timestep.

    Returns:
      blip_times : list of t where the blip occurs (the middle timestep)
    """
    tmax = len(n_total)
    blip_times = []
    for t in range(max(t_blip_start, 1), tmax - 1):
        if (n_total[t] == n_total[t-1] + 1) and (n_total[t+1] == n_total[t-1]):
            blip_times.append(t)
    return blip_times



def classify_blips(blip_times, n_pos, n_neg, n_zero, n_total):
    """
    Classify each one-step total blip by how the type-counts change at the blip step.

    Returns:
      dict with counts and per-event records.
    """
    records = []
    n_zero_blip = 0
    n_other = 0

    for t in blip_times:
        dpos  = int(n_pos[t]  - n_pos[t-1])
        dneg  = int(n_neg[t]  - n_neg[t-1])
        dzero = int(n_zero[t] - n_zero[t-1])
        dtot  = int(n_total[t]- n_total[t-1])

        kind = "OTHER"
        if dtot == 1 and dpos == 0 and dneg == 0 and dzero == 1:
            kind = "ZERO_VORTEX_BLIP"
            n_zero_blip += 1
        else:
            n_other += 1

        records.append((t, dtot, dpos, dneg, dzero, kind))

    return {
        "n_blips": len(blip_times),
        "n_zero_vortex_blips": n_zero_blip,
        "n_other_blips": n_other,
        "records": records,
    }




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







