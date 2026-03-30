"""
Analyzes the output produced by perturb_initial_state_oneByone.py.

For each single-cell perturbation of an initial configuration, loads the
resulting final pattern type and final binary configuration, then produces
three histograms:

  1. Distribution of final pattern types across all perturbations.
  2. Distribution of run times (timesteps to final pattern) across all perturbations.
  3. Distribution of the number of cells whose final state differs from the
     final state of the unperturbed (original) run.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
folderName_parent             = 'OUTPUT_perturb_initial_state_oneByone'
folderName_finalType_Labels   = os.path.join(folderName_parent, 'finalType_Labels')
folderName_FinalBinaryConfig  = os.path.join(folderName_parent, 'finalConfig_inBinary_Array')
folderName_OUTPUT_Figures     = os.path.join(folderName_parent, 'OUTPUT_Figures')
os.makedirs(folderName_OUTPUT_Figures, exist_ok=True)

fileName_finalPatternLabels        = 'FinalPattern_labels_for_perturbedRuns.npy'
fileName_originalPatternLabel      = 'FinalPatternLabel_for_originalCA.npy'
fileName_parameters                = 'OUTPUT_parameters.npy'
fileName_runTimes_for_Perturbed    = 'RunTimes_for_perturbedRuns.npy'
fileName_runTime_for_Original      = 'RunTime_for_originalCA.npy'
fileName_finalConfig_for_Original  = 'original_finalConfigBinary.npy'


# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
label_finalPatterns    = np.load(os.path.join(folderName_finalType_Labels, fileName_finalPatternLabels),     allow_pickle=True)
label_initialPattern   = np.load(os.path.join(folderName_finalType_Labels, fileName_originalPatternLabel),   allow_pickle=True)
parameters_used        = np.load(os.path.join(folderName_finalType_Labels, fileName_parameters),             allow_pickle=True)
runTimes_for_Perturbed = np.load(os.path.join(folderName_finalType_Labels, fileName_runTimes_for_Perturbed), allow_pickle=True)
runTime_for_Original   = np.load(os.path.join(folderName_finalType_Labels, fileName_runTime_for_Original),   allow_pickle=True)
original_final_config  = np.load(os.path.join(folderName_FinalBinaryConfig, fileName_finalConfig_for_Original), allow_pickle=True)

print(f'Final pattern type for the original (unperturbed) run: {label_initialPattern}')
print(f'Runtime for original run: {runTime_for_Original}')
print(f'Final pattern types for perturbed runs: {label_finalPatterns}')

num_samples = len(runTimes_for_Perturbed)
print(f'Number of perturbed runs: {num_samples}')


# -----------------------------------------------------------------------
# Compute cell-by-cell distance between each perturbed final config
# and the original final config
# -----------------------------------------------------------------------
distance_between_finalConfigs = np.zeros_like(runTimes_for_Perturbed)

for i in range(num_samples):
    fileName_finalConfig_for_perturbed = f'finalConfigBinary_Perturbation_{i+1}.npy'
    perturbed_final_config = np.load(
        os.path.join(folderName_FinalBinaryConfig, fileName_finalConfig_for_perturbed),
        allow_pickle=True
    )
    row_comparison = np.all(original_final_config == perturbed_final_config, axis=1)
    distance_between_finalConfigs[i] = np.sum(~row_comparison)

np.save(
    os.path.join(folderName_OUTPUT_Figures, 'distance_between_finalConfigs.npy'),
    distance_between_finalConfigs
)


# -----------------------------------------------------------------------
# Figure 1: histogram of final pattern types across all perturbations
# -----------------------------------------------------------------------
plt.figure(1, figsize=(10, 6))

counts, bins, patches = plt.hist(
    label_finalPatterns,
    bins=[0.5, 1.5, 2.5, 3.5],
    edgecolor='black',
    align='mid'
)

plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=len(runTimes_for_Perturbed)))
plt.xticks([1, 2, 3])
plt.title(
    f'Final pattern types after single-cell perturbations\n'
    f'(n = {label_finalPatterns.shape[0]}, original type = {label_initialPattern})'
)
plt.xlabel('Final pattern type  (1 = static, 2 = spiral wave, 3 = rectilinear wave)')
plt.ylabel('% of perturbed runs')

plt.savefig(os.path.join(folderName_OUTPUT_Figures, 'histogram_finalPatterns_afterPerturbation.jpg'), format='jpg')
plt.savefig(os.path.join(folderName_OUTPUT_Figures, 'histogram_finalPatterns_afterPerturbation.pdf'), format='pdf')
plt.show()


# -----------------------------------------------------------------------
# Figure 2: histogram of run times across all perturbations
# -----------------------------------------------------------------------
plt.figure(2, figsize=(10, 6))

counts, bins, patches = plt.hist(
    runTimes_for_Perturbed,
    bins=25,
    edgecolor='black',
    align='mid'
)

plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=len(runTimes_for_Perturbed)))
plt.title(
    f'Run times after single-cell perturbations\n'
    f'(n = {runTimes_for_Perturbed.shape[0]}, original runtime = {runTime_for_Original})'
)
plt.xlabel('Timesteps to final pattern formation')
plt.ylabel('% of perturbed runs')

plt.savefig(os.path.join(folderName_OUTPUT_Figures, 'histogram_runTimes_afterPerturbation.jpg'), format='jpg')
plt.savefig(os.path.join(folderName_OUTPUT_Figures, 'histogram_runTimes_afterPerturbation.pdf'), format='pdf')
plt.show()


# -----------------------------------------------------------------------
# Figure 3: histogram of number of cells whose final state differs
# from the final state of the original (unperturbed) run
# -----------------------------------------------------------------------
plt.figure(3, figsize=(10, 6))

counts, bins, patches = plt.hist(
    distance_between_finalConfigs,
    bins=100,
    edgecolor='black',
    align='mid'
)

plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=len(distance_between_finalConfigs)))
plt.title(
    f'Number of cells in final configuration differing from the original run\n'
    f'(n = {distance_between_finalConfigs.shape[0]})'
)
plt.xlabel('Number of cells differing from final configuration of original run')
plt.ylabel('% of perturbed runs')

plt.savefig(os.path.join(folderName_OUTPUT_Figures, 'histogram_numCellsDifferent.jpg'), format='jpg')
plt.savefig(os.path.join(folderName_OUTPUT_Figures, 'histogram_numCellsDifferent.pdf'), format='pdf')
plt.show()
