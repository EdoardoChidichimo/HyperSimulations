import os
from config import *
from itertools import product
import matplotlib.pyplot as plt
import pickle
import pandas as pd

sensors_labels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz',
                  'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO9', 'PO1', 'OZ', 'PO2', 'PO10']

def lab2idx(sensor):
    matches = [i for i, s in enumerate(sensors_labels) if s.lower() == sensor.lower()]
    if matches:
        return matches[0]
    else:
        raise ValueError(f"Sensor label '{sensor}' not found in sensors_labels")


all_clusters = [
    ['Fp1', 'F3', 'F7', 'Fz'], # Frontal Left
    ['Fp2', 'F4', 'F8', 'Fz'], # Frontal Right

    ['CP5', 'P3', 'P7'], # Parietal Left
    ['CP6', 'P4', 'P8'], # Parietal Right

    ['PO9', 'PO1', 'Oz'], # Occipital Left
    ['PO10', 'PO2', 'Oz'], # Occipital Right

    ['P3', 'PO9'],  # Parieto-Occipital Left
    ['P4', 'PO10'], # Parieto-Occipital Right
]

all_labels = ['FL', 'FR', 'PL', 'PR', 'OL', 'OR', 'POL', 'POR']

all_cluster_combinations = []
all_label_combinations = []

for idx1, cluster1 in enumerate(all_clusters):
    for idx2, cluster2 in enumerate(all_clusters):
        if idx2 < idx1:
            continue

        combo = [cluster1, cluster2]
        label_combo = [all_labels[idx1], all_labels[idx2]]

        all_cluster_combinations.append(combo)
        all_label_combinations.append(label_combo)



mask = np.array([s for s in np.arange(32)])

columns = ['cinter', 'phase_noise', 'freq_std', 'amp_noise', 'sensor_noise', 'iteration', 'brain12', 
           'measure', 'freq_band', 'cluster', 'value_real', 'value_shuffled', 'diff']
accu = []


def add_results():

    accu = []
    freq_filt = None

    for directory in ["standard", "mi", "phyid"]:

        if directory == "standard":
            mode_names = standard_modes
            freq_filt = True

        elif directory == "mi":
            mode_names = mi_modes
            freq_filt = False

        elif directory == "phyid":
            mode_names = phyid_modes
            freq_filt = False

        # REMINDER:
        # standard_modes = ['plv', 'pli', 'wpli', 'ccorr', 'coh', 'imaginary_coh', 'envelope_corr', 'pow_corr']
        # mi_modes = ['mi_histogram', 'mi_gaussian', 'mi_kernel', 'mi_ksg']
        # phyid_modes = ['phyid_tdmi', 'phyid_te', 'phyid_pure', 'phyid_redundancy', 'phyid_synergy']

        combination = product(cinter_dict.values(),
                              [0.0,0.5,1.0]) # noise

        for cinter, noise in combination:

            phase_noise = noise*0.1
            freq_std = noise
            amp_noise = noise
            sensor_noise = noise

            for it in range(n_it):

                filename = f'cinter_{cinter}_phase_noise{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}_it_{it}.pkl' # notice the accidental missing _ after phase_noise

                full_file_h1 = os.path.join(f'simulations_0.53/results/IB_{directory}', filename)
                full_file_h0 = os.path.join(f'simulations_0.53/results/null/IB_{directory}', filename)

                with open(full_file_h1, 'rb') as f1:
                    results_h1 = pickle.load(f1) # shape (modes, freq_bands, 2*ch, 2*ch) or (modes, 2*ch, 2*ch)

                with open(full_file_h0, 'rb') as f0:
                    results_h0 = pickle.load(f0)

                if not freq_filt:
                    results_h1 = np.expand_dims(results_h1, axis=1)
                    results_h0 = np.expand_dims(results_h0, axis=1)
                    freqs = broadband
                else:
                    freqs = freq_bands

                for mode_idx, mode in enumerate(mode_names):
                    for freq_band_idx, freq_band in enumerate(freqs):
                        for cluster_idx, cluster in enumerate(all_cluster_combinations):
                            idx1 = np.array([lab2idx(s) for s in cluster[0]])
                            idx2 = np.array([lab2idx(s) for s in cluster[1]])
                            combo_label = "-".join(all_label_combinations[cluster_idx])

                            val_inter12_h1 = np.nanmean(results_h1[mode_idx][freq_band_idx, mask[idx1], :][:, mask[idx2] + 32])
                            val_inter21_h1 = np.nanmean(results_h1[mode_idx][freq_band_idx, mask[idx1] + 32, :][:, mask[idx2]])
                            val_inter12_h0 = np.nanmean(results_h0[mode_idx][freq_band_idx, mask[idx1], :][:, mask[idx2] + 32])
                            val_inter21_h0 = np.nanmean(results_h0[mode_idx][freq_band_idx, mask[idx1] + 32, :][:, mask[idx2]])

                            diff_12 = (val_inter12_h1 - val_inter12_h0) / (val_inter12_h1 + val_inter12_h0)
                            diff_21 = (val_inter21_h1 - val_inter21_h0) / (val_inter21_h1 + val_inter21_h0)

                            # ['cinter', 'phase_noise', 'freq_std', 'amp_noise', 'sensor_noise', 'iteration', 'measure', 'freq_band', 'cluster', 'brain12', 'value_real', 'value_shuffled', 'diff']

                            accu.append([cinter, phase_noise, freq_std, amp_noise, sensor_noise, it, mode, freq_band, combo_label, 1, val_inter12_h1, val_inter12_h0, diff_12])
                            accu.append([cinter, phase_noise, freq_std, amp_noise, sensor_noise, it, mode, freq_band, combo_label, 0, val_inter21_h1, val_inter21_h0, diff_21])

    pd.DataFrame(data=accu, columns=columns).to_csv(f'simulations_0.53/results/analysis/summary.csv')

if __name__ == "__main__":
    add_results()
