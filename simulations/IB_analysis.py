#!/usr/bin/env python
# coding=utf-8
# ====================================================================================================
# title           : IB_analysis.py
# description     : Conduct Inter-Brain analysis on simulations w/ standard, MI, and phyid measures
# author          : Edoardo Chidichimo, Guillaume Dumas
# date            : 2024-11-01
# version         : 0.1
# usage           : python IB_analysis.py
# python_version  : >3.10
# ====================================================================================================

import os
import numpy as np
from config import *
import pickle
import mne
from hyperit import HyperIT
import warnings
from itertools import product
from joblib import Parallel, delayed
from hypyp import analyses

mne.set_log_level(verbose=False)
warnings.filterwarnings("ignore", category=RuntimeWarning, module='mne', message='.*filter length.*')
mne.set_log_level(verbose=False)

def shuffle_epochs(data: np.ndarray):

    # data has shape (2 participants, 60 epochs, 32 channels, 1000 timepoints)
    for participant in range(data.shape[0]):
        np.random.shuffle(data[participant])
    
    return data

def IB_analysis_standard(data_inter: np.ndarray) -> np.ndarray:

    IB_analysis_results = []
    
    for mode in standard_modes:
        connectivity = analyses.pair_connectivity(data=data_inter,
                                                sampling_rate=sfreq,
                                                frequencies=freq_bands,
                                                mode=mode)
        IB_analysis_results.append(connectivity)

    return IB_analysis_results

def IB_analysis_IT(data_inter: np.ndarray):

    IB_analysis_results_IT = []
    data1, data2 = data_inter

    HyperIT.setup_JVM()
    itClass = HyperIT(data1, data2, show_tqdm=False, verbose=False)

    for mode in ['histogram', 'gaussian', 'kernel', 'ksg', 'symbolic']:
        IB_analysis_results_IT.append(itClass.compute_mi(estimator=mode, include_intra=True, epoch_average=True))

    return IB_analysis_results_IT


def IB_analysis_phyid(data_inter: np.ndarray):

    data1, data2 = data_inter

    HyperIT.setup_JVM()
    itClass = HyperIT(data1, data2, show_tqdm=False, verbose=False)

    IB_phyid_results = []
    atoms = itClass.compute_atoms(tau=23, include_intra=True, epoch_average=True)

    tdmi = np.sum(atoms, axis=-1)

    te_atoms = [4,6,12,14] 
    te = np.sum(atoms[..., te_atoms], axis=-1)
    
    pure = atoms[..., 6]

    redundancy = atoms[..., 0]

    synergy = atoms[..., 15]

    IB_phyid_results.append(tdmi)
    IB_phyid_results.append(te)
    IB_phyid_results.append(pure)
    IB_phyid_results.append(redundancy)
    IB_phyid_results.append(synergy)

    return np.array(IB_phyid_results)

def IB_analysis_main(cinter: float,
                            phase_noise: float, 
                            freq_std: float, 
                            amp_noise: float,
                            sensor_noise: float,
                            it: int,
                            analysis: str,
                            calc_null: bool):

    np.random.seed(it)

    full_configuration = f'cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}/it_{it}.pkl'

    sim_file_path = os.path.join(f'simulations_0.53/sim/', full_configuration)
    if not os.path.exists(sim_file_path):
        print(f'Simulation still needed for {full_configuration}', flush=True)
        return

    with open(sim_file_path, 'rb') as file:
        simulation = np.array(pickle.load(file)) # shape (2, 60, 32, 1000)

    if calc_null:
        simulation = shuffle_epochs(simulation)

    if analysis == "IB_standard":
        IB_analysis_results = IB_analysis_standard(simulation)
    elif analysis == "IB_mi":
        IB_analysis_results = IB_analysis_IT(simulation)
    elif analysis == "IB_phyid":
        IB_analysis_results = IB_analysis_phyid(simulation)

    if calc_null:
        with open(f'simulations_0.53/results/null/{analysis}/cinter_{cinter}_phase_noise{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}_it_{it}.pkl', 'wb') as file:
            pickle.dump(IB_analysis_results, file)
        return

    with open(f'simulations_0.53/results/{analysis}/cinter_{cinter}_phase_noise{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}_it_{it}.pkl', 'wb') as file:
        pickle.dump(IB_analysis_results, file)


def main():

    analysis = "IB_mi" # "IB_standard", "IB_phyid"
    calc_null = False # i.e., whether to shuffle epochs

    combination_values = product(cinter_dict.values(),
                                 phase_noise_dict.values(),
                                 freq_std_dict.values(),
                                 amp_noise_dict.values(),
                                 sensor_noise_dict.values(),
                                 range(0,n_it)
                                )
    params = []

    for cinter, phase_noise, freq_std, amp_noise, sensor_noise, it in combination_values:

        file_name = f'simulations_0.53/results/{"null/" if calc_null else ""}IB_mi/cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}_it_{it}.pkl'
        
        if not os.path.exists(file_name):
            params.append((cinter, phase_noise, freq_std, amp_noise, sensor_noise, it, analysis, calc_null))

    Parallel(n_jobs=25, backend="loky")(delayed(IB_analysis_main)(*p) for p in params)


if __name__ == "__main__":
    main()
