import os
import numpy as np
from config import *
import pickle
import mne
from hypyp import analyses
# from hyperit import HyperIT

# from hyperit import HyperIT

import time as ti

mne.set_log_level(verbose=False)


def virtual_dyad_scalp(phi: np.ndarray, sensor_noise: float, seed: int) -> np.ndarray:

    s1_source, s2_source = phi[:, :n_osc_intra], phi[:, n_osc_intra:]

    s1_eeg = np.matmul(s1_source, eeg_mat[0:n_osc_intra, :])  
    s2_eeg = np.matmul(s2_source, eeg_mat[0:n_osc_intra, :])  

    hyper_eeg = np.concatenate((s1_eeg, s2_eeg), axis=1)  
    hyper_eeg /= gain.shape[1]

    eeg = hyper_eeg + sensor_noise * np.random.randn(*hyper_eeg.shape) 

    return eeg


def epoch_data(data: np.ndarray):

    epoch_length = 2 
    n_samples_per_epoch = int(sfreq * epoch_length)
    n_epochs = data.shape[0] // n_samples_per_epoch
    n_samples_used = n_epochs * n_samples_per_epoch

    epochs = data[:n_samples_used].reshape(n_epochs, data.shape[1], n_samples_per_epoch)
    
    return epochs


def IB_analysis_standard(data_inter: np.ndarray):

    IB_analysis_results = []
    for mode in modes:
        connectivity = analyses.pair_connectivity(data=data_inter,
                                                  sampling_rate=sfreq,
                                                  frequencies=freq_bands,
                                                  mode=mode)
        IB_analysis_results.append(connectivity)

    return IB_analysis_results


# def IB_analysis_IT(data_inter: np.ndarray):

#     IB_analysis_results_IT = []
#     data1, data2 = data_inter

#     HyperIT.setup_JVM()
#     it = HyperIT(data1, data2, sfreq=sfreq, freq_bands=freq_bands)

#     # mi_modes = ['histogram', 'gaussian', 'ksg', 'kernel', 'symbolic']
#     mi_modes = ['histogram', 'gaussian']
#     for mode in mi_modes:
#         IB_analysis_results_IT.append(np.mean(it.compute_mi(estimator=mode, include_intra=True), axis=1))

#     return IB_analysis_results_IT



def simulations_complete(cintra: float,
                         cinter: float,
                         phase_noise: float, 
                         freq_std: float, 
                         amp_noise: float,
                         sensor_noise: float,
                         n_it: int):

    phase_configuration = f'cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}'
    full_configuration = f'cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}'
    
    directory_name = os.path.join('phases', phase_configuration)

    for it in range(n_it):

        file_name = f'cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_it{it}.pickle'
        file_path = os.path.join(directory_name, file_name)

        with open(file_path, 'rb') as file:
            phases = np.array(pickle.load(file))

        print(f'phases: {phases.shape}', flush=True)
        
        t0 = ti.time()

        np.random.seed(it)
        phi = np.sin(phases) + amp_noise * np.random.randn(*phases.shape)

        eeg = virtual_dyad_scalp(phi=phi, sensor_noise=sensor_noise, seed=it)
        print(f'eeg: {eeg.shape}', flush=True)
        simulation = np.array([epoch_data(eeg[:,:32]), epoch_data(eeg[:,32:])]) * 10e-6
        print(f'sim: {simulation.shape}', flush=True)
        IB_analysis_results = IB_analysis_standard(simulation)
        print(f'IB: {IB_analysis_results.shape}', flush=True)



        sim_config = os.path.join('results/sim', full_configuration)
        if not os.path.exists(sim_config):
            os.makedirs(sim_config)

        with open(os.path.join(sim_config, f'it_{it}.pickle'), 'wb') as file:
            pickle.dump(simulation, file)


        IBS_config = os.path.join('results/IB_standard', full_configuration)
        if not os.path.exists(IBS_config):
            os.makedirs(IBS_config)

        with open(os.path.join(IBS_config, f'it_{it}.pickle'), 'wb') as file:
            pickle.dump(IB_analysis_results, file)

        print(f'Complete in {ti.time()-t0:.1f} seconds: Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std}, Amp Noise {amp_noise}, Sensor Noise {sensor_noise}', flush=True)


if __name__ == "__main__":

    cintra = 0.0
    cinter = 0.0
    phase_noise = 0.0
    freq_std = 0.0
    amp_noise = 0.0
    sensor_noise = 0.0
    n_it = 20

    simulations_complete(cintra, cinter, phase_noise, freq_std, amp_noise, sensor_noise, n_it)
