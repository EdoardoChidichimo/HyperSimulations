import os
import numpy as np
from config import *
import pickle
import mne
from hypyp import analyses
import time as ti
import warnings
from itertools import product
from joblib import Parallel, delayed


warnings.filterwarnings("ignore", category=RuntimeWarning, module='mne', message='.*filter length.*')
mne.set_log_level(verbose=False)


def IB_analysis_standard(data_inter: np.ndarray):

    IB_analysis_results = []
    
    for mode in modes:
        connectivity = analyses.pair_connectivity(data=data_inter,
                                                sampling_rate=sfreq,
                                                frequencies=freq_bands,
                                                mode=mode)
        IB_analysis_results.append(connectivity)

    return IB_analysis_results


def IB_standard_analysis_complete(cintra: float,
                                cinter: float,
                                phase_noise: float, 
                                freq_std: float, 
                                amp_noise: float,
                                sensor_noise: float):
    
    for it in range(n_it):
    # for it in range(n_it - 1, -1, -1):

        full_configuration = f'cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}/it_{it}.pickle'

        IB_file = os.path.join('results/IB_standard', full_configuration)

        if os.path.exists(IB_file):
            print(f"Skipping computation for existing file: {full_configuration}")
            continue

        sim_file_path = os.path.join(f'results/sim/', full_configuration)
        if not os.path.exists(sim_file_path):
            print(f'Simulation still needed for {full_configuration}', flush=True)
            continue

        with open(sim_file_path, 'rb') as file:
            simulation = np.array(pickle.load(file)) 

        print(f'Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std}, Amp Noise {amp_noise}, Sensor Noise {sensor_noise} — STARTING', flush=True)
        
        t0 = ti.time()

        np.random.seed(it)
        IB_analysis_results = IB_analysis_standard(simulation) # (8 modes, 5 freqbands, 64, 64)

        with open(IB_file, 'wb') as file:
            pickle.dump(IB_analysis_results, file)

        print(f'Complete in {ti.time()-t0:.1f} seconds: Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std}, Amp Noise {amp_noise}, Sensor Noise {sensor_noise}', flush=True)


def main():
    # combination_values = product(cintra_dict.values(),
    #                             cinter_dict.values(),
    #                             phase_noise_dict.values(),
    #                             freq_std_dict.values(),
    #                             amp_noise_dict.values(),
    #                             sensor_noise_dict.values()
    #                             )

    combination_values = product(cintra_dict.values(),
                                cinter_dict.values(),
                                [0.0,0.5,1.0])
    n_it = 20
    params = []

    # for cintra, cinter, phase_noise, freq_std, amp_noise, sensor_noise in combination_values:
    for cintra, cinter, noise in combination_values:

        directory_name = 'results/IB_standard/cintra_{}_cinter_{}_phase_noise_{}_freq_std_{}_amp_noise_{}_sensor_noise_{}'.format(cintra, cinter, noise*0.1, noise, noise, noise)

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            print(f"Directory created: {directory_name}", flush=True)
        else:
            if len(os.listdir(directory_name)) < n_it:
                print(f"Processing needed for: {directory_name}", flush=True)
            else:
                print(f"Already processed: {directory_name}", flush=True)
                continue
        
        params.append((cintra, cinter, noise*0.1, noise, noise, noise))

    Parallel(n_jobs=4, backend="loky")(delayed(IB_standard_analysis_complete)(*p) for p in params)

if __name__ == '__main__':

    main()
