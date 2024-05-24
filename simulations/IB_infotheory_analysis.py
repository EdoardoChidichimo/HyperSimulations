import os
import numpy as np
from config import *
import pickle
import mne
from hyperit import HyperIT
import time as ti
from itertools import product
from joblib import Parallel, delayed

mne.set_log_level(verbose=False)


def IB_analysis_IT(data_inter: np.ndarray, full_configuration: str):

    IB_analysis_results_IT = []
    data1, data2 = data_inter

    HyperIT.setup_JVM()
    it = HyperIT(data1, data2, sfreq=sfreq, freq_bands=freq_bands, show_tqdm=False)

    mi_modes = ['histogram', 'gaussian', 'kernel', 'ksg', 'symbolic']
    for mode in mi_modes:
        IB_analysis_results_IT.append(np.mean(it.compute_mi(estimator=mode, include_intra=True), axis=1))
        print(f'Finished {mode} for {full_configuration}', flush=True)

    return IB_analysis_results_IT


def IB_infotheory_analysis_complete(cintra: float,
                         cinter: float,
                         phase_noise: float, 
                         freq_std: float, 
                         amp_noise: float,
                         sensor_noise: float):

    full_configuration = f'cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}'

    for it in range(n_it):

        if os.path.exists(f'results/IB_infotheory/{full_configuration}/it_{it}.pickle'):
            print(f'Already computed {full_configuration}, iteration {it}', flush=True)
            continue
        
        sim_config = os.path.join('results/sim', full_configuration)

        if not os.path.exists(os.path.join(sim_config, f'it_{it}.pickle')):
            continue

        with open(os.path.join(sim_config, f'it_{it}.pickle'), 'rb') as file:
            simulation = np.array(pickle.load(file))

        print(f'Computing Iteration {it} — {full_configuration}', flush=True)
        t0 = ti.time()

        IB_analysis_results = IB_analysis_IT(simulation, full_configuration)

        IBS_config = os.path.join('results/IB_infotheory', full_configuration)
        if not os.path.exists(IBS_config):
            os.makedirs(IBS_config)

        with open(os.path.join(IBS_config, f'it_{it}.pickle'), 'wb') as file:
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

        directory_name = 'results/IB_infotheory/cintra_{}_cinter_{}_phase_noise_{}_freq_std_{}_amp_noise_{}_sensor_noise_{}'.format(cintra, cinter, noise*0.1, noise, noise, noise)

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

    Parallel(n_jobs=15, backend="loky")(delayed(IB_infotheory_analysis_complete)(*p) for p in params)



if __name__ == "__main__":
    main()
