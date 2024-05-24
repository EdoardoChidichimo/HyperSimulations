import os
import numpy as np
from config import *
import pickle
import mne
import time as ti
import warnings
from joblib import Parallel, delayed
from itertools import product

warnings.filterwarnings("ignore", category=RuntimeWarning, module='mne', message='.*filter length.*')

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


def simulations_complete(cintra: float,
                         cinter: float,
                         phase_noise: float, 
                         freq_std: float, 
                         amp_noise: float,
                         sensor_noise: float):

    phase_directory_name = f'phases/cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}'
    full_configuration = f'cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}'

    for it in range(n_it):

        IB_file = os.path.join('results/sim', full_configuration, f'it_{it}.pickle')

        if os.path.exists(IB_file):
            print(f"Skipping computation for existing file: {full_configuration, f'it_{it}'}")
            continue

        phase_file_path = os.path.join(phase_directory_name, f'it_{it}.pickle')

        if not os.path.exists(phase_file_path):
            print(f'Phase calculation still needed for {phase_directory_name}, Iteration {it}', flush=True)
            continue

        with open(phase_file_path, 'rb') as file:
            phases = np.array(pickle.load(file))[-int(120*sfreq):,:] # take last two minutes (120seconds) 
        # (60_000, 180)

        print(f'Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std}, Amp Noise {amp_noise}, Sensor Noise {sensor_noise} — STARTING', flush=True)
        
        t0 = ti.time()

        np.random.seed(it)
        phi = np.sin(phases) + amp_noise * np.random.randn(*phases.shape)
        # (60_000, 180 osc)

        eeg = virtual_dyad_scalp(phi=phi, sensor_noise=sensor_noise, seed=it)
        # (60_000, 64 chan)

        simulation = np.array([epoch_data(eeg[:,:32]), epoch_data(eeg[:,32:])]) * 10e-6
        # (2 pp, 60 epochs, 32 channels, 1000 timepoints)

        sim_config = os.path.join('results/sim', full_configuration)
        if not os.path.exists(sim_config):
            os.makedirs(sim_config)

        with open(os.path.join(sim_config, f'it_{it}.pickle'), 'wb') as file:
            pickle.dump(simulation, file)

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

        # directory_name = 'results/sim/cintra_{}_cinter_{}_phase_noise_{}_freq_std_{}_amp_noise_{}_sensor_noise_{}'.format(cintra, cinter, phase_noise, freq_std, amp_noise, sensor_noise)
        directory_name = 'results/sim/cintra_{}_cinter_{}_phase_noise_{}_freq_std_{}_amp_noise_{}_sensor_noise_{}'.format(cintra, cinter, noise*0.1, noise, noise, noise)

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            print(f"Directory created: {directory_name}", flush=True)
        else:
            if len(os.listdir(directory_name)) < n_it:
                print(f"Processing needed for: {directory_name}", flush=True)
            else:
                print(f"Already processed: {directory_name}", flush=True)
                continue
        
        # params.append((cintra, cinter, phase_noise, freq_std, amp_noise, sensor_noise))
        params.append((cintra, cinter, noise*0.1, noise, noise, noise))

    Parallel(n_jobs=14, backend="loky")(delayed(simulations_complete)(*p) for p in params)


if __name__ == '__main__':
    main()
