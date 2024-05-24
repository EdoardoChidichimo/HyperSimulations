import os
import numpy as np
from config import *
import pickle
from hyperit import HyperIT
import time as ti
from itertools import product
from joblib import Parallel, delayed


def IB_phyid_analysis(data_inter: np.ndarray):

    IB_analysis_results_phyid = []
    data1, data2 = data_inter

    HyperIT.setup_JVM()
    it = HyperIT(data1, data2, sfreq=sfreq, freq_bands=freq_bands, standardise_data=True, show_tqdm=False)

    atoms = it.compute_atoms(tau=23, redundancy='MMI', include_intra=True)
    results = np.mean(np.array(atoms), axis=0) # average across epochs

    # Time-delayed mutual information
    tdmi = np.sum(results, axis=-1)

    # TE
    te_atoms = [4,6,12,14]
    te = np.sum(results[..., te_atoms], axis=-1)

    # Pure information transfer
    pure = results[..., 6]

    # Redundancy atom
    redundancy = results[..., 0]

    # Synergy atom
    synergy = results[..., 15]

    IB_analysis_results_phyid.append(tdmi)
    IB_analysis_results_phyid.append(te)
    IB_analysis_results_phyid.append(pure)
    IB_analysis_results_phyid.append(redundancy)
    IB_analysis_results_phyid.append(synergy)

    return IB_analysis_results_phyid



def IB_phyid_analysis_complete(cintra: float,
                         cinter: float,
                         phase_noise: float, 
                         freq_std: float, 
                         amp_noise: float,
                         sensor_noise: float):

    full_configuration = f'cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}'

    for it in range(n_it):

        if os.path.exists(f'results/IB_phyid/{full_configuration}/it_{it}.pickle'):
            print(f'Already computed {full_configuration}, iteration {it}', flush=True)
            continue
        
        sim_config = os.path.join('results/sim', full_configuration)

        if not os.path.exists(os.path.join(sim_config, f'it_{it}.pickle')):
            print(f'Simulation still needs computing: {sim_config}, Iteration {it}', flush=True)
            continue

        with open(os.path.join(sim_config, f'it_{it}.pickle'), 'rb') as file:
            simulation = np.array(pickle.load(file))

        print(f'Computing Iteration {it} — {full_configuration}', flush=True)
        t0 = ti.time()

        IB_analysis_results = IB_phyid_analysis(simulation)

        IBS_config = os.path.join('results/IB_phyid', full_configuration)
        if not os.path.exists(IBS_config):
            os.makedirs(IBS_config)

        with open(os.path.join(IBS_config, f'it_{it}.pickle'), 'wb') as file:
            pickle.dump(IB_analysis_results, file)

        print(f'Complete in {ti.time()-t0:.1f} seconds: Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std}, Amp Noise {amp_noise}, Sensor Noise {sensor_noise}', flush=True)


def main():

    combination_values = product(cintra_dict.values(),
                                cinter_dict.values(),
                                [0.0,0.5,1.0])
    n_it = 20
    params = []

    # for cintra, cinter, phase_noise, freq_std, amp_noise, sensor_noise in combination_values:
    for cintra, cinter, noise in combination_values:

        directory_name = 'results/IB_phyid/cintra_{}_cinter_{}_phase_noise_{}_freq_std_{}_amp_noise_{}_sensor_noise_{}'.format(cintra, cinter, noise*0.1, noise, noise, noise)

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

    Parallel(n_jobs=1, backend="loky")(delayed(IB_phyid_analysis_complete)(*p) for p in params)


if __name__ == "__main__":
    main()
