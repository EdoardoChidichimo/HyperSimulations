from config import *
import pickle
import mne
import numpy as np
from numpy import pi, random, max
import os
os.environ["CC"] = "gcc"
from jitcdde import jitcdde_input, y, t, input
from symengine import sin
from chspy import CubicHermiteSpline
import time as ti
from itertools import product
from joblib import Parallel, delayed

mne.set_log_level(verbose=False)


def source_simulation(cintra: float,
                      cinter: float,
                      phase_noise: float, 
                      freq_std: float):

    intra_matrix = dti * cintra
    inter_matrix = np.zeros((n_osc_intra, n_osc_intra))
    connectivity_inter = dti.mean() * cinter
    for m_idx in motor:
        for v_idx in visual:
            inter_matrix[m_idx, v_idx] += connectivity_inter
            inter_matrix[v_idx, m_idx] += connectivity_inter
    
    A = np.block([[intra_matrix, inter_matrix], [inter_matrix, intra_matrix]])
    ω = ((np.random.randn(1, n) * freq_std_factor * freq_std + freq_mean) * (2 * pi) / (sfreq)).flatten()

    for it in range(n_it): 

        file_path = f'phases/cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}/it_{it}.pickle'

        if os.path.exists(file_path):
            print(f"Skipping computation for existing file: {file_path}")
            continue

        print(f'Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std} — STARTING', flush=True)

        t0 = ti.time()
        np.random.seed(it)

        def kuramotos():
            for i in range(n):
                yield ω[i] + (sum(
                    A[j, i] * sin(y(j, t - τ[i, j]) - y(i))
                    for j in range(n) if A[j,i] != 0
                ) + phase_noise * input(i))
        
        input_data = np.random.normal(size=(len(times), n))
        input_spline = CubicHermiteSpline.from_data(times, input_data)
        DDE = jitcdde_input(kuramotos, n=n, input=input_spline, verbose=False)
        DDE.compile_C(simplify=False, do_cse=False, chunk_size=90)
        DDE.set_integration_parameters(rtol=0, atol=1e-6)
        DDE.constant_past(random.uniform(0, 2 * pi, n), time=0.0)
        DDE.integrate_blindly(max(τ), 1)
        
        print(f'Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std} — SETUP COMPLETE in {ti.time()-t0:.1f}', flush=True)

        output = []
        for time in np.arange(DDE.t, n_times - DDE.t, 1/sfreq):
            output.append([*DDE.integrate(time) % (2 * pi)])

        phases = np.array(output)

        print(f'Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std} — INTEGRATION COMPLETE', flush=True)

        with open(file_path, 'wb') as file:
            pickle.dump(phases, file)

        print(f'Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std} — COMPLETE in {ti.time()-t0:.1f}', flush=True)


def main():

    # combination_values = product(cintra_dict.values(),
    #                             cinter_dict.values(),
    #                             phase_noise_dict.values(),
    #                             freq_std_dict.values()
    #                             )

    combination_values = product(cintra_dict.values(),
                                 cinter_dict.values(),
                                 [0.0, 0.5, 1.0])
    
    n_it = 20 
    params = []
    
    # for cintra, cinter, phase_noise, freq_std in combination_values:
    for cintra, cinter, noise in combination_values:

        directory_name = 'phases/cintra_{}_cinter_{}_phase_noise_{}_freq_std_{}'.format(cintra, cinter, noise*0.1, noise)
        
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            print(f"Directory created: {directory_name}", flush=True)
        else:
            # Check if the directory contains the expected number of files
            if len(os.listdir(directory_name)) < n_it:
                print(f"Processing needed for: {directory_name}", flush=True)
            else:
                print(f"Already processed: {directory_name}", flush=True)
                continue

        params.append((cintra, cinter, noise*0.1, noise))
        params = params[::-1]


    Parallel(n_jobs=7, backend="loky")(delayed(source_simulation)(*p) for p in params)

if __name__ == '__main__':
    main()
