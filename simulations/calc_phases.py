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

mne.set_log_level(verbose=False)

# To prevent huge redundancy, we only need to run jitcdde and store phases of source simulation for ONLY freq_std, phase_noise, cintra, and cinter values 3^4 (since these are the biological parameters)
# We can later unpickle the file (which stores the phases) and then add external noise (amp_noise, sensor_noise).

def source_simulation(cintra: float,
                      cinter: float,
                      phase_noise: float, 
                      freq_std: float, 
                      n_it: int):

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
    # for it in range(n_it - 1, -1, -1):

        directory_name = os.path.join('phases', f'cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}')
        file_name = f'cintra_{cintra}_cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_it{it}.pickle'
        file_path = os.path.join(directory_name, file_name)

        if os.path.exists(file_path):
            print(f"Skipping computation for existing file: {file_name}")
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
        DDE.compile_C(simplify=False, do_cse=False, chunk_size=180)
        DDE.set_integration_parameters(rtol=0, atol=1e-6)
        DDE.constant_past(random.uniform(0, 2 * pi, n), time=0.0)
        DDE.integrate_blindly(max(τ), 1)
        
        print(f'Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std} — SETUP COMPLETE in {ti.time()-t0:.1f}', flush=True)

        output = []
        for time in np.arange(DDE.t, DDE.t + n_times, 1/sfreq):
            output.append([*DDE.integrate(time) % (2 * pi)])

        phases = np.array(output)

        print(f'Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std} — INTEGRATION COMPLETE', flush=True)

        with open(file_path, 'wb') as file:
            pickle.dump(phases, file)

        print(f'Iteration {it}, Cintra {cintra}, Cinter {cinter}, Phase Noise {phase_noise}, Freq std {freq_std} — COMPLETE in {ti.time()-t0:.1f}', flush=True)
