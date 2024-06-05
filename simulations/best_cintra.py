import os
import numpy as np
from config import *
import pickle
import time as ti
from numpy import pi, random, max
os.environ["CC"] = "gcc"
from jitcdde import jitcdde_input, y, t, input
from symengine import sin
from chspy import CubicHermiteSpline
from itertools import product
from joblib import Parallel, delayed
from hyperit import HyperIT


def epoch_data(data: np.ndarray, epoch_length: int) -> np.ndarray:

    n_channels, n_tot_samples = data.shape
    n_epo = int(np.floor((n_tot_samples)/(sfreq*epoch_length)))
    n_samples = int(sfreq * epoch_length)

    epoched_data = np.zeros((n_epo, n_channels, n_samples))

    for ch in range(n_channels):
        for ep in range(n_epo):
            epoched_data[ep, ch, :] = data[ch, ep*n_samples:(ep+1)*n_samples]
    
    return epoched_data

def calc_gauss_mi(simulation: np.ndarray) -> np.ndarray:

    HyperIT.setup_JVM()
    itClass = HyperIT(simulation[:, :16, :], simulation[:, 16:, :], verbose=False, show_tqdm=False)
    gauss_matrix = itClass.compute_mi(mode='gaussian', include_intra=True, epoch_average=True)

    return gauss_matrix
    

def simulate_brain(cintra: float, 
                phase_noise: float,
                freq_std: float,
                amp_noise: float,
                sensor_noise: float):

    n_intra = 90
    A_intra = np.array(dti * cintra)
    ω_intra = ((np.random.randn(1, n_intra) * freq_std_factor * freq_std + freq_mean) * (2 * pi) / (sfreq)).flatten()
    τ_intra = sfreq * w_d / velocity


    for it in range(n_it):
        np.random.seed(it)

        # Our model of single-brain dynamics using Kuramoto model following stochastic delay differential equation
        def kuramotos():
            for i in range(n_intra):
                yield ω_intra[i] + (sum(
                    A_intra[j, i] * sin(y(j, t - τ_intra[i, j]) - y(i))
                    for j in range(n_intra) if A_intra[j,i] != 0
                ) + phase_noise * input(i))
        
        input_data = np.random.normal(size=(len(times), n_intra))
        input_spline = CubicHermiteSpline.from_data(times, input_data)
        DDE = jitcdde_input(kuramotos, n=n_intra, input=input_spline, verbose=False)
        DDE.compile_C(simplify=False, do_cse=False, chunk_size=45)
        DDE.set_integration_parameters(rtol=0, atol=1e-6)
        DDE.constant_past(random.uniform(0, 2 * pi, n_intra), time=0.0)
        DDE.integrate_blindly(max(τ_intra), 1)

        # Integrate right-side of equation to get instantaneous phase angle over time
        output = []
        for time in np.arange(DDE.t, n_times-DDE.t, 1/sfreq):
            output.append([*DDE.integrate(time) % (2 * pi)])
        phases = np.array(output)[-int(120*sfreq):,:] # Take last 120 seconds (same as real data)
        phases = phases.T

        with open(f'best_cintra/phases/cintra_{cintra}_noise_{freq_std}_it_{it}.pkl', 'wb') as file:
            pickle.dump(phases, file) 

        # Convert phases to timeseries signal
        phi = np.sin(2*pi*phases) + amp_noise * np.random.randn(*phases.shape)

        # Forward modelling
        s1_eeg = np.matmul(phi.T, eeg_mat) / gain.shape[1]
        eeg = s1_eeg + sensor_noise * np.random.randn(*s1_eeg.shape) 

        # Epoch the data in 2 second windows
        simulation = np.array(epoch_data(eeg.T, 2)) * 10e-6

        with open(f'best_cintra/sim/cintra_{cintra}_noise_{freq_std}_it_{it}.pkl', 'wb') as file:
            pickle.dump(simulation, file)

        # Calculate the Gaussian MI Matrix
        gauss_matrix = calc_gauss_mi(simulation) # shape (32,32)

        with open(f'best_cintra/mi_results/cintra_{cintra}_noise_{noise}_it_{it}.pkl', 'wb') as file:
            pickle.dump(gauss_matrix, file)

def main():

    cintra_range = np.round(np.linspace(0.45, 0.70, 26) * scale_factor, 3)
    
    # 26 x 3 (none, mid, high for phase_noise, freq_std, amp_noise, and sensor_noise) = 78 simulations
    combination = product(cintra_range, [0.0, 0.5, 1.0]) 
    params = []

    for cintra, noise in combination:
        params.append((cintra, noise*.1, noise, noise, noise))

    Parallel(n_jobs=30, backend="loky")(delayed(simulate_brain)(*p) for p in params)

    
if __name__ == "__main__":
    main()
