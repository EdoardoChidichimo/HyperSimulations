#!/usr/bin/env python
# coding=utf-8
# ====================================================================================================
# title           : simulations.py
# description     : Simulate EEG-Hyperscanning Data using Kuramoto model with Forward Model + Noise!
# author          : Edoardo Chidichimo, Guillaume Dumas
# date            : 2024-11-01
# version         : 0.1
# usage           : python simulations.py
# python_version  : >3.10
# ====================================================================================================

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
from itertools import product
from joblib import Parallel, delayed

mne.set_log_level(verbose=False)

def virtual_dyad_scalp(phi: np.ndarray, sensor_noise: float) -> np.ndarray:

    s1_source, s2_source = phi[:n_osc_intra, :], phi[n_osc_intra:, :] # s1_source and s2_source have shape (90, 60_000)

    s1_eeg = np.matmul(s1_source.T, eeg_mat)  # now with shape (60_000, 32)
    s2_eeg = np.matmul(s2_source.T, eeg_mat)  

    hyper_eeg = np.concatenate((s1_eeg, s2_eeg), axis=1)  # (60_000, 64)
    hyper_eeg /= gain.shape[1]

    eeg = hyper_eeg + sensor_noise * np.random.randn(*hyper_eeg.shape) 

    return eeg

def epoch_data(data: np.ndarray) -> np.ndarray:

    epoch_length = 2
    n_channels, n_tot_samples = data.shape
    n_epo = int(np.floor((n_tot_samples)/(sfreq*epoch_length)))
    n_samples = int(sfreq * epoch_length)

    epoched_data = np.zeros((n_epo, n_channels, n_samples))

    for ch in range(n_channels):
        for ep in range(n_epo):
            epoched_data[ep, ch, :] = data[ch, ep*n_samples:(ep+1)*n_samples]
    
    return epoched_data


def simulate(cintra: float,
             cinter: float,
             phase_noise: float, 
             freq_std: float) -> None:

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
        
        np.random.seed(it)
        file_path = f'simulations_0.53/phases/cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}/it_{it}.pickle'

        if os.path.exists(file_path):
            continue

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
        
        output = []
        for time in np.arange(DDE.t, n_times - DDE.t, 1/sfreq):
            output.append([*DDE.integrate(time) % (2 * pi)])
        phases = np.array(output).T # shape (180, 104250)

        with open(file_path, 'wb') as file:
            pickle.dump(phases, file) 

        # Take only stable dynamics which are often the last 2 minutes of data
        phases = np.array(phases)[:,-int(120*sfreq):] # shape (180 osc, 60_000) 

        for amp_noise, sensor_noise in product(amp_noise_dict.values(), sensor_noise_dict.values()):

            phi = np.sin(2 * np.pi * phases) + amp_noise * np.random.randn(*phases.shape)
            # (180 osc, 60_000)
    
            hyper_eeg = virtual_dyad_scalp(phi=phi, sensor_noise=sensor_noise)
            # (60_000, 64 chan)
    
            simulation = np.array([epoch_data(hyper_eeg[:,:32].T), epoch_data(hyper_eeg[:,32:].T)]) * 10e-6
            # (2 pp, 60 epochs, 32 channels, 1000 timepoints)
    
            sim_config_directory = f'simulations_0.53/sim/cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{freq_std}_amp_noise_{amp_noise}_sensor_noise_{sensor_noise}'
            if not os.path.exists(sim_config_directory):
                os.mkdir(sim_config_directory)
    
            with open(os.path.join(sim_config_directory, f'it_{it}.pkl'), 'wb') as file:
                pickle.dump(simulation, file)


def main():

    combination_values = product(cinter_dict.values(),
                                 phase_noise_dict.values(),
                                 sensor_noise_dict.values())
    cintra = 0.53 * scale_factor
    
    params = []

    for cinter, phase_noise, sensor_noise in combination_values:

        directory_name = f'simulations_0.53/phases/cinter_{cinter}_phase_noise_{phase_noise}_freq_std_{sensor_noise}'

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        if len(os.listdir(directory_name)) == n_it:
            continue

        params.append((cintra, cinter, phase_noise, sensor_noise))

    Parallel(n_jobs=20, backend="loky")(delayed(simulate)(*p) for p in params)

if __name__ == '__main__':
    main()
