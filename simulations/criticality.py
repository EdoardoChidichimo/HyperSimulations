#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : criticality.py
# description     : Plot criticality dynamics for single-brain model (stochastic delay Kuramoto model)
# author          : Edoardo Chidichimo
# date            : 2024-06-01
# version         : 0.1
# usage           : python criticality.py
# python_version  : >3.10
# ==============================================================================


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


def compute_order_param_range(cintra: float, 
                              phase_noise: float,
                              freq_std: float):

    file_name = f'criticality/cintra_{cintra}_phase_noise_{phase_noise}_freq_std_{freq_std}'
    
    n_intra = 90
    A_intra = np.array(dti * cintra)
    ω_intra = ((np.random.randn(1, n_intra) * freq_std_factor * freq_std + freq_mean) * (2 * pi) / (sfreq)).flatten()
    τ_intra = sfreq * w_d / velocity

    results = []

    t0 = ti.time()

    for it in range(n_it):

        np.random.seed(it)

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
        DDE.set_integration_parameters(rtol=0, atol=1e-5)
        DDE.constant_past(random.uniform(0, 2 * pi, n_intra), time=0.0)
        DDE.integrate_blindly(max(τ_intra), 1)

        output = []
        for time in np.arange(DDE.t, n_times-DDE.t, 1/sfreq):
            output.append([*DDE.integrate(time) % (2 * pi)])
        phases = np.array(output)

        R = np.mean(np.abs(np.mean(np.exp(1j * phases), axis=1))) # average R over time. (Remember that it first has to be averaged across all oscillators as its a measure of synchrony)
        results.append(R)

    overall_R_mean = np.mean(results)
    overall_R_std = np.std(results)

    final = np.array((overall_R_mean, overall_R_std))

    with open(file_name, 'wb') as file:
        pickle.dump(final, file)


def main():

    cintra_range = np.linspace(0, 1, 51) * scale_factor
  
    # 51 x 3 (none, mid, high for both phase_noise and freq_std) = 153 combinations
    combination = product(cintra_range, [0.0, 0.5, 1.0]) 
    params = []

    for cintra, noise in combination:
        params.append((cintra, noise*.1, noise))

    Parallel(n_jobs=28, backend="loky")(delayed(compute_order_param_range)(*p) for p in params)

    
if __name__ == "__main__":
    main()
