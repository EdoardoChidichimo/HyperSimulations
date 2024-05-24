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


cintra_range = np.linspace(0, 1, 51) * scale_factor
iterations = 100
n_intra = 90


def compute_order_param_range(cintra: float, 
                              phase_noise: float,
                              freq_std: float,
                              file_name: str):
    
    A_intra = np.array(dti * cintra)
    ω_intra = ((np.random.randn(1, n_intra) * freq_std_factor * freq_std + freq_mean) * (2 * pi) / (sfreq)).flatten()
    τ_intra = sfreq * w_d / velocity

    results = []

    t0 = ti.time()

    for it in range(iterations):

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
        DDE.compile_C(simplify=False, do_cse=False, chunk_size=1)
        DDE.set_integration_parameters(rtol=0, atol=1e-5)
        DDE.constant_past(random.uniform(0, 2 * pi, n_intra), time=0.0)
        DDE.integrate_blindly(max(τ_intra), 1)

        output = []
        for time in np.arange(DDE.t, n_times-DDE.t, 1/sfreq):
            output.append([*DDE.integrate(time) % (2 * pi)])
        phases = np.array(output)

        R = np.mean(np.abs(np.mean(np.exp(1j * phases), axis=1))) # average R over time. (Remember that it first has to be averaged across all oscillators as its a measure of synchrony)
        results.append(R)

        print(f'Iteration {it+1}/100 for {file_name}', flush=True)

    overall_R_mean = np.mean(results)
    overall_R_std = np.std(results)

    final = np.array((overall_R_mean, overall_R_std))

    with open(file_name, 'wb') as file:
        pickle.dump(final, file)

    print(f'Complete in {ti.time()-t0} seconds: {file_name} — MEAN: {final[0]}, STD: {final[1]}', flush=True)



def main():

    combination = product(cintra_range, [0.0, 0.5, 1.0]) # 50 x 3 (none, mid, high for both phase_noise and freq_std) = 150 combinations
    params = []

    for cintra, noise in combination:
            
        file_name = f'order_params/cintra_{cintra}_phase_noise_{noise*.1}_freq_std_{noise}.pickle'

        if os.path.exists(file_name):
            print(f'Already complete, skipping: {file_name}', flush=True)
            continue

        print(f'Needs computing: {file_name}', flush=True)
        params.append((cintra, noise*.1, noise, file_name))
        # params = params[::-1]

    Parallel(n_jobs=25, backend="loky")(delayed(compute_order_param_range)(*p) for p in params)

    
if __name__ == "__main__":
    main()
