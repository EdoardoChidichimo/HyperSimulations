import os
import itertools
from config import *
from itertools import product
from calc_all_phases import source_simulation
from joblib import Parallel, delayed


if __name__ == '__main__':

    # generate a list with all possible combinations of all factors length 3^4 (81)
    combination_values = itertools.product(cintra_dict.values(),
                                           cinter_dict.values(),
                                           phase_noise_dict.values(),
                                           freq_std_dict.values()
                                          )
    n_it = 20  # iterations per set (p = 0.05)
    params = []
    
    for cintra, cinter, phase_noise, freq_std in combination_values:

        directory_name = 'phases/cintra_{}_cinter_{}_phase_noise_{}_freq_std_{}'.format(cintra, cinter, phase_noise, freq_std)
        
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

        params.append((cintra, cinter, phase_noise, freq_std, n_it))
        # params = params[::-1]

    Parallel(n_jobs=10, backend="loky")(delayed(source_simulation)(*p) for p in params)
