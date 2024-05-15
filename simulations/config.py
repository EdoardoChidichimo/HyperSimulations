import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

basefolder = ""
scale_factor = 0.1

# set number of oscillators to simulate
n = 180 # Inter-brain oscillators
n_osc_intra = int(n / 2) # 90


# Constants
freq_mean = 40.0 # Gamma oscillations
freq_std_factor = 8.0
velocity = 1.65 # (Dumas e2012) or 6? (Axonal velocity)
sfreq = 500.0 # sampling frequency
n_times = 600.0  # total seconds
total_samp = int(sfreq * n_times)
times = np.arange(0, (n_times + 2), 1/sfreq) 
n_ch_intra = 32
n_ch_inter = 2 * n_ch_intra


# Define connectivity and distance matrices
dti = loadmat("biosim_data/connectomes.mat")['cuban']
dti = np.array((dti - np.min(dti)) / (np.max(dti) - np.min(dti))) # Normalising 0 <= dti <= 1
distance = np.array(loadmat("biosim_data/distance.mat")['distance'])
w_d = np.block([[distance, np.zeros((n_osc_intra, n_osc_intra))],
                [np.zeros((n_osc_intra, n_osc_intra)), distance]])
τ = sfreq * w_d / velocity # Mean Delay = 22.82, Max Delay = 91.50


passage_mat = loadmat("biosim_data/FinePassageTZ.mat")['passageFine'].T # 90, 15028
head_model = loadmat("biosim_data/Passage.mat")['HeadModel'] 
gain = np.array(head_model['Gain'][0][0])  # 32, v
eeg_mat = np.matmul(passage_mat, gain.T) # matrix shaped 90 x 32 to multiply with

# Handling of logging & warnings
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='jitcdde')

# create one dictionary per factor for systematic manipulation, 3 levels per factor

cintra_dict = {'cintra_low': 0.0,
               'cintra_med': 0.5,
               'cintra_hig': 1.0}

cinter_dict = {'cinter_low': 0.0,
               'cinter_med': 1.0,
               'cinter_hig': 2.0}

cinter_extended_dict = {'cinter_low': 0.0,
                        'cinter_med_low': 0.5,
                        'cinter_med': 1.0,
                        'cinter_med_hig': 1.5,
                        'cinter_hig': 2.0}

phase_noise_dict = {'phase_noise_low': 0.00,
                    'phase_noise_med': 0.05,
                    'phase_noise_hig': 0.10}

freq_std_dict = {'freq_std_low': 0.0,
                 'freq_std_med': 0.5,
                 'freq_std_hig': 1.0}

amp_noise_dict = {'amp_noise_low': 0.0,
                  'amp_noise_med': 0.5,
                  'amp_noise_hig': 1.0}               

sensor_noise_dict = {'sensor_noise_low': 0.0,
                     'sensor_noise_med': 0.5,
                     'sensor_noise_hig': 1.0}

modes = ['plv', 'pli', 'wpli', 'ccorr', 'coh', 'imaginary_coh', 'envelope_corr', 'pow_corr']

freq_bands = {'Delta': [1, 4],
              'Theta': [4, 8],
              'Alpha': [8, 12],
              'Beta': [15, 30],
              'Gamma': [32, 48]}

# Define motor and visual areas of connectivity matrix
motor = np.array([57, 58, 59, 60, 61, 62, 67, 68, 69, 70]) - 1 
visual = np.array([43, 44, 45, 46, 49, 50, 51, 52, 53, 54]) - 1
