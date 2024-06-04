# HyperSimulations
Simulating EEG hyperscanning data using coupled Kuramoto oscillators following stochastic delay differential dynamics and using connectome data. Benchmarking information-theoretic measures against standard connectivity measures.

## Preparation and Finding Best Biophysical Model of Single-Brain Dynamics 

A. **Brain Criticality Dynamics** (*order_param_cintra_criticality.py*)
Plot criticality dynamics of Kuramoto model of simulated source signals. Varying cintra 0:1 in 50 steps, 100 iterations, and three noise conditions (none, medium, high).

B. **Finding Best Cintra** (*best_cintra.py*, *mahalanobis_distance.py*)
Calculating the Mahalanobis distance between the MI Gaussian connectivity matrices of various single-brain simulations (Cintra: 0.45–7, 25steps) and a real resting-state EEG dataset.


## Simulating Brain-to-Brain Interaction and Evaluating Various Connectivity Measures

1. **Simulating source neural dynamics** (*calc_phases.py*)
Calculate the phases of each oscillator (N=180) in a large Kuramoto model which follows stochastic delayed differential dynamics. 
Specifying connectivity (cintra, cinter) and biological noise (phase_noise, freq_std).

2. **Simulating EEG signals** (*eeg_simulations.py*)
Convert all the phases into simulated EEG data using forward gain model (N=64).
Specifying amp_noise and sensor_noise.

3. **Calculating Connectivity Measures** 

   a. Calculate standard connectivity measures: PLV, PLI, wPLI, CCorr, COH, iCOH, envCorr, powCorr (*IB_standard_analysis.py*)
   b. Calculate information-theoretic measures: Mutual Information with Histogram, Box Kernel, Gaussian, KSG, and Symbolic Estimators (*IB_infotheory_analysis.py*)
   c. Calculate integrated-information-decomposition: Time-Delayed Mutual Information, Transfer Entropy, Pure Information Transfer, Redundancy, Synergy (*IB_phyid_analysis.py*)
   d. Calculate state-space transfer entropy: Transfer Entropy using State-Space Modelling (*state_space_TE.m*)

4. **Analysis** (*analysis.Rmd*)
