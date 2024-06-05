# HyperSimulations
Simulating EEG hyperscanning data using coupled Kuramoto oscillators following stochastic delay differential dynamics and using connectome data. Benchmarking information-theoretic measures against standard connectivity measures.

## Preparation and Finding Best Biophysical Model of Single-Brain Dynamics 

A. **Brain Criticality Dynamics** (*order_param_cintra_criticality.py*)
- Plot criticality dynamics of Kuramoto model of simulated source signals. Varying $C_{\text{intra}}$ [0,1] in 50 steps, 20 iterations, and three noise conditions (none, medium, high).

B. **Finding Best Cintra** (*best_cintra.py*, *mahalanobis_distance.py*)
- Calculating the Mahalanobis distance between the MI Gaussian connectivity matrices of various single-brain simulations — $C_{\text{intra}}$ [0.45,7], 25steps — and real resting-state EEG datasets (Gifford, Pérez).


## Simulating Brain-to-Brain Interaction and Evaluating Various Connectivity Measures

1. **Simulating source neural dynamics** (*calc_phases.py*)
- Calculate the phases of each oscillator (N=180) in a large Kuramoto model which follows stochastic delayed differential dynamics. 
- Specifying connectivity ($C_{\text{intra}}$, $C_{\text{inter}}$) and biological noise (phase_noise, freq_std).

2. **Simulating EEG signals** (*eeg_simulations.py*)
- Convert all the phases into simulated EEG data using forward gain model (N=64).
- Specifying amp_noise and sensor_noise.

3. **Calculating Connectivity Measures** 

   a. Calculate Standard Connectivity Measures: PLV, PLI, wPLI, CCorr, COH, iCOH, envCorr, powCorr (*IB_standard_analysis.py*)
   
   b. Calculate Mutual Information with Histogram, Box Kernel, Gaussian, KSG, and Symbolic Estimators (*IB_infotheory_analysis.py*)
   
   c. Calculate Integrated Information Decomposition Measures: Time-Delayed Mutual Information, Transfer Entropy, Pure Information Transfer, Redundancy, Synergy (*IB_phyid_analysis.py*)
   
   d. Calculate State-Space Transfer Entropy/Granger Causality (*state_space_TE.m*)

5. **Statistical Analysis** (*analysis.Rmd*)
