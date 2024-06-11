# HyperSimulations
Simulating EEG hyperscanning data using coupled Kuramoto oscillators following stochastic delay differential dynamics and using connectome data. Benchmarking information-theoretic measures against standard connectivity measures.

## Preparation and Finding Best Biophysical Model of Single-Brain Dynamics 

A. **Brain Criticality Dynamics** (*criticality.py*)
- Plot criticality dynamics of Kuramoto model of simulated source signals. Varying $C_{\text{intra}}$ [0,1] in 50 steps, 20 iterations, and three noise conditions (none, medium, high).

B. **Finding Best Cintra** (*best_cintra.py*, *mahalanobis_distance.py*)
- Calculating the Mahalanobis distance between the MI Gaussian connectivity matrices of various single-brain simulations — $C_{\text{intra}}$ [0.45,7], 25steps — and real resting-state EEG datasets (Gifford, Pérez).


## Simulating Brain-to-Brain Interaction and Evaluating Various Connectivity Measures

1. **Simulating source neural dynamics** (*simulations.py*)
- Calculate the phases of each oscillator (N=180) in a large Kuramoto model which follows stochastic delayed differential dynamics.
- Convert all the phases into simulated EEG data using forward gain model (N=64).
- Plotting extensive parameter space, varying inter-brain connectivity ($C_{\text{inter}}$), biological noise (phase_noise, freq_std), and external/recording noise (amp_noise, sensor_noise).

2. **Calculating Connectivity Measures** (*IB_analysis.py*)

   a. Calculate Standard Connectivity Measures: PLV, PLI, wPLI, CCorr, COH, iCOH, envCorr, and powCorr
   
   b. Calculate Mutual Information: Histogram, Box Kernel, Gaussian, KSG, and Symbolic Estimators 
   
   c. Calculate Integrated Information Decomposition Measures: Time-Delayed Mutual Information, Transfer Entropy, Pure Information Transfer, Redundancy, and Synergy 
   

3. **Statistical Analysis** (*statistical_analysis.R*)
