# Fiber-Optic Transmission Systems Modeling

This library models the single-pol and dual-pol fiber-optic channels governed by nonlinear Schrödinger equation (NLSE) and coupled NLSE, accompanied with TX and RX DSP chain implementation.

NLS equation is solved using split-step Fourier method in this library. 

The TX and RX DSP chain module, for single-polarization, includes the following modules:

- Modulation
- RRC Pulse shaping
- Sampling
- Matched filtering
- Chromatic dispersion compensation
- Digital back-propgation
- Carrier phase estimation
- Detection
- Demodulation

For dual-polarization systems, in addition to the modules above, the following modules are implemented:

- Polarization multiplexation and demultiplixation
- Radius-directed-equalization (RDE) -based multiple-input-multiple-output (MIMO) algorithm to compensate for the PMD



***This library is under documentation!***

# Licence and Authors
This project is developed by Abtin Shahkarami, PhD candidate at Institute Polytechnique de Paris, Telecom Paris, with the contribution of Prof. Mansoor Yousefi, Prof. Yves Jaouen, and Jamal Darweesh, under GNU GPLv3 license.


# Acknowledment 

This project has received funding from the European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie grant agreement No 766115.


