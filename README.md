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
This project is developed by Abtin Shahkarami, PhD candidate at Institute Polytechnique de Paris, Telecom Paris, with the contribution of Prof. Mansoor Yousefi and Prof. Yves Jaouen, under GNU GPLv3 license.

# Citing
Please cite the following papers if you use or modify this library:

- A. Shahkarami, M. I. Yousefi, and Y. Jaouen, "Efficient Deep Learning of Nonlinear Fiber-Optic Communications Using a Convolutional Recurrent Neural Network," in *IEEE International Conference on Machine Learning and Applications (ICMLA)*, 2021, pp. 668-673, DOI: 10.1109/ICMLA52953.2021.00112.
- A. Shahkarami, M. Yousefi, and Yves Jaouen, "Attention-Based Neural Network Equalization in Fiber-Optic Communications." in *Asia Communications and Photonics Conference (ACP)*, Optical Society of America, 2021, p. M5H.3, ISBN: 978-1-957171-00-5


# Acknowledment 

This project has received funding from the European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie grant agreement No 766115.


