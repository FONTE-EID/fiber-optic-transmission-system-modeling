import numpy as np
import sys
import constellations_set as cons
from numpy import exp, sinc, mean, abs, pi, inf, sum, arange, linspace, sqrt, log2, log, log10


Ns = 2**14           # no. of symbols, in your commentary, basis_count
Rs = 64e9;

# T = 2 * (1/Rs) * Ns  # R is baud rate

SpS_fw = 8; #number of samples/symbol in forward propgation
SpS_rx = 2;

# Ns = 2**7              
T = (1/Rs) * Ns
# dt = (1/Rs)/SpS_fw        
# Nfft = T/dt
# t = arange(-T/2, T/2, dt)

Nfft = Ns * SpS_fw   # len of signal. The factor 2 is due to the factor 2 in T

dt = T/Nfft #
t = arange(-T/2, T/2, dt)

# basis_count=Ns #number of basis - the same as the numbet if symbols as each stmbik us carried by one basis

# test_powers = [-18, -16, -14, -12, -10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10]
# test_powers = [-10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10]
test_powers = [-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8]
# test_powers = [4, 6, 8, 10]
# test_powers = [0]
# num_taps = [21, 21, 21]
cnstl_list_init = cons.constellation[3]; #Initial constlations

# data_set_size = 5120; #the number of samples(bits) you want to process - lcm val equals to 13440
itr_count = 10
#data_set_size=lcm.reduce([basis_count*x for x in range(1,9)])*10**2; #the number of samples(bits) you want to process - lcm val equals to 13440


laser_linewidth = 100e3 # line-width of laser for calculating phase noise

# phsical constants
h = 6.626*10**-34
c = 3 * 10**8
lambda0 = 1.55 * 10**-6
nu0 = c/lambda0;


# fiber parameters
loss_dB = 0.2*1e-3; # dB/m
# loss_dB = 0; # dB/m
loss = loss_dB * log(10)/10;
Chrom_Disp = 17 * 10**-12/10**-9/10**3; #s/m2
# Chrom_Disp = 0; #s/m2
beta2 = -lambda0**2/(2*pi*c) * Chrom_Disp;         # s**2/m
# Gam = 1.27*1e-3
Gam = 1.4*1e-3
# Gam = 2.5*1e-3
# Gam = 0


# Noise PSD and amplifier gain

SpanLength = 80e3 # 100km
nbr_of_span=14;
StPS = 80
StPS_bw = 8

Noise_Figure = 5
# Noise_Figure = 0
NF = 10**(Noise_Figure/10);
Gain_dB = loss_dB*SpanLength;  # Noise figure in dB
gain = 10**(Gain_dB/10);
PSD = (gain-1)*h*nu0*(NF/2); # Pase is the Power of ASE noise

Fs= SpS_fw * Rs

Nsp = (NF*gain)/(2*(gain-1)) # spantanous emission factor
Pase = Nsp * (gain-1)*h*nu0*Fs



PMD = 0.05 * 10**-12/sqrt(10**3) # 0.1 ps/sqrt(km)
# PMD = 0
# SNR_1mW_10G_10span = 10*log10(1e-3/(PSD*10e9*10))

# print(SNR_1mW_10G_10span)
