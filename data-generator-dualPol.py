import numpy as np
from numpy.random import randn, randint
from numpy.fft import fft, ifft, fftshift, ifftshift
from numpy import exp, sinc, mean, abs, pi, inf, sum, arange, linspace, sqrt, log2, log, log10
from numpy.linalg import norm 

# import tensorflow as tf
# import matplotlib.pyplot as plt
#import optics_lib as optics
import constellations_set as cons
import sys
import time
import params_dualPol as params
from params_dualPol import * 
from fiber_transmission_dualPol import *
from utils_dualPol import *
# from scipy import signal


np.set_printoptions(threshold=np.inf);


Ns = params.Ns              # no. of symbols, in your commentary, basis_count
Rs = params.Rs
T=params.T #the period of the signals - the optimal T is 24:-12 to +12

SpS_fw = params.SpS_fw;
SpS_rx = params.SpS_rx;

Nfft = params.Nfft        # len of signal. The factor 2 is due to the factor 2 in T

dt = params.dt
t = params.t

gap_sps = int(SpS_fw/SpS_rx)

# basis_count=params.basis_count #number of basis

test_powers = params.test_powers;
# num_taps = params.num_taps;
cnstl_list_init = params.cnstl_list_init; #Initial constlations
# data_set_size = params.data_set_size; #the number of samples(bits) you want to process - lcm val equals to 13440
itr_count = params.itr_count

#fiber
c = params.c
# loss_dB = 0.2*10**-3;
loss_dB = params.loss_dB;
loss = params.loss
lambda0 = params.lambda0
Chrom_Disp = params.Chrom_Disp
beta2 = params.beta2


h = params.h
# Seff = 90
# Gam = 2*np.pi*n2/(lambda0*1e-9)/(Seff*1e-12) * mode;
Gam = params.Gam


# Noise
nu0 = params.nu0
Noise_Figure = params.Noise_Figure
NF = params.NF
SpanLength = params.SpanLength # 100km
nbr_of_span=params.nbr_of_span;
Lf = SpanLength * nbr_of_span
StPS = params.StPS;
StPS_bw = params.StPS_bw;
Gain_dB = params.Gain_dB  # Noise figure in dB
gain = params.gain
# Rs = 25*10**9
# Pase1 = params.Pase1; # Pase is the Power of ASE noise
# ASEnoise_effect = params.ASEnoise_effect;


PMD = params.PMD

# n2=1

output_path_dir="dataset/{}x80km/{}sps/".format(nbr_of_span,SpS_rx);


# symbl_rx_qe_total = np.zeros(Ns*200, dtype="complex")
# symbl_rx_qzt_total = np.zeros(Ns*200, dtype="complex")
# symbl_rx_cdc_total = np.zeros(Ns*200, dtype="complex")


def transmit_receive(u):

	# global symbl_rx_qe_total
	# global symbl_rx_qzt_total

	error_Ax=0; #Initializing the error value
	error_Ay=0; #Initializing the error value
	Q_factor_Ax = 0; # initializing Q-factor
	Q_factor_Ay = 0; # initializing Q-factor
	# global bit_stream_tx
	
	data_tx_symbl = np.zeros((itr_count,2,Ns), dtype="complex")
	data_qzt = np.zeros((itr_count,2,2*Ns), dtype="complex")
	data_qe_DBP = np.zeros((itr_count,2,2*Ns), dtype="complex")
	data_qe_CDC = np.zeros((itr_count,2,2*Ns), dtype="complex")
	data_cpe_output = np.zeros((itr_count,2,Ns-5), dtype="complex");
	
	for x in range(itr_count):

		bit_stream_tx_Ax = randint(2, size=frame_size)
		bit_stream_tx_Ay = randint(2, size=frame_size)

		symbl_tx_Ax = map_QAM(bit_stream_tx_Ax,cnstl)
		symbl_tx_Ay = map_QAM(bit_stream_tx_Ay,cnstl)
		
		data_tx_symbl[x,0,:]=symbl_tx_Ax
		data_tx_symbl[x,1,:]=symbl_tx_Ay
		
		# pss = mean(abs(symbl_tx_Ax)**2)
		# print("pwr symbol:", pss)
		# input()
		
		q0t_Ax = modulate(symbl_tx_Ax);
		q0t_Ay = modulate(symbl_tx_Ay);
		q0t_Axy = [q0t_Ax,q0t_Ay];
		
		# plt.plot(q0t_Ax)
		# if x ==0:
			# print("sig power: {} dbm".format(10*log10(mean(abs(q0t)**2)/1e-3)))
		# print("transmission..")
		qzt_Ax2, qzt_Ay2 = ssfm_dpol(q0t_Axy,Lf,nbr_of_span,StPS,Gam,loss_dB,beta2,dt,PMD,Po); #mode: 1 denotes forward and mode:-1 denote inverse of the function
		
		#phase noise
		phi = phase_noise(Nfft, laser_linewidth, dt)
		qzt_Ax2 = qzt_Ax2*exp(1j*phi)
		qzt_Ay2 = qzt_Ay2*exp(1j*phi)
		#end phase noise
		
		qzt_Ax = match_filtering(qzt_Ax2)
		qzt_Ay = match_filtering(qzt_Ay2)
		# plt.plot(qzt_Ax)
		# plt.show()
		
		if SpS_rx != SpS_fw:
			qzt_Ax = qzt_Ax[::gap_sps]
			qzt_Ay = qzt_Ay[::gap_sps]
		
		data_qzt[x,0,:] = qzt_Ax
		data_qzt[x,1,:] = qzt_Ay
		
		
		qzt_Axy = [qzt_Ax,qzt_Ay]
		qe_DBP_Ax, qe_DBP_Ay = DBP_dpol(qzt_Axy,Lf,nbr_of_span,StPS_bw,-Gam*0.2,-loss_dB,-beta2,dt_s); 
		qe_CDC_Ax, qe_CDC_Ay = cd_comp(qzt_Axy,Lf,-beta2,dt_s); 
		
		data_qe_DBP[x,0,:] = qe_DBP_Ax
		data_qe_DBP[x,1,:] = qe_DBP_Ay
		
		data_qe_CDC[x,0,:] = qe_CDC_Ax
		data_qe_CDC[x,1,:] = qe_CDC_Ay
		

		qe_Axy = [qe_CDC_Ax, qe_CDC_Ay]
		
		symbl_rx_Ax, symbl_rx_Ay = CMA_RDEM(qe_Axy,SpS_rx,Ns,5,Po)
		
		symbl_tx_Ax = symbl_tx_Ax[:-5]
		symbl_tx_Ay = symbl_tx_Ay[:-5]		
		
		symbl_rx_Ax, symbl_rx_Ay = pol_dmux(symbl_rx_Ax,symbl_rx_Ay,symbl_tx_Ax,symbl_tx_Ay)
		
		# symbl_rx_Ax, symbl_rx_Ay = cpe([symbl_rx_Ax,symbl_rx_Ay],[symbl_tx_Ax,symbl_tx_Ay],51,Po)
		symbl_rx_Ax = phase_offset(symbl_rx_Ax, symbl_tx_Ax, 102, 51)
		symbl_rx_Ay = phase_offset(symbl_rx_Ay, symbl_tx_Ay, 102, 51)
		
		data_cpe_output[x,0,:]=symbl_rx_Ax
		data_cpe_output[x,1,:]=symbl_rx_Ay

		print(round(100*x/itr_count,1), "% / pw_itr: ", _itr_01+1 ,end="\r");
		
	
	output_filename=output_path_dir+"/{}dbm-data".format(u);
	np.savez(output_filename, qzt = data_qzt, qe_CDC = data_qe_CDC, qe_DBP = data_qe_DBP, symbl_tx=data_tx_symbl, cpe_output = data_cpe_output)
	

#--------------
#End function
#--------------

# BASIS_init = make_pulses()

#ssfm config
F = 1/dt
df = 1/T
f = arange(-F/2,F/2, df)
w = 2*pi*ifftshift(f)


dt_s=T/(Nfft/gap_sps);
F_s = 1/dt_s
df_s = 1/T
f_s = arange(-F_s/2,F_s/2, df_s)
w_s = 2*pi*fftshift(f_s)


cnstl_init = np.asarray(cnstl_list_init,dtype="complex128");
cnstl_unit = cnstl_init*1/np.sqrt(mean(abs(cnstl_init)**2)); # normalize the constelation power to 1
cnstl_size = cnstl_unit.size; #constellation size


frame_size=int(log2(cnstl_size)*Ns); #the number of bits that each message can carry based on the number of basis and constellation size
# itr_count=int(data_set_size/frame_size);

error_arr_Ax = np.zeros(len(test_powers)); #array to save error rate for different power
error_arr_Ay = np.zeros(len(test_powers)); #array to save error rate for different power
qfactor_arr_Ax = np.zeros(len(test_powers)); #array to save error rate for different power
qfactor_arr_Ay = np.zeros(len(test_powers)); #array to save error rate for different power
snr_val = np.zeros(len(test_powers)); #array to different power values
cnstl=cnstl_init.copy();#initialzing cnstl as a used constellation for different operation
_itr_01 = 0;

start_time=time.process_time();


for i in test_powers:

	Po = 10**(0.1*i-3)

	cnstl=cnstl_unit;
	
	pwr_dbm = 10*log10(Po/1e-3)
	print("\nsig pwr: {} dbm".format(np.round(pwr_dbm,2)))
	# print("\ntaps: {}".format(num_taps[_itr_01]))
	
	cnstl = sqrt(Po)*cnstl
	cnstl_list=list(cnstl); #helful for demaping - should be optimized in future

	transmit_receive(i);
	# _itr_01+=1


end_time = time.process_time();

#storing results

