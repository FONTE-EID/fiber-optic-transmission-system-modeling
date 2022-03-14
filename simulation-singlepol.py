import numpy as np
from numpy.random import randn, randint
from numpy.fft import fft, ifft, fftshift, ifftshift
from numpy import exp, sinc, mean, abs, pi, inf, sum, arange, linspace, sqrt, log2, log, log10
from numpy.linalg import norm 

# import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
#import optics_lib as optics
import constellations_set as cons
import sys
import time
import params_singlePol as params
from fiber_transmission_singlePol import *
from utils_singlePol import *
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

output_path_dir="BER/{}x80km/{}sps/nophase/".format(nbr_of_span,SpS_rx);

output_filename = sys.argv[1]
gam_fac = int(output_filename)*1e-2

# symbl_rx_qe_total = np.zeros(Ns*200, dtype="complex")
# symbl_rx_qzt_total = np.zeros(Ns*200, dtype="complex")
# symbl_rx_cdc_total = np.zeros(Ns*200, dtype="complex")
# symbl_rx_cpe_total = np.zeros(Ns*2, dtype="complex")


def myplot(x, y, s, bins=2000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def transmit_receive(u):

	# global symbl_rx_qe_total
	# global symbl_rx_qzt_total
	global symbl_rx_cpe_total


	error_Ax=0; #Initializing the error value
	error_Ay=0; #Initializing the error value
	Eff_SNR_Ax = 0; # initializing Q-factor
	Eff_SNR_Ay = 0; # initializing Q-factor
	# global bit_stream_tx
	for x in range(itr_count):

		bit_stream_tx_Ax = randint(2, size=frame_size)

		symbl_tx_Ax = map_QAM(bit_stream_tx_Ax,cnstl)
		
		
		q0t_Ax = modulate(symbl_tx_Ax);
				
		# plt.plot(q0t_Ax)
		# if x ==0:
			# print("sig power: {} dbm".format(10*log10(mean(abs(q0t)**2)/1e-3)))
		# print("transmission..")
		qzt_Ax2 = ssfm_dpol(q0t_Ax,Lf,nbr_of_span,StPS,Gam,loss_dB,beta2,dt,PMD,Po); #mode: 1 denotes forward and mode:-1 denote inverse of the function
		
		#phase noise
		phi = phase_noise(Nfft, laser_linewidth, dt)
		qzt_Ax2 = qzt_Ax2*exp(1j*phi)
		#end phase noise
		
		
		qzt_Ax = match_filtering(qzt_Ax2)
		# plt.plot(qzt_Ax)
		# plt.show()
		
		if SpS_rx != SpS_fw:
			qzt_Ax = qzt_Ax[::gap_sps]
		
		
		qe_Ax = DBP_dpol(qzt_Ax,Lf,nbr_of_span,StPS_bw,-Gam*gam_fac,-loss_dB,-beta2,dt_s); #mode: 1 denotes forward and mode:-1 denote inverse of the function, for equalization with negated params
		# qe_Ax = cd_comp(qzt_Ax,Lf,-beta2,dt_s); #mode: 1 denotes forward and mode:-1 denote inverse of the function, for equalization with negated params
				
				
		symbl_rx_Ax = qe_Ax[::SpS_rx];
		# symbl_rx_Ay = qe_Ay[::SpS_rx];
		
		# symbl_rx_Ax = demod(qe_Ax);
		# symbl_rx_Ay = demod(qe_Ay);
		
		
		# symbl_rx_cdc_total [x*Ns:(x+1)*Ns] = symbl_rx
		
		# symbl_rx_Ax, symbl_rx_Ay = pol_dmux(symbl_rx_Ax,symbl_rx_Ay,symbl_tx_Ax,symbl_tx_Ay)
		
		# plt.plot(symbl_rx_Ax.real,symbl_rx_Ax.imag, '.', label='rx');
		# plt.plot(symbl_tx_Ax.real, symbl_tx_Ax.imag, '.', label="tx");
		# plt.legend(loc="best")
		# plt.title("200kHz Before CPE")
		# plt.show();
		# augmented_symbl_rx_Ax = np.zeros(len(symbl_rx_Ax)*200,dtype="complex")
		# len_vec  =len(symbl_rx_Ax)
		# for i in range(200):
			# vec = symbl_rx_Ax+symbl_rx_Ax*np.random.randn(len_vec)*0.0025
			# augmented_symbl_rx_Ax[i*len_vec:(i+1)*len_vec] = vec
		
		# fig, axs = plt.subplots(1, 1)
		

		# img, extent = myplot(augmented_symbl_rx_Ax.real, augmented_symbl_rx_Ax.imag, 1)
		# axs.imshow(img, extent=extent, origin='lower', cmap=cm.hot)
		# axs.set_title("200kHz Before CPE")
		
		# plt.show()
		
		
		symbl_rx_Ax = phase_offset(symbl_rx_Ax, symbl_tx_Ax, 102, 51)		
		
		# symbl_rx_cpe_total[x*Ns:(x+1)*Ns] = symbl_rx_Ax

		
		# del augmented_symbl_rx_Ax
		
		# if x == 1:
			# augmented_symbl_rx_Ax = np.zeros(len(symbl_rx_cpe_total)*200,dtype="complex")
			# len_vec  =len(symbl_rx_cpe_total)
			# for i in range(200):
				# vec = symbl_rx_cpe_total+symbl_rx_cpe_total*np.random.randn(len_vec)*0.001
				# augmented_symbl_rx_Ax[i*len_vec:(i+1)*len_vec] = vec
			
			# fig, axs = plt.subplots(1, 1)
		
			# img, extent = myplot(augmented_symbl_rx_Ax.real, augmented_symbl_rx_Ax.imag, 1)
			# axs.imshow(img, extent=extent, origin='lower', cmap=cm.hot)
			# axs.set_title("100kHz Before CPE")
			
			# plt.show()
		
		# plt.plot(symbl_rx_Ax.real,symbl_rx_Ax.imag, '.', label='rx');
		# plt.plot(symbl_tx_Ax.real, symbl_tx_Ax.imag, '.', label="tx");
		# plt.legend(loc="best")
		# plt.title("200kHz After CPE")
		# plt.show();
		# input() 
		
		# print("symbl_rx_Ax1: ", symbl_rx_Ax[4000:4016])
		# print("symbl_tx_Ax1: ", symbl_tx_Ax[4000:4016])
		
		symbl_rx_Ax_dec = detect(symbl_rx_Ax,cnstl);
		 
		# plt.plot(symbl_rx_Ax_dec.real,symbl_rx_Ax_dec.imag, '.', label='rx');
		# plt.plot(symbl_tx_Ax.real, symbl_tx_Ax.imag, '.', label="tx");
		# plt.legend(loc="best")
		# plt.show();
		# plt.plot(symbl_rx_Ay_dec.real,symbl_rx_Ay_dec.imag, '.', label='rx');
		# plt.plot(symbl_tx_Ay.real, symbl_tx_Ay.imag, '.', label="tx");
		# plt.legend(loc="best")
		# plt.show();
		# input() 
		 
		symbl_tx_Ax_eval = symbl_tx_Ax[1000:15000]
		
		symbl_rx_Ax_eval = symbl_rx_Ax[1000:15000]
		
		symbl_rx_Ax_dec_eval = symbl_rx_Ax_dec[1000:15000]
		
		 
		bit_stream_rx_Ax_eval = demap_QAM(symbl_rx_Ax_dec_eval, cnstl)*1;
		
		
		bit_stream_tx_Ax_eval = demap_QAM(symbl_tx_Ax_eval, cnstl)*1;
		
		
		Eff_SNR_Ax += norm(symbl_tx_Ax_eval)/norm(symbl_rx_Ax_eval-symbl_tx_Ax_eval)
		
		
		error_Ax += mean(abs(bit_stream_tx_Ax_eval-bit_stream_rx_Ax_eval))
		

		print(round(100*x/itr_count,1), "% / pw_itr: ", _itr_01+1 ,end="\r");
		
	Eff_SNR_rate_Ax = Eff_SNR_Ax/itr_count;
	error_rate_Ax = error_Ax/itr_count;
	print("\r\t\t\t\t\rEff. SNR (dB): {}".format((10*log10(Eff_SNR_rate_Ax**2))));
	print("BER: {}".format(error_rate_Ax));
	
	return error_rate_Ax, 10*log10(Eff_SNR_rate_Ax**2);
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
qfactor_arr_Ax = np.zeros(len(test_powers)); #array to save error rate for different power
snr_val = np.zeros(len(test_powers)); #array to different power values
cnstl=cnstl_init.copy();#initialzing cnstl as a used constellation for different operation
_itr_01 = 0;

start_time=time.process_time();


for i in test_powers:

	Po = 10**(0.1*i-3)
	# BASIS = sqrt(Po)* BASIS_init
	
	# if SpS_rx != SpS_fw:
		# BASIS_sampled = BASIS[:,1::gap_sps]
	# else:
		# BASIS_sampled = BASIS

	cnstl=cnstl_unit;
	
	pwr_dbm = 10*log10(Po/1e-3)
	print("\nsig pwr: {} dbm".format(np.round(pwr_dbm,2)))
	
	cnstl = sqrt(Po)*cnstl
	cnstl_list=list(cnstl); #helful for demaping - should be optimized in future

	# update_mod_var(Po, cnstl, cnstl_list)

	error_arr_Ax[_itr_01], qfactor_arr_Ax[_itr_01] = transmit_receive(i);
	snr_val[_itr_01] = pwr_dbm;

	_itr_01+=1;
	


end_time = time.process_time();

#storing results

print("\nOperation done. Time: ",end_time-start_time,"\nStoring results ... \n");

# output_file = "/error-"+output_filename+"fn"+str(StPS_bw)+"-PN"+str(len(test_powers))+"-P."+str(test_powers[0])+"-"+str(test_powers[-1])+".val";
output_file = "/error-"+output_filename+"-.val";

_f=open(output_path_dir+output_file, "w");
_f.write(np.array2string(error_arr_Ax, separator=','));
_f.close();


output_file = "/Qfactor-"+output_filename+"-.val";

_f=open(output_path_dir+output_file, "w");
_f.write(np.array2string(qfactor_arr_Ax, separator=','));
_f.close();


output_file = "/snr-PN"+str(len(test_powers))+"-P."+str(test_powers[0])+"-"+str(test_powers[-1])+".val";

_f=open(output_path_dir+output_file, "w");
_f.write(np.array2string(snr_val, separator=','));
_f.close();

