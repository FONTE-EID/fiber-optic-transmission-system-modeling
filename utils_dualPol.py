import numpy as np
from numpy.random import randn, randint
from numpy.fft import fft, ifft, fftshift, ifftshift
from numpy import exp, sinc, mean, abs, pi, inf, sum, arange, linspace, sqrt, log2, log, log10, sin, cos, sinc, argmin, sqrt
from numpy.linalg import norm 
import params_dualPol as params
# from scipy import interpolate

# import matplotlib.pyplot as plt

# from scipy import signal

Ns=params.Ns #number of basis - must be an Even number
T=params.T #the period of the signals - the optimal T is 24:-12 to +12
Nfft=params.Nfft; #sample rate used for making basis signal - Must be power of 2

# SpS=4;

cnstl_list_init = params.cnstl_list_init; #Initial constlations
# data_set_size = params.data_set_size; #the number of samples(bits) you want to process - lcm val equals to 13440
#data_set_size=np.lcm.reduce([Ns*x for x in range(1,9)])*10**2; #the number of samples(bits) you want to process - lcm val equals to 13440

# Rs = 25*10**9
Rs = params.Rs

cnstl_init = np.asarray(cnstl_list_init);
cnstl_size = cnstl_init.size; #constellation size

frame_size=int(np.log2(cnstl_size)*Ns); #the number of bits that each message can carry based on the number of basis and constellation size

bsr=int(np.log2(cnstl_size)); # bits to symbols ratio

mapper_vec=[2**x for x in range(bsr)]; #the vector which is used for map_ing process


#--------------
#Begin function - RRC
#--------------
def rrc(t, dt, beta, Ts):
	
	t0 = argmin(abs(t));
	
	t=np.where(abs(t) == Ts/(4*beta),
	(beta/(Ts*sqrt(2)))*((1+(2/pi))*sin(pi/(4*beta))+(1-(2/pi))*cos(pi/(4*beta))),
	(1/Ts)*((sin((pi*t/Ts)*(1-beta))+(4*beta*t/Ts)*cos((pi*t/Ts)*(1+beta)))/((pi*t/Ts)*(1-(4*beta*t/Ts)**2)))
	)
	
	t[t0] = (1/Ts)*(1+beta*((4/pi)-1))
	
	e=sum(abs(t)**2)*dt
	
	t=sqrt(Ts/e)*t
	
	return t;
#--------------
#End function
#--------------
	
	
#--------------
#Begin function - Map at TX
#--------------
def map_(arr,cnstl):
    #tre=np.dot(arr,mapper_vec)
    #print("symbl: ",tre, "cntl: ",cnstl[tre])
    #input()
    return cnstl[np.dot(arr,mapper_vec)]

#--------------
#End function
#--------------


#--------------
#Begin function
#--------------
def demap_QAM(arr, cnstl):

    cnstl_list=list(cnstl)
	
    bit_stream = np.zeros(frame_size);

    for i in range(len(arr)):
        arr[i]=cnstl_list[argmin(abs(cnstl-arr[i]))];

    for i in range(len(arr)):
        _temp=[int(x) for x in bin(cnstl_list.index(arr[i]))[:1:-1]];
        #note that in converting binary to decimal in map_ func. we were converting in reverse order compare in paper, but bin() gives the binary of a decimal as on the paper, so we should reverse its format
        #there should be better way than search for the value in the list, having the order O(n) - should optimized in future - important
        bit_stream[i*bsr:i*bsr+len(_temp)] = _temp
        #print("symbl val: ", arr[i],"#index: ",cnstl_list.index(arr[i]), "#_temp: ",_temp, "#bit stream: ", bit_stream[i*bsr:(i+1)*bsr]);
        #input();

    return bit_stream
#--------------
#End function
#--------------
	
#--------------
#Begin function
#--------------
def map_QAM(stream,cnstl):
	symbl_tx = np.asarray([map_(stream[i*bsr:(i+1)*bsr],cnstl) for i in range(int(len(stream)/bsr))])
	return symbl_tx
#--------------
#End function
#--------------


#--------------
#Begin function - Demaping function based on Nearest neighbourhood
#--------------
def detect(vec,cnstl):
	arr = vec*1
	cnstl_list=list(cnstl)
	for i in range(len(arr)):
		arr[i]=cnstl_list[argmin(abs(cnstl-arr[i]))];
	return arr;
#--------------
#End function
#--------------


#--------------
#Begin function - RX
#--------------
def demod(sig):

	dt = T/len(sig)
	
	e = 1/Rs * 1/dt
	
	T0=1/Rs;
	k_range=np.arange(-Ns/2,Ns/2);
	t=np.arange(-T/2,T/2,dt);
	
	# pulse = rrc(t, dt, 0.1, T0)
	
	# symbl_rx = fft(sig)*fft(rrc(t, dt, 0.1, T0))/e
	symbl_rx = np.zeros(Ns, dtype=complex);
	
	for k in k_range.astype(int):
		# symbl_rx[k] = np.dot(sig,np.roll(pulse,int(k*T0/dt)));
		symbl_rx[k] = np.dot(sig,rrc(t-T0*k, dt, 0.25, T0));
	# symbl_rx = np.dot(sig,BASIS_sampled.T).flatten()/e;    #Detection Process
	
	symbl_rx = symbl_rx/e

	return symbl_rx
#--------------
#End function
#--------------

	

#--------------
#Begin function - TX
# --------------
def modulate(symbl_tx):

	T0=1/Rs;
	k_range=np.arange(-Ns/2,Ns/2);
	dt=T/Nfft;
	t=np.arange(-T/2,T/2,dt);
	
	pulse = rrc(t, dt, 0.25, T0)
	
	symbl_tx_upsampled = np.zeros(t.shape[0], dtype="complex")
	symbl_tx_upsampled[::int(Nfft/Ns)] = symbl_tx
	
	q0t = ifft(ifftshift(fftshift(fft(symbl_tx_upsampled)) * fftshift(fft(pulse))))

	return q0t;
#--------------
#End function
#--------------

def match_filtering(sig):
	
	p = mean(abs(sig)**2)
	
	T0=1/Rs;
	k_range=np.arange(-Ns/2,Ns/2);
	dt=T/Nfft;
	t=np.arange(-T/2,T/2,dt);
	
	pulse = rrc(t, dt, 0.25, T0)
	
	q =  ifft(ifftshift(fftshift(fft(sig)) * fftshift(fft(pulse))))
	
	q = sqrt(p) * q/sqrt(mean(abs(q)**2))
	
	return q