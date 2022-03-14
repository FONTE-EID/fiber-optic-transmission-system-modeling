# import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, sinc, mean, abs, pi, inf, sum, arange, linspace, sqrt, log2, log, log10, sin, cos, sinc, argmin, sqrt, cos, sin, floor, ceil, argmax, angle
from numpy.random import randn
import scipy
import math
import cmath
from scipy import special as sp
import scipy.integrate as integrate
from numpy.fft import fft, ifft, fftshift, ifftshift
from utils_singlePol import *
from params_singlePol import *

#########################################################################################################
def noise_type1():

	noise_vec = sqrt(PSD*Rs/2) * (randn(Ns)+ 1j*randn(Ns))
	noise_sig = modulate(noise_vec)
	return noise_sig

def noise_type2():

	noise_vec = sqrt(PSD*Rs*SpS_fw/2) * (randn(Ns)+ 1j*randn(Ns))
	noise_sig = modulate(noise_vec)
	return noise_sig

def ssfm_dpol(signal,l_f,n_span,n_seg,gamma,alpha_db,beta2,dt,PMD,p):
    #Noise addition
    # p = mean(abs(signal[0])**2)
    # print("input SSFM power:", p)
    # input()
    alpha=(log(10)/10)*(alpha_db)
    l_span=(l_f/n_span)
    G_db=alpha_db*l_span
    # print("G_db: ", G_db)
    g=10**(G_db/10)
    signal_recx=signal
    l_s=int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * fftshift(np.arange(-l_s/2,l_s/2))
    step_s=(l_f/((n_seg)*(n_span)))
    dis=np.exp(-(1j/2)*beta2*(w**2)*step_s)#dispersion.
    att=np.exp((-alpha/2)*step_s)#loss
    for j in range(n_span):
        # print("FwP SSFM: {}km".format(int(j*l_span/1e3)), end="\r")
        for jj in range(n_seg):
            signal_recfx=fft(signal_recx)
            #add dispersion 
            signal_recfx=signal_recfx*dis

            # add loss +nonlinearity 
            signal_recx=ifft(signal_recfx)
            signal_recx=signal_recx*att
            signal_recx=signal_recx*np.exp(1j*gamma*(abs(signal_recx)**2)*step_s)
        # signal_recx=sqrt(g)*signal_recx + noise_type1()
        # signal_recx=sqrt(g)*signal_recx + noise_type2()
        signal_recx=sqrt(g)*signal_recx + sqrt(Pase/2) * (randn(signal_recx.size)+ 1j*randn(signal_recx.size))

    return signal_recx


#########################################################################################################


def DBP_dpol(signal,l_f,n_span,n_seg,gamma,alpha_db,beta2,dt):
    #Noise addition
    alpha=(log(10)/10)*(alpha_db)
    l_span=(l_f/n_span)
    signal_recx=signal
    l_s=int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * fftshift(np.arange(-l_s/2,l_s/2))
    step_s=(l_f/((n_seg)*(n_span)))
    dis=np.exp(-(1j/2)*(beta2)*(w**2)*step_s)#dispersion.
    for j in range(n_span):
        for jj in range(n_seg):
            signal_recfx=fft(signal_recx)
            #add dispersion 
            signal_recfx=signal_recfx*dis

            signal_recx=ifft(signal_recfx)
			
            signal_recx=signal_recx*np.exp(1j*gamma*(abs(signal_recx)**2)*step_s)
			
    return signal_recx


#########################################################################################################


def cd_comp(signal,l_f,beta2,dt):
    signal_recx=signal
    l_s=int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * fftshift(np.arange(-l_s/2,l_s/2))
    dis_com=np.exp(-(1j/2)*(beta2)*(w**2)*l_f)#dispersion.
    signal_recfx=fft(signal_recx)*dis_com
    signal_recx=ifft(signal_recfx)
    return signal_recx

#########################################################################################################

def CMA_RDEM(signal,Nsps,N_s,N_taps,p):
    p = mean(abs(signal[0])**2)
    # print("p: ", p)
    s_inx=signal[0]/sqrt(p)
    # s_inx=signal[0]/mean(abs(signal[0])**2)
    # print("signal power: {} dbm".format(10*log10(mean(abs(signal[0])**2)/1e-3)))
    # print("s_inx power: {} dbm".format(10*log10(mean(abs(s_inx)**2)/1e-3)))
    # print("po: ", p)
    # input()
    # s_iny=signal[1]/mean(abs(signal[1])**2)
    s_iny=signal[1]/sqrt(p)
    l_s=len(s_inx)
    mu=0.001
    itr=15000 # training symbols.
    th=5000  # 
    # R1=abs(1/3+1j/3)
    # R2=abs(1/3+1j)
    # R3=abs(1+1j)
    R1=0.4470035621347161
    R2=0.9995303606046092
    R3=1.3410107005462
    g1=((R1)+(R2))/2
    g3=((R2)+(R3))/2
#************************************************************
#Convergance of the X_pol
    F_1=np.zeros((N_taps))+0j#inpx__outx
    F_2=np.zeros((N_taps))+0j#inpy__outx
    F_1[int(floor(N_taps/2))]=1+0j
    #CMA
    for l in range(th):
      idx_s=l*(Nsps)
      s_x=s_inx[idx_s:idx_s+N_taps]
      s_y=s_iny[idx_s:idx_s+N_taps]
      st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
      e_x=2*(abs(st_outx)**2-sqrt(2))*st_outx
      F_1=F_1-mu*e_x*np.conjugate(s_x)
      F_2=F_2-mu*e_x*np.conjugate(s_y)
#*************************************************************
#RDE
    
    for l in range(th,itr):  
       idx_s=l*(Nsps)
       s_x=s_inx[idx_s:idx_s+N_taps]
       s_y=s_iny[idx_s:idx_s+N_taps]
       st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
#*************************************************************
       if (abs(st_outx)<g1):
         Rs_x=R1**2       
       elif (abs(st_outx)>g3):
         Rs_x=R3**2
       else:
         Rs_x=R2**2 

       e_x=2*(abs(st_outx)**2-Rs_x)*st_outx
       F_1=F_1-mu*e_x*np.conjugate(s_x)
       F_2=F_2-mu*e_x*np.conjugate(s_y)   
###############################################################################
#Convergance for X and Y pol.
    F_4=np.conjugate(np.flip(F_1))
    F_3=-1*np.conjugate(np.flip(F_2))
#******************************************************************************
    for l in range(th):
      idx_s=l*(Nsps)
      s_x=s_inx[idx_s:idx_s+N_taps]
      s_y=s_iny[idx_s:idx_s+N_taps]
      st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
      st_outy=np.dot(F_3,s_x)+np.dot(F_4,s_y)
      e_x=2*(abs(st_outx)**2-sqrt(2))*st_outx
      e_y=2*(abs(st_outy)**2-sqrt(2))*st_outy
      F_1=F_1-mu*e_x*np.conjugate(s_x)
      F_2=F_2-mu*e_x*np.conjugate(s_y)
      F_3=F_3-mu*e_y*np.conjugate(s_x)
      F_4=F_4-mu*e_y*np.conjugate(s_y)
#*************************************************************
#RDE
    for l in range(th,itr):  
       idx_s=l*(Nsps)
       s_x=s_inx[idx_s:idx_s+N_taps]
       s_y=s_iny[idx_s:idx_s+N_taps]
       st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
       st_outy=np.dot(F_3,s_x)+np.dot(F_4,s_y)
#*************************************************************
       if (abs(st_outx)<g1):
         Rs_x=R1**2       
       elif (abs(st_outx)>g3):
         Rs_x=R3**2
       else:
         Rs_x=R2**2 
       if (abs(st_outy)<g1):
         Rs_y=R1**2       
       elif (abs(st_outy)>g3):
         Rs_y=R3**2
       else:
         Rs_y=R2**2 
       
       e_x=2*(abs(st_outx)**2-Rs_x)*st_outx
       e_y=2*(abs(st_outy)**2-Rs_y)*st_outy
       F_1=F_1-mu*e_x*np.conjugate(s_x)
       F_2=F_2-mu*e_x*np.conjugate(s_y)
       F_3=F_3-mu*e_y*np.conjugate(s_x)
       F_4=F_4-mu*e_y*np.conjugate(s_y)
    
    limit=N_s-N_taps
    s_outx=np.zeros((1,limit))+0j
    s_outy=np.zeros((1,limit))+0j
    for l in range (limit):
         idx_s=l*2
         s_x=s_inx[idx_s:idx_s+N_taps]
         s_y=s_iny[idx_s:idx_s+N_taps]
         s_outx[0,l]=np.dot(F_1,s_x)+np.dot(F_2,s_y)
         s_outy[0,l]=np.dot(F_3,s_x)+np.dot(F_4,s_y)
         
    sx_out=s_outx[0]*sqrt(p)
    sy_out=s_outy[0]*sqrt(p)  
    return sx_out,sy_out
    
#########################################################################################################


def cpe(sym_r,sym_t,s_wind,p):
    sr_x=sym_r[0]
    sr_y=sym_r[1]
    st_x=sym_t[0]
    st_y=sym_t[1]
    B=32 # number of angles to be tested.
    b=0
    e=1
    theta_t=(np.arange(start=b, stop=e, step=(1/B))*(pi/2))-(pi/4)
    l_theta=len(theta_t)
    l_sr=len(sr_x)
    s_begin=0
    d_x=np.zeros((l_sr,l_theta))
    d_y=np.zeros((l_sr,l_theta))
    for l in range(s_begin,l_sr):
        srx_test=sr_x[l]*exp(-1j*theta_t)#applay phase shifts to the lth symbol.
        srx_test_d=deQAM16(srx_test,p)#demodulate.
        sry_test=sr_y[l]*exp(-1j*theta_t)#applay phase shifts to the lth symbol.
        sry_test_d=deQAM16(sry_test,p)#demodulate.
        #*******************************************
        #calculate the distances.
        d_x[l]=abs(srx_test-srx_test_d)**2#for x_pol.
        d_y[l]=abs(sry_test-sry_test_d)**2#for y_pol.
    #*******************************************
    #avg distances.
    dx_avg=np.zeros((l_sr,l_theta))
    dy_avg=np.zeros((l_sr,l_theta))
    theta_estx=np.zeros((1,l_sr))
    theta_esty=np.zeros((1,l_sr))
    lim1=int((s_wind/2))
    lim2=int(l_sr-(s_wind/2))
    for l in range(lim1,lim2):
      for k in range(l_theta):
        dx_avg[l,k]=mean(d_x[l-lim1:l+lim1,k])
        dy_avg[l,k]=mean(d_y[l-lim1:l+lim1,k])
    ind_x=np.where(dx_avg[l]==min(dx_avg[l]))
    # print("idx: ", ind_x)
    theta_estx[0,l]=theta_t[int(ind_x[0])]
    ind_y=np.where(dy_avg[l]==min(dy_avg[l]))
    theta_esty[0,l]=theta_t[int(ind_y[0])]
    # #******************************************************************
    # # Cs mitigation 
    phase_trx=diff(theta_estx[0])
    phase_try=diff(theta_esty[0])
    phase_shiftx=np.cumsum((phase_trx<(-pi/4))*(-pi/2)+(phase_trx>(pi/4))*(pi/2))
    phase_shifty=np.cumsum((phase_try<(-pi/4))*(-pi/2)+(phase_try>(pi/4))*(pi/2))
    thet_x=theta_estx[0]-phase_shiftx
    thet_y=theta_esty[0]-phase_shifty
    s_outtx=np.multiply(sr_x,exp(-1j*thet_x))
    s_outty=np.multiply(sr_y,exp(-1j*thet_y))
# #***********************************************************************
    
#***************************************************************************
# #seconed cpe stage
    v_1x =s_outtx[lim1:lim2]
    v_2x =st_x[lim1:lim2]
    ratio_x= np.divide(v_1x,v_2x)
    anglex = cmath.phase(mean(ratio_x))
    ss_x=np.multiply(s_outtx, exp(-1j*anglex))
    v_1y =s_outty[lim1:lim2]
    v_2y =st_y[lim1:lim2]
    ratio_y= np.divide(v_1y,v_2y)
    angley = cmath.phase(mean(ratio_y))
    ss_y=np.multiply(s_outty, exp(-1j*angley))
    #*******************************************************************
    return ss_x,ss_y

	

#########################################################################################################

def phase_offset(Rec_polX,Sequence_baud_X, Tbegin, Twindow):
	
	N = len(Rec_polX)
	
	ratioXX = Rec_polX/Sequence_baud_X; 
	
	CPE_PolX_DA = angle(mean(ratioXX[Tbegin:]));

	CPE_PolX_window = np.zeros(N);
	CPE_PolX_window[:Tbegin] = CPE_PolX_DA;    
  

	for iter in range(int(floor(Twindow/2)), int(N-floor(Twindow/2))):  
		CPE_PolX_window[iter] = angle(mean(ratioXX[int(iter-floor(Twindow/2)):int(iter+floor(Twindow/2))]));

	X_out_window = Rec_polX * exp(-1j*CPE_PolX_window);

	return X_out_window
	

def phase_offset_simple(symbl_rx,symbl_tx):

	mean_angle = angle(symbl_rx/symbl_tx)
	rotated_symbols = symbl_rx*np.exp(-1j*mean_angle)

	return rotated_symbols

	
def phase_offset_x(symbl_rx,symbl_tx):

	mean_angle = np.mean(np.angle(symbl_rx[1000:1200]*np.conj(symbl_tx[1000:1200])))
	rotated_symbols = symbl_rx*np.exp(-1j*mean_angle)

	return rotated_symbols	

def phase_offset_y(symbl_rx,symbl_tx):

	mean_angle = np.mean(np.angle(symbl_rx[1000:1200]*np.conj(symbl_tx[1000:1200])))
	rotated_symbols = symbl_rx*np.exp(-1j*mean_angle*1.55)

	return rotated_symbols		
	
#########################################################################################

def phase_noise(Nfft, linewidth, dt):
    
    phi = np.zeros(Nfft)

    noise = randn(Nfft)   # normalized Gaussian noise 
    dphi  =  np.sqrt(2*pi*linewidth*dt)*noise  # Gaussian noise
    phi   = np.cumsum(dphi)  # Wiener process 
    
    return phi	