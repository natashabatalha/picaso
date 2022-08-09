import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import astropy.units as u
from picaso import test as test

#picaso
from picaso import justdoit as jdi 
from picaso import justplotit as jpi
#plotting
from bokeh.io import output_notebook
output_notebook()
from bokeh.plotting import show
 
from picaso import fluxes as fl
import pickle as pk

wno = None
nwno = 1
nt = 1

nlayer = 1; nlevel = nlayer+1
dtau = np.zeros((nlayer, 1))
tau = np.zeros((dtau.shape[0]+1, dtau.shape[1]))
g = 0.75
g0 = np.zeros(dtau.shape) + g
ng = 1
fbeam = 1.
b_top = 0.

output_dir = '/Users/crooney/Documents/codes/picaso/docs/notebooks/picaso_data/liou_test/'
        
dtau_dedd =  np.zeros(dtau.shape) 
dtau_dedd4 =  np.zeros(dtau.shape) 
g0_dedd = np.zeros(dtau.shape) 
g0_dedd4 = np.zeros(dtau.shape) 
w0_dedd = np.zeros(dtau.shape)
w0_dedd4 = np.zeros(dtau.shape)

albedo = 0.
cos_theta = 1.
single_phase = 1
multi_phase = 0
gcos2 = 0
ftau_cld = None
ftau_ray = None
frac_a = None 
frac_b = None 
frac_c = None 
constant_back = None 
constant_forward = None


for w in [0.8, 1]:  
    if w == 1:
        w = 0.999999
    w0 = np.zeros(dtau.shape) + w

    for u0 in [0.1,0.5,0.9]:

        for tauN in 4**np.linspace(-1,2,4):
            #nlayer = 4; nlevel = nlayer+1
            #tau[1:,0] = 4**np.linspace(-1,2,4)
            tau[1,0] = tauN
            dtau[:,0] = tau[1:,0] - tau[:-1,0]
        
            ubar0 = np.array([[u0]])
            ubar1 = np.array([[u0]]) # this shouldn't matter for reflection and transmission
            
            for i in range(nlayer):
                w0_dedd[i]=w0[i]*(1.-g0[i]**2)/(1.0-w0[i]*g0[i]**2)
                w0_dedd4[i]=w0[i]*(1.-g0[i]**4)/(1.0-w0[i]*g0[i]**4)
                g0_dedd[i]=(g0[i]-g0[i]**2)/(1.-g0[i]**2)
                g0_dedd4[i]=(g0[i]-g0[i]**4)/(1.-g0[i]**4)
                dtau_dedd[i]=dtau[i]*(1.-w0[i]*g0[i]**2) 
                dtau_dedd4[i]=dtau[i]*(1.-w0[i]*g0[i]**4) 

            tau_dedd = np.zeros((dtau_dedd.shape[0]+1, dtau_dedd.shape[1]))
            tau_dedd4 = np.zeros((dtau_dedd.shape[0]+1, dtau_dedd.shape[1]))
            tau_dedd[1:,0] = np.cumsum(dtau_dedd)
            tau_dedd4[1:,0] = np.cumsum(dtau_dedd4)
            
            (xint, flux, intensity) = fl.get_reflected_1d(nlevel, wno, nwno, ng, nt,
                                                #dtau_dedd, tau_dedd, w0_dedd, g0_dedd, gcos2, ftau_cld, ftau_ray,
                                                dtau, tau, w0, g0, gcos2, ftau_cld, ftau_ray,
                                                dtau, tau, w0, g0 ,
                                                albedo, ubar0, ubar1, cos_theta, fbeam,
                                                single_phase, multi_phase,
                                                frac_a, frac_b, frac_c, constant_back, constant_forward, approximation=0,
                                                b_top = b_top)
            
            (xint_SH2, flux_SH2, intensity_SH2) = fl.get_reflected_new(nlevel, wno, nwno, ng, nt, 
                                        #dtau_dedd, tau_dedd, w0_dedd, g0_dedd, gcos2, ftau_cld, ftau_ray,
                                        dtau, tau, w0, g0, gcos2, ftau_cld, ftau_ray,
                                        dtau, tau, w0, g0 ,
                                        albedo, ubar0, ubar1, cos_theta, fbeam,
                                        single_phase, multi_phase,
                                        frac_a, frac_b, frac_c, constant_back, constant_forward,
                                        '2d', 2, b_top = b_top)
            
            (xint_SH4, flux_SH4, intensity_SH4) = fl.get_reflected_new(nlevel, wno, nwno, ng, nt, 
                                        #dtau_dedd4, tau_dedd4, w0_dedd4, g0_dedd4, gcos2, ftau_cld, ftau_ray,
                                        dtau, tau, w0, g0, gcos2, ftau_cld, ftau_ray,
                                        dtau, tau, w0, g0 ,
                                        albedo, ubar0, ubar1, cos_theta, fbeam,
                                        single_phase, multi_phase,
                                        frac_a, frac_b, frac_c, constant_back, constant_forward,
                                        '2d', 4, b_top = b_top)
            
            picaso_up = np.zeros(nlevel)
            picaso_down = np.zeros(nlevel)
            SH4_up = np.zeros(nlevel)
            SH4_down = np.zeros(nlevel)
            picaso_up = flux[0,0,1::2,0]
            picaso_down = flux[0,0,::2,0]
            SH2_up = flux_SH2[0,0,1::2,0]
            SH2_down = flux_SH2[0,0,::2,0]
            SH4_up = flux_SH4[0,0,2::4,0]
            SH4_down = flux_SH4[0,0,::4,0]
            picaso_net = picaso_down - picaso_up
            SH2_net = SH2_down - SH2_up
            SH4_net = SH4_down - SH4_up

            picaso_ref = picaso_up[0] / (u0 * fbeam)
            SH2_ref = SH2_up[0] / (u0 * fbeam)
            SH4_ref = SH4_up[0] / (u0 * fbeam)

            # unsure about these calculations -- should there be a pi beside fbeam?
            #picaso_trans = picaso_down[-1]/ (u0 * fbeam) + np.exp(-tau_dedd[-1]/u0)
            #SH2_trans = SH2_down[-1]/ (u0 * fbeam) + np.exp(-tau_dedd[-1]/u0)
            #SH4_trans = SH4_down[-1]/ (u0 * fbeam) + np.exp(-tau_dedd4[-1]/u0)
            picaso_trans = picaso_down[-1]/ (u0 * fbeam) + np.exp(-tauN/u0)
            SH2_trans = SH2_down[-1]/ (u0 * fbeam) + np.exp(-tauN/u0)
            SH4_trans = SH4_down[-1]/ (u0 * fbeam) + np.exp(-tauN/u0)
            print('SH4 transmission = ', SH4_trans)
            
            #headers = ['Optical','Depth','Upward','Downward','Picaso','SH2','SH4','Net']
            #print('                    <----------------------- FLUXES ----------------------->                                      ' )
            #print('{:<8s}{:>10}{:>12s}{:>12s}{:>10s}{:>12s}{:>12s}{:>10s}{:>12s}{:>12s}'.format(
            #            headers[0],headers[3],headers[3],headers[3],headers[2],headers[2],headers[2],
            #            headers[7],headers[7],headers[7]))
            #print('{:<9s}{:>8}{:>10s}{:>12s}{:>13s}{:>10s}{:>13s}{:>13s}{:>10s}{:>12s}'.format(
            #            headers[1],headers[4],headers[5],headers[6],headers[4],headers[5],headers[6],
            #            headers[4],headers[5],headers[6]))
            #print(''                                                                                      )
            #for j in range(len(tau)):
            #    print('%.4f' % tau[j,0] + '{:>12.3e}{:>12.3e}{:>12.3e}{:>12.3e}{:>12.3e}{:>12.3e}{:>12.3e}{:>12.3e}{:>12.3e}'.format(
            #                picaso_down[j], SH2_down[j], SH4_down[j], picaso_up[j], SH2_up[j], SH4_up[j],
            #                picaso_net[j], SH2_net[j], SH4_net[j])) 

            #print('\n'                                                                                                                 )
            #print(''                                                                                                                   )
            #headers = ['Optical', 'Polar Angle', 'Picaso', 'SH2', 'SH4', 'Depth', 'Cosine']
            #print('{:<10s}{:>4s}{:>10s}{:>13s}{:>16s}'.format(headers[0],headers[1],headers[2],headers[3],headers[4]))
            #print('{:<12s}{:>4s}'.format(headers[5],headers[6]))
            #print('\n')
            #for j in range(len(tau)):
            #    for i in range(len(ubar1)):
            #        if i==0:
            #            print('{:<10.4f}{:>4.4f}{:>15.3e}{:>15.3e}{:>15.3e}'.format(
            #                tau[j,0], ubar1[i,0], intensity[i,0,j,0], intensity_SH2[i,0,j,0], intensity_SH4[i,0,j,0])) 
            #        else:
            #            print('{:>17.4f}{:>15.3e}{:>15.3e}{:>15.3e}'.format(
            #                ubar1[i,0], intensity[i,0,j,0], intensity_SH2[i,0,j,0], intensity_SH4[i,0,j,0])) 
            #    print('\n')

            
            #output_filename = output_dir + 'data_test_%.3f_%.3f_%.1f.pk' % (g, w, ubar0[0][0]) 
            output_filename = output_dir + 'data_test_%.3f_%.1f_%.2f.pk' % (w, ubar0[0][0], tauN) 
            pk.dump({'xint_picaso':intensity,'xint_SH2':intensity_SH2, 'xint_SH4': intensity_SH4,
                        'flux_dwn_picaso':picaso_down,'flux_dwn_SH2':SH2_down, 'flux_dwn_SH4': SH4_down,
                        'flux_up_picaso':picaso_up,'flux_up_SH2':SH2_up, 'flux_up_SH4': SH4_up,
                        #'multiple_SH2':MS_SH2, 'single_SH2':SS_SH2, 'multiple_SH4':MS_SH4, 'single_SH4':SS_SH4,
                        #'multiple_picaso':MS_picaso, 'single_picaso':SS_picaso, 
                        'tau': tau, 
                        'picaso_ref': picaso_ref, 'SH2_ref': SH2_ref, 'SH4_ref': SH4_ref,
                        'picaso_trans': picaso_trans, 'SH2_trans': SH2_trans, 'SH4_trans': SH4_trans,
                        },
                        open(output_filename,'wb'))#, protocol=2)
