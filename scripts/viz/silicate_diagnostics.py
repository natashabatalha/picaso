# %%
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import virga.justdoit as vj
import virga.justplotit as cldplt
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
import xarray
from copy import deepcopy
from bokeh.plotting import show
import h5py

#1 ck tables from roxana
mh = '+000'#'+0.0' #log metallicity
CtoO = '100'#'1.0' # CtoO ratio

ck_db = f"../data/kcoeff_2020/sonora_2020_feh{mh}_co_{CtoO}.data.196"

#sonora bobcat cloud free structures file
sonora_profile_db = '../data/sonora_bobcat/structures_m+0.0'
sonora_dat_db = '../data/sonora_bobcat/structures_m+0.0'

teff = 1500
print(f"Effective temperature = {teff} K")
cl_run = jdi.inputs(calculation="browndwarf", climate = True) # start a calculation
grav = 100 # Gravity of your brown dwarf in m/s/s
cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)')) # input gravity
cl_run.effective_temp(teff) # input effective temperature

opacity_ck = jdi.opannection(ck_db=ck_db) # grab your opacities

nlevel = 91 # number of plane-parallel levels in your code
pressure_bobcat,temp_bobcat = np.loadtxt(jdi.os.path.join(
                            sonora_profile_db,f"t{teff}g{grav}nc_m0.0.dat"),
                            usecols=[1,2],unpack=True, skiprows = 1)

nofczns = 1 # number of convective zones initially
nstr_upper = 65 # top most level of guessed convective zone
nstr_deep = nlevel - 2 # this is always the case. Dont change this
nstr = np.array([0,nstr_upper,nstr_deep,0,0,0])
rfacv = 0.5

T_star =5326.6 # K, star effective temperature
logg =4.38933 #logg , cgs
metal =-0.03 # metallicity of star
r_star = 0.932 # solar radius
semi_major = 0.0486 # star planet distance, AU

cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star, 
            radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU)#opacity db, pysynphot database, temp, metallicity, logg

def twod_to_threed(arr, reps=4):
    """
    Takes in a 2D array of size (r, c) and repeats it along the last axis to make an array of size (r, c, reps).
    """
    return np.repeat(arr[:, :, np.newaxis], reps, axis=2)

print("Self-consistent run")
cl_run.inputs_climate(temp_guess=deepcopy(temp_bobcat), pressure= pressure_bobcat,
                    nstr = nstr, nofczns = nofczns , rfacv = rfacv, cloudy = "selfconsistent", mh = '0.0', 
                    CtoO = '1.0',species = ['MgSiO3'], fsed = 8.0, beta = 0.1, virga_param = 'const',
                    mieff_dir = "~/projects/clouds/virga/refrind", do_holes = False, fhole = 0.5, fthin_cld = 0.9, moistgrad = False,
                    )
out_selfconsistent = deepcopy(cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True))
# %%
df_selfconsistent = out_selfconsistent['cld_output_picaso']
all_wavenumbers = np.unique(out_selfconsistent['cld_output_picaso'].wavenumber.values)
opd_selfconsistent = df_selfconsistent[df_selfconsistent.wavenumber == all_wavenumbers[150]].opd.values

# %%
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') 
from matplotlib import colors, ticker, cm
from numpy import *
from pylab import *
import getopt
import os
import re
import glob
import pandas as pd
import matplotlib.colors as colors

import math as math
from scipy import interpolate
from matplotlib.animation import FFMpegWriter
# import imageio
from scipy import integrate
import gc

# root_path = "/Users/jjm6243/Documents/"
root_path = "./"

# sys.path.append('/home/jjm6243/picaso/')
# os.environ['picaso_refdata'] = "/home/jjm6243/picaso/reference/"
# os.environ['PYSYN_CDBS'] = "/home/jjm6243/picaso/grp/hst/cdbs/"

sys.path.append(root_path + "picaso/")
sys.path.append(root_path + "virga/")
os.environ['picaso_refdata'] = root_path + "picaso/reference/"
os.environ['PYSYN_CDBS'] = root_path + "picaso/grp/hst/cdbs/"

# print(os.listdir(os.path.join(os.getenv('PYSYN_CDBS'), 'grid', 'ck04models')))

from scipy.stats import pearsonr
from bokeh.models import LogColorMapper, ColorBar, LogTicker
from bokeh.layouts import row
from bokeh.palettes import Colorblind, viridis
from bokeh.plotting import show, figure
from bokeh.io import export_png
from bokeh.plotting import show
from bokeh.io import output_notebook
import picaso.opacity_factory as opa
from virga import justplotit as cldplt
from picaso import justplotit as jpi
from virga import justdoit as vj
from picaso import justdoit as jdi
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')
# plotting
output_notebook()


def climate_plots(load_profile, load_inputs, savepath, summarypath, case, temp_bobcat,pressure_bobcat,teff_guess_bob,
                  grav_guess_bob, temp_elfowl = None,pressure_elfowl = None,teff_guess_elf = None,
                  grav_guess_elf = None, diseq_chem = False, cloudy = False, egp = False, egp_p = None, egp_t = None,
                  egp_ad = None, egp_fnet = None, egp_td = None):
    
    # kcl_cond_p, kcl_cond_t = vj.condensation_t('KCl', 1, 2.2, pressure = load_profile['pressure'])
    h2o_cond_p, h2o_cond_t = vj.condensation_t('H2O', 1, 2.2, pressure = load_profile['pressure'])

    # ani = jpi.animate_convergence(load_profile, load_inputs, opacity_ck,
    #     molecules=['H2O','CH4','CO','NH3'])
    # ani.save('animation.gif', writer='imagemagick', fps=10)
    # ani.save("animation.mp4", dpi=300, writer=PillowWriter(fps=10))

    # plot PT Profile
    plt.figure(figsize=(8, 6))
    # plt.plot(kcl_cond_t,kcl_cond_p, 'k', label = 'KCl Condensation Curve')
    plt.semilogy(temp_bobcat,pressure_bobcat,color="b",label="Bobcat T={}, g={}".format(teff_guess_bob,grav_guess_bob))
    plt.semilogy(temp_elfowl,pressure_elfowl,color="r",label="Elf Owl T={}, g={}".format(teff_guess_elf,grav_guess_elf))
    if cloudy == True:
        plt.plot(h2o_cond_t,h2o_cond_p, 'k--')
    plt.semilogy(load_profile['temperature'],load_profile['pressure'],label="Our Run", alpha=0.8)

    plt.ylabel("Pressure [bar]")
    plt.xlabel('Temperature [K]')
    plt.xlim(0,max(load_profile['temperature'])+50)
    plt.ylim(max(load_profile['pressure']),min(load_profile['pressure']))
    plt.legend()
    # plt.title(r"T$_{\rm eff}$= 250 K, log(g)=4.5")
    plt.savefig(savepath + 'PT.png',dpi = 400, bbox_inches='tight')
    plt.close()
    # plt.show()

    # Plot adiabat
    cp_test, grad_test, dtdp_test, layer_p_test = jpi.pt_adiabat(load_profile,load_inputs)
    # find the convective zone in bobcat model
    threshold = 0.01             # Your chosen threshold for divergence
    #divergence_index = find_divergence_index(grad_test, dtdp_test, threshold)
    #if divergence_index >= 91:
    divergence_index = 90
    
    plt.figure(figsize=(8,6))
    plt.ylabel("Pressure [bar]")
    plt.xlabel(r'dT/dP vs Adiabat')
    plt.ylim(max(load_profile['pressure']),min(load_profile['pressure']))
    # plt.xlim(0,1200)
    plt.semilogy(dtdp_test, load_profile['pressure'][:-1], label = 'Model')
    plt.semilogy(grad_test, load_profile['pressure'][:-1], 'k', label = 'Adiabat')

    plt.legend()
    plt.savefig(savepath + 'adiabat.png',dpi = 400, bbox_inches='tight')
    plt.close()
    # plt.show()

    # Plot FnetIR
    plt.figure(figsize=(8,6))
    plt.ylabel("Pressure [bar]")
    plt.xlabel(r'F$_{\rm net}$/F$_{\rm net IR}$')
    plt.ylim(max(load_profile['pressure']),min(load_profile['pressure']))
    # plt.xlim(0,1200)
    plt.loglog(abs(load_profile['fnet/fnetir']),load_profile['pressure'])
    plt.axhline(load_profile['pressure'][divergence_index], color = 'k', linestyle = '--')

    # plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + 'Fnet.png',dpi = 400, bbox_inches='tight')
    plt.close()
    # plt.show()

    #Plot brightness temperature
    brightness_temp, figure= jpi.brightness_temperature(load_profile['spectrum_output'])

    fig = plt.figure(figsize=(12,6))
    plt.ylabel("Brightness Temperature [K]")
    plt.xlabel(r'Wavelength [$\mu$m]')

    wno = 1e4/load_profile['spectrum_output']['wavenumber']
    plt.xlim(min(wno),max(wno))

    plt.semilogx(wno,brightness_temp)
    plt.axhline(np.max(load_profile['spectrum_output']['full_output']['layer']['temperature']), color = 'r', label = 'Max Temperature')
    plt.axhline(np.min(load_profile['spectrum_output']['full_output']['layer']['temperature']), color = 'b', label = 'Min Temperature')
    plt.savefig(savepath + 'Brightness_temp.png',dpi = 400, bbox_inches='tight')
    plt.close()

    # Plot low res spectra
    opa_mon = jdi.opannection()
    hi_res = jdi.inputs(calculation="browndwarf") # start a calculation
    hi_res.gravity(gravity=load_inputs.inputs['planet']['gravity'], gravity_unit=u.Unit(load_inputs.inputs['planet']['gravity_unit'])) # input gravity

    hi_res.atmosphere(df=load_profile['ptchem_df'])
    if cloudy == True:
        hi_res.clouds(df=load_profile['cld_output_picaso'])
    df_spec = hi_res.spectrum(opa_mon, calculation='thermal')

    wno, fp = df_spec['wavenumber'], df_spec['thermal'] #erg/cm2/s/cm
    xmicron = 1e4/wno

    flamy = fp*1e-8  # per anstrom instead of per cm
    sp = jdi.psyn.ArraySpectrum(xmicron, flamy,
                            waveunits='um',
                            fluxunits='FLAM')
    sp.convert("um")
    sp.convert('Fnu')  # erg/cm2/s/Hz

    wno = sp.wave  # micron
    fp = sp.flux  # erg/cm2/s/Hz
    df_spec['fluxnu'] = fp
    wno,fp = jdi.mean_regrid(1e4/wno,fp, R=200)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    # plt.ylim(1e4,1e-4)
    plt.xlim(0.3,12)

    fp_W = fp/1000*2.99792458e14/(1e4/wno)**2
    plt.loglog(1e4/wno,fp_W)

    ax.set_ylabel(r'Flux [W/m$^2$/$\mu$m]')
    ax.set_xlabel(r'Wavelength [$\mu$m]')
    ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.savefig(savepath + 'spectra.png',dpi = 400, bbox_inches='tight')
    plt.close()
    # plt.show()

    with open(savepath + 'R200spectra.txt', 'w') as new:
        new.write("Wno      Flux[erg/cm2/s/Hz]       Flux[W/m2/um]" + '\n')
        for i in range(len(wno)):
            line = '    ' + str(wno[i]) + '    ' + str(fp[i]) + '    ' + str(fp_W[i])+ '\n'
            new.write(line)

    # Plot abundance profiles
    plt.figure(figsize=(8,6))
    plt.ylabel("Pressure [bar]")
    plt.xlabel('Abundance')
    plt.ylim(max(load_profile['pressure']),min(load_profile['pressure']))
    plt.xlim(1e-16,1e-1)

    plt.loglog(load_profile['ptchem_df']['H2O'],load_profile['pressure'],label="H$_2$O")
    plt.loglog(load_profile['ptchem_df']['CH4'],load_profile['pressure'],label="CH$_4$")
    plt.loglog(load_profile['ptchem_df']['CO'],load_profile['pressure'],label="CO")
    plt.loglog(load_profile['ptchem_df']['PH3'],load_profile['pressure'],label="PH$_3$")
    plt.loglog(load_profile['ptchem_df']['CO2'],load_profile['pressure'],label="CO$_2$")
    plt.loglog(load_profile['ptchem_df']['NH3'],load_profile['pressure'],label="NH$_3$")
    plt.loglog(load_profile['ptchem_df']['N2'],load_profile['pressure'],color = 'gray',label="N$_2$")

    plt.legend(ncol=2)

    # plt.title(r"T$_{\rm eff}$= 250 K, log(g)=4.5")
    # plt.tight_layout()
    plt.savefig(savepath + 'abundances.png',dpi = 400, bbox_inches='tight')
    plt.close()
    # plt.show()

    if cloudy == True:
        if diseq_chem == True:
            # Plot Cloud properties
            td_ind = zeros((91, 661))
            g0_ind = zeros((91, 661))
            w0_ind = zeros((91, 661))

            for i in range(90):
                for ii in range(661):
                    td_ind[i, ii] = load_profile['cld_output_picaso']['opd'][i*661 + ii]
                    g0_ind[i, ii] = load_profile['cld_output_picaso']['g0'][i*661 + ii]
                    w0_ind[i, ii] = load_profile['cld_output_picaso']['w0'][i*661 + ii]
        else:
            # Plot Cloud properties
            td_ind = zeros((91, 196))
            g0_ind = zeros((91, 196))
            w0_ind = zeros((91, 196))

            for i in range(90):
                for ii in range(196):
                    td_ind[i, ii] = load_profile['cld_output_picaso']['opd'][i*196 + ii]
                    g0_ind[i, ii] = load_profile['cld_output_picaso']['g0'][i*196 + ii]
                    w0_ind[i, ii] = load_profile['cld_output_picaso']['w0'][i*196 + ii]

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        ax[0].loglog(td_ind[:,55], load_profile['pressure'], label='PICASO')

        ax[0].set_ylabel('Pressure [bar]')
        ax[0].set_xlabel('Optical Depth')
        # ax[0].set_ylim(ax[0].get_ylim()[::-1])
        ax[0].set_ylim(max(load_profile['pressure']),min(load_profile['pressure']))
        ax[0].set_xlim(1e-5, max(td_ind[:,55])* 10)

        ax[1].plot(w0_ind[:,55], load_profile['pressure'], label='PICASO')

        ax[1].set_yscale('log')
        ax[1].set_ylabel('Pressure [bar]')
        ax[1].set_xlabel('Single Scattering Albedo')
        ax[1].set_ylim(max(load_profile['pressure']),min(load_profile['pressure']))

        ax[2].plot(g0_ind[:,55], load_profile['pressure'], label='PICASO')

        ax[2].set_yscale('log')
        ax[2].set_ylabel('Pressure [bar]')
        ax[2].set_xlabel('Asymmetry Parameter')
        ax[2].set_ylim(max(load_profile['pressure']),min(load_profile['pressure']))

        plt.tight_layout()
        # plt.legend()
        plt.savefig(savepath + 'cld_properties.png',dpi = 400, bbox_inches='tight')
        plt.close()
        # plt.show()

    # Plot diagnostic figure
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(22, 12))

    ax[0,0].semilogy(dtdp_test, load_profile['pressure'][:-1], label = 'Model')
    ax[0,0].semilogy(grad_test, load_profile['pressure'][:-1], 'k', label = 'Adiabat')

    # ax[0,1].semilogy(temperature[1], pressure[0], label='EGP')
    ax[0,1].semilogy(temp_bobcat,pressure_bobcat,color="b",label="Bobcat T={}, g={}".format(teff_guess_bob,grav_guess_bob))

    ax[0,1].semilogy(temp_elfowl,pressure_elfowl,color="r",label="Elf Owl T={}, g={}".format(teff_guess_elf,grav_guess_elf))
    if cloudy == True:
        ax[0,1].plot(h2o_cond_t,h2o_cond_p, 'k--')
    ax[0,1].semilogy(load_profile['temperature'],load_profile['pressure'], label = 'Our PICASO Run',alpha=0.8)

    ax[1,0].loglog(abs(load_profile['fnet/fnetir']),load_profile['pressure'])
    ax[1,0].axhline(load_profile['pressure'][divergence_index], color = 'k', linestyle = '--')

    ax[1,1].semilogx(1e4/load_profile['spectrum_output']['wavenumber'],brightness_temp)
    ax[1,1].axhline(np.max(load_profile['spectrum_output']['full_output']['layer']['temperature']), color = 'r', label = 'Max Temperature')
    ax[1,1].axhline(np.min(load_profile['spectrum_output']['full_output']['layer']['temperature']), color = 'b', label = 'Min Temperature')

    ax[0,2].loglog(load_profile['ptchem_df']['H2O'],load_profile['pressure'],label="H$_2$O")
    ax[0,2].loglog(load_profile['ptchem_df']['CH4'],load_profile['pressure'],label="CH$_4$")
    ax[0,2].loglog(load_profile['ptchem_df']['CO'],load_profile['pressure'],label="CO")
    ax[0,2].loglog(load_profile['ptchem_df']['PH3'],load_profile['pressure'],label="PH$_3$")
    ax[0,2].loglog(load_profile['ptchem_df']['CO2'],load_profile['pressure'],label="CO$_2$")
    ax[0,2].loglog(load_profile['ptchem_df']['NH3'],load_profile['pressure'],label="NH$_3$")
    ax[0,2].loglog(load_profile['ptchem_df']['N2'],load_profile['pressure'],color = 'gray',label="N$_2$")

    if cloudy == True:
        ax[1,2].loglog(td_ind[:,55], load_profile['pressure'])
        ax[1,2].set_xlim(1e-5, max(td_ind[:,55])* 10)

    # ax[0,0].set_ylim(ax[0,0].get_ylim()[::-1])
    # ax[0,1].set_ylim(ax[0,1].get_ylim()[::-1])
    # ax[1,0].set_ylim(ax[1,0].get_ylim()[::-1])

    ax[0,1].set_xlim(0,max(load_profile['temperature'])+50)
    ax[0,2].set_xlim(1e-16,1e-1)

    ax[0,0].set_ylim(max(load_profile['pressure']),min(load_profile['pressure']))
    ax[1,0].set_ylim(max(load_profile['pressure']),min(load_profile['pressure']))
    ax[0,1].set_ylim(max(load_profile['pressure']),min(load_profile['pressure']))
    ax[0,2].set_ylim(max(load_profile['pressure']),min(load_profile['pressure']))
    ax[1,2].set_ylim(max(load_profile['pressure']),min(load_profile['pressure']))
    # ax[1,1].set_ylim(max(load_profile['pressure']),min(load_profile['pressure']))

    ax[0,0].set_xlabel('dT/dP vs Adiabat')
    ax[0,1].set_xlabel('Temperature [K]')
    ax[1,0].set_xlabel(r'F$_{\rm net}$/F$_{\rm net~IR}$')
    ax[1,1].set_xlabel('Wavelength [$\mu$m]')
    ax[0,2].set_xlabel('Abundance')
    ax[1,2].set_xlabel('Optical Depth')

    ax[0,0].set_ylabel('Pressure [bar]')
    ax[0,1].set_ylabel('Pressure [bar]')
    ax[1,0].set_ylabel('Pressure [bar]')
    ax[1,1].set_ylabel('Brightness Temperature [K]')
    ax[0,2].set_ylabel("Pressure [bar]")
    ax[1,2].set_ylabel("Pressure [bar]")

    if egp == True:
        ax[0,0].semilogy(egp_ad, egp_p, label = 'EGP')
        ax[0,1].semilogy(egp_t,egp_p, label = 'EGP',alpha=0.8)
        ax[1,0].loglog(abs(egp_fnet),egp_p, label = 'EGP')
        if cloudy == True:
            ax[1,2].loglog(egp_td[:,55], egp_p[:-1], label = 'EGP')

    # plt.tight_layload_profile()
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,1].legend()
    ax[0,2].legend(ncol=2)
    plt.savefig(savepath + 'Summary.png',dpi = 400, bbox_inches='tight')
    plt.savefig(summarypath + case + '.png',dpi = 400, bbox_inches='tight')
    plt.close()
# %%
climate_plots(out_selfconsistent, cl_run, ".", ".", "first", temp_bobcat,pressure_bobcat,1500,
                  100, temp_elfowl=temp_bobcat, pressure_elfowl=pressure_bobcat)

# %%
np.save("p_temp.npy", out_selfconsistent["pressure"][1:])
np.save("opd_temp.npy", opd_selfconsistent)
# %%
import numpy as np
from matplotlib import pyplot as plt

p = np.load("p_temp.npy")
opd = np.load("opd_temp.npy")
plt.loglog(opd, p)
plt.gca().invert_yaxis()
plt.xlabel("Optical depth")
plt.ylabel("Pressure (bar)")

# %%
# PICASO time 29/10: turn parent star on, 0.05 AU separation. Should induce some heating and some cooling, which will balance things out