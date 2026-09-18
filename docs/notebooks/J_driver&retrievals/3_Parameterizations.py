# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pic312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PT, Chemistry, and Cloud Parameterizations in PICASO 
#
# Parameterizations can be used via the driver.toml specification. But, they can also be run inline directly using the parameterizations class. In this tutorial you will: 
#
# 1. Learn to use the parameterizations class to run spectra using in line Python code instead of the driver function
#
# You should already be well versed with PICASO spectra generation 

# %%
from picaso import justplotit as jpi
from picaso import justdoit as jdi
import picaso.parameterizations as pr
import numpy as np
import os

jpi.output_notebook()

# %% [markdown]
# ## Introducing the parameterizations class
#
# The purpose of this class is to preload any data, specifically cloud optical properties. In other tutorials this is used to pre load PT/chem grids. This is why the class must be established outside the main retrieval call, if you are running retrievals.

# %%
#all species whose optical properties you want to preload
virga_mieff_files = os.path.join(os.getenv('picaso_refdata'),'virga')
cloud_species = ['MgSiO3','Al2O3']
param_tools = pr.Parameterize(load_cld_optical=cloud_species,
        mieff_dir=virga_mieff_files)

# %% [markdown]
# Now we can proceed with setting picaso inputs 

# %%
pic = jdi.inputs(calculation='brown')
grav = 1000
pic.gravity(gravity=grav, gravity_unit = jdi.u.Unit('cm/s**2'))

#initialize pressure grid
nlevel = 91
pic.add_pt(P=np.logspace(-6,3,nlevel))

#add this class to param_tools
param_tools.add_class(pic)

# %% [markdown]
# ## Accessing P-T Parameterizations

# %%
df_pt_knots = param_tools.pt_knots(P_knots=[1e2,1e1,1e0,1e-1,1e-3,1e-5],
                                    T_knots=[1000, 700, 400, 300, 250, 200],
                                    interpolation='brewster')

df_pt_guillot = param_tools.pt_guillot(Teq=1000,T_int=100,logg1=-1, logKir=-1.5, alpha=0.5)

madhu_seager09_noinv = param_tools.pt_madhu_seager_09_noinversion(P_1=1e1,P_3=1e-5, T_3=1000,
                                                                  alpha_1=1 , alpha_2=0.5, beta=0.5)

madhu_seager09_inv = param_tools.pt_madhu_seager_09_inversion(P_1=1e1,P_3=1e-5, T_3=1000, alpha_1=0.5 ,
                                                              alpha_2=0.5, beta=0.5,P_2=1e-3)

save_pts = {'df_pt_knots':df_pt_knots
            ,'df_pt_guillot':df_pt_guillot
            , 'madhu_seager09_noinv':madhu_seager09_noinv
            ,'madhu_seager09_inv':madhu_seager09_inv
            }

# %%
for i in save_pts.keys():
    jpi.plt.semilogy(save_pts[i]['temperature'],save_pts[i]['pressure'],label=i)
jpi.plt.ylim([1e2,1e-6])
jpi.plt.legend()

# %% [markdown]
# ## Accessing Cloud Parameterizations

# %%
df_cld_slab = param_tools.cloud_brewster_grey(decay_type='slab',alpha=-4,ssa=1,reference_wave=1,
                                         slab_kwargs={'ptop': -1, 'dp': 1, 'reference_tau': 1})

df_cld_deck = param_tools.cloud_brewster_grey(decay_type='deck',alpha=-4,ssa=1,reference_wave=1,
                                         deck_kwargs={'ptop': -1, 'dp': 1})

df_cld_SiO2_deck = param_tools.cloud_brewster_mie('Al2O3',
                                             distribution='lognorm',lognorm_kwargs={'sigma': 1, 'lograd':1},
                                             decay_type='deck',deck_kwargs={'ptop': -1, 'dp': 1})

df_cld_SiO2_slab = param_tools.cloud_brewster_mie('Al2O3',
                                             distribution='lognorm',lognorm_kwargs={'sigma': 1, 'lograd': 1},
                                             decay_type='slab',slab_kwargs={'ptop': -1, 'dp': 1, 'reference_tau': 1})


# %%
clouds = {'slab grey':df_cld_slab,
          'deck grey':df_cld_deck,
          'slab sio2':df_cld_SiO2_slab,
          'deck sio2':df_cld_SiO2_deck}

# %%
for ikey in clouds.keys():
    print(ikey)
    nlayer = (nlevel-1)
    nwno = int(clouds[ikey].shape[0]/nlayer)
    fig = jpi.plot_cld_input(nwno, nlayer,df=clouds[ikey])

# %% [markdown]
# ## Compute Spectra Parameterizations

# %%
opa = jdi.opannection()
bd = jdi.inputs(calculation='browndwarf')
bd.gravity(gravity=grav , gravity_unit=jdi.u.Unit('cm/s**2'))
bd.atmosphere(df_pt_guillot) #just testing one out here, but feel free to loop this too!
bd.chemeq_visscher_2121(cto_absolute=0.55,log_mh=0)

output={}
for ikey in clouds.keys():
    bd.clouds(df=clouds[ikey])

    output[ikey] = bd.spectrum(opa,full_output=True)
    output[ikey]['lowres'] = jdi.mean_regrid(output[ikey]['wavenumber'],
                                             output[ikey]['thermal'],R=200)

# %%
xs = [output[ikey]['lowres'][0] for ikey in output.keys()]
ys = [output[ikey]['lowres'][1] for ikey in output.keys()]
jpi.show(jpi.spectrum(xs,ys, legend=list(output.keys()),y_axis_type='log',x_axis_type='log'))

# %%
