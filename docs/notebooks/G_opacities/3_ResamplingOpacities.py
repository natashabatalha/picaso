# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # What Resampling Do I Need for My Data??
#
# This notebook is purely used to help people understand what resampling they need for their specific dataset. It will show you:
#
# 1. Given the precision, and resolution of your data, what resampling do I need?
#
# **Note:** This notebook relies on having a R=500,000 database and therefore is **not** executable based on public data. For context, a R=500,000 database for 1-5$\mu$m is around 0.5 Tb. It is also not needed for most use cases. However, these are valuable tests for users to see in order to judge the required accuracy of a resampled opacity database

# %%
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import picaso.opacity_factory as opa_fac
jpi.output_notebook()

#tells us how long picaso will take to run
import time
import tracemalloc
import numpy as np

# %% [markdown]
# ## Test set up
#
# Now let's run a simple transmission model for each resampling:
#
# 1. LBL: 500,000
# 2. 100,000
# 3. 60,000
# 4. 20,000
# 5. 10,000
# 6. Lupu insert direct (option #2) in [this tutorial](https://natashabatalha.github.io/picaso/notebooks/10_CreatingOpacityDb.html)
#
# To determine what resolution to use for:
#
# 1. 100
# 2. 500
# 3. 1000
# 4. 3000
# 5. 5000
# 6. 10000

# %%
#this is where your opacity file should be located if you've set your environments correctly
db_filename = '/data2/picaso_dbs/R500k/all_opacities_0,3_5,3_R500k.db'
R=500000
opas = {}
# I had a 500k table on hand so let's first add samplings of
resampled_at = [500000,100000,20000,10000]
for inewr in resampled_at:
    isamp = int(R/inewr)
    opas[inewr] = jdi.opannection(filename_db=db_filename,
                                  resample=isamp,wave_range=[1,3])

#lets pull in this other one I ran for 1-5 um at 60,000 (since it wasnt a multiple of 500k :)
opas[60000] = jdi.opannection(filename_db='/data2/picaso_dbs/R60000/all_opacities_0.6_6_R60000.db',
                             wave_range=[1,3])
#and lets test the lupu "insert direct" we explored in
#the previous tutorial
opas['lupu'] = jdi.opannection(filename_db='/data2/picaso_dbs/lupu_1_3_OG_R.db')


# %% [markdown]
# Note: picaso will really yell at you for resampling your data further than what it has already done. This is because as you will see below, it strongly affects the accuracy of your calculations.
#
# ## Simple Transit Model
#
# - Mixture of H2, H2O, CH4
# - Basic Hot Jupiter

# %%
log_g = 4.38933
metallicity = -0.03
t_eff =  5326.6 #K
r_star =0.932#895 #rsun#josh=
m_star = 0.934 #msun
m_planet = 0.281 #mjup
r_planet = 1.279 #rjup

outs = {}

for ikey in opas.keys():
    tracemalloc.start()
    start = time.time()
    pl=jdi.inputs()

    pl.star(opas[ikey],
            t_eff,metallicity,log_g,radius=r_star,
            radius_unit = jdi.u.Unit('R_sun') )

    pl.gravity(mass=m_planet, mass_unit=jdi.u.Unit('M_jup'),
              radius=r_planet, radius_unit=jdi.u.Unit('R_jup'))

    pl.approx(p_reference=10)

    df = {'pressure':np.logspace(-7,2,40)}
    df['temperature'] = np.logspace(-7,2,40)*0 + 500
    df['H2O'] = np.logspace(-7,2,40)*0 + 1e-4
    df['CH4'] = np.logspace(-7,2,40)*0 + 1e-4
    df['H2'] = np.logspace(-7,2,40)*0 + 1-2e-4

    pl.atmosphere(df=jdi.pd.DataFrame(df))
    outs[ikey] = pl.spectrum(opas[ikey], calculation='transmission')
    mem = tracemalloc.get_traced_memory()
    print("Resampling: ", ikey, 'Took (s):',(time.time()-start)/60,
         'Peak Memory:', mem)


# %% [markdown]
# ## Regridding to Data Resolution
#
# Here we will regrid everything to various resolutions (100,500,1000,3000) so that the user can see how this shapes the spectra

# %%
#medium to low resolution tests
r_test_low = [100,500,1000,3000]
rebin = {isamp:{} for isamp in opas.keys()}
for isamp in opas.keys():
    for iR in r_test_low:
        w,f = jdi.mean_regrid(outs[isamp]['wavenumber'],outs[isamp]['transit_depth']
                              , R=iR)
        rebin[isamp][iR] = [w,f]

#high resolution resolution tests
r_test_hi = [10000, 50000]
for iR in r_test_hi:
    for isamp in [500000, 100000,'lupu']:
        if isamp==500000:
            w,f = jdi.mean_regrid(outs[isamp]['wavenumber'],outs[isamp]['transit_depth']
                      , R=iR)
        else:
            w,f = jdi.mean_regrid(outs[isamp]['wavenumber'],outs[isamp]['transit_depth']
                      , newx=w)
        rebin[isamp][iR] = [w,f]


# %%
for iR in r_test_low:
    w,f,l=[],[],[]
    w+=[ rebin[500000][iR][0]]
    f+=[ 1e6*(rebin[500000][iR][1]-np.min(rebin[500000][iR][1]))]
    l+=['Line-by-Line at 500k']
    for isamp in [i for i in opas.keys() if i not in [500000]]:
        w+=[rebin[isamp][iR][0]]
        f+=[1e6*(rebin[isamp][iR][1]-np.min(rebin[isamp][iR][1]))]
        l+=['Resampled R='+str(isamp)]
    jpi.show(jpi.spectrum(w,f,plot_width=600,legend=l,muted_alpha=0,
                          title=f'Data is R={iR}'), y_axis_label='Spectrum(ppm)')

for iR in r_test_hi:
    w,f,l=[],[],[]
    w+=[ rebin[500000][iR][0]]
    f+=[ 1e6*(rebin[500000][iR][1]-np.min(rebin[500000][iR][1]))]
    l+=['Line-by-Line at 500k']
    for isamp in [100000,'lupu']:
        w+=[rebin[isamp][iR][0]]
        f+=[1e6*(rebin[isamp][iR][1]-np.min(rebin[isamp][iR][1]))]
        l+=['Resampled R='+str(isamp)]
    jpi.show(jpi.spectrum(w,f,plot_width=600,legend=l, muted_alpha=0,
                          title=f'Data is R={iR}', y_axis_label='Spectrum(ppm)'))

# %% [markdown]
# ## Plot differences

# %%
errs = {isamp:{} for isamp in list(opas.keys())[1:]}
for iR in r_test_low:
    w,f,l=[],[],[]
    for isamp in list(opas.keys())[1:]:
        w+=[rebin[isamp][iR][0]]
        err = 1e6*(rebin[isamp][iR][1] - rebin[500000][iR][1])
        f+=[err]
        l+=['Resampled R='+str(isamp)]
        errs[isamp][iR] = np.std(err)
    jpi.show(jpi.spectrum(w,f,plot_width=600,legend=l,
                          #y_range=[10,10],
                          title=f'Data is R={iR}',
                        y_axis_label='Delta LBL-Resampled(ppm)')
            )
for iR in r_test_hi:
    w,f,l=[],[],[]
    for isamp in [100000,'lupu']:
        w+=[rebin[isamp][iR][0]]
        err = 1e6*(rebin[isamp][iR][1] - rebin[500000][iR][1])
        f+=[err]
        l+=['Resampled R='+str(isamp)]
        errs[isamp][iR] = np.std(err)
    jpi.show(jpi.spectrum(w,f,plot_width=600,legend=l,
                          title=f'Data is R={iR}',
                          y_axis_label='Delta LBL-Resampled(ppm)')
                        )

# %% [markdown]
# ## Takeaways
#
# 1. Largest "error" from resampling comes in the window regions of opacity
# 2. Your resampling is really dependent on what level of data precision you expect
# 3. Errors associated with under sampling can be seen in the spectra as "jaggedness". These should not be mistaken for features.

# %%
err_fig = jpi.figure(x_axis_type='log', height=400, width=500,
                    x_axis_label='Data Resolution',
                    y_axis_label='1 Sigma Error (ppm)')
for i,isamp in enumerate(errs.keys()):
    x = list(errs[isamp].keys())
    y = list(errs[isamp].values())
    err_fig.line(x,y,line_width=2, color=jpi.Colorblind8[i],legend_label=f'Resampling={isamp}')
jpi.show(err_fig)

# %%
