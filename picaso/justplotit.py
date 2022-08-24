from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Colorblind8
import numpy as np
import pandas as pd
from bokeh.layouts import column,row
import numpy as np

from bokeh.palettes import gray as colfun3
from bokeh.palettes import Spectral11,Category20,viridis,magma,RdBu11
from bokeh.models import HoverTool
from bokeh.models import LinearColorMapper, LogTicker,BasicTicker, ColorBar,LogColorMapper,Legend
from bokeh.models import ColumnDataSource,LinearAxis,Range1d

from bokeh.layouts import row,column
from bokeh.io import output_notebook
import os 
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.stats.stats import pearsonr  
from scipy.stats import binned_statistic

from .fluxes import blackbody
from .opacity_factory import *

def mean_regrid(x, y, newx=None, R=None):
    """
    Rebin the spectrum at a minimum R or on a fixed grid 

    Parameters
    ----------
    x : array 
        Wavenumbers
    y : array 
        Anything (e.g. albedo, flux)
    newx : array 
        new array to regrid on. 
    R : float 
        create grid with constant R

    Returns
    -------
    final x, and final y
    """
    if (isinstance(newx, type(None)) & (not isinstance(R, type(None)))) :
        newx = create_grid(1e4/max(x), 1e4/min(x), R)
    elif (not isinstance(newx, type(None)) & (isinstance(R, type(None)))) :  
        d = np.diff(newx)
        binedges = np.array([newx[0]-d[0]/2] + list(newx[0:-1]+d/2.0) + [newx[-1]+d[-1]/2])
        newx = binedges
    else: 
        raise Exception('Please either enter a newx or a R') 
    y, edges, binnum = binned_statistic(x,y,bins=newx)
    newx = (edges[0:-1]+edges[1:])/2.0

    return newx, y

def plot_errorbar(x,y,e,plot,point_kwargs={}, error_kwargs={}):
    """
    Plot only symmetric y error bars in bokeh plot

    Parameters
    ----------
    x : array 
        x data 
    y : array 
        y data 
    e : array 
        +- error for y which will be distributed as y+e, y-e on data point
    plot : bokeh.figure 
        Bokeh figure to add error bars to 
    point_kwargs : dict 
        formatting for circles 
    error_kwargs : dict 
        formatting for error bar lines
    """

    y_err = []
    x_err = []
    for px, py, yerr in zip(x, y, e):
        np.array(x_err.append((px , px )))
        np.array(y_err.append((py - yerr, py + yerr)))

    plot.multi_line(x_err, y_err, **error_kwargs)
    plot.circle(x, y, **point_kwargs)
    return

def plot_multierror(x,y,plot, dx_up=0, dx_low=0, dy_up=0, dy_low=0, 
    point_kwargs={}, error_kwargs={}):
    """
    Plot non-symmetric x and y error bars in bokeh plot

    Parameters
    ----------
    x : array 
        x data 
    y : array 
        y data 
    dx_up : array or int or float 
        upper error bar to be distributed as x + dx_up
    dx_low : array or int or float 
        lower error bar to be distributed as x + dx_low 
    dy_up : array or int or float 
        upper error bar to be distributed as y + dy_up
    dy_low : array or int or float 
        lower error bar to be distributed as y + dy_low 
    plot : bokeh.figure 
        Bokeh figure to add error bars to 
    point_kwargs : dict 
        formatting for circles 
    error_kwargs : dict 
        formatting for error bar lines
    """
    #first turn everything into lists 
    for i in [dx_up, dx_low, dy_up, dy_low]:
        if isinstance(i, (float, int)):
            i = [i]*len(x)

    #first x error
    y_err = []
    x_err = []
    for px, py, x_up, x_low in zip(x, y, dx_up, dx_low):
        np.array(x_err.append((px - x_low, px + x_up)))
        np.array(y_err.append((py, py )))

    plot.multi_line(x_err, y_err, **error_kwargs)

    #first y error
    y_err = []
    x_err = []
    for px, py, y_up, y_low in zip(x, y, dy_up, dy_low):
        np.array(x_err.append((px , px )))
        np.array(y_err.append((py - y_low, py + y_up)))

    plot.multi_line(x_err, y_err, **error_kwargs)

    plot.circle(x, y, **point_kwargs)
    return

def bin_errors(newx, oldx, dy):
    """
    Bin errors properly to account for reduction in noise 
    
    Parameters
    ----------
    newx : array 
        New x axis (either micron or wavenumber)
    oldx : array 
        Old x axis (either micron or wavenumber) 
    dy : array 
        Error bars 

    Returns
    -------
    array
        new dy
    """
    newx =[newx[0] -  np.diff(newx)[0]/2] +  list(newx[0:-1] + np.diff(newx)/2) + [newx[-1] +  np.diff(newx)[-1]/2]
    err = []
    for i in range(len(newx)-1):
        loc = np.where(((oldx>newx[i]) & (oldx<=newx[i+1])))[0]
        err += [np.sqrt(np.sum(dy[loc]**2.0))/len(dy[loc])]
    return err


def mixing_ratio(full_output,limit=50, **kwargs):
    """Returns plot of mixing ratios 

    Parameters
    ----------
    full_output : class
        picaso.atmsetup.ATMSETUP
    limit : int
        Limits the number of curves to 20. Will plot the top 20 molecules 
        with highest max(abundance). Limit must be >=3. 
    **kwargs : dict 
        Any key word argument for bokeh.figure() 
    """
    #set plot defaults
    molecules = full_output['weights'].keys()
    pressure = full_output['layer']['pressure']

    kwargs['plot_height'] = kwargs.get('plot_height',300)
    kwargs['plot_width'] = kwargs.get('plot_width',400)
    kwargs['title'] = kwargs.get('title','Mixing Ratios')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure(Bars)')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Mixing Ratio(v/v)')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log') 
    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure),np.min(pressure)])
    kwargs['x_range'] = kwargs.get('x_range',[1e-20, 1e2])

    #to plot (incl limit)
    to_plot=full_output['layer']['mixingratios'].max().sort_values(ascending=False)[0:limit].keys()

    fig = figure(**kwargs)
    if len(molecules) < 3: ncol = 5
    else: ncol = len(molecules)
    if limit<3: 
        cols = magma(5) #magma needs at least 5 colors
    else: 
        cols = magma(min([ncol,limit]))
    legend_it=[]    
    for mol , c in zip(to_plot,cols):
        ind = np.where(mol==np.array(molecules))[0][0]
        f = fig.line(full_output['layer']['mixingratios'][mol],pressure, color=c, line_width=3,
                    muted_color=c, muted_alpha=0.2)
        legend_it.append((mol, [f]))

    legend = Legend(items=legend_it, location=(0, -20))
    legend.click_policy="mute"
    fig.add_layout(legend, 'left')  

    return fig
    
def pt(full_output,ng=None, nt=None, **kwargs):
    """Returns plot of pressure temperature profile

    Parameters
    ----------
    full_output : class
        picaso.atmsetup.ATMSETUP
    **kwargs : dict 
        Any key word argument for bokeh.figure() 
    """
    #set plot defaults
    if ((ng==None) & (nt==None)):
        pressure = full_output['layer']['pressure']
        temperature = full_output['layer']['temperature']
    else: 
        pressure = full_output['layer']['pressure'][:,ng,nt]
        temperature = full_output['layer']['temperature'][:,ng,nt]

    kwargs['plot_height'] = kwargs.get('plot_height',300)
    kwargs['plot_width'] = kwargs.get('plot_width',400)
    kwargs['title'] = kwargs.get('title','Pressure-Temperature Profile')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure(Bars)')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Temperature (K)')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    #kwargs['x_axis_type'] = kwargs.get('x_axis_type','log') 
    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure),np.min(pressure)])

    fig = figure(**kwargs)

    f = fig.line(temperature,pressure,line_width=3) 
    plot_format(fig)
    return fig

def spectrum(xarray, yarray,legend=None,wno_to_micron=True, palette = Colorblind8, **kwargs):
    """Plot formated albedo spectrum

    Parameters
    ----------
    xarray : float array, list of arrays
        wavenumber or micron 
    yarray : float array, list of arrays 
        albedo or fluxes 
    legend : list of str , optional
        legends for plotting 
    wno_to_micron : bool , optional
        Converts wavenumber to micron
    palette : list,optional
        List of colors for lines. Default only has 8 colors so if you input more lines, you must
        give a different pallete 
    **kwargs : dict     
        Any key word argument for bokeh.plotting.figure()

    Returns
    -------
    bokeh plot
    """ 
    if len(yarray)==len(xarray):
        Y = [yarray]
    else:
        Y = yarray

    if wno_to_micron : 
        x_axis_label = 'Wavelength [μm]'
        def conv(x):
            return 1e4/x
    else: 
        x_axis_label = 'Wavenumber [(]cm-1]'
        def conv(x):
            return x

    kwargs['plot_height'] = kwargs.get('plot_height',345)
    kwargs['plot_width'] = kwargs.get('plot_width',1000)
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Spectrum')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label',x_axis_label)
    #kwargs['y_range'] = kwargs.get('y_range',[0,1.2])
    #kwargs['x_range'] = kwargs.get('x_range',[0.3,1])

    fig = figure(**kwargs)

    i = 0
    legend_it=[] 
    for yarray in Y:
        if isinstance(xarray, list):
            if isinstance(legend,type(None)): legend=[None]*len(xarray[0])
            for w, a,i,l in zip(xarray, yarray, range(len(xarray)), legend):
                if l == None: 
                    fig.line(conv(w),  a,  color=palette[np.mod(i, len(palette))], line_width=3)
                else:
                    f = fig.line(conv(w), a, color=palette[np.mod(i, len(palette))], line_width=3,
                                muted_color=palette[np.mod(i, len(palette))], muted_alpha=0.2)
                    legend_it.append((l, [f]))
        else: 
            if isinstance(legend,type(None)):
                fig.line(conv(xarray), yarray,  color=palette[i], line_width=3)
            else:
                f = fig.line(conv(xarray), yarray, color=palette[i], line_width=3,
                                muted_color=palette[np.mod(i, len(palette))], muted_alpha=0.2)
                legend_it.append((l, [f]))
        i = i+1

    if not isinstance(legend,type(None)):
        plt_legend = Legend(items=legend_it, location=(0, 0))
        plt_legend.click_policy="mute"
        fig.add_layout(plt_legend, 'left')

    plot_format(fig)
    return fig


def photon_attenuation(full_output, at_tau=0.5,return_output=False,igauss=0, **kwargs):
    """
    Plot breakdown of gas opacity, cloud opacity, 
    Rayleigh scattering opacity at a specified pressure level. 
    
    Parameters
    ----------
    full_output : class 
        picaso.atmsetup.ATMSETUP
    at_tau : float 
        Opacity at which to plot the cumulative opacity. 
        Default = 0.5. 
    return_output : bool 
        Return photon attenuation plot values 
    igauss : int 
        Gauss angle to plot if using correlated-k method. If not, should always be 0.
    **kwargs : dict 
        Any key word argument for bokeh.plotting.figure()

    Returns
    -------
    if return_output=False: bokeh plot
    else: bokeh plot,wave,at_pressures_gas,at_pressures_cld,at_pressures_ray
    """
    wave = 1e4/full_output['wavenumber']

    dtaugas = full_output['taugas'][:,:,igauss]
    dtaucld = full_output['taucld'][:,:,igauss]*full_output['layer']['cloud']['w0']
    dtauray = full_output['tauray'][:,:,igauss]
    shape = dtauray.shape
    taugas = np.zeros((shape[0]+1, shape[1]))
    taucld = np.zeros((shape[0]+1, shape[1]))
    tauray = np.zeros((shape[0]+1, shape[1]))

    #comptue cumulative opacity
    taugas[1:,:]=numba_cumsum(dtaugas)
    taucld[1:,:]=numba_cumsum(dtaucld)
    tauray[1:,:]=numba_cumsum(dtauray)


    pressure = full_output['level']['pressure']

    at_pressures = np.zeros(shape[1]) #pressure for each wave point

    ind_gas = find_nearest_2d(taugas, at_tau)
    ind_cld = find_nearest_2d(taucld, at_tau)
    ind_ray = find_nearest_2d(tauray, at_tau)

    at_pressures_gas = np.zeros(shape[1])
    at_pressures_cld = np.zeros(shape[1])
    at_pressures_ray = np.zeros(shape[1])

    for i in range(shape[1]):
        at_pressures_gas[i] = pressure[ind_gas[i]]
        at_pressures_cld[i] = pressure[ind_cld[i]]
        at_pressures_ray[i] = pressure[ind_ray[i]]

    kwargs['plot_height'] = kwargs.get('plot_height',300)
    kwargs['plot_width'] = kwargs.get('plot_width',1000)
    kwargs['title'] = kwargs.get('title','Pressure at 𝞽 =' +str(at_tau))
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure(Bars)')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Wavelength [μm]')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure),1e-2])

    fig = figure(**kwargs)

    legend_it = []

    f = fig.line(wave,at_pressures_gas,line_width=3, color=Colorblind8[0]) 
    legend_it.append(('Gas Opacity', [f]))
    f = fig.line(wave,at_pressures_cld,line_width=3, color=Colorblind8[3]) 
    legend_it.append(('Cloud Opacity', [f]))
    f = fig.line(wave,at_pressures_ray,line_width=3,color=Colorblind8[6]) 
    legend_it.append(('Rayleigh Opacity', [f]))

    legend = Legend(items=legend_it, location=(0, -20))
    legend.click_policy="mute"
    fig.add_layout(legend, 'right')   

    #finally add color sections 
    gas_dominate_ind = np.where((at_pressures_gas<at_pressures_cld) & (at_pressures_gas<at_pressures_ray))[0]
    cld_dominate_ind = np.where((at_pressures_cld<at_pressures_gas) & (at_pressures_cld<at_pressures_ray))[0]
    ray_dominate_ind = np.where((at_pressures_ray<at_pressures_cld) & (at_pressures_ray<at_pressures_gas))[0]

    gas_dominate = np.zeros(shape[1]) + 1e-8
    cld_dominate = np.zeros(shape[1]) + 1e-8
    ray_dominate = np.zeros(shape[1])+ 1e-8

    gas_dominate[gas_dominate_ind] = at_pressures_gas[gas_dominate_ind]
    cld_dominate[cld_dominate_ind] = at_pressures_cld[cld_dominate_ind]
    ray_dominate[ray_dominate_ind] = at_pressures_ray[ray_dominate_ind]

    if len(gas_dominate) > 0  :
        band_x = np.append(np.array(wave), np.array(wave[::-1]))
        band_y = np.append(np.array(gas_dominate), np.array(gas_dominate)[::-1]*0+1e-8)
        fig.patch(band_x,band_y, color=Colorblind8[0], alpha=0.3)
    if len(cld_dominate) > 0  :
        band_x = np.append(np.array(wave), np.array(wave[::-1]))
        band_y = np.append(np.array(cld_dominate), np.array(cld_dominate)[::-1]*0+1e-8)
        fig.patch(band_x,band_y, color=Colorblind8[3], alpha=0.3)
    if len(ray_dominate) > 0  :
        band_x = np.append(np.array(wave), np.array(wave[::-1]))
        band_y = np.append(np.array(ray_dominate), np.array(ray_dominate)[::-1]*0+1e-8)
        fig.patch(band_x,band_y, color=Colorblind8[6], alpha=0.3)

    plot_format(fig)
    if return_output: return fig ,wave,at_pressures_gas,at_pressures_cld,at_pressures_ray
    else: return fig

def plot_format(df):
    """Function to reformat plots"""
    df.xaxis.axis_label_text_font='times'
    df.yaxis.axis_label_text_font='times'
    df.xaxis.major_label_text_font_size='14pt'
    df.yaxis.major_label_text_font_size='14pt'
    df.xaxis.axis_label_text_font_size='14pt'
    df.yaxis.axis_label_text_font_size='14pt'
    df.xaxis.major_label_text_font='times'
    df.yaxis.major_label_text_font='times'
    df.xaxis.axis_label_text_font_style = 'bold'
    df.yaxis.axis_label_text_font_style = 'bold'


def plot_cld_input(nwno, nlayer, filename=None,df=None,pressure=None, wavelength=None, **pd_kwargs):
    """
    This function was created to investigate CLD input file for PICASO. 

    The plot itselfs creates maps of the wavelength dependent single scattering albedo 
    and cloud opacity and assymetry parameter as a function of altitude. 


    Parameters
    ----------
    nwno : int 
        Number of wavenumber points. For runs from Ackerman & Marley, this will always be 196. 
    nlayer : int 
        Should be one less than the number of levels in your pressure temperature grid. Cloud 
        opacity is assigned for slabs. 
    file : str , optional
        (Optional)Path to cloud input file
    df : str 
        (Optional)Dataframe of cloud input file
    wavelength : array , optional
        (Optional) this allows you to reset the tick marks to wavelengths instead of indicies 
    pressure : array, optional 
        (Optional) this allows you to reset the tick marks to pressure instead of indicies  
    pd_kwargs : kwargs
        Pandas key word arguments for `pandas.read_csv`

    Returns
    -------
    Three bokeh plots with the single scattering, optical depth, and assymetry maps
    """
    if (pressure is not None):
        pressure_label = 'Pressure (units by user)'
    else: 
        pressure_label = 'Pressure Grid, TOA ->'
    if (wavelength is not None):
        wavelength_label = 'Wavelength (units by user)'
    else: 
        wavelength_label = 'Wavenumber Grid'
    cols = magma(200)
    color_mapper = LinearColorMapper(palette=cols, low=0, high=1)

    if not isinstance(filename,type(None)):
        dat01 = pd.read_csv(filename, **pd_kwargs)
    elif not isinstance(df,type(None)):
        dat01=df

    #PLOT W0
    scat01 = np.flip(np.reshape(dat01['w0'].values,(nlayer,nwno)),0)
    xr, yr = scat01.shape
    f01a = figure(x_range=[0, yr], y_range=[0,xr],
                           x_axis_label=wavelength_label, y_axis_label=pressure_label,
                           title="Single Scattering Albedo",
                          plot_width=300, plot_height=300)


    f01a.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw =yr )

    color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
                       label_standoff=12, border_line_color=None, location=(0,0))

    f01a.add_layout(color_bar, 'left')


    #PLOT OPD
    scat01 = np.flip(np.reshape(dat01['opd'].values,(nlayer,nwno)),0)

    xr, yr = scat01.shape
    cols = viridis(200)[::-1]
    color_mapper = LogColorMapper(palette=cols, low=1e-3, high=10)


    f01 = figure(x_range=[0, yr], y_range=[0,xr],
                           x_axis_label=wavelength_label, y_axis_label=pressure_label,
                           title="Cloud Optical Depth Per Layer",
                          plot_width=300, plot_height=300)

    f01.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw =yr )

    color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                       label_standoff=12, border_line_color=None, location=(0,0))
    f01.add_layout(color_bar, 'left')

    #PLOT G0
    scat01 = np.flip(np.reshape(dat01['g0'].values,(nlayer,nwno)),0)

    xr, yr = scat01.shape
    cols = colfun3(200)[::-1]
    color_mapper = LinearColorMapper(palette=cols, low=0, high=1)


    f01b = figure(x_range=[0, yr], y_range=[0,xr],
                           x_axis_label=wavelength_label, y_axis_label=pressure_label,
                           title="Assymetry Parameter",
                          plot_width=300, plot_height=300)

    f01b.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw =yr )

    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                       label_standoff=12, border_line_color=None, location=(0,0))
    f01b.add_layout(color_bar, 'left')

    #CHANGE X AND Y AXIS TO BE PHYSICAL UNITS 
    #indexes for pressure plot 
    if (pressure is not None):
        pressure = ["{:.1E}".format(i) for i in pressure[::-1]] #flip since we are also flipping matrices
        npres = len(pressure)
        ipres = np.array(range(npres))
        #set how many we actually want to put on the figure 
        #hard code ten on each.. 
        ipres = ipres[::int(npres/10)]
        pressure = pressure[::int(npres/10)]
        #create dictionary for tick marks 
        ptick = {int(i):j for i,j in zip(ipres,pressure)}
        for i in [f01a, f01, f01b]:
            i.yaxis.ticker = ipres
            i.yaxis.major_label_overrides = ptick
    if (wavelength is not None):
        wave = ["{:.2F}".format(i) for i in wavelength]
        nwave = len(wave)
        iwave = np.array(range(nwave))
        iwave = iwave[::int(nwave/10)]
        wave = wave[::int(nwave/10)]
        wtick = {int(i):j for i,j in zip(iwave,wave)}
        for i in [f01a, f01, f01b]:
            i.xaxis.ticker = iwave
            i.xaxis.major_label_overrides = wtick       

    return row(f01a, f01,f01b)

def cloud(full_output):
    """
    Plotting the cloud input from ``picaso``. 

    The plot itselfs creates maps of the wavelength dependent single scattering albedo 
    and cloud opacity as a function of altitude. 


    Parameters
    ----------
    full_output

    Returns
    -------
    A row of two bokeh plots with the single scattering and optical depth map
    """
    cols = magma(200)
    color_mapper = LinearColorMapper(palette=cols, low=0, high=1)

    dat01 = full_output['layer']['cloud']


    #PLOT W0
    scat01 = np.flip(dat01['w0'],0)#[0:10,:]
    xr, yr = scat01.shape
    f01a = figure(x_range=[0, yr], y_range=[0,xr],
                           x_axis_label='Wavelength (micron)', y_axis_label='Pressure (bar)',
                           title="Single Scattering Albedo",
                          plot_width=300, plot_height=300)


    f01a.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)

    color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
                       label_standoff=12, border_line_color=None, location=(0,0))

    f01a.add_layout(color_bar, 'left')


    #PLOT OPD
    scat01 = np.flip(dat01['opd']+1e-60,0)

    xr, yr = scat01.shape
    cols = viridis(200)[::-1]
    color_mapper = LogColorMapper(palette=cols, low=1e-3, high=10)


    f01 = figure(x_range=[0, yr], y_range=[0,xr],
                           x_axis_label='Wavelength (micron)', y_axis_label='Pressure (bar)',
                           title="Cloud Optical Depth Per Layer",
                          plot_width=300, plot_height=300)

    f01.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)

    color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                       label_standoff=12, border_line_color=None, location=(0,0))
    f01.add_layout(color_bar, 'left')

    #PLOT G0
    scat01 = np.flip(dat01['g0']+1e-60,0)

    xr, yr = scat01.shape
    cols = colfun3(200)[::-1]
    color_mapper = LinearColorMapper(palette=cols, low=0, high=1)


    f01b = figure(x_range=[0, yr], y_range=[0,xr],
                           x_axis_label='Wavelength (micron)', y_axis_label='Pressure (bar)',
                           title="Assymetry Parameter",
                          plot_width=300, plot_height=300)

    f01b.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)

    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                       label_standoff=12, border_line_color=None, location=(0,0))
    f01b.add_layout(color_bar, 'left')

    #CHANGE X AND Y AXIS TO BE PHYSICAL UNITS 
    #indexes for pressure plot 
    pressure = ["{:.1E}".format(i) for i in full_output['layer']['pressure'][::-1]] #flip since we are also flipping matrices
    wave = ["{:.2F}".format(i) for i in 1e4/full_output['wavenumber']]
    nwave = len(wave)
    npres = len(pressure)
    iwave = np.array(range(nwave))
    ipres = np.array(range(npres))
    #set how many we actually want to put on the figure 
    #hard code ten on each.. 
    iwave = iwave[::int(nwave/10)]
    ipres = ipres[::int(npres/10)]
    pressure = pressure[::int(npres/10)]
    wave = wave[::int(nwave/10)]
    #create dictionary for tick marks 
    ptick = {int(i):j for i,j in zip(ipres,pressure)}
    wtick = {int(i):j for i,j in zip(iwave,wave)}
    for i in [f01a, f01, f01b]:
        i.xaxis.ticker = iwave
        i.yaxis.ticker = ipres
        i.xaxis.major_label_overrides = wtick
        i.yaxis.major_label_overrides = ptick


    return row(f01a, f01, f01b)

def lon_lat_to_cartesian(lon_r, lat_r, R = 1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z

def disco(full_output,wavelength,calculation='reflected'):
    """
    Plot disco ball with facets. Bokeh is not good with 3D things. So this is in matplotlib

    Parameters
    ----------
    full_output : class 
        Full output from picaso

    wavelength : list 
        Where to plot 3d facets. Can input as many wavelengths as wanted. 
        Must be a list, must be in microns. 
    calculation : str, optional 
        Default is to plot 'reflected' light but can also switch to 'thermal' if it has been computed

    """
    if calculation=='reflected':to_plot='albedo_3d'
    elif calculation=='thermal':to_plot='thermal_3d'

    if isinstance(wavelength,(float,int)): wavelength = [wavelength]

    nrow = int(np.ceil(len(wavelength)/3))
    ncol = int(np.min([3,len(wavelength)])) #at most 3 columns
    fig = plt.figure(figsize=(6*ncol,4*nrow))
    for i,w in zip(range(len(wavelength)),wavelength):
        ax = fig.add_subplot(nrow,ncol,i+1, projection='3d')
        #else:ax = fig.gca(projection='3d')
        wave = 1e4/full_output['wavenumber']
        indw = find_nearest_1d(wave,w)
        #[umg, numt, nwno] this is xint_at_top

        xint_at_top = full_output[to_plot][:,:,indw]

        latitude = full_output['latitude']  #tangle
        longitude = full_output['longitude'] #gangle

        cm = plt.cm.get_cmap('plasma')
        u, v = np.meshgrid(longitude, latitude)
        
        x,y,z = lon_lat_to_cartesian(u, v)

        ax.plot_wireframe(x, y, z, color="gray")

        sc = ax.scatter(x,y,z, c = xint_at_top.T.ravel(),cmap=cm,s=150)

        fig.colorbar(sc)
        ax.set_zlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4]
        ax.set_ylim3d(-1, 1)                    # viewrange for y-axis should be [-2,2]
        ax.set_xlim3d(-1, 1)
        ax.view_init(0, 0)
        ax.set_title(str(wave[indw])+' Microns')
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

def map(full_output,pressure=[0.1], plot='temperature', wavelength = None,igauss=0):
    """
    Plot disco ball with facets. Bokeh is not good with 3D things. So this is in matplotlib

    Parameters
    ----------
    full_output : class 
        Full output from picaso
    pressure : list 
        What pressure (in bars) to make the map on. Note: this assumes that the 
        pressure grid is the same for each ng,nt point accross the grid. 
    plot : str, optional 
        Default is to plot 'temperature' map but can also switch to any 
        3d [nlayer, nlong, nlat] or 4d [nlayer, nwave, nlong, nlat] output in full_output. You can check what is available to plot by printing:
        `print(full_output['layer'].keys()`. 
        If you are plotting something that is ALSO wavelength dependent you have to 
        also supply a single wavelength. 
    wavelength, float, optional
        This allows users to plot maps of things that are wavelength dependent, like 
        `taugas` and `taucld`. 
    igauss : int 
        Gauss point to plot if using ktables this can be greater than 0 up to ngauss-1. Otherwise, 
        This must be zero for monochromatic opacities.  
    """
    
    to_plot = explore(full_output, plot)

    if isinstance(to_plot,np.ndarray):
        if len(to_plot.shape) < 3: 
            raise Exception("The key you are search for is not a 3D matrix. This function \
                is used to plot out a map of a matrix that is [nlayer, nlong, nlat] or \
                [nlayer, nwave, nlong, nlat, ].")
        #here the four dimentions are nlayer, nwave, nlong, nlat
        elif len(to_plot.shape) == 4: 
            wave = 1e4/full_output['wavenumber']
            indw = find_nearest_1d(wave,wavelength)
            to_plot= to_plot[:,indw,:,:]
        #here the five dimentions are nlayer, nwave, nlong, nlat, gauss
        elif len(to_plot.shape) == 5: 
            wave = 1e4/full_output['wavenumber']
            indw = find_nearest_1d(wave,wavelength)
            to_plot= to_plot[:,indw,:,:,igauss]
    else:
        raise Exception ("The key you are search for is not an np.ndarray. This function \
                is used to plot out a map of an numpy.ndarray matrix that is [nlayer, nlong, nlat] or \
                [nlayer, nwave, nlong, nlat, ]")

    nrow = int(np.ceil(len(pressure)/3))
    ncol = int(np.min([3,len(pressure)])) #at most 3 columns
    fig = plt.figure(figsize=(6*ncol,4*nrow))
    for i,p in zip(range(len(pressure)),pressure):
        ax = fig.add_subplot(nrow,ncol,i+1, projection='3d')
        #else:ax = fig.gca(projection='3d')
        pressure = full_output['layer']['pressure'][:,0,0]
        indp = find_nearest_1d(np.log10(pressure),np.log10(p))
        
        to_map = to_plot[indp, :,:]

        latitude = full_output['latitude']  #tangle
        longitude = full_output['longitude'] #gangle

        cm = plt.cm.get_cmap('plasma')
        u, v = np.meshgrid(longitude, latitude)
        
        x,y,z = lon_lat_to_cartesian(u, v)

        ax.plot_wireframe(x, y, z, color="gray")

        sc = ax.scatter(x,y,z, c = to_map.T.ravel(),cmap=cm,s=150)

        fig.colorbar(sc)
        ax.set_zlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4]
        ax.set_ylim3d(-1, 1)                    # viewrange for y-axis should be [-2,2]
        ax.set_xlim3d(-1, 1)
        ax.view_init(0, 0)
        ax.set_title(str(p)+' bars')
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

def find_nearest_old(array,value):
    #small program to find the nearest neighbor in a matrix
    idx = (np.abs(array-value)).argmin(axis=0)
    return idx

def find_nearest_2d(array,value,axis=1):
    #small program to find the nearest neighbor in a matrix
    all_out = []
    for i in range(array.shape[axis]):
        ar , iar ,ic = np.unique(array[:,i],return_index=True,return_counts=True)
        idx = (np.abs(ar-value)).argmin(axis=0)
        if ic[idx]>1: 
            idx = iar[idx] + (ic[idx]-1)
        else: 
            idx = iar[idx]
        all_out+=[idx]
    return all_out

def find_nearest_1d(array,value):
    #small program to find the nearest neighbor in a matrix
    ar , iar ,ic = np.unique(array,return_index=True,return_counts=True)
    idx = (np.abs(ar-value)).argmin(axis=0)
    if ic[idx]>1: 
        idx = iar[idx] + (ic[idx]-1)
    else: 
        idx = iar[idx]
    return idx

@jit(nopython=True, cache=True)
def numba_cumsum(mat):
    """Function to compute cumsum along axis=0 to bypass numba not allowing kwargs in 
    cumsum 
    """
    new_mat = np.zeros(mat.shape)
    for i in range(mat.shape[1]):
        new_mat[:,i] = np.cumsum(mat[:,i])
    return new_mat

def spectrum_hires(wno, alb,legend=None, **kwargs):
    """Plot formated albedo spectrum

    Parameters
    ----------
    wno : float array, list of arrays
        wavenumber 
    alb : float array, list of arrays 
        albedo 
    legend : list of str 
        legends for plotting 
    **kwargs : dict     
        Any key word argument for hv.opts

    Returns
    -------
    bokeh plot
    """
    import holoviews as hv
    from holoviews.operation.datashader import datashade

    hv.extension('bokeh')

    kwargs['plot_height'] = kwargs.get('plot_height',345)
    kwargs['plot_width'] = kwargs.get('plot_width',1000)
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Albedo')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Wavelength [μm]')
    kwargs['y_range'] = kwargs.get('y_range',[0,1.2])
    kwargs['x_range'] = kwargs.get('x_range',[0.3,1])

    points_og = datashade(hv.Curve((1e4/wno, alb)))

    return points_og

def flux_at_top(full_output, plot_bb = True, R=None, pressures = [1e-1,1e-2,1e-3],ng=None, nt=None, **kwargs):
    """
    Routine to plot the OLR with overlaying black bodies. 

    Flux units are CGS = erg/s/cm^3 
    
    Parameters
    ----------

    full_output : dict 
        full dictionary output with {'wavenumber','thermal','full_output'}
    plot_bb : bool , optional 
        Default is to plot black bodies for three pressures specified by `pressures`
    R : float 
        New constant R to bin to 
    pressures : list, optional 
        Default is a list of three pressures (in bars) =  [1e-1,1e-2,1e-3]
    ng : int    
        Used for 3D calculations to select point on the sphere (equivalent to longitude point)
    nt : int    
        Used for 3D calculations to select point on the sphere (equivalent to latitude point)
    **kwargs : dict 
        Any key word argument for bokeh.figure() 
    """
    if ((ng==None) & (nt ==None)):
        pressure_all = full_output['full_output']['layer']['pressure']
        temperature_all = full_output['full_output']['layer']['temperature']
    else: 
        pressure_all = full_output['full_output']['layer']['pressure'][:,ng,nt]
        temperature_all = full_output['full_output']['layer']['temperature'][:,ng,nt]

    if not isinstance(pressures, (np.ndarray, list)): 
        raise Exception('check pressure input. It must be list or array. You can still input a single value as `pressures = [1e-3]`')

    kwargs['plot_height'] = kwargs.get('plot_height',300)
    kwargs['plot_width'] = kwargs.get('plot_width',400)
    kwargs['title'] = kwargs.get('title','Outgoing Thermal Radiation')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Flux (erg/s/cm^3)')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Wavelength [μm]')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log') 

    fig = figure(**kwargs)
    if len(pressures) < 3: ncol = 5
    else: ncol = len(pressures)
    cols = magma(ncol)

    wno = full_output['wavenumber']
    if isinstance(R,(int, float)): 
        wno, thermal = mean_regrid(wno, full_output['thermal'], R=R)
    else: 
        thermal =  full_output['thermal']
    fig.line(1e4/wno, thermal, color='black', line_width=4)

    for p,c in zip(pressures,cols): 
        ip = find_nearest_1d(pressure_all, p)
        t = temperature_all[ip]
        intensity = blackbody(t, 1/wno)[0] 
        flux = np.pi * intensity
        fig.line(1e4/wno, flux, color=c, alpha=0.5, legend_label=str(int(t))+' K at '+str(p)+' bars' , line_width=4)

    return fig

def explore(df, key):
    """Function to explore a dictionary that is THREE levels deep and return the data sitting at 
    the end of key. 

    Parameters
    ----------
    df : dict 
        Dictionary that you want to search through. 
    
    Examples
    ---------
    Consider a dictionary that has `df['layer']['cloud']['w0'] = [0,0,0]` 
    
    >>>explore(df, 'w0')
    [0,0,0]
    """
    check=[False,True,True]
    if df.get(key) is None: 
        for i in df.keys():
            try:
                if df[i].get(key) is None: 
                    for ii in df[i].keys(): 
                        try:
                            if df[i][ii].get(key) is not None:
                                return df[i][ii].get(key)
                        except AttributeError:
                            check[2] = False
                else:
                    return df[i].get(key)
            except AttributeError:
                check[1]=False
    elif df.get(key) is not None: 
        return df.get(key)
    
    if True not in check: 
            raise Exception ('The key that was entered cloud not be found within three layers of the specified dictionary')

def taumap(full_output, at_tau=1, wavelength=1,igauss=0):
    """
    Plot breakdown of gas opacity, cloud opacity, 
    Rayleigh scattering opacity at a specified pressure level. 
    
    Parameters
    ----------
    full_output : class 
        full_output from dictionary picaso output
    at_tau : float 
        Opacity at which to plot the cumulative opacity. 
        Default = 0.5. 
    igauss : int 
        Gauss point to plot if using ktables this can be greater than 0 up to ngauss-1. Otherwise, 
        This must be zero for monochromatic opacities. 
    **kwargs : dict 
        Any key word argument for bokeh.plotting.figure()

    Returns
    -------
    bokeh plot
    """ 
    all_dtau_gas = full_output['taugas'][:,:,:,:,igauss]
    all_dtau_cld = full_output['taucld'][:,:,:,:,igauss]*full_output['layer']['cloud']['w0'][:,:,:,:]
    all_dtau_ray = full_output['tauray'][:,:,:,:,igauss]

    ng = all_dtau_gas.shape[2]
    nt = all_dtau_gas.shape[3]

    map_gas = np.zeros((ng, nt))
    map_cld = np.zeros((ng, nt))
    map_ray = np.zeros((ng, nt))

    wave = 1e4/full_output['wavenumber']

    iw = find_nearest_1d(wave, wavelength)

    #build tau 1 map 
    for ig in range(ng):
        for it in range(nt):

            dtaugas = all_dtau_gas[:,iw, ig, it]
            dtaucld = all_dtau_cld[:,iw, ig, it]
            dtauray = all_dtau_ray[:,iw, ig, it]
            
            shape = len(dtaugas)

            taugas = np.zeros(shape+1)
            taucld = np.zeros(shape+1)
            tauray = np.zeros(shape+1)

            #comptue cumulative opacity
            taugas[1:]=np.cumsum(dtaugas)
            taucld[1:]=np.cumsum(dtaucld)
            tauray[1:]=np.cumsum(dtauray)


            pressure = full_output['level']['pressure'][:,ig,it]

            ind_gas = find_nearest_1d(taugas, at_tau)
            ind_cld = find_nearest_1d(taucld, at_tau)
            ind_ray = find_nearest_1d(tauray, at_tau)

            if (len(taucld[taucld == 0]) == len(taucld.shape)) : 
                ind_cld = ind_cld*0 + shape

            map_gas[ig, it] = pressure[ind_gas]
            map_cld[ig, it] = pressure[ind_cld]
            map_ray[ig, it] = pressure[ind_ray]

    #now build three panel plot 
    all_maps = [map_gas, map_cld, map_ray]
    labels = ['Molecular Opacity','Cloud Opacity','Rayleigh Opacity']

    nrow = 1
    ncol = 3 #at most 3 columns
    fig = plt.figure(figsize=(6*ncol,4*nrow))
    for i,w,l in zip(range(len(all_maps)),all_maps,labels):
        ax = fig.add_subplot(nrow,ncol,i+1, projection='3d')


        #[umg, numt, nwno] this is xint_at_top
        xint_at_top = w

        latitude = full_output['latitude']  #tangle
        longitude = full_output['longitude'] #gangle

        cm = plt.cm.get_cmap('plasma')
        u, v = np.meshgrid(longitude, latitude)
        
        x,y,z = lon_lat_to_cartesian(u, v)

        ax.plot_wireframe(x, y, z, color="gray")

        sc = ax.scatter(x,y,z, c = xint_at_top.T.ravel(),cmap=cm,s=150)

        fig.colorbar(sc)
        ax.set_zlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4]
        ax.set_ylim3d(-1, 1)                    # viewrange for y-axis should be [-2,2]
        ax.set_xlim3d(-1, 1)
        ax.view_init(0, 0)
        ax.set_title(l)
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()  

def plot_evolution(evo, y = "Teff",**kwargs):
    """
    Plot evolution of tracks. Requires input from justdoit: 

    evo = justdoit.evolution_track(mass='all',age='all')

    Parameters 
    ----------
    evo : dict 
        Output from the function justdoit.evolution_track(mass='all',age='all')
    y : str 
        What to plot on the y axis. Can be anything in the pandas that as the mass 
        attached to the end. E.g. "Teff" is an option because there exists "Teff1Mj". 
        But, age_years is not an option as it is not a function of mass. 
        Current options : [logL, Teff, grav_cgs]
    """
    kwargs['plot_height'] = kwargs.get('plot_height',400)
    kwargs['plot_width'] = kwargs.get('plot_width',500)
    kwargs['title'] = kwargs.get('title','Thermal Evolution')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label',y)
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Age(years)')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','log') 

    f = figure(**kwargs)

    lp = len(y)#used to find where mass tag starts
    evo_hot=evo['hot']
    evo_cold=evo['cold']
    source_hot = ColumnDataSource(data=dict(
        evo_hot))
    source_cold = ColumnDataSource(data=dict(
        evo_cold))

    colors = viridis(10)
    for i, ikey in enumerate(list(evo_hot.keys())[1:]):
        if y in ikey:
            mass = int(ikey[ikey.rfind(y[-1])+1:ikey.find('M')])
            icolor = mass -1
            f1 = f.line(x='age_years',y=ikey,line_width=2,
                   color=colors[icolor]
                  ,legend_label='Hot Start', source = source_hot)
            f.add_tools(HoverTool(renderers=[f1], tooltips=[('Teff',f'@Teff{ikey[lp:]} K'),
                                                            ('Age','@age_years Yrs'),
                                                            ('Gravity' , f'@grav_cgs{ikey[lp:]} cm/s2'),
                                                           ('Mass',str(mass)+" Mj")]
                                  ))#,mode='vline'
            f2 = f.line('age_years',ikey,line_width=2,
                   color=colors[icolor],
                  line_dash='dashed',legend_label='Cold Start' , source = source_cold)
            f.add_tools(HoverTool(renderers=[f2], tooltips=[('Teff',f'@Teff{ikey[lp:]} K'),
                                                            ('Age','@age_years Yrs'),
                                                            ('Gravity', f'@grav_cgs{ikey[lp:]} cm/s2'),
                                                           ('Mass',str(mass)+" Mj")]))

    color_bar = ColorBar(title='Mass (Mj)',
        color_mapper=LinearColorMapper(palette="Viridis256", 
                     low=1, high=10), 
        label_standoff=12,location=(0,0))

    f.add_layout(color_bar, 'right')
    return f
        
def heatmap_taus(out, R=0):
    """
    Plots a heatmap of the tau fields (taugas, taucld, tauray)

    Parameters
    ----------
    out : dict 
        full_ouput dictionary
    R : int 
        Resolution to bin to (if zero, no binning)
    """
    nrow = 1
    ncol = 3 #at most 3 columns
    fig = plt.figure(figsize=(6*ncol,4*nrow))
    for it, itau in enumerate(['taugas','taucld','tauray']):
        ax = fig.add_subplot(nrow,ncol,it+1)
        tau_bin = []
        for i in range(out['full_output'][itau].shape[0]):
            if R == 0 : 
                x,y = out['wavenumber'], out['full_output'][itau][i,:,0]
            else: 
                x,y = mean_regrid(out['wavenumber'],
                                  out['full_output'][itau][i,:,0], R=R)
            tau_bin += [[y]]
        tau_bin = np.array(tau_bin)
        tau_bin[tau_bin==0]=1e-100
        tau_bin = np.log10(tau_bin)[:,0,:]
        X,Y = np.meshgrid(1e4/x,out['full_output']['layer']['pressure'])
        Z = tau_bin
        pcm=ax.pcolormesh(X, Y, Z,shading='auto',cmap='RdBu_r')
        cbar=fig.colorbar(pcm, ax=ax)
        pcm.set_clim(-3.0, 3.0)
        ax.set_title(itau)
        ax.set_yscale('log')
        ax.set_ylim([1e2,1e-3])
        ax.set_ylabel('Pressure(bars)')
        ax.set_ylabel('Wavelength(um)')
        cbar.set_label('log Opacity')

def phase_snaps(allout, x = 'longitude', y = 'pressure', z='temperature',palette='RdBu_r',
    y_log=True, x_log=False,z_log=False,
    col_wrap = 3,collapse='np.mean',igauss=0):
    """

    Parameters
    ----------
    x : str 
        What to plot on the x axis options = ('longitude' or 'latitude' or 'pressure')
    y : str 
        What to plot on the y axis ('longitude' or 'latitude' or 'pressure')
    z : str 
        What to plot in the heatmap ('temperature','taugas','taucld','tauray','w0','g0','opd') 
    y_log : bool 
        Makes y axis log
    x_log : bool 
        Makes x axis log
    z_log : bool 
        Makes z axis log (colorbar)
    palette : str 
        Color pallete 
    col_wrap : int 
        Column wrap, determines number of columns to split runs into
    collapse : str or int
        Collapse lets us know how to collapse the axis, not used. For instance, if plotting 
        x=longitude, and y=pressure, with collapse=mean, it will take an average along the latitude 
        axis. If collapse=0, it will take the 0th latitude point. 
        Allowed collapse functions = np.mean, np.median, np.min, np.max
    igauss : int 
        If using k-coeff gauss points, this can be changed to get different 
        gauss quadrature points. 
    """
    allowed_xy = ['longitude','latitude','pressure']
    if x not in allowed_xy:
        raise Exception(f'Allowable x options are {allowed_xy}')

    if y not in allowed_xy:
        raise Exception(f'Allowable y options are {allowed_xy}')

    allowed_z = ['temperature','taugas','taucld','tauray','w0','g0','opd']
    if z not in allowed_z:
        raise Exception(f'Allowable z options are {allowed_z}')

    phases = list(allout.keys())
        
    nrows=int(np.ceil(len(phases) / col_wrap))
    #gs = gridspec.GridSpec(nrows, col_wrap)
    fig = plt.figure(figsize=(4*nrows, 3*col_wrap), dpi=80)

    for ind in range(len(phases)):
        
        iphase = phases[ind]
        full_output = allout[iphase]['full_output']

        xd = explore(full_output,x)#returns either longitude or latitude or pressure grid
        yd = explore(full_output,y)
        #one dimension means user has selected long/lat
        #convert to degrees
        #or in the case of pressure grab one axis for the meshgrid
        if len(xd.shape)==1:
            x_1d=xd*180/np.pi
        else: 
            x_1d=xd[:,0,0]
        #same with y 
        if len(yd.shape)==1:
            y_1d=yd*180/np.pi  
        else: 
            y_1d=yd[:,0,0]

        x_mesh,y_mesh = np.meshgrid(x_1d, y_1d)


        zd = explore(full_output,z)
        len_zd = len(zd.shape)
        #now to collapse zd to only the axes we need 
        if len_zd==3:
            #indicates [pressure x longitude x latitude ]
            to_collapse = [i for i,key in enumerate(['pressure','longitude','latitude']) if key not in [x,y]]
        elif len_zd==4:
            #indicates [pressure x wavelength x longitude x latitude ]]
            to_collapse = [i for i,key in enumerate(['pressure','wavelength','longitude','latitude']) if key not in [x,y]]
        elif len_zd==5:
            zd = zd[:,:,:,:,igauss]
            #indicates [pressure x wavelength x longitude x latitude x gauss]
            to_collapse = [i for i,key in enumerate(['pressure','wavelength','longitude','latitude']) if key not in [x,y]]

         
        allowed_collapse = ['np.mean','np.max', 'np.min', 'np.median']
        #allow users to collapse different axes with different methods
        if ((len(to_collapse)>=1) & (not isinstance(collapse, list))): 
            collapse = [collapse]*len(to_collapse)
        else: 
            assert len(collapse) == len(to_collapse), 'A list was give to collapse but it is not the same size as the number of axes that need to be collapsed.'

        count = 0
        for i,method in zip(to_collapse,collapse): 
            if ((isinstance(method , str)) & (method in allowed_collapse)):
                foo = eval(method)
                zd = foo(zd, axis=i-count);count+=1
            elif isinstance(method , int):
                #zd = zd[i-count];count+=1
                select = [':']*len(zd.shape)
                select[i-count] = str(method);count+=1
                #take the right axis if user asks for int
                zd = eval('zd['+','.join(select)+']')
            else: 
                raise Exception(f'Collapse not allowed. Choose an int or {allowed_collapse}')

        minmax = {  'z':[zd.min(), zd.max()],
                    'x': [x_mesh.min(), x_mesh.max()],
                    'y': [y_mesh.min(), y_mesh.max()]}
        #flip pressure axis
        if x=='pressure': minmax['x'] = minmax['x'][::-1]
        if y=='pressure': minmax['y'] = minmax['y'][::-1]
        
        ax = fig.add_subplot(col_wrap,nrows, ind+1)
        if z_log: 
            c = ax.pcolormesh(x_mesh, y_mesh, zd, cmap=palette, 
                          norm=colors.LogNorm(vmin=minmax['z'][0], vmax=minmax['z'][1]))#,
        else: 
            c = ax.pcolormesh(x_mesh, y_mesh, zd, cmap=palette, 
                          vmin=minmax['z'][0], vmax=minmax['z'][1])

        ax.set_title(f'Phase={int(iphase*180/np.pi)}')
        # set the limits of the plot to the limits of the data

        ax.axis([minmax['x'][0], minmax['x'][1], minmax['y'][0], minmax['y'][1]])

        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label(z)
        if y_log: ax.set_yscale('log')
        if x_log: ax.set_xscale('log')
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    fig.tight_layout()
    return fig
def phase_curve(allout, to_plot, collapse=None, R=100, palette=Spectral11,verbose=True, **kwargs):
    """
    Plots phase curves
    
    Parameters
    ----------
    allouts : dict
        picaso allouts element that comes from jdi.phase_curve
    to_plot : str 
        either fpfs_reflected, fpfs_thermal, or thermal, or albedo 
    collapse : str or float or list of float
        Allowable options to collapse wavelength axis:
        - `'np.mean'` or `np.sum`
        - float or list of float: wavelength(s) in microns (will find the nearest value to this wavelength). Must be in wavenumber range. 
    R : float 
        Resolution to regrid before finding nearest wavelength element
    palette : list
        list of hex from bokeh or other palette 
    verbose : bool 
        Print out low level warnings 
    kwargs : dict 
        Bokeh plotting kwargs for bokeh.Figure
    """
    kwargs['plot_height'] = kwargs.get('plot_height',400)
    kwargs['plot_width'] = kwargs.get('plot_width',600)
    kwargs['title'] = kwargs.get('title','Phase Curves')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label',to_plot)
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Orbital Phase')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','linear')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type','linear') 

    fig = figure(**kwargs)

    #check collapse
    if (isinstance(collapse, (float,int)) or isinstance(collapse, str)):
        collapse = [collapse]
    elif isinstance(collapse, list): 
        for i in collapse : assert isinstance(i,(float,int)), 'Can only supply list if it is a list of floats that represent the wavelength in micron.'
    else: 
        raise Exception('Collapse must either be float, str or list')
    if len(collapse)>len(palette): 
        if verbose: print('Switched color palette to accomodate more collapse input options')
        palette = magma(len(collapse))

    all_curves = np.zeros((len(allout.keys()), len(collapse)))
    all_ws = np.zeros(len(collapse))
    phases = np.array(list(allout.keys()))
    
    for i,iphase in enumerate(phases):
        for j,icol in enumerate(collapse): 
            if icol in ['np.mean', 'np.sum']:
                w,f =eval(icol)(allout[iphase]['wavenumber']), eval(icol)(allout[iphase][to_plot])
                all_curves[i,j] = f 
                all_ws[j] = w
            else: 
                w,f = mean_regrid(allout[iphase]['wavenumber'],
                                   allout[iphase][to_plot],R=R)
                iw = np.argmin(abs(1e4/w-icol)) 
                w,f = w[iw],f[iw]
                all_curves[i,j] = f
                all_ws[j] = w
    legend_it=[]  
    for i in range(len(collapse)): 
        f = fig.line(phases*180/np.pi,all_curves[:,i],line_width=3,color=palette[i],
                )
        
        legend_it.append((str(int(1e4/all_ws[i]*100)/100)+'um', [f]))

    legend = Legend(items=legend_it, location=(0, -20))
    legend.click_policy="mute"
    fig.add_layout(legend, 'left') 
        
    fig.xgrid.grid_line_alpha=0
    fig.ygrid.grid_line_alpha=0
    plot_format(fig)
    return phases, all_curves, all_ws, fig
