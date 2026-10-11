import pandas as pd
import numpy as np

from bokeh.palettes import Colorblind8
import bokeh.palettes as pals
from bokeh.models import HoverTool
from bokeh.models import LinearColorMapper, LogTicker,BasicTicker, ColorBar,LogColorMapper,Legend
from bokeh.models import ColumnDataSource,LinearAxis,Range1d
from bokeh.layouts import row,column,gridplot
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show
Colorblind8 = pals.Colorblind8

import os 
import warnings

import astropy.units as u
import astropy.constants as const
import copy
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import rc

from scipy.stats import pearsonr, binned_statistic

from .fluxes import blackbody, get_transit_1d
from .opacity_factory import *
from .climate import convec, namedtuple, calculate_atm

# plotly is an optional dependency, so it is only imported when backend='plotly' is requested
BACKENDS = ('bokeh', 'matplotlib', 'plotly')
# max number of wavelength points drawn in bokeh/plotly heatmaps (matplotlib is unaffected)
MAX_HEATMAP_COLUMNS = 500

def _check_backend(backend):
    """Make sure the requested plotting backend is supported"""
    if backend not in BACKENDS:
        raise ValueError(f"backend='{backend}' is not supported. Options are {BACKENDS}")
    return backend

def _plotly():
    """Import plotly only when it is needed so that it remains optional"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("backend='plotly' requires plotly. Install it with `pip install plotly`")
    return go, make_subplots

def _bokeh_size(kwargs, width, height):
    """Pop bokeh's legacy plot_width/plot_height and set width/height defaults"""
    kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', height))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', width))
    kwargs.pop('plot_width', None)
    kwargs.pop('plot_height', None)
    return kwargs

def _mpl_format(ax, kwargs):
    """Translate the common bokeh.figure keywords (labels, ranges, axis types, title)
    so that the same kwargs can be used with backend='matplotlib'"""
    if kwargs.get('x_axis_type') == 'log': ax.set_xscale('log')
    if kwargs.get('y_axis_type') == 'log': ax.set_yscale('log')
    if isinstance(kwargs.get('x_range'), (list, tuple)): ax.set_xlim(kwargs['x_range'])
    if isinstance(kwargs.get('y_range'), (list, tuple)): ax.set_ylim(kwargs['y_range'])
    if kwargs.get('x_axis_label') is not None: ax.set_xlabel(kwargs['x_axis_label'])
    if kwargs.get('y_axis_label') is not None: ax.set_ylabel(kwargs['y_axis_label'])
    if kwargs.get('title') is not None: ax.set_title(kwargs['title'])

def _plotly_format(fig, kwargs):
    """Translate the common bokeh.figure keywords (labels, ranges, axis types, title, size)
    so that the same kwargs can be used with backend='plotly'"""
    for axis, update in zip(['x', 'y'], [fig.update_xaxes, fig.update_yaxes]):
        log = kwargs.get(f'{axis}_axis_type') == 'log'
        rng = kwargs.get(f'{axis}_range')
        axis_kwargs = {'title_text': kwargs.get(f'{axis}_axis_label')}
        if log: axis_kwargs['type'] = 'log'
        if isinstance(rng, (list, tuple)):
            #plotly expects log axis ranges in log10 units. a reversed range flips the axis
            axis_kwargs['range'] = list(np.log10(rng)) if log else list(rng)
        update(**axis_kwargs)
    fig.update_layout(title=kwargs.get('title'), width=kwargs.get('width'), height=kwargs.get('height'))

def _bokeh_image(x, y, z, palette, low=None, high=None, log_color=False, log_y=True, **kwargs):
    """
    Bokeh heatmap of z[y,x]. Bokeh images assume evenly spaced pixels, so z is resampled
    (nearest neighbor) onto an evenly spaced x grid and an evenly spaced (log) y grid
    before it is drawn. Pressure is plotted increasing downward.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    ix = np.argsort(x); iy = np.argsort(y)
    x, y, z = x[ix], y[iy], z[iy][:, ix]

    x_even = np.linspace(x.min(), x.max(), min(len(x), MAX_HEATMAP_COLUMNS))
    if log_y:
        y_even = np.logspace(np.log10(y.min()), np.log10(y.max()), len(y))
        y_near = [find_nearest_1d(np.log10(y), np.log10(i)) for i in y_even]
    else:
        y_even = np.linspace(y.min(), y.max(), len(y))
        y_near = [find_nearest_1d(y, i) for i in y_even]
    x_near = [find_nearest_1d(x, i) for i in x_even]
    z = z[np.ix_(y_near, x_near)]

    if log_color:
        positive = z[z > 0]
        low = positive.min() if low is None else low
        high = positive.max() if high is None else high
        mapper = LogColorMapper(palette=palette, low=low, high=high)
    else:
        mapper = LinearColorMapper(palette=palette,
                                   low=np.nanmin(z) if low is None else low,
                                   high=np.nanmax(z) if high is None else high)

    kwargs['y_axis_type'] = kwargs.get('y_axis_type', 'log' if log_y else 'linear')
    kwargs['y_range'] = kwargs.get('y_range', [y.max(), y.min()])
    kwargs['x_range'] = kwargs.get('x_range', [x.min(), x.max()])
    fig = figure(**kwargs)
    fig.image(image=[z], x=x.min(), y=y.min(), dw=x.max()-x.min(), dh=y.max()-y.min(),
              color_mapper=mapper)
    fig.add_layout(ColorBar(color_mapper=mapper, label_standoff=8), 'right')
    return fig

def _thin_columns(x, z, n=None):
    """
    Sort by x and keep at most n (default MAX_HEATMAP_COLUMNS) evenly spaced columns of z[:, x]. 
    Interactive heatmaps embed every point in the notebook, so this keeps them small. 
    """
    n = MAX_HEATMAP_COLUMNS if n is None else n
    x = np.asarray(x); z = np.asarray(z)
    order = np.argsort(x)
    keep = order[np.unique(np.linspace(0, len(x)-1, min(len(x), n)).astype(int))]
    return x[keep], z[:, keep]

def _plotly_colorbar_x(fig, col):
    """x position just right of subplot `col` so each heatmap gets its own colorbar"""
    axis = 'xaxis' if col == 1 else f'xaxis{col}'
    return fig.layout[axis].domain[1] + 0.01


def _get_legend_cols(n_items, plot_height):
    """
    Calculate the number of columns needed for a legend to fit within the plot height.
    This is only supported in bokeh >= 3.1.0
    """
    max_items_per_col = max(1, (plot_height - 17) // 23)
    ncols = int(np.ceil(n_items / max_items_per_col))
    return {'ncols': max([1,ncols])}

def mean_regrid(x, y, newx=None, R=None):
    """
    Rebin the spectrum at a minimum R or on a fixed grid! 

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
    binwidths : array 
        bin widths centered at x 

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

def plot_errorbar(x,y,e,plot=None,point_kwargs={}, error_kwargs={},
    plot_type='bokeh', plot_kwargs={}, backend=None):
    """
    Plot only symmetric y error bars

    Parameters
    ----------
    x : array
        x data
    y : array
        y data
    e : array
        +- error for y which will be distributed as y+e, y-e on data point
    plot : bokeh.figure or plotly.graph_objects.Figure, optional
        Figure to add error bars to
    plot_type : str, optional
        type of plot (either bokeh, matplotlib or plotly)
    point_kwargs : dict
        formatting for circles
    error_kwargs : dict
        formatting for error bar lines
    plot_kwargs : dict
        plot attriutes for bokeh figure
    backend : str, optional
        Same as plot_type. If supplied, it takes precedence over plot_type
    """
    if backend is not None: plot_type = backend
    #copy so the mutable defaults are not modified between calls
    point_kwargs, error_kwargs, plot_kwargs = dict(point_kwargs), dict(error_kwargs), dict(plot_kwargs)
    if plot_type=='bokeh':
        if isinstance(plot, type(None)):
            plot_kwargs['height'] = plot_kwargs.get('plot_height',plot_kwargs.get('height',345))
            plot_kwargs['width'] = plot_kwargs.get('plot_width', plot_kwargs.get('width',1000))
            if 'plot_width' in plot_kwargs.keys() : plot_kwargs.pop('plot_width')
            if 'plot_height' in plot_kwargs.keys() : plot_kwargs.pop('plot_height')
            plot_kwargs['y_axis_label'] = plot_kwargs.get('y_axis_label','Spectrum')
            plot_kwargs['x_axis_label'] = plot_kwargs.get('x_axis_label','Wavelength')
            plot = figure(**plot_kwargs)
        y_err = []
        x_err = []
        for px, py, yerr in zip(x, y, e):
            np.array(x_err.append((px , px )))
            np.array(y_err.append((py - yerr, py + yerr)))

        plot.multi_line(x_err, y_err, **error_kwargs)
        plot.scatter(x, y, **point_kwargs)
        return plot
    elif plot_type=='matplotlib':
        point_kwargs['color'] = point_kwargs.get('color','k')

        plot_kwargs['xlabel'] = plot_kwargs.get('xlabel',r'Wavelength [$\mu$m]')
        plot_kwargs['ylabel'] = plot_kwargs.get('ylabel',r'(R$_p$/R$_*$)$^2$')
        plot_kwargs['figsize'] = plot_kwargs.get('figsize',(20,10))
        plot_kwargs['fontsize'] = plot_kwargs.get('fontsize',25)


        point_kwargs.get('color','k')
        plt.figure(figsize=plot_kwargs['figsize'])
        plt.errorbar(x,y,e,**point_kwargs)
        plt.xlabel(plot_kwargs['xlabel'],fontsize=plot_kwargs['fontsize'])
        plt.ylabel(plot_kwargs['ylabel'],fontsize=plot_kwargs['fontsize'])
        plt.minorticks_on()
        plt.tick_params(axis='y',which='major',length =20, width=3,direction='in',labelsize=20)
        plt.tick_params(axis='y',which='minor',length =10, width=2,direction='in',labelsize=20)
        plt.tick_params(axis='x',which='major',length =20, width=3,direction='in',labelsize=20)
        plt.tick_params(axis='x',which='minor',length =10, width=2,direction='in',labelsize=20)
        return
    elif plot_type=='plotly':
        go, _ = _plotly()
        fig = plot if plot is not None else go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=point_kwargs,
                                 error_y=dict(type='data', array=e, visible=True, **error_kwargs)))
        #only label axes if this is a new figure, or the user asked for it
        x_label = plot_kwargs.get('x_axis_label', plot_kwargs.get('xlabel', None if plot is not None else 'Wavelength'))
        y_label = plot_kwargs.get('y_axis_label', plot_kwargs.get('ylabel', None if plot is not None else 'Spectrum'))
        if x_label is not None: fig.update_xaxes(title_text=x_label)
        if y_label is not None: fig.update_yaxes(title_text=y_label)
        if plot is None:
            fig.update_layout(height=plot_kwargs.get('height', plot_kwargs.get('plot_height', 400)),
                              width=plot_kwargs.get('width', plot_kwargs.get('plot_width', 600)))
        return fig
    else:
        _check_backend(plot_type)

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
    (dx_up, dx_low, dy_up, dy_low) = [[i]*len(x)
                                      if isinstance(i, (float, int)) else i
                                      for i in [dx_up, dx_low, dy_up, dy_low]]


    #first x error
    y_err = []
    x_err = []
    for px, py, x_up, x_low in zip(x, y, dx_up, dx_low):
        np.array(x_err.append((px - x_low, px + x_up)))
        np.array(y_err.append((py, py )))

    plot.multi_line(x_err, y_err, **error_kwargs)

    #then y error
    y_err = []
    x_err = []
    for px, py, y_up, y_low in zip(x, y, dy_up, dy_low):
        np.array(x_err.append((px , px )))
        np.array(y_err.append((py - y_low, py + y_up)))

    plot.multi_line(x_err, y_err, **error_kwargs)

    plot.scatter(x, y, **point_kwargs)
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


def mixing_ratio(full_output,limit=50,ng=None,nt=None, plot_type='bokeh',
        molecules=None, backend=None,
        **kwargs):
    """Returns plot of mixing ratios

    Parameters
    ----------
    full_output : class
        picaso.atmsetup.ATMSETUP
    limit : int
        Limits the number of curves to 20. Will plot the top 20 molecules
        with highest max(abundance). Limit must be >=3.
    molecules : list
        Directly specify which molecule to plot
    plot_type : str
        Options are "bokeh", "matplotlib" or "plotly". Default is bokeh
    backend : str
        Same as plot_type. If supplied, it takes precedence over plot_type
    **kwargs : dict
        Any key word argument for bokeh.figure()
    """
    backend = plot_type if backend is None else backend
    return Analyzer(full_output, backend=backend).mixing_ratio(limit=limit, ng=ng, nt=nt,
                                                               molecules=molecules, **kwargs)
    
def pt(full_output,ng=None, nt=None, backend='bokeh', **kwargs):
    """Returns plot of pressure temperature profile

    Parameters
    ----------
    full_output : class
        picaso.atmsetup.ATMSETUP
    backend : str
        Options are "bokeh" (default), "matplotlib" or "plotly"
    **kwargs : dict
        Any key word argument for bokeh.figure()
    """
    return Analyzer(full_output, backend=backend).pt(ng=ng, nt=nt, **kwargs)

def spectrum(xarray, yarray,legend=None,wno_to_micron=True, palette = Colorblind8,muted_alpha=0.2,
    backend='bokeh', **kwargs):
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
    muted_alpha : float
        number 0-1 to indicate how muted you want the click functionaity (bokeh only)
    backend : str, optional
        Plotting library to use: 'bokeh' (default), 'matplotlib', or 'plotly'
    **kwargs : dict
        Any key word argument for bokeh.plotting.figure(). For matplotlib and plotly,
        the common ones (height, width, title, x/y_axis_label, x/y_axis_type, x/y_range) are
        translated, and the rest are ignored.

    Returns
    -------
    bokeh, matplotlib or plotly figure
    """
    _check_backend(backend)
    if len(yarray)==len(xarray):
        Y = [yarray]
    else:
        Y = yarray

    if wno_to_micron :
        x_axis_label = 'Wavelength [μm]'
        def conv(x):
            return 1e4/x
    else:
        x_axis_label = 'Wavenumber [cm-1]'
        def conv(x):
            return x
    if isinstance(legend, str): legend=[legend]
    kwargs = _bokeh_size(kwargs, 1000, 345)
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Spectrum')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label',x_axis_label)

    if backend == 'bokeh':
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
                                    muted_color=palette[np.mod(i, len(palette))], muted_alpha=muted_alpha)
                        legend_it.append((l, [f]))
            else:
                if isinstance(legend,type(None)):
                    fig.line(conv(xarray), yarray,  color=palette[i], line_width=3)
                else:
                    f = fig.line(conv(xarray), yarray, color=palette[i], line_width=3,
                                    muted_color=palette[np.mod(i, len(palette))], muted_alpha=muted_alpha)
                    legend_it.append((legend[i], [f]))
            i = i+1

        if not isinstance(legend,type(None)):
            plt_legend = Legend(items=legend_it, location=(0, 0),
                            **_get_legend_cols(len(legend_it), kwargs.get('height')))

            plt_legend.click_policy="mute"
            fig.add_layout(plt_legend, 'left')

        plot_format(fig)
        return fig

    #matplotlib and plotly: collect (x, y, label, color) for each line first
    lines = []
    for i, yarray in enumerate(Y):
        if isinstance(xarray, list):
            labels = legend if legend is not None else [None]*len(xarray)
            for j, (w, a, l) in enumerate(zip(xarray, yarray, labels)):
                lines.append((conv(w), a, l, palette[np.mod(j, len(palette))]))
        else:
            l = legend[i] if legend is not None else None
            lines.append((conv(xarray), yarray, l, palette[np.mod(i, len(palette))]))

    if backend == 'matplotlib':
        fig, ax = plt.subplots(figsize=(kwargs['width']/100, kwargs['height']/100))
        for x, y, l, c in lines:
            ax.plot(x, y, color=c, label=l, linewidth=3)
        _mpl_format(ax, kwargs)
        if legend is not None: ax.legend()
        return fig

    elif backend == 'plotly':
        go, _ = _plotly()
        fig = go.Figure()
        for x, y, l, c in lines:
            fig.add_trace(go.Scatter(x=x, y=y, name=l, mode='lines', showlegend=l is not None,
                                     line=dict(color=c, width=3)))
        _plotly_format(fig, kwargs)
        return fig


def photon_attenuation(full_output, at_tau=0.5,return_output=False,igauss=0, backend='bokeh', **kwargs):
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
    backend : str
        Options are "bokeh" (default), "matplotlib" or "plotly"
    **kwargs : dict
        Any key word argument for bokeh.plotting.figure()

    Returns
    -------
    if return_output=False: plot
    else: plot,wave,at_pressures_gas,at_pressures_cld,at_pressures_ray
    """
    return Analyzer(full_output, backend=backend).photon_attenuation(at_tau=at_tau,
                    return_output=return_output, igauss=igauss, **kwargs)

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
        pressure_label = 'Pressure (bars)'
        yaxis = 'log'
    else: 
        pressure_label = 'Pressure Index Grid, TOA ->'
        pressure = np.array(range(nlayer))
        yaxis = 'linear'
    if (wavelength is not None):
        wavelength_label = 'Wavelength (um)'
        wave=wavelength
    else: 
        wavelength_label = 'Wavenumber Index Grid'
        wave = np.array(range(nwno))

    if not isinstance(filename,type(None)):
        dat01 = pd.read_csv(filename, **pd_kwargs)
    elif not isinstance(df,type(None)):
        dat01=df

    #PLOT W0
        
    w0 = np.reshape(dat01['w0'].astype(float).values,(nlayer,nwno))
    opd = np.reshape(dat01['opd'].astype(float).values,(nlayer,nwno)) + 1e-60
    g0 = np.reshape(dat01['g0'].astype(float).values,(nlayer,nwno))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot W0
    im1 = ax1.pcolormesh(wave, pressure, w0,  cmap='magma', vmin=0, vmax=1)
    ax1.set_title("Single Scattering Albedo")
    fig.colorbar(im1, ax=ax1)

    # Plot OPD
    im2 = ax2.pcolormesh(wave, pressure, opd,  cmap='viridis_r', norm=colors.LogNorm(vmin=1e-3, vmax=10))
    ax2.set_title("Cloud Optical Depth Per Layer")
    fig.colorbar(im2, ax=ax2)

    # Plot G0
    im3 = ax3.pcolormesh(wave, pressure, g0,  cmap='gray_r', vmin=0, vmax=1)
    ax3.set_title("Asymmetry Parameter")
    fig.colorbar(im3, ax=ax3)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(wavelength_label)
        ax.set_ylabel(pressure_label)
        if yaxis=='log': ax.set_yscale('log')
        ax.set_ylim(pressure.max(), pressure.min())

    return fig


def cloud(full_output, backend='matplotlib'):
    """
    Plotting the cloud input from ``picaso``.

    The plot itselfs creates maps of the wavelength dependent single scattering albedo
    and cloud opacity as a function of altitude.


    Parameters
    ----------
    full_output
    backend : str
        Options are "matplotlib" (default), "bokeh" or "plotly"

    Returns
    -------
    A figure with three subplots showing heatmaps of single scattering, optical depth, and asymmetry.
    """
    return Analyzer(full_output, backend=backend).cloud()

def lon_lat_to_cartesian(lon_r, lat_r, R = 1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z

def disco(full_output,wavelength,calculation='reflected', backend='matplotlib'):
    """
    Plot disco ball with facets.

    Parameters
    ----------
    full_output : class
        Full output from picaso

    wavelength : list
        Where to plot 3d facets. Can input as many wavelengths as wanted.
        Must be a list, must be in microns.
    calculation : str, optional
        Default is to plot 'reflected' light but can also switch to 'thermal' if it has been computed
    backend : str
        Options are "matplotlib" (default, shows the plot and returns None),
        "plotly" (3D, returns figure) or "bokeh" (2D latitude-longitude maps, returns figure)
    """
    fig = Analyzer(full_output, backend=backend).disco(wavelength, calculation=calculation)
    if backend == 'matplotlib':
        plt.show()
        return
    return fig

def map(full_output,pressure=[0.1], plot='temperature', wavelength = None,igauss=0, backend='matplotlib'):
    """
    Plot disco ball with facets.

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
    backend : str
        Options are "matplotlib" (default, shows the plot and returns None),
        "plotly" (3D, returns figure) or "bokeh" (2D latitude-longitude maps, returns figure)
    """
    fig = Analyzer(full_output, backend=backend).map(pressure=pressure, plot=plot,
                                                    wavelength=wavelength, igauss=igauss)
    if backend == 'matplotlib':
        plt.show()
        return
    return fig

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

def flux_at_top(full_output, plot_bb = True, R=None, pressures = [1e-1,1e-2,1e-3],ng=None, nt=None,
    backend='bokeh', **kwargs):
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
    backend : str
        Options are "bokeh" (default), "matplotlib" or "plotly"
    **kwargs : dict
        Any key word argument for bokeh.figure()
    """
    return Analyzer(full_output, backend=backend).flux_at_top(plot_bb=plot_bb, R=R,
                    pressures=pressures, ng=ng, nt=nt, **kwargs)

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

def taumap(full_output, at_tau=1, wavelength=1,igauss=0, backend='matplotlib'):
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
    backend : str
        Options are "matplotlib" (default, shows the plot and returns None),
        "plotly" (3D, returns figure) or "bokeh" (2D latitude-longitude maps, returns figure)
    """
    fig = Analyzer(full_output, backend=backend).taumap(at_tau=at_tau, wavelength=wavelength, igauss=igauss)
    if backend == 'matplotlib':
        plt.show()
        return
    return fig

def transit_geometry(full_output, plane='day-night', sector=0, zenith=None, plot='temperature',
    unit='R_jup', cmap='RdYlBu_r', ax=None, **kwargs):
    """
    Plot the columns of a 3D transmission calculation (to scale), colored by temperature
    or a molecule's mixing ratio. Use this to check the geometry specified in
    `transit_geometry`. Reproduces MacDonald & Lewis (2022) Figure 3 and Figure 9 (right).
    Requires `spectrum(...,calculation='transmission', dimension='3d', full_output=True)`.

    Parameters
    ----------
    full_output : dict
        full_output from dictionary picaso output
    plane : str
        'day-night' (Default) plots the slice through the planet containing the observer
        line of sight, for the azimuthal sector `sector` (top) and the opposite sector (bottom).
        The star is to the left and the observer is to the right.
        'terminator' plots the plane of the sky as seen by the observer, for the zenith
        slice `zenith`. The leading limb (morning terminator) is to the right.
    sector : int
        Azimuthal sector index to plot for plane='day-night'. Default=0
    zenith : int
        Zenith slice index to plot for plane='terminator'. Default is the slice on the
        dayside of the terminator plane.
    plot : str
        'temperature' (Default), or the name of a molecule to plot its mixing ratio
    unit : str
        Any astropy length unit for the axes. Default='R_jup'
    cmap : str
        Matplotlib colormap
    ax : matplotlib axis
        (Optional) Axis to plot on. If not supplied a new figure is created and shown.
    kwargs : dict
        Any of the common figure keywords (x_range, y_range, title, x_axis_label, y_axis_label),
        and vmin, vmax for the color scale
    """
    from matplotlib.collections import PolyCollection
    import astropy.units as u

    if 'transit' not in full_output.keys():
        raise Exception('full_output does not contain a 3D transit geometry. Run transit_geometry() and spectrum(...,dimension="3d", full_output=True)')

    theta_edges = full_output['transit']['theta_edges']
    phi_edges = full_output['transit']['phi_edges']
    z = full_output['level']['z'] / (1*u.Unit(unit)).to(u.cm).value
    vmin, vmax = kwargs.pop('vmin', None), kwargs.pop('vmax', None)
    if plot == 'temperature':
        value = full_output['layer']['temperature']
        label, norm = 'Temperature (K)', colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        imol = list(full_output['weights'].keys()).index(plot)
        value = full_output['layer']['mixingratios'][:,imol,:,:]
        label, norm = f'{plot} Mixing Ratio', colors.LogNorm(vmin=vmin, vmax=vmax)
    nlayer, nzen, nsec = value.shape

    def wedge(r_low, r_up, ang_a, ang_b, sign=1):
        #polygon for an annular wedge, sampled every ~1 degree
        ang = np.linspace(ang_a, ang_b, max(2, int(np.ceil(np.degrees(ang_b-ang_a)))+1))
        r = np.concatenate([np.full(len(ang), r_up), np.full(len(ang), r_low)])
        ang = np.concatenate([ang, ang[::-1]])
        return np.column_stack([r*np.sin(ang), sign*r*np.cos(ang)])

    polys, vals = [], []
    if plane == 'day-night':
        #the opposite sector is drawn on the bottom half
        phi_center = 0.5*(phi_edges[1:] + phi_edges[:-1])
        dist = np.abs(np.angle(np.exp(1j*(phi_center - phi_center[sector] - np.pi))))
        opposite = np.argmin(dist)
        for j, sign in [(sector, 1), (opposite, -1)]:
            for k in range(nzen):
                for l in range(nlayer):
                    polys += [wedge(z[l+1,k,j], z[l,k,j], theta_edges[k], theta_edges[k+1], sign)]
                    vals += [value[l,k,j]]
        title = f'Day-Night Plane (sector {sector} and {opposite})'
        xlabel, ylabel = f'Distance Toward Observer ({unit})', f'Impact Parameter ({unit})'
    elif plane == 'terminator':
        if zenith is None: zenith = np.argmin(np.abs(theta_edges[1:]))
        for j in range(nsec):
            for l in range(nlayer):
                polys += [wedge(z[l+1,zenith,j], z[l,zenith,j], phi_edges[j], phi_edges[j+1])]
                vals += [value[l,zenith,j]]
        th_a, th_b = np.degrees(theta_edges[zenith:zenith+2])
        title = f'Terminator Plane (zenith slice {zenith}: {th_a:.1f} to {th_b:.1f} deg)'
        xlabel, ylabel = f'Morning Terminator Direction ({unit})', f'North ({unit})'
    else:
        raise Exception("plane must be 'day-night' or 'terminator'")

    show = ax is None
    if show: fig, ax = plt.subplots(figsize=(6,6))
    collection = PolyCollection(polys, array=np.array(vals), cmap=cmap, norm=norm, edgecolors='face', linewidths=0.2)
    ax.add_collection(collection)
    ax.add_patch(plt.Circle((0,0), np.min(z[-1]), color='0.3'))
    rmax = np.max(z[0])*1.05
    ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax)
    ax.set_aspect('equal')
    if plane == 'day-night':
        ax.axvline(0, color='k', ls=':', lw=1)
        ax.annotate('to star', xy=(-rmax*0.95, -rmax*0.95), ha='left')
        ax.annotate('to observer', xy=(rmax*0.95, -rmax*0.95), ha='right')
    plt.colorbar(collection, ax=ax, label=label, shrink=0.8)
    kwargs['title'] = kwargs.get('title', title)
    kwargs['x_axis_label'] = kwargs.get('x_axis_label', xlabel)
    kwargs['y_axis_label'] = kwargs.get('y_axis_label', ylabel)
    _mpl_format(ax, kwargs)
    if show:
        plt.show()
    return

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
    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',400))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',500))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
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

    colors = pals.viridis(10)
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

def all_optics_1d(full_output, wave_range, return_output = False,legend=None,
    ng=None, nt=None,
    colors = Colorblind8, backend='bokeh', **kwargs):
    """
    Plots 1d profiles of optical depth per layer, single scattering, and
    asymmetry averaged over the user input wave_range.

    Parameters
    ----------
    full_output : list,dict
        Dictonary of outputs or list of dicts
    wave_range : list
        min and max wavelength in microns
    return_output : bool
        Default is just to return a figure but you can also
        return all the 1d profiles
    legend : bool
        Default is none. Legend for each component of out
    ng : int
        gauss index
    nt : int
        chebychev intex
    backend : str
        Options are "bokeh" (default), "matplotlib" or "plotly"
    **kwargs : keyword arguments
        Key word arguments will be supplied to each bokeh figure function
    """
    return Analyzer(full_output, backend=backend).all_optics_1d(wave_range, return_output=return_output,
                    legend=legend, ng=ng, nt=nt, colors=colors, **kwargs)
      
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
        ax.set_ylim([np.max(out['full_output']['layer']['pressure']),np.min(out['full_output']['layer']['pressure'])])
        ax.set_ylabel('Pressure(bars)')
        ax.set_xlabel('Wavelength(um)')
        cbar.set_label('log Opacity')
    
    return ax

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

def phase_curve(allout, to_plot, collapse=None, R=100, 
    palette=pals.Spectral11,verbose=True, 
    reorder_output=False, **kwargs):
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
    reorder_output : bool
        Returns sorted outputs, for better phase curve plotting.
        re-orders phases such that brightest value is at phase=0
    """

    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',400))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',600))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
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
        palette = pals.magma(len(collapse))

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

    legend = Legend(items=legend_it, location=(0, -20),
                        **_get_legend_cols(len(legend_it), kwargs.get('height')))
    legend.click_policy="mute"
    fig.add_layout(legend, 'left') 
    
    fig.xgrid.grid_line_alpha=0
    fig.ygrid.grid_line_alpha=0
    plot_format(fig)

    #if verbose: print("phases", phases)
    #if verbose: print("all_curves",all_curves)

    #re-order phases such that brightest value is at phase=0 (only way to do this for reflected case)
    front_half_phases = phases[:len(phases)//2]
    #print("front_half_phases", front_half_phases)
    back_half_phases = phases[len(phases)//2:] - (2*np.pi)
    #print("back_half_phases", back_half_phases)
    reorder_phases = np.concatenate((back_half_phases, front_half_phases))

    #re-order all_curves such that brightest value is at phase=0 (only way to do this for reflected case)
    front_half_all_curves = all_curves[:len(all_curves)//2]
    back_half_all_curves = all_curves[len(all_curves)//2:]
    reorder_all_curves = np.concatenate((back_half_all_curves, front_half_all_curves))
    #print("Reorder all curves",reorder_all_curves)

    if to_plot == "fpfs_reflected" or to_plot == "albedo":
        fig2 = figure(**kwargs)

        for i in range(len(collapse)): 
            f2 = fig2.line(reorder_phases,reorder_all_curves[:,i],line_width=3,color=palette[i],
                    )
    
        legend2 = Legend(items=legend_it, location=(0, -20),
                        **_get_legend_cols(len(legend_it), kwargs.get('height')))
        legend2.click_policy="mute"
        fig2.add_layout(legend2, 'left')

        fig2.xgrid.grid_line_alpha=0
        fig2.ygrid.grid_line_alpha=0
        plot_format(fig2)
        #show(fig2)
    
    if reorder_output:
        return reorder_phases, reorder_all_curves, all_ws, fig
    
    return phases, all_curves, all_ws, fig

def thermal_contribution(full_output, tau_max=1.0,R=100, backend='matplotlib', **kwargs):
    """
    Computer the contribution function from https://doi.org/10.3847/1538-4357/aadd9e equation 4
    Note the equation in the paper is missing the - sign in the exponent
    Parameters
    ----------
    full_output : dict
        full dictionary output with {'wavenumber','thermal','full_output'}
    tau_max : float, optional
        Maximum tau to consider as opaque, largely here to help prevent NaNs and the colorbar from being weird.
    backend : str
        Options are "matplotlib" (default), "bokeh" or "plotly"
    **kwargs : dict
        Any key word argument for pcolormesh (matplotlib only)

    Returns
    -------
    fig, ax, CF_bin
        ax is None for bokeh and plotly
    """
    return Analyzer(full_output, backend=backend).thermal_contribution(tau_max=tau_max, R=R, **kwargs)
 

def molecule_contribution(contribution_out, opa, min_pressure=4.5, R=100, **kwargs):
    """
    Function to plot & graph the Tau~1 Pressure (bars) of various elements
    
    Parameters
    ----------
    contribution_out : dict
        contribution_out from jdi.get_contribution. 
        This function will grab contribution_out['tau_p_surface']
        Pressure vs. wavelength optical depth surface at tau specified by user in 
        get_contribution function (user input for at_tau)
        
    opa : picaso.opannection 
        Picaso opacity connection to get wavelength
    
    min_pressure : float, int
        Minimum pressure contribution in bars for molecules you want to plot. Ignores all molecules that 
        are optically thick higher than min_pressure (bars)
    
    R : int
        Resolution defined as lambda/dlambda 
        
    Outputs
    -------
    figure : bokeh.plotting.figure.Figure
        Shows a default graph of Tau 1 Surface of various molecules and a graph based on user input based on their parameters
        
    """
    kwargs['height'] = kwargs.get('plot_height',kwargs.get('height',400))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width',500))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label','Tau Pressure (bars)')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label','Wavelength')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
    kwargs['y_range'] = kwargs.get('y_range',[1e2,1e-4])
    kwargs['title'] = kwargs.get('title','User Input Tau Pressure Surface')

    tau_p_surface = contribution_out['tau_p_surface']
    wno=[]
    spec=[]
    labels=[]
    for j in tau_p_surface.keys(): 
        x,y = mean_regrid(opa.wno, tau_p_surface[j],R=R) 
        if np.min(y)<min_pressure: # Bars 
            wno+=[x]
            spec+=[y]
            labels +=[j]
    fig = spectrum(wno,spec, legend=labels, **kwargs)
    return fig

def transmission_contribution(full_output ,R=None, backend='matplotlib', **kwargs):
    """
    Compute the transmission contribution function from

    https://petitradtrans.readthedocs.io/en/latest/content/notebooks/analysis.html#Transmission-contribution-functions



    Parameters
    ----------
    full_output : dict
        full dictionary output with {'wavenumber','thermal','full_output'}
    R : float, optional
        Resolution for rebinning
    backend : str
        Options are "matplotlib" (default), "bokeh" or "plotly"
    **kwargs : dict
        Any key word argument for pcolormesh (matplotlib only)

    Returns
    -------
    fig
        figure
    ax
        matplotlib ax (None for bokeh and plotly)
    um
        micron array
    CF_bin
        contribution function
    """
    return Analyzer(full_output, backend=backend).transmission_contribution(R=R, **kwargs)

def brightness_temperature(out_dict,plot=True, R = None, with_guide=None, dimension='1d',
    wavelength=None, backend='matplotlib'): 
    r"""
    Plots and returns brightness temperature

    $T_{\rm bright}=\dfrac{a}{{\lambda}log\left(\dfrac{{b}}{F(\lambda){\lambda}^5}+1\right)}$

    where a = 1.43877735$\times$10$^{-2}$ m.K and b = 11.91042952$\times$10$^{-17}$ m$^4$kg/s$^3$ 
    
    Parameters
    ----------
    out_dict : dict 
        output of bundle.spectrum(opa,full_output=True) or a dictionary with 
        {"wavenumber": np.array, "thermal": np.array of standard output in erg/cm2/s/cm}
    plot : bool 
        If true creates and returns a plot 
    R : float 
        If not None, rebins the brightness temperature. In 3d mode every 
        facet (ng, nt) is rebinned 
    with_guide : bool 
        Plots points from the minimum and maximum pt values. Default is True in 1d 
        mode. Not available in 3d mode (a warning is raised if requested).
    dimension : str 
        '1d' (default) uses the disk integrated output['thermal']. '3d' uses 
        output['full_output']['thermal_3d'] and plots a latitude-longitude map 
        of the brightness temperature of each facet
    wavelength : float or list, optional
        3d mode only. Wavelength in micron to map. Either a single value (maps the nearest 
        wavelength) or [min, max] (maps the brightness temperature averaged over that 
        wavelength region). Default (None) averages over all wavelengths. Only one map is created.
    backend : str 
        'matplotlib' (default), 'bokeh', or 'plotly'. 3d mode is only available 
        with 'matplotlib'

    Returns
    -------
    T_B, fig if plot=True, else T_B. 
        In 1d mode T_B is [nwave], in 3d mode T_B is [ng, nt, nwave]
    """
    analyze = Analyzer(out_dict, backend=backend)
    if plot: 
        fig, T_B = analyze.brightness_temperature(R=R, with_guide=with_guide, dimension=dimension,
                                                  wavelength=wavelength)
        return T_B , fig
    else: 
        _, T_B = analyze._brightness_temperature_data(R=R, dimension=dimension)
        return T_B

def _brightness_temperature(wno, thermal):
    """
    Brightness temperature from thermal flux in erg/cm2/s/cm. 
    Wavelength must be the last axis of thermal
    """
    flux = thermal/np.pi*1e-7
    lam = 1e4/wno
    a=1.43877735e-2  #m K
    hc2=2*5.95521476e-17   # m^4 kg/s^3
    ## since flux is in W/m^2/microns hence need to multiple 1e6 to the flux to make it W/m^2/m
    flux=flux*1e6
    ## converting wv to m from microns
    lam=lam*1e-6

    return (a/lam)/np.log(1+(hc2/flux/lam**5))


def animate_convergence(clima_out, picaso_bundle, opacity, calculation='thermal',
    wave_range=[0.3,6],
    molecules=['H2O','CH4','CO','NH3']):
    """
    Function to animate climate convergence given all profiles that were 
    computed throughout the climate run 
    Animates from the first guess through to the final converged state. 
    Runs spectra for each of those cases. 
    
    Parameters
    ----------
    clima_out : dict 
        Output from, for example: clima_out = cl_run.climate(opacity_ck, save_all_profiles=True)
    picaso_bundle : <picaso.justdoit.inputs>
        This would be the equivalent of cl_run that you used to run climate_model(). This ensures that 
        your spectra can be rerun
    opacity : <picaso.optics.RetrieveCKs>
        This is the jdi.opannection that you used as input in the run_climate_model  
    wave_range : list 
        sets the wavelngth range of the spectra 
    molecules : list 
        list of strings for what molecules to animate
    
    Returns
    -------
    matplotlib.animation
    """
    map_calc = {'thermal':'thermal','reflected':'albedo','transmission':'transit_depth'}
    t_eq,p_eq,all_profiles_eq = (
                np.copy(clima_out['temperature']), 
                np.copy(clima_out['pressure']), 
                np.copy(clima_out['all_profiles']))
    
    if 'cld_output_final' in clima_out:
        # Code to be executed if clima_out['all_opd'] exists
        all_opd = np.copy(clima_out['all_opd'])
        cld_p = np.copy(clima_out['cld_output_picaso']['pressure'][0::196])
    
    nlevel = len(t_eq)
    # added this so that it can actually plot up the PT profile that's broken up into chunks of profiles
    nstep = int(all_profiles_eq.shape[0]/nlevel)
    split_profiles = np.array_split(all_profiles_eq, nstep)

    mols_to_plot = {i:np.zeros(all_profiles_eq.size) for i in molecules}
    spec = np.zeros(shape =(nstep,opacity.nwno))
    
    for i in range(nstep):
        picaso_bundle.add_pt(split_profiles[i], 
                             p_eq)

        picaso_bundle.premix_atmosphere(opacity)

        if 'cld_output_picaso' in clima_out:
            picaso_bundle.clouds(df=clima_out['cld_output_picaso'])

        df_spec = picaso_bundle.spectrum(opacity,calculation=calculation,full_output=True)
        spec[i,:] = df_spec[map_calc[calculation]]
        for imol in molecules:
            mols_to_plot[imol][i*nlevel:(i+1)*nlevel] = picaso_bundle.inputs['atmosphere']['profile'][imol]
    
    wh = np.where( (1e4/df_spec['wavenumber'] > wave_range[0]) & (1e4/df_spec['wavenumber'] < wave_range[1]))
    wv = 1e4/df_spec['wavenumber'][wh]    


    writergif = animation.PillowWriter(fps=3) 
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig = plt.figure(figsize=(50,10))

    # plot optical depth profile as well
    if 'cld_output_picaso' in clima_out:
        x='''
        AA.BB.CC.DD
        '''

        ax = fig.subplot_mosaic(x,gridspec_kw={
            # set the height ratios between the rows
            "height_ratios": [1],
            # set the width ratios between the columns
            "width_ratios": [1,1,0.1,1,1,0.1,1,1,0.1,1,1]})

    else:
        x='''
        AA.BB.CC
        '''
        ax = fig.subplot_mosaic(x,gridspec_kw={
            # set the height ratios between the rows
            "height_ratios": [1],
            # set the width ratios between the columns
            "width_ratios": [1,1,0.1,1,1,0.1,1,1]})

    temp = all_profiles_eq[0]
    temp = split_profiles[0]
    lines = {}
    for imol,col in zip(molecules,Colorblind8):
        lines[imol], = ax['B'].loglog(mols_to_plot[imol][0:nlevel], p_eq,linewidth=3,color=col, label=imol)

    lines['temp'], = ax['A'].semilogy(temp, p_eq,linewidth=3,color='k')
    if calculation=='thermal':
        lines['spec'], = ax['C'].semilogy(1e4/df_spec['wavenumber'], spec[0,:],linewidth=3,color="k")
    else: 
        lines['spec'], = ax['C'].plot(1e4/df_spec['wavenumber'], spec[0,:],linewidth=3,color="k")
    
    if 'cld_output_picaso' in clima_out: #bad hack to make shapes match but won't affect the plot
        lines['opd'], = ax['D'].loglog(all_opd[:91], np.append(cld_p,0),linewidth=3,color='k')

    def init():
        #line.set_ydata(np.ma.array(x, mask=True))

        ax['A'].set_xlabel('Temperature [K]',fontsize=30)
        ax['A'].set_ylabel('Pressure [Bars]',fontsize=30)
        ax['A'].set_xlim(0,max(t_eq))
        ax['A'].set_ylim(max(p_eq),min(p_eq))
        ax['B'].set_xlabel('Abundance [V/V]',fontsize=30)
        ax['B'].set_ylabel('Pressure [Bars]',fontsize=30)
        ax['B'].set_xlim(1e-6,1e-2)
        ax['B'].set_ylim(max(p_eq),min(p_eq))
        ax['B'].legend(fontsize=20)
        ax['C'].set_xlabel(r'Wavelength [$\mu$m]',fontsize=30)
        ax['C'].set_ylabel('Spectrum',fontsize=30)
        ax['C'].set_xlim(0,6)
        #ax['C'].set_ylim(1e7,1e14)
        ax['A'].minorticks_on()
        ax['A'].tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=30)
        ax['A'].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
        ax['B'].minorticks_on()
        ax['B'].tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=30)
        ax['B'].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)
        ax['C'].minorticks_on()
        ax['C'].tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=30)
        ax['C'].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)

        if 'cld_output_picaso' in clima_out:
            ax['D'].set_xlabel('Optical Depth',fontsize=30)
            ax['D'].set_ylabel('Pressure [Bars]',fontsize=30)
            ax['D'].set_xlim(1e-7,1e3)
            ax['D'].set_ylim(max(p_eq),min(p_eq))
            ax['D'].minorticks_on()
            ax['D'].tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=30)
            ax['D'].tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=30)

        for ikey in molecules+['temp']:
            lines[ikey].set_ydata(p_eq)
        
        lines['spec'].set_xdata(wv)

        if 'cld_output_picaso' in clima_out:
            lines['opd'].set_ydata(np.append(cld_p,max(cld_p)+1))
        return lines
    
    def animate(i):                       
        lines['temp'].set_xdata(split_profiles[i])
        
        for imol in molecules:
            lines[imol].set_xdata(mols_to_plot[imol][i*nlevel:(i+1)*nlevel])
        
        lines['spec'].set_ydata(spec[i,:][wh])

        if 'cld_output_picaso' in clima_out:
            lines['opd'].set_xdata(np.append(all_opd[i*90:(i+1)*90],1e-50))
        return lines

    ani = animation.FuncAnimation(fig, animate, frames=nstep,init_func=init,interval=50, blit=False)
    plt.close()
    return ani




def rt_heatmap(data,figure_kwargs={},cmap_kwargs={}):
    """
    This creates a heat map of the dataframe created from 
    picaso.test.thermal_sh_test() or picaso.test.dlugach_test()
    The tutorial 10c_AnalyzingApproximationsThermal.ipynb 
    and 10b_AnalyzingApproximationsReflectedLightSH.ipynb 
    shows it in use. 
    
    Parameters
    ----------
    data : pandas.DataFrame 
        Technically, this can be any dataframe, but this function was specifically built to 
        replicate Figure 9 in Batalha et al.  2019 and Figure 6 from Rooney et al. 2023. Part II Thermal. It is assumed that the 
        index of the dataframe is asymmetry and the columns are single scattering albedo. 
    figure_kwargs : dict 
        keywords for bokeh.plotting.figure function
    cmap_kwargs : dict 
        keywords for bokeh.models.LinearColorMapper funtion. Most common parameters to input are 
        palette (list or tuple of colors), high (upper lim of color map), and low (lower lim of color map)
    """
    reverse = True
    data.columns.name = 'w0' 
    data.index.name = 'g0' 
    data.index=data.index.astype(str)
    x_range = list(data.index)
    if reverse:
        y_range =  list(reversed(data.columns))
    else:
        y_range =  list(data.columns)

    df = pd.DataFrame(data.stack(), columns=['albedo']).reset_index()
    bd = max(abs(df.albedo.min()), abs(df.albedo.max()))

    cmap_kwargs['palette'] = cmap_kwargs.get('palette',pals.RdGy[11])
    cmap_kwargs['low'] = cmap_kwargs.get('low',-bd)
    cmap_kwargs['high'] = cmap_kwargs.get('high',bd)    

    mapper = LinearColorMapper(**cmap_kwargs)
    colors = cmap_kwargs['palette']
    
    figure_kwargs['height'] = figure_kwargs.get('height',400)
    figure_kwargs['width'] = figure_kwargs.get('width',300)
    figure_kwargs['x_axis_location'] = figure_kwargs.get('x_axis_location','above')
    figure_kwargs['tools'] = figure_kwargs.get('tools',"hover,save,pan,box_zoom,reset,wheel_zoom")
    figure_kwargs['toolbar_location'] =  figure_kwargs.get('toolbar_location','below') 
    figure_kwargs['y_range'] =  figure_kwargs.get('y_range',y_range) 
    figure_kwargs['x_range'] =  figure_kwargs.get('x_range',x_range) 
    figure_kwargs['y_axis_label'] =  figure_kwargs.get('y_axis_label','Single Scattering Albedo') 
    figure_kwargs['x_axis_label'] =  figure_kwargs.get('x_axis_label','Asymmetry')
    figure_kwargs['title'] =  figure_kwargs.get('title','% Diff')
    
    p = figure(**figure_kwargs)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "7px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect(x="g0", y="w0", width=1, height=1,
       source=df,
       fill_color={'field': 'albedo', 'transform': mapper},
       line_color=None)
    
    
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="12px",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'below')
    p.axis.major_label_text_font_size='12px'
    return p

    
def pt_adiabat(clima_out, input_class, opacityclass, plot=True):
    """
    Plot the PT profile with the adiabat 

    Parameters
    ----------
    clima_out : dict
        Full output from picaso 
    input_class : justdoit.input
        PICASO input class (e.g. the result of jdi.inputs()) 

    Returns
    -------
    adiabat, dTdP, pressure 
    """
    t_table = input_class.inputs['climate']['t_table']
    p_table = input_class.inputs['climate']['p_table']
    grad = input_class.inputs['climate']['grad']
    cp = input_class.inputs['climate']['cp']
    moist = input_class.inputs['climate']['moistgrad']
    AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad','cp'])
    AdiabatBundle = AdiabatBundle(t_table,p_table,grad,cp)

    Atmosphere = calculate_atm(input_class,opacityclass,only_atmosphere=True)

    layer_p = clima_out['spectrum_output']['full_output']['layer']['pressure']
    
    grad, cp = convec(clima_out['temperature'],clima_out['pressure'],
                      AdiabatBundle, Atmosphere,moist=moist)
    if plot: 
        plt.semilogy(clima_out['dtdp'], layer_p)
        plt.semilogy(grad,layer_p) 
        plt.ylim([1e4,1e-4]), 
        plt.xlabel('dT/dP vs adiabat')
        plt.ylabel('Pressure(bars)')
    return cp, grad, clima_out['dtdp'], layer_p

class Analyzer:
    """
    Diagnostic plots of a ``picaso`` run with a choice of plotting library.

    Each method returns a figure from the chosen backend: a bokeh figure/layout
    (display with ``show``), a matplotlib figure, or a plotly figure (display
    by returning it from a cell or with ``fig.show()``).

    Parameters
    ----------
    output : dict
        Either the full dictionary returned by ``spectrum(opa, full_output=True)``
        (which has the keys 'wavenumber', 'full_output', ...), or ``output['full_output']``.
        Only ``flux_at_top`` and 1d ``brightness_temperature`` require the full dictionary
        (or for ``brightness_temperature``, a dict with {'wavenumber','thermal'}).
    backend : str
        Default plotting library for every method: 'bokeh' (default), 'matplotlib', or
        'plotly' (requires ``pip install plotly``). Every method also takes ``backend``
        to override this for a single plot.

    Examples
    --------
    >>> df = case.spectrum(opa, full_output=True)
    >>> analyze = jpi.Analyzer(df, backend='matplotlib')
    >>> analyze.pt()
    >>> analyze.mixing_ratio(backend='plotly')
    """
    def __init__(self, output, backend='bokeh'):
        self.backend = _check_backend(backend)
        if isinstance(output, dict) and 'full_output' in output:
            self.output = output
            self.full_output = output['full_output']
        else:
            self.output = None
            self.full_output = output

    def _backend(self, backend):
        return _check_backend(self.backend if backend is None else backend)

    def mixing_ratio(self, limit=50, ng=None, nt=None, molecules=None, backend=None, **kwargs):
        """Returns plot of mixing ratios. See ``justplotit.mixing_ratio``"""
        backend = self._backend(backend)
        full_output = self.full_output
        if ((ng==None) & (nt==None)):
            pressure = full_output['layer']['pressure']
            mixingratios = full_output['layer']['mixingratios']
        else:
            pressure = full_output['layer']['pressure'][:,ng,nt]
            mixingratios = pd.DataFrame(full_output['layer']['mixingratios'][:,:,ng,nt],columns=molecules)

        if isinstance(molecules, type(None)):
            # remove 'pressure' and 'temperature'
            to_plot = [mol for mol in mixingratios.keys() if mol not in ['pressure', 'temperature', 'kz']][:limit]
        elif isinstance(molecules , str):
            to_plot = [molecules]
        elif isinstance((molecules) , list):
            to_plot = molecules

        if len(to_plot) < 3: ncol = 5
        else: ncol = len(to_plot)
        if limit<3:
            cols = pals.magma(5) #magma needs at least 5 colors
        else:
            cols = pals.magma(min([ncol,limit]))

        if backend=='matplotlib':
            #matplotlib sizes are in inches
            kwargs.setdefault('height', kwargs.get('plot_height', kwargs.get('height', 5)))
            kwargs.setdefault('width', kwargs.get('plot_width', kwargs.get('width', 7)))
            kwargs.pop('plot_width', None)
            kwargs.pop('plot_height', None)
            kwargs.setdefault('x_range', [1e-25, 5])
        else:
            kwargs = _bokeh_size(kwargs, 400, 300)
            kwargs['x_range'] = kwargs.get('x_range',[1e-20, 5])
        kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure(Bars)')
        kwargs['x_axis_label'] = kwargs.get('x_axis_label','Mixing Ratio(v/v)')
        kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
        kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')
        kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure),np.min(pressure)])

        if backend=='bokeh':
            fig = figure(**kwargs)
            legend_it=[]
            for mol , c in zip(to_plot,cols):
                f = fig.line(mixingratios[mol],pressure, color=c, line_width=3,
                            muted_color=c, muted_alpha=0.2)
                legend_it.append((mol, [f]))

            legend = Legend(items=legend_it, location=(0, -20),
                            **_get_legend_cols(len(legend_it), kwargs.get('height')))

            legend.click_policy="mute"
            fig.add_layout(legend, 'left')

        elif backend=='matplotlib':
            fig, ax = plt.subplots(figsize=(kwargs['width'], kwargs['height']))
            for mol, c in zip(to_plot, cols):
                ax.plot(mixingratios[mol], pressure, color=c, label=mol)
            _mpl_format(ax, kwargs)
            ax.legend()

        elif backend=='plotly':
            go, _ = _plotly()
            fig = go.Figure()
            for mol, c in zip(to_plot, cols):
                fig.add_trace(go.Scatter(x=mixingratios[mol], y=pressure, mode='lines', name=mol,
                                         line=dict(color=c, width=3)))
            _plotly_format(fig, kwargs)

        return fig

    def pt(self, ng=None, nt=None, backend=None, **kwargs):
        """Returns plot of pressure temperature profile. See ``justplotit.pt``"""
        backend = self._backend(backend)
        if ((ng==None) & (nt==None)):
            pressure = self.full_output['layer']['pressure']
            temperature = self.full_output['layer']['temperature']
        else:
            pressure = self.full_output['layer']['pressure'][:,ng,nt]
            temperature = self.full_output['layer']['temperature'][:,ng,nt]

        kwargs = _bokeh_size(kwargs, 400, 300)
        kwargs['title'] = kwargs.get('title','Pressure-Temperature Profile')
        kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure(Bars)')
        kwargs['x_axis_label'] = kwargs.get('x_axis_label','Temperature (K)')
        kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
        kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure),np.min(pressure)])

        if backend == 'bokeh':
            fig = figure(**kwargs)
            fig.line(temperature,pressure,line_width=3)
            plot_format(fig)

        elif backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=(kwargs['width']/100, kwargs['height']/100))
            ax.plot(temperature, pressure, linewidth=3)
            _mpl_format(ax, kwargs)

        elif backend == 'plotly':
            go, _ = _plotly()
            fig = go.Figure(go.Scatter(x=temperature, y=pressure, mode='lines', line=dict(width=3)))
            _plotly_format(fig, kwargs)

        return fig

    def photon_attenuation(self, at_tau=0.5, return_output=False, igauss=0, backend=None, **kwargs):
        """
        Plot breakdown of gas opacity, cloud opacity, Rayleigh scattering opacity
        at a specified pressure level. See ``justplotit.photon_attenuation``
        """
        backend = self._backend(backend)
        full_output = self.full_output
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

        #color sections for whichever opacity source dominates
        gas_dominate_ind = np.where((at_pressures_gas<at_pressures_cld) & (at_pressures_gas<at_pressures_ray))[0]
        cld_dominate_ind = np.where((at_pressures_cld<at_pressures_gas) & (at_pressures_cld<at_pressures_ray))[0]
        ray_dominate_ind = np.where((at_pressures_ray<at_pressures_cld) & (at_pressures_ray<at_pressures_gas))[0]

        gas_dominate = np.zeros(shape[1]) + 1e-8
        cld_dominate = np.zeros(shape[1]) + 1e-8
        ray_dominate = np.zeros(shape[1])+ 1e-8

        gas_dominate[gas_dominate_ind] = at_pressures_gas[gas_dominate_ind]
        cld_dominate[cld_dominate_ind] = at_pressures_cld[cld_dominate_ind]
        ray_dominate[ray_dominate_ind] = at_pressures_ray[ray_dominate_ind]

        kwargs = _bokeh_size(kwargs, 1000, 345)
        kwargs['title'] = kwargs.get('title','Pressure at 𝞽 =' +str(at_tau))
        kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure(Bars)')
        kwargs['x_axis_label'] = kwargs.get('x_axis_label','Wavelength [μm]')
        kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
        kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure),1e-2])

        labels = ['Gas Opacity', 'Cloud Opacity', 'Rayleigh Opacity']
        cols = [Colorblind8[0], Colorblind8[3], Colorblind8[6]]
        at_pressures = [at_pressures_gas, at_pressures_cld, at_pressures_ray]
        dominates = [gas_dominate, cld_dominate, ray_dominate]

        if backend == 'bokeh':
            fig = figure(**kwargs)
            legend_it = []
            for l, c, p in zip(labels, cols, at_pressures):
                f = fig.line(wave, p, line_width=3, color=c)
                legend_it.append((l, [f]))

            legend = Legend(items=legend_it, location=(0, -20),
                                **_get_legend_cols(len(legend_it), kwargs.get('height')))
            legend.click_policy="mute"
            fig.add_layout(legend, 'right')

            for c, d in zip(cols, dominates):
                band_x = np.append(np.array(wave), np.array(wave[::-1]))
                band_y = np.append(np.array(d), np.array(d)[::-1]*0+1e-8)
                fig.patch(band_x,band_y, color=c, alpha=0.3)
            plot_format(fig)

        elif backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=(kwargs['width']/100, kwargs['height']/100))
            for l, c, p, d in zip(labels, cols, at_pressures, dominates):
                ax.plot(wave, p, label=l, color=c, linewidth=3)
                ax.fill_between(wave, 1e-8, d, color=c, alpha=0.3)
            _mpl_format(ax, kwargs)
            ax.set_title(kwargs['title'].replace('𝞽', r'$\tau$')) #default matplotlib font has no 𝞽
            ax.legend()

        elif backend == 'plotly':
            go, _ = _plotly()
            fig = go.Figure()
            for l, c, p, d in zip(labels, cols, at_pressures, dominates):
                fig.add_trace(go.Scatter(x=wave, y=p, name=l, mode='lines', line=dict(color=c, width=3)))
                band_x = np.append(np.array(wave), np.array(wave[::-1]))
                band_y = np.append(np.array(d), np.array(d)[::-1]*0+1e-8)
                fig.add_trace(go.Scatter(x=band_x, y=band_y, fill='toself', fillcolor=c, opacity=0.3,
                                         mode='none', showlegend=False, hoverinfo='skip'))
            _plotly_format(fig, kwargs)

        if return_output: return fig ,wave,at_pressures_gas,at_pressures_cld,at_pressures_ray
        else: return fig

    def cloud(self, backend=None):
        """
        Heatmaps of the cloud single scattering albedo, optical depth, and asymmetry
        as a function of wavelength and pressure. See ``justplotit.cloud``
        """
        backend = self._backend(backend)
        dat01 = self.full_output['layer']['cloud']
        pressure = self.full_output['layer']['pressure']
        wave = 1e4 / self.full_output['wavenumber']

        w0 = dat01['w0']
        opd = dat01['opd'] + 1e-60
        g0 = dat01['g0']

        titles = ["Single Scattering Albedo", "Cloud Optical Depth Per Layer", "Asymmetry Parameter"]

        if backend == 'matplotlib':
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Plot W0
            im1 = ax1.pcolormesh(wave, pressure, w0,  cmap='magma', vmin=0, vmax=1)
            ax1.set_title(titles[0])
            fig.colorbar(im1, ax=ax1)

            # Plot OPD
            im2 = ax2.pcolormesh(wave, pressure, opd,  cmap='viridis_r', norm=colors.LogNorm(vmin=1e-3, vmax=10))
            ax2.set_title(titles[1])
            fig.colorbar(im2, ax=ax2)

            # Plot G0
            im3 = ax3.pcolormesh(wave, pressure, g0,  cmap='gray_r', vmin=0, vmax=1)
            ax3.set_title(titles[2])
            fig.colorbar(im3, ax=ax3)

            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel('Wavelength (micron)')
                ax.set_ylabel('Pressure (bar)')
                ax.set_yscale('log')
                ax.invert_yaxis()

        elif backend == 'bokeh':
            fig_kwargs = dict(x_axis_label='Wavelength (micron)', y_axis_label='Pressure (bar)',
                              width=400, height=350)
            fig1 = _bokeh_image(wave, pressure, w0, pals.magma(256), low=0, high=1,
                                title=titles[0], **fig_kwargs)
            fig2 = _bokeh_image(wave, pressure, opd, pals.viridis(256)[::-1], low=1e-3, high=10, log_color=True,
                                title=titles[1], **fig_kwargs)
            fig3 = _bokeh_image(wave, pressure, g0, pals.gray(256)[::-1], low=0, high=1,
                                title=titles[2], **fig_kwargs)
            fig = row(fig1, fig2, fig3)

        elif backend == 'plotly':
            go, make_subplots = _plotly()
            fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.1)
            heatmaps = [dict(z=w0, colorscale='Magma', zmin=0, zmax=1),
                        dict(z=np.log10(opd), colorscale='Viridis_r', zmin=-3, zmax=1),
                        dict(z=g0, colorscale='Greys', zmin=0, zmax=1)]
            for i, h in enumerate(heatmaps):
                x_thin, h['z'] = _thin_columns(wave, h['z'])
                fig.add_trace(go.Heatmap(x=x_thin, y=pressure,
                    colorbar=dict(x=_plotly_colorbar_x(fig, i+1), len=0.9,
                                  title='log' if i==1 else None), **h), row=1, col=i+1)
            fig.update_xaxes(title_text='Wavelength (micron)')
            fig.update_yaxes(type="log", autorange="reversed", title_text='Pressure (bar)')
            fig.update_layout(height=450, width=1300)

        return fig

    def disco(self, wavelength, calculation='reflected', backend=None):
        """
        Plot disco ball with facets. See ``justplotit.disco``.
        The bokeh backend shows 2D latitude-longitude maps instead of 3D spheres.
        """
        backend = self._backend(backend)
        if calculation=='reflected':to_plot='albedo_3d'
        elif calculation=='thermal':to_plot='thermal_3d'

        if isinstance(wavelength,(float,int)): wavelength = [wavelength]

        wave = 1e4/self.full_output['wavenumber']
        maps = []
        titles = []
        for w in wavelength:
            indw = find_nearest_1d(wave,w)
            #[umg, numt, nwno] this is xint_at_top
            maps.append(self.full_output[to_plot][:,:,indw])
            titles.append(str(wave[indw])+' Microns')

        return self._sphere_maps(maps, titles, backend)

    def map(self, pressure=[0.1], plot='temperature', wavelength=None, igauss=0, backend=None):
        """
        Plot a map of any 3d [nlayer, nlong, nlat] or 4d [nlayer, nwave, nlong, nlat]
        output at several pressures. See ``justplotit.map``.
        The bokeh backend shows 2D latitude-longitude maps instead of 3D spheres.
        """
        backend = self._backend(backend)
        to_plot = explore(self.full_output, plot)

        if isinstance(to_plot,np.ndarray):
            if len(to_plot.shape) < 3:
                raise Exception("The key you are search for is not a 3D matrix. This function \
                    is used to plot out a map of a matrix that is [nlayer, nlong, nlat] or \
                    [nlayer, nwave, nlong, nlat, ].")
            #here the four dimentions are nlayer, nwave, nlong, nlat
            elif len(to_plot.shape) == 4:
                wave = 1e4/self.full_output['wavenumber']
                indw = find_nearest_1d(wave,wavelength)
                to_plot= to_plot[:,indw,:,:]
            #here the five dimentions are nlayer, nwave, nlong, nlat, gauss
            elif len(to_plot.shape) == 5:
                wave = 1e4/self.full_output['wavenumber']
                indw = find_nearest_1d(wave,wavelength)
                to_plot= to_plot[:,indw,:,:,igauss]
        else:
            raise Exception ("The key you are search for is not an np.ndarray. This function \
                    is used to plot out a map of an numpy.ndarray matrix that is [nlayer, nlong, nlat] or \
                    [nlayer, nwave, nlong, nlat, ]")

        pressure_grid = self.full_output['layer']['pressure'][:,0,0]
        maps = []
        for p in pressure:
            indp = find_nearest_1d(np.log10(pressure_grid),np.log10(p))
            maps.append(to_plot[indp, :,:])
        titles = [str(p)+' bars' for p in pressure]

        return self._sphere_maps(maps, titles, backend)

    def taumap(self, at_tau=1, wavelength=1, igauss=0, backend=None):
        """
        Map of the pressure where the molecular, cloud, and Rayleigh opacity
        reach ``at_tau``. See ``justplotit.taumap``.
        The bokeh backend shows 2D latitude-longitude maps instead of 3D spheres.
        """
        backend = self._backend(backend)
        full_output = self.full_output
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

        return self._sphere_maps([map_gas, map_cld, map_ray],
                                 ['Molecular Opacity','Cloud Opacity','Rayleigh Opacity'], backend)

    def _sphere_maps(self, maps, titles, backend):
        """
        Shared plotting for disco, map and taumap. Each map is [nlong, nlat].
        matplotlib and plotly draw facets on a sphere, bokeh draws flat latitude-longitude maps.
        """
        latitude = self.full_output['latitude']  #tangle
        longitude = self.full_output['longitude'] #gangle
        u, v = np.meshgrid(longitude, latitude)
        x,y,z = lon_lat_to_cartesian(u, v)

        if backend == 'matplotlib':
            nrow = int(np.ceil(len(maps)/3))
            ncol = int(np.min([3,len(maps)])) #at most 3 columns
            fig = plt.figure(figsize=(6*ncol,4*nrow))
            for i, (to_map, title) in enumerate(zip(maps, titles)):
                ax = fig.add_subplot(nrow,ncol,i+1, projection='3d')
                ax.plot_wireframe(x, y, z, color="gray")
                sc = ax.scatter(x,y,z, c = to_map.T.ravel(),cmap=plt.get_cmap('plasma'),s=150)
                fig.colorbar(sc)
                ax.set_zlim3d(-1, 1)
                ax.set_ylim3d(-1, 1)
                ax.set_xlim3d(-1, 1)
                ax.view_init(0, 0)
                ax.set_title(title)
                # Hide grid lines and ticks
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.axis('off')
            plt.subplots_adjust(wspace=0.3, hspace=0.3)

        elif backend == 'plotly':
            go, make_subplots = _plotly()
            fig = make_subplots(rows=1, cols=len(maps), subplot_titles=titles,
                                specs=[[{'type': 'scene'}]*len(maps)])
            for i, to_map in enumerate(maps):
                fig.add_trace(go.Scatter3d(x=x.ravel(), y=y.ravel(), z=z.ravel(), mode='markers',
                    name=titles[i], showlegend=False,
                    marker=dict(size=6, color=to_map.T.ravel(), colorscale='Plasma',
                                colorbar=dict(x=(i+1)/len(maps), len=0.7))), row=1, col=i+1)
            fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                              camera=dict(eye=dict(x=1.6, y=0, z=0)))
            fig.update_layout(height=450, width=450*len(maps))

        elif backend == 'bokeh':
            figs = []
            for to_map, title in zip(maps, titles):
                figs.append(_bokeh_image(longitude*180/np.pi, latitude*180/np.pi, to_map.T,
                                         pals.plasma(256), log_y=False, title=title,
                                         x_axis_label='Longitude', y_axis_label='Latitude',
                                         y_range=[np.min(latitude)*180/np.pi, np.max(latitude)*180/np.pi],
                                         width=350, height=300))
            fig = row(*figs)

        return fig

    def flux_at_top(self, plot_bb=True, R=None, pressures=[1e-1,1e-2,1e-3], ng=None, nt=None,
                    backend=None, **kwargs):
        """
        Plot the OLR with overlaying black bodies. See ``justplotit.flux_at_top``.
        Requires the full dictionary output with {'wavenumber','thermal','full_output'}
        """
        backend = self._backend(backend)
        if self.output is None:
            raise Exception("flux_at_top needs the full dictionary returned by spectrum(opa, full_output=True), "
                            "with keys {'wavenumber','thermal','full_output'}, not just full_output")
        if ((ng==None) & (nt ==None)):
            pressure_all = self.full_output['layer']['pressure']
            temperature_all = self.full_output['layer']['temperature']
        else:
            pressure_all = self.full_output['layer']['pressure'][:,ng,nt]
            temperature_all = self.full_output['layer']['temperature'][:,ng,nt]

        if not isinstance(pressures, (np.ndarray, list)):
            raise Exception('check pressure input. It must be list or array. You can still input a single value as `pressures = [1e-3]`')

        kwargs = _bokeh_size(kwargs, 400, 300)
        kwargs['title'] = kwargs.get('title','Outgoing Thermal Radiation')
        kwargs['y_axis_label'] = kwargs.get('y_axis_label','Flux (erg/s/cm^3)')
        kwargs['x_axis_label'] = kwargs.get('x_axis_label','Wavelength [μm]')
        kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
        kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')

        if len(pressures) < 3: ncol = 5
        else: ncol = len(pressures)
        cols = pals.magma(ncol)

        wno = self.output['wavenumber']
        if isinstance(R,(int, float)):
            wno, thermal = mean_regrid(wno, self.output['thermal'], R=R)
        else:
            thermal =  self.output['thermal']

        #black bodies at each requested pressure
        bbs = []
        if plot_bb:
            for p,c in zip(pressures,cols):
                ip = find_nearest_1d(pressure_all, p)
                t = temperature_all[ip]
                intensity = blackbody(t, 1/wno)[0]
                bbs.append((np.pi * intensity, c, str(int(t))+' K at '+str(p)+' bars'))

        if backend == 'bokeh':
            fig = figure(**kwargs)
            fig.line(1e4/wno, thermal, color='black', line_width=4)
            for flux, c, l in bbs:
                fig.line(1e4/wno, flux, color=c, alpha=0.5, legend_label=l, line_width=4)

        elif backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=(kwargs['width']/100, kwargs['height']/100))
            ax.plot(1e4/wno, thermal, color='black', linewidth=4, label='Thermal')
            for flux, c, l in bbs:
                ax.plot(1e4/wno, flux, color=c, alpha=0.5, linewidth=4, label=l)
            _mpl_format(ax, kwargs)
            ax.legend()

        elif backend == 'plotly':
            go, _ = _plotly()
            fig = go.Figure(go.Scatter(x=1e4/wno, y=thermal, name='Thermal', mode='lines',
                                       line=dict(color='black', width=4)))
            for flux, c, l in bbs:
                fig.add_trace(go.Scatter(x=1e4/wno, y=flux, name=l, mode='lines',
                                         line=dict(color=c, width=4), opacity=0.5))
            _plotly_format(fig, kwargs)

        return fig

    def all_optics_1d(self, wave_range, return_output=False, legend=None, ng=None, nt=None,
                      colors=Colorblind8, backend=None, **kwargs):
        """
        Plots 1d profiles of optical depth per layer, single scattering, and
        asymmetry averaged over the user input wave_range. See ``justplotit.all_optics_1d``
        """
        backend = self._backend(backend)
        kwargs = _bokeh_size(kwargs, 300, 300)
        kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')

        full_output = self.full_output
        if not isinstance(full_output, list):
            full_output = [full_output]

        if ((ng==None) & (nt==None)):
            pressure = full_output[0]['layer']['pressure']
        else:
            pressure = full_output[0]['layer']['pressure'][:,ng,nt]

        kwargs['y_range'] = kwargs.get('y_range',[max(pressure),min(pressure)])

        #wavelength averaged profiles for each output
        profiles = []
        for i,results in enumerate(full_output):
            if ((ng==None) & (nt==None)):
                pressure = results['layer']['pressure']
                ssa = results['layer']['cloud']['w0']
                g0 = results['layer']['cloud']['g0']
                opd = results['layer']['cloud']['opd']
            else:
                pressure = results['layer']['pressure'][:,ng,nt]
                ssa = results['layer']['cloud']['w0'][:,:,ng,nt]
                g0 = results['layer']['cloud']['g0'][:,:,ng,nt]
                opd = results['layer']['cloud']['opd'][:,:,ng,nt]

            wno = results['wavenumber']

            inds = np.where((1e4/wno>wave_range[0]) &
                (1e4/wno<wave_range[1]))

            profiles.append(dict(pressure=pressure,
                                 opd=np.mean(opd[:,inds],axis=2)[:,0],
                                 ssa=np.mean(ssa[:,inds],axis=2)[:,0],
                                 g0=np.mean(g0[:,inds],axis=2)[:,0],
                                 color=colors[np.mod(i, len(colors))],
                                 label=None if legend is None else legend[i]))

        if backend == 'bokeh':
            fssa = figure(x_axis_label='Single Scattering Albedo',**kwargs)
            fg0 = figure(x_axis_label='Asymmetry',**kwargs)
            fopd = figure(x_axis_label='Optical Depth',y_axis_label='Pressure (bars)',
                x_axis_type='log',**kwargs)
            for p in profiles:
                fopd.line(p['opd'], p['pressure'], color=p['color'],line_width=3)
                fg0.line(p['g0'], p['pressure'], color=p['color'],line_width=3)
                if p['label'] is None:
                    fssa.line(p['ssa'], p['pressure'], color=p['color'],line_width=3)
                else:
                    fssa.line(p['ssa'], p['pressure'], color=p['color'],line_width=3,
                         legend_label=p['label'])
                    fssa.legend.location='top_left'
            fig = gridplot([[fopd,fssa,fg0]])

        elif backend == 'matplotlib':
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,
                                    figsize=(3*kwargs['width']/100, kwargs['height']/100))
            for p in profiles:
                ax1.plot(p['opd'], p['pressure'], color=p['color'], linewidth=3)
                ax2.plot(p['ssa'], p['pressure'], color=p['color'], linewidth=3, label=p['label'])
                ax3.plot(p['g0'], p['pressure'], color=p['color'], linewidth=3)
            ax1.set_xscale('log')
            ax1.set_yscale(kwargs['y_axis_type'])
            ax1.set_ylim(kwargs['y_range'])
            ax1.set_ylabel('Pressure (bars)')
            ax1.set_xlabel('Optical Depth')
            ax2.set_xlabel('Single Scattering Albedo')
            ax3.set_xlabel('Asymmetry')
            if legend is not None: ax2.legend()
            fig.tight_layout()

        elif backend == 'plotly':
            go, make_subplots = _plotly()
            fig = make_subplots(rows=1, cols=3, shared_yaxes=True)
            for p in profiles:
                for col, key in enumerate(['opd', 'ssa', 'g0']):
                    fig.add_trace(go.Scatter(x=p[key], y=p['pressure'], name=p['label'], mode='lines',
                                             showlegend=(col==1) & (p['label'] is not None),
                                             line=dict(color=p['color'], width=3)), row=1, col=col+1)
            for col, label in enumerate(['Optical Depth', 'Single Scattering Albedo', 'Asymmetry']):
                fig.update_xaxes(title_text=label, row=1, col=col+1)
            fig.update_xaxes(type='log', row=1, col=1)
            fig.update_yaxes(type=kwargs['y_axis_type'], title_text='Pressure (bars)', row=1, col=1)
            fig.update_yaxes(range=list(np.log10(kwargs['y_range'])) if kwargs['y_axis_type']=='log'
                                    else kwargs['y_range'])
            fig.update_layout(height=kwargs['height'], width=3*kwargs['width'])

        if return_output:
            #same as previous versions: the un-averaged profiles of the last output
            return fig, [opd,ssa,g0]
        else:
            return fig

    def thermal_contribution(self, tau_max=1.0, R=100, backend=None, **kwargs):
        """
        Emission contribution function from https://doi.org/10.3847/1538-4357/aadd9e equation 4.
        See ``justplotit.thermal_contribution``

        Returns
        -------
        fig, ax, CF_bin
            ax is None for bokeh and plotly
        """
        backend = self._backend(backend)
        full_output = self.full_output
        kwargs['norm'] = kwargs.get('norm',colors.LogNorm())
        kwargs['shading'] = kwargs.get('shading','auto')

        all_taus = np.squeeze(full_output['taugas']+full_output['taucld']+full_output['tauray'])
        all_taus[all_taus > tau_max] = tau_max
        sum_taus = np.cumsum(all_taus, axis=0)

        pressure = full_output['layer']['pressure']
        press2D = np.transpose(np.repeat(pressure[np.newaxis], np.shape(sum_taus)[1], axis=0))

        bb = np.ones(np.shape(press2D))
        for i, temp in enumerate(full_output['layer']['temperature']):
            for j, wave in enumerate(1/full_output['wavenumber']):
                bb[i, j] = blackbody(temp, wave)[0][0]
        CF = bb[0:-1, :] * np.exp(-sum_taus[0:-1, :]) * all_taus[0:-1, :] / np.diff(np.log(press2D), axis=0)

        if not isinstance(R,type(None)):
            wno,_ = mean_regrid(full_output['wavenumber'],full_output['wavenumber'],R=R)
            CF_bin = np.zeros((len(pressure)-1, len(wno)))
            for i in range(len(pressure)-1):
                _,CF_bin[i,:] = mean_regrid(full_output['wavenumber'], CF[i,:],newx=wno)
        else:
            CF_bin = CF
            wno=full_output['wavenumber']

        if backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=(15,10))
            smap = ax.pcolormesh(1e4/wno, pressure[0:-1], CF_bin, **kwargs)
            ax.set_ylim(np.max(pressure), np.min(pressure))
            ax.set_yscale('log')
            ax.set_ylabel('Pressure (bar)', fontsize=20)
            ax.set_xlabel(r'Wavelength ($\mu$m)', fontdict={'fontsize':20})
            cm = plt.colorbar(smap)
            cm.ax.set_ylabel('Emission Contribution Function', fontdict={'fontsize':18} )
            for l in cm.ax.yaxis.get_ticklabels():
                l.set_fontsize(16)
            ax.set_yticks(np.logspace(-5,1,7), minor=False)
            ax.set_yticklabels(np.logspace(-5,1,7), fontdict={'fontsize':18})
            return fig, ax, CF_bin

        fig = self._contribution_heatmap(1e4/wno, pressure[0:-1], CF_bin,
                                         'Emission Contribution Function', backend)
        return fig, None, CF_bin

    def transmission_contribution(self, R=None, backend=None, **kwargs):
        """
        Transmission contribution function. See ``justplotit.transmission_contribution``

        Returns
        -------
        fig, ax, um, CF_bin
            ax is None for bokeh and plotly
        """
        backend = self._backend(backend)
        full_output = self.full_output
        DTAU = (full_output['taugas'][:,:,0] +
                full_output['taucld'][:,:,0] +
                full_output['tauray'][:,:,0])
        z = full_output['level']['z']
        dz= full_output['level']['dz']
        nlevel = len(dz)
        nwno = DTAU.shape[1]
        rstar=1
        k_b=1
        amu=1
        player = full_output['layer']['pressure']
        tlayer = full_output['layer']['temperature']
        colden = full_output['layer']['column_density']
        mmw = full_output['layer']['mmw']

        zs = []
        for i in range(DTAU.shape[0]):
            dtau_copy = copy.deepcopy(DTAU)
            dtau_copy[i,:] = 0
            if i == 0 :
                norm = get_transit_1d(z, dz,nlevel, nwno, rstar, mmw, k_b,amu,
                                player, tlayer, colden, DTAU)
            zs+=[get_transit_1d(z, dz,nlevel, nwno, rstar, mmw, k_b,amu,
                                player, tlayer, colden, dtau_copy)]

        CF=(norm-np.array(zs))/np.sum(norm-np.array(zs),axis=0)

        if not isinstance(R,type(None)):
            wno,_ = mean_regrid(full_output['wavenumber'],full_output['wavenumber'],R=R)
            CF_bin = np.zeros((len(player), len(wno)))
            for i in range(len(player)):
                _,CF_bin[i,:] = mean_regrid(full_output['wavenumber'], CF[i,:],newx=wno)
        else:
            CF_bin = CF
            wno=full_output['wavenumber']

        if backend == 'matplotlib':
            kwargs['norm'] = kwargs.get('norm',colors.LogNorm())
            kwargs['shading'] = kwargs.get('shading','auto')
            fig, ax = plt.subplots()
            smap = ax.pcolormesh(1e4/wno, player, CF_bin, **kwargs)
            ax.set_ylim(np.max(player), np.min(player))
            ax.set_yscale('log')
            ax.set_ylabel('Pressure (bar)')
            ax.set_xlabel(r'Wavelength ($\mu$m)')
            plt.colorbar(smap, label='Transmission CF')
            return fig, ax, 1e4/wno, CF_bin

        fig = self._contribution_heatmap(1e4/wno, player, CF_bin, 'Transmission Contribution Function', backend)
        return fig, None, 1e4/wno, CF_bin

    def _brightness_temperature_data(self, R=None, dimension='1d'):
        """
        Returns wavenumber and brightness temperature. In 1d mode T_B is [nwave] computed from 
        output['thermal']. In 3d mode T_B is [ng, nt, nwave] computed from full_output['thermal_3d']
        """
        if dimension == '1d':
            #accepts the full dictionary or a dict with only {'wavenumber','thermal'}
            spec = self.full_output if self.output is None else self.output
            if 'thermal' not in spec:
                raise Exception("brightness_temperature in 1d mode needs the dictionary returned by "
                                "spectrum(opa, full_output=True) or one with keys {'wavenumber','thermal'}")
            wno = spec['wavenumber']
            T_B = _brightness_temperature(wno, spec['thermal'])
            if not isinstance(R, type(None)):
                wno, T_B = mean_regrid(wno, T_B, R=R)

        elif dimension == '3d':
            if 'thermal_3d' not in self.full_output:
                raise Exception("brightness_temperature in 3d mode needs 'thermal_3d' in full_output. "
                                "Run spectrum(opa, calculation='thermal', full_output=True)")
            wno = self.full_output['wavenumber']
            #thermal_3d is the [ng, nt, nwave] intensity of each facet, which carries a factor of 2pi
            #(see compress_thermal). Dividing by 2 puts it in the same units as the 1d thermal flux
            T_B = _brightness_temperature(wno, self.full_output['thermal_3d']/2)
            if not isinstance(R, type(None)):
                ng, nt, _ = T_B.shape
                new_wno, _ = mean_regrid(wno, wno, R=R)
                T_B_bin = np.zeros((ng, nt, len(new_wno)))
                for ig in range(ng):
                    for it in range(nt):
                        _, T_B_bin[ig, it, :] = mean_regrid(wno, T_B[ig, it, :], newx=new_wno)
                wno, T_B = new_wno, T_B_bin
        else:
            raise ValueError(f"dimension='{dimension}' is not supported. Options are '1d' or '3d'")

        return wno, T_B

    def brightness_temperature(self, R=None, with_guide=None, dimension='1d', wavelength=None, backend=None):
        """
        Plots brightness temperature. See ``justplotit.brightness_temperature``.
        In 3d mode a latitude-longitude map of the brightness temperature of each facet is plotted
        (matplotlib only) at a single wavelength or averaged over a [min, max] wavelength region.

        Returns
        -------
        fig, T_B
            In 1d mode T_B is [nwave], in 3d mode T_B is [ng, nt, nwave]
        """
        backend = self._backend(backend)
        wno, T_B = self._brightness_temperature_data(R=R, dimension=dimension)

        if dimension == '3d':
            if with_guide:
                warnings.warn("with_guide is not available for 3d brightness temperature maps. "
                              "Setting with_guide=False.")
            if backend != 'matplotlib':
                warnings.warn(f"backend='{backend}' is not available for 3d brightness temperature maps. "
                              "Using backend='matplotlib'.")
            return self._brightness_temperature_map(wno, T_B, wavelength), T_B

        with_guide = True if with_guide is None else with_guide
        t_eq = None
        if with_guide:
            if 'layer' in self.full_output:
                t_eq = self.full_output['layer']['temperature']
            else:
                warnings.warn("with_guide requires the full_output to get the min and max temperature. "
                              "Plotting without it.")

        wave = 1e4/wno
        if backend == 'matplotlib':
            fig = plt.figure(figsize=(15,8))
            plt.xlabel("Wavelength [microns]",fontsize=20)
            plt.ylabel("Brightness Temperature [K]",fontsize=20)
            plt.xlim(min(wave),max(wave))
            if t_eq is not None: plt.ylim(np.min(t_eq)-0.1*np.min(t_eq),np.max(t_eq)+0.1*np.min(t_eq))

            plt.semilogx(wave,T_B,color='k', label="Brightness Temperature")
            if t_eq is not None: plt.axhline(np.min(t_eq),linewidth=5,color="blue",label="Minimum Temperature")
            if t_eq is not None: plt.axhline(np.max(t_eq),linewidth=5,color="red",label="Maximum Temperature")

            plt.legend(fontsize=10)

        elif backend == 'bokeh':
            kwargs = dict(width=800, height=400, x_axis_type='log',
                          x_axis_label='Wavelength [μm]', y_axis_label='Brightness Temperature [K]',
                          x_range=[min(wave), max(wave)])
            if t_eq is not None:
                kwargs['y_range'] = [np.min(t_eq)-0.1*np.min(t_eq), np.max(t_eq)+0.1*np.min(t_eq)]
            fig = figure(**kwargs)
            fig.line(wave, T_B, color='black', line_width=2, legend_label='Brightness Temperature')
            if t_eq is not None:
                for t, c, l in [(np.min(t_eq),'blue','Minimum Temperature'),
                                (np.max(t_eq),'red','Maximum Temperature')]:
                    fig.line([min(wave), max(wave)], [t, t], color=c, line_width=5, legend_label=l)

        elif backend == 'plotly':
            go, _ = _plotly()
            fig = go.Figure(go.Scatter(x=wave, y=T_B, name='Brightness Temperature', mode='lines',
                                       line=dict(color='black', width=2)))
            if t_eq is not None:
                for t, c, l in [(np.min(t_eq),'blue','Minimum Temperature'),
                                (np.max(t_eq),'red','Maximum Temperature')]:
                    fig.add_trace(go.Scatter(x=[min(wave), max(wave)], y=[t, t], name=l, mode='lines',
                                             line=dict(color=c, width=5)))
                fig.update_yaxes(range=[np.min(t_eq)-0.1*np.min(t_eq), np.max(t_eq)+0.1*np.min(t_eq)])
            fig.update_xaxes(type='log', title_text='Wavelength [μm]')
            fig.update_yaxes(title_text='Brightness Temperature [K]')
            fig.update_layout(width=800, height=400)

        return fig, T_B

    def _brightness_temperature_map(self, wno, T_B, wavelength=None):
        """Matplotlib latitude-longitude heatmap of 3d brightness temperature [ng, nt, nwave]
        at a single wavelength or averaged over [min, max]"""
        wave = 1e4/wno
        if wavelength is None:
            wavelength = [np.min(wave), np.max(wave)]
        if isinstance(wavelength, (float, int, np.floating, np.integer)):
            indw = find_nearest_1d(wave, wavelength)
            to_map = T_B[:, :, indw]
            title = f'{wave[indw]:.3f} μm'
        elif len(wavelength) == 2:
            wmin, wmax = np.min(wavelength), np.max(wavelength)
            inds = np.where((wave >= wmin) & (wave <= wmax))[0]
            if len(inds) == 0:
                raise Exception(f"No wavelengths between {wmin} and {wmax} μm. "
                                f"The spectrum covers {np.min(wave):.2f}-{np.max(wave):.2f} μm")
            to_map = np.nanmean(T_B[:, :, inds], axis=2)
            title = f'Mean over {wmin:.2f}-{wmax:.2f} μm'
        else:
            raise Exception("wavelength must be a single value or [min, max] in micron")

        #full_output angles are in radians. maps are [ng, nt] = [nlong, nlat]
        longitude = np.atleast_1d(self.full_output['longitude'])*180/np.pi
        latitude = np.atleast_1d(self.full_output['latitude'])*180/np.pi

        fig, ax = plt.subplots(figsize=(6,4))
        smap = ax.pcolormesh(longitude, latitude, to_map.T, shading='nearest',
                             cmap=plt.get_cmap('plasma'))
        fig.colorbar(smap, ax=ax, label='Brightness Temperature [K]')
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def _contribution_heatmap(self, wave, pressure, CF, title, backend):
        """Log-scaled heatmap of a contribution function for the bokeh and plotly backends"""
        if backend == 'bokeh':
            return _bokeh_image(wave, pressure, CF, pals.viridis(256), log_color=True, title=title,
                                x_axis_label='Wavelength (μm)', y_axis_label='Pressure (bar)',
                                width=800, height=500)
        elif backend == 'plotly':
            go, _ = _plotly()
            with np.errstate(divide='ignore'):
                logCF = np.log10(CF)
            logCF[~np.isfinite(logCF)] = np.nan
            wave, logCF = _thin_columns(wave, logCF)
            fig = go.Figure(go.Heatmap(z=logCF, x=wave, y=pressure, colorscale='Viridis',
                                       colorbar=dict(title='log CF')))
            fig.update_yaxes(type="log", autorange="reversed", title_text='Pressure (bar)')
            fig.update_layout(title=title, xaxis_title="Wavelength (μm)", width=800, height=500)
            return fig

def flux_analyzer(clima_out, clima_out_compare=None, labels=None, 
    gravity=None, opacityclass=None, colors=('k','tab:red'), title=None, plot=True):
    """
    Analyzes the converged energy budget of a climate model (output of `cl_run.climate()`) 
    to show where the atmosphere absorbs starlight and where it loses energy to space. 

    If one climate output is given, creates a 3 panel figure: 

    1. Net absorbed stellar flux and net upward thermal (IR) flux at each level 
    2. Radiative heating (stellar) and cooling (IR) rates (in K/s if `gravity` is supplied, 
       otherwise the flux divergence per decade of pressure is plotted)
    3. Outgoing thermal flux spectrum at the top of the atmosphere 

    If a second climate output is given via `clima_out_compare` (e.g. the same model with 
    `exclude_mol`), both are plotted on the top row and a second row shows the difference 
    (compare - clima_out) of each quantity. Convective zones are shaded in the pressure panels. 

    Parameters
    ----------
    clima_out : dict 
        Output of `jdi.inputs.climate()`
    clima_out_compare : dict 
        (Optional) Output of `jdi.inputs.climate()` for a second model to compare against
    labels : tuple of str
        (Optional) Legend labels for (clima_out, clima_out_compare)
    gravity : float or astropy.units.Quantity
        (Optional) Planet gravity. If float, it is assumed to be cgs (cm/s/s). 
        Needed to convert flux divergences into heating rates [K/s]. 
    opacityclass : class 
        (Optional) Opacity class from `jdi.opannection` used for the climate runs. 
        Gives exact wavenumber bin widths for the outgoing flux spectrum. If not supplied 
        the bin widths are approximated with numpy.gradient and `with_spec=True` must have 
        been used in the climate runs so that the wavenumber grid can be found. 
    colors : tuple of str 
        Colors for (clima_out, clima_out_compare)
    title : str 
        (Optional) Figure title 
    plot : bool 
        If True, creates and returns the figure

    Returns
    -------
    dict (or list of two dicts if clima_out_compare is given), (matplotlib.figure.Figure if plot=True)
        Computed diagnostics for each model containing: 
        pressure (levels, bars), pressure_layer (bars), 
        flux_stellar_absorbed (net downward stellar flux at each level, erg/cm2/s), 
        flux_ir_net (net upward thermal flux at each level, erg/cm2/s), 
        heating_stellar, cooling_ir, heating_net (per layer, K/s if gravity supplied 
        else erg/cm2/s per dex of pressure), wavelength (um), 
        flux_toa (outgoing thermal flux in erg/cm2/s/cm), olr (bolometric outgoing thermal flux, erg/cm2/s)

    Notes
    -----
    Heating rates use dT/dt = (g/cp) dF/dP with net fluxes at the level boundaries. Here cp 
    is approximated as an ideal diatomic gas, cp = 7/2 R/mmw, with the mean molecular weight 
    from `spectrum_output` (requires `with_spec=True`), otherwise mmw=2.3 g/mol is assumed. 
    In radiative zones of a converged model stellar heating and thermal cooling balance 
    (heating_net ~ 0). In convective zones they do not, since convection carries the remaining energy.
    """
    #Ratio of specific heat at constant pressure to the gas constant, cp/R, for an ideal diatomic gas [dimensionless]
    CP_OVER_R_DIATOMIC = 3.5
    #Default mean molecular weight used when it is not available from the climate output [g/mol]
    MMW_DEFAULT = 2.3*u.g/u.mol

    if not isinstance(gravity, type(None)): 
        if not isinstance(gravity, u.Quantity): 
            gravity = gravity*u.cm/u.s**2
    
    def get_diagnostics(out): 
        pressure = np.array(out['pressure'])
        fb = out['flux_balance']
        #net upward stellar flux is negative, flip it so this is the stellar flux absorbed below each level 
        f_v = -fb['rfacv']*np.array(fb['flux_net_v'])
        f_ir = fb['rfaci']*np.array(fb['flux_net_ir'])
        p_layer = np.sqrt(pressure[1:]*pressure[:-1])

        #energy gained by each layer is the net upward flux entering from below 
        #minus the net upward flux leaving out the top 
        d_v = f_v[:-1] - f_v[1:] #stellar 
        d_ir = f_ir[1:] - f_ir[:-1] #thermal 
        if isinstance(gravity, type(None)): 
            dlogp = np.log10(pressure[1:]) - np.log10(pressure[:-1])
            heat_v = d_v/dlogp
            cool_ir = d_ir/dlogp
        else: 
            mmw = MMW_DEFAULT
            spec = out.get('spectrum_output', None)
            if isinstance(spec, dict) and ('full_output' in spec.keys()): 
                mmw = np.array(spec['full_output']['layer']['mmw'])*u.g/u.mol
            cp = CP_OVER_R_DIATOMIC*const.R/mmw
            dp = (pressure[1:] - pressure[:-1])*u.bar
            flux_unit = u.erg/u.cm**2/u.s
            heat_v = (gravity/cp*d_v*flux_unit/dp).to(u.K/u.s).value
            cool_ir = (gravity/cp*d_ir*flux_unit/dp).to(u.K/u.s).value

        flux_toa = np.array(out['flux_ir_attop'])
        if not isinstance(opacityclass, type(None)): 
            wno = opacityclass.wno
            dwno = opacityclass.delta_wno
        elif isinstance(out.get('spectrum_output', None), dict): 
            wno = out['spectrum_output']['wavenumber']
            dwno = np.gradient(wno)
        else: 
            raise Exception('Need either opacityclass or climate output with `with_spec=True` to get the wavenumber grid')
        #erg/cm2/s per bin -> erg/cm2/s/cm to match units of spectrum thermal output 
        flux_toa_spec = flux_toa/dwno*wno**2

        return dict(pressure=pressure, pressure_layer=p_layer, 
                    flux_stellar_absorbed=f_v, flux_ir_net=f_ir, 
                    heating_stellar=heat_v, cooling_ir=cool_ir, heating_net=heat_v+cool_ir,
                    wavelength=1e4/wno, flux_toa=flux_toa_spec, olr=np.sum(flux_toa),
                    cvz_locs=out.get('cvz_locs',None))

    compare = not isinstance(clima_out_compare, type(None))
    diags = [get_diagnostics(clima_out)]
    if compare: diags += [get_diagnostics(clima_out_compare)]
    if isinstance(labels, type(None)): 
        labels = ('Model A','Model B') if compare else ('Model',)
    returns = diags if compare else diags[0]

    if not plot: 
        return returns

    nrow = 2 if compare else 1
    fig, ax = plt.subplots(nrow, 3, figsize=(18,5.5*nrow), squeeze=False)
    heat_unit = 'K/s' if not isinstance(gravity, type(None)) else 'erg/cm$^2$/s/dex'
    pmin = np.min([d['pressure'].min() for d in diags])
    pmax = np.max([d['pressure'].max() for d in diags])

    for d, lab, col in zip(diags, labels, colors): 
        #net fluxes 
        ax[0,0].loglog(d['flux_stellar_absorbed'], d['pressure'], color=col, lw=2.5, ls='--')
        ax[0,0].loglog(d['flux_ir_net'], d['pressure'], color=col, lw=2.5, ls='-')
        #heating rates 
        ax[0,1].semilogy(d['heating_stellar'], d['pressure_layer'], color=col, lw=2.5, ls='--')
        ax[0,1].semilogy(d['cooling_ir'], d['pressure_layer'], color=col, lw=2.5, ls='-')
        #outgoing thermal spectrum 
        ax[0,2].loglog(d['wavelength'], d['flux_toa'], color=col, lw=1.5, 
                     label=f"{lab}: OLR={d['olr']:.3e} erg/cm$^2$/s")
        #shade convective zones 
        #nstr format: convective zones are [nstr[1]:nstr[2]] and, if nstr[3]!=0, [nstr[4]:nstr[5]]
        if not isinstance(d['cvz_locs'], type(None)): 
            nstr = np.array(d['cvz_locs']).astype(int)
            zones = [(nstr[1],nstr[2])] + ([(nstr[4],nstr[5])] if nstr[3]!=0 else [])
            for itop, ibot in zones: 
                ibot = min(ibot+1, len(d['pressure'])-1)
                for a in ax[:,:2].ravel(): 
                    a.axhspan(d['pressure'][itop], d['pressure'][ibot], color=col, alpha=0.08, lw=0)

    #legend entries for line styles 
    for lab, col in zip(labels, colors[:len(diags)]): 
        ax[0,0].plot([],[],color=col,lw=2.5,label=lab)
    ax[0,0].plot([],[],color='grey',ls='--',label='Net absorbed stellar')
    ax[0,0].plot([],[],color='grey',ls='-',label='Net upward thermal')
    ax[0,0].fill_between([],[],color='grey',alpha=0.2,label='Convective zone')
    ax[0,1].plot([],[],color='grey',ls='--',label='Stellar heating')
    ax[0,1].plot([],[],color='grey',ls='-',label='Thermal cooling')

    #flux axis: stellar flux goes to ~0 at depth so clip to something sensible 
    fmax = np.max([max(d['flux_stellar_absorbed'].max(), d['flux_ir_net'].max()) for d in diags])
    ax[0,0].set_xlim(fmax*1e-5, fmax*3)
    #heating rates span many orders of magnitude so use symlog 
    hmax = np.max([np.abs(d['heating_stellar']).max() for d in diags])
    #spectra: only show where there is meaningful emission 
    toa_max = np.max([d['flux_toa'].max() for d in diags])
    wv = np.concatenate([d['wavelength'][d['flux_toa']>toa_max*1e-4] for d in diags])
    ax[0,2].set_ylim(toa_max*1e-4, toa_max*3)

    ax[0,0].set_xlabel('Flux [erg/cm$^2$/s]'); ax[0,0].set_title('Net fluxes', fontsize=13)
    ax[0,1].set_xlabel(f'Rate [{heat_unit}]'); ax[0,1].set_title('Radiative heating (+) / cooling (-)', fontsize=13)
    ax[0,2].set_ylabel('Outgoing thermal flux [erg/cm$^2$/s/cm]')
    ax[0,2].set_title('Top-of-atmosphere thermal emission', fontsize=13)

    if compare: 
        da, db = diags
        dlab = f'({labels[1]} - {labels[0]})'
        ax[1,0].semilogy(db['flux_stellar_absorbed']-da['flux_stellar_absorbed'], da['pressure'], color='k', lw=2.5, ls='--', label='$\\Delta$ Net absorbed stellar')
        ax[1,0].semilogy(db['flux_ir_net']-da['flux_ir_net'], da['pressure'], color='k', lw=2.5, ls='-', label='$\\Delta$ Net upward thermal')
        ax[1,1].semilogy(db['heating_stellar']-da['heating_stellar'], da['pressure_layer'], color='k', lw=2.5, ls='--', label='$\\Delta$ Stellar heating')
        ax[1,1].semilogy(db['cooling_ir']-da['cooling_ir'], da['pressure_layer'], color='k', lw=2.5, ls='-', label='$\\Delta$ Thermal cooling')
        ax[1,2].semilogx(da['wavelength'], db['flux_toa']-da['flux_toa'], color='k', lw=1.5)
        ax[1,0].axvline(0, color='grey', lw=1); ax[1,1].axvline(0, color='grey', lw=1)
        ax[1,2].axhline(0, color='grey', lw=1)
        ax[1,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax[1,0].set_xlabel('$\\Delta$ Flux [erg/cm$^2$/s]'); ax[1,0].set_title(f'Change in net fluxes\n{dlab}', fontsize=13)
        ax[1,1].set_xlabel(f'$\\Delta$ Rate [{heat_unit}]'); ax[1,1].set_title(f'Change in heating / cooling rates\n{dlab}', fontsize=13)
        ax[1,2].set_ylabel('$\\Delta$ Flux [erg/cm$^2$/s/cm]')
        ax[1,2].set_title(f'Change in thermal emission\n{dlab}', fontsize=13)
        ax[1,0].legend(fontsize=11); ax[1,1].legend(fontsize=11)

    for irow in range(nrow): 
        ax[irow,1].set_xscale('symlog', linthresh=hmax*1e-4)
        ax[irow,1].axvline(0, color='grey', lw=1)
        ax[irow,2].set_xlim(wv.min(), wv.max())
        ax[irow,2].set_xlabel('Wavelength [$\\mu$m]')
        for icol in range(2): 
            ax[irow,icol].set_ylim(pmax, pmin)
            ax[irow,icol].set_ylabel('Pressure [bars]')
    for a in ax[0,:]: 
        a.legend(fontsize=11)
    for a in ax.ravel(): 
        a.minorticks_on()
        a.tick_params(axis='both',which='both',direction='in',labelsize=12)
    if not isinstance(title, type(None)): 
        fig.suptitle(title, fontsize=18)
    fig.tight_layout()

    return returns, fig

