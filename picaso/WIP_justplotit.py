import pandas as pd
import numpy as np

from bokeh.palettes import Colorblind8, RdGy
import bokeh.palettes as pals
from bokeh.models import HoverTool
from bokeh.models import LinearColorMapper, LogTicker, BasicTicker, ColorBar, LogColorMapper, Legend
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.layouts import row, column, gridplot
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show
Colorblind8 = pals.Colorblind8
RdGy = pals.RdGy

import os 
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

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ==================== HELPERS ====================

def _get_legend_cols(n_items, plot_height):
    """
    Calculate the number of columns needed for a legend to fit within the plot height.
    This is only supported in bokeh >= 3.1.0
    """
    max_items_per_col = max(1, (plot_height - 17) // 23)
    ncols = int(np.ceil(n_items / max_items_per_col))
    return {'ncols': max([1, ncols])}

def mean_regrid(x, y, newx=None, R=None):
    """
    Rebin the spectrum at a minimum R or on a fixed grid! 
    """
    if (isinstance(newx, type(None)) & (not isinstance(R, type(None)))) :
        newx = create_grid(1e4/max(x), 1e4/min(x), R)
    elif (not isinstance(newx, type(None)) & (isinstance(R, type(None)))) :  
        d = np.diff(newx)
        binedges = np.array([newx[0]-d[0]/2] + list(newx[0:-1]+d/2.0) + [newx[-1]+d[-1]/2])
        newx = binedges
    else: 
        raise Exception('Please either enter a newx or a R') 
    y, edges, binnum = binned_statistic(x, y, bins=newx)
    newx = (edges[0:-1]+edges[1:])/2.0
    return newx, y

def plot_multierror(x, y, plot, dx_up=0, dx_low=0, dy_up=0, dy_low=0, 
    point_kwargs={}, error_kwargs={}):
    """
    Plot non-symmetric x and y error bars in bokeh plot
    """
    (dx_up, dx_low, dy_up, dy_low) = [[i]*len(x)
                                      if isinstance(i, (float, int)) else i
                                      for i in [dx_up, dx_low, dy_up, dy_low]]

    # first x error
    y_err = []
    x_err = []
    for px, py, x_up, x_low in zip(x, y, dx_up, dx_low):
        np.array(x_err.append((px - x_low, px + x_up)))
        np.array(y_err.append((py, py )))

    plot.multi_line(x_err, y_err, **error_kwargs)

    # then y error
    y_err = []
    x_err = []
    for px, py, y_up, y_low in zip(x, y, dy_up, dy_low):
        np.array(x_err.append((px , px )))
        np.array(y_err.append((py - y_low, py + y_up)))

    plot.multi_line(x_err, y_err, **error_kwargs)
    plot.scatter(x, y, **point_kwargs)
    return

def bin_errors(newx, oldx, dy):
    newx =[newx[0] -  np.diff(newx)[0]/2] +  list(newx[0:-1] + np.diff(newx)/2) + [newx[-1] +  np.diff(newx)[-1]/2]
    err = []
    for i in range(len(newx)-1):
        loc = np.where(((oldx>newx[i]) & (oldx<=newx[i+1])))[0]
        err += [np.sqrt(np.sum(dy[loc]**2.0))/len(dy[loc])]
    return err

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

def lon_lat_to_cartesian(lon_r, lat_r, R = 1):
    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z

def find_nearest_old(array, value):
    idx = (np.abs(array-value)).argmin(axis=0)
    return idx

def find_nearest_2d(array, value, axis=1):
    all_out = []
    for i in range(array.shape[axis]):
        ar, iar, ic = np.unique(array[:, i], return_index=True, return_counts=True)
        idx = (np.abs(ar-value)).argmin(axis=0)
        if ic[idx]>1: 
            idx = iar[idx] + (ic[idx]-1)
        else: 
            idx = iar[idx]
        all_out += [idx]
    return all_out

def find_nearest_1d(array, value):
    ar, iar, ic = np.unique(array, return_index=True, return_counts=True)
    idx = (np.abs(ar-value)).argmin(axis=0)
    if ic[idx]>1: 
        idx = iar[idx] + (ic[idx]-1)
    else: 
        idx = iar[idx]
    return idx

@jit(nopython=True, cache=True)
def numba_cumsum(mat):
    new_mat = np.zeros(mat.shape)
    for i in range(mat.shape[1]):
        new_mat[:, i] = np.cumsum(mat[:, i])
    return new_mat

def explore(df, key):
    check=[False, True, True]
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

# ==================== STANDALONE FUNCTIONS ====================

def plot_errorbar(x, y, e, plot=None, point_kwargs={}, error_kwargs={},
    plot_type='bokeh', plot_kwargs={}, backend=None):
    """
    Plot symmetric error bars across bokeh, matplotlib, or plotly.
    """
    if backend is not None:
        plot_type = backend

    if plot_type == 'bokeh':
        if isinstance(plot, type(None)):
            plot_kwargs['height'] = plot_kwargs.get('plot_height', plot_kwargs.get('height', 345))
            plot_kwargs['width'] = plot_kwargs.get('plot_width', plot_kwargs.get('width', 1000))
            if 'plot_width' in plot_kwargs.keys() : plot_kwargs.pop('plot_width')
            if 'plot_height' in plot_kwargs.keys() : plot_kwargs.pop('plot_height')
            plot_kwargs['y_axis_label'] = plot_kwargs.get('y_axis_label', 'Spectrum')
            plot_kwargs['x_axis_label'] = plot_kwargs.get('x_axis_label', 'Wavelength')
            plot = figure(**plot_kwargs) 
        y_err = []
        x_err = []
        for px, py, yerr in zip(x, y, e):
            np.array(x_err.append((px , px )))
            np.array(y_err.append((py - yerr, py + yerr)))

        plot.multi_line(x_err, y_err, **error_kwargs)
        plot.scatter(x, y, **point_kwargs)
        return plot

    elif plot_type == 'matplotlib':
        point_kwargs = point_kwargs.copy()
        point_kwargs['color'] = point_kwargs.get('color', 'k')
        
        plot_kwargs = plot_kwargs.copy()
        plot_kwargs['xlabel'] = plot_kwargs.get('xlabel', r'Wavelength [$\mu$m]')
        plot_kwargs['ylabel'] = plot_kwargs.get('ylabel', r'(R$_p$/R$_*$)$^2$')
        plot_kwargs['figsize'] = plot_kwargs.get('figsize', (20, 10))
        plot_kwargs['fontsize'] = plot_kwargs.get('fontsize', 25)

        fig = plt.figure(figsize=plot_kwargs['figsize'])
        plt.errorbar(x, y, e, **point_kwargs)
        plt.xlabel(plot_kwargs['xlabel'], fontsize=plot_kwargs['fontsize'])
        plt.ylabel(plot_kwargs['ylabel'], fontsize=plot_kwargs['fontsize'])
        plt.minorticks_on()
        plt.tick_params(axis='y', which='major', length=20, width=3, direction='in', labelsize=20)
        plt.tick_params(axis='y', which='minor', length=10, width=2, direction='in', labelsize=20)
        plt.tick_params(axis='x', which='major', length=20, width=3, direction='in', labelsize=20)
        plt.tick_params(axis='x', which='minor', length=10, width=2, direction='in', labelsize=20)
        return fig

    elif plot_type in ['plotly', 'plotly_m']:
        fig = plot if plot is not None else go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            error_y=dict(
                type='data',
                array=e,
                visible=True,
                **error_kwargs
            ),
            mode='markers',
            marker=point_kwargs
        ))
        x_label = plot_kwargs.get('x_axis_label', plot_kwargs.get('xlabel', None if plot is not None else 'Wavelength'))
        y_label = plot_kwargs.get('y_axis_label', plot_kwargs.get('ylabel', None if plot is not None else 'Spectrum'))
        if x_label is not None:
            fig.update_xaxes(title_text=x_label)
        if y_label is not None:
            fig.update_yaxes(title_text=y_label)
        height = plot_kwargs.get('height', plot_kwargs.get('plot_height', 400))
        width = plot_kwargs.get('width', plot_kwargs.get('plot_width', 600))
        fig.update_layout(height=height, width=width)
        return fig

def spectrum(xarray, yarray, legend=None, wno_to_micron=True, palette=Colorblind8, muted_alpha=0.2, backend='plotly', **kwargs):
    """
    Plot formatted albedo spectrum across bokeh, matplotlib, or plotly.
    """
    if len(yarray) == len(xarray):
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

    if isinstance(legend, str): 
        legend = [legend]

    if backend == 'bokeh':
        kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 345))
        kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 1000))
        if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
        if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
        kwargs['y_axis_label'] = kwargs.get('y_axis_label', 'Spectrum')
        kwargs['x_axis_label'] = kwargs.get('x_axis_label', x_axis_label)

        fig = figure(**kwargs)

        i = 0
        legend_it = [] 
        for yarray in Y:
            if isinstance(xarray, list):
                if isinstance(legend, type(None)): legend = [None]*len(xarray[0])
                for w, a, i, l in zip(xarray, yarray, range(len(xarray)), legend):
                    if l == None: 
                        fig.line(conv(w), a, color=palette[np.mod(i, len(palette))], line_width=3)
                    else:
                        f = fig.line(conv(w), a, color=palette[np.mod(i, len(palette))], line_width=3,
                                    muted_color=palette[np.mod(i, len(palette))], muted_alpha=muted_alpha)
                        legend_it.append((l, [f]))
            else: 
                if isinstance(legend, type(None)):
                    fig.line(conv(xarray), yarray, color=palette[i], line_width=3)
                else:
                    f = fig.line(conv(xarray), yarray, color=palette[i], line_width=3,
                                    muted_color=palette[np.mod(i, len(palette))], muted_alpha=muted_alpha)
                    legend_it.append((legend[i], [f]))
            i = i+1

        if not isinstance(legend, type(None)):
            plt_legend = Legend(items=legend_it, location=(0, 0),
                            **_get_legend_cols(len(legend_it), kwargs.get('height')))
            plt_legend.click_policy="mute"
            fig.add_layout(plt_legend, 'left')

        plot_format(fig)
        return fig

    elif backend == 'matplotlib':
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        i = 0
        for yarray in Y:
            if isinstance(xarray, list):
                for w, a, i, l in zip(xarray, yarray, range(len(xarray)), legend or [None]*len(xarray)):
                    ax.plot(conv(w), a, color=palette[np.mod(i, len(palette))], label=l, linewidth=3)
            else:
                label = legend[i] if legend is not None and i < len(legend) else None
                ax.plot(conv(xarray), yarray, color=palette[np.mod(i, len(palette))], label=label, linewidth=3)
            i += 1
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(kwargs.get('y_axis_label', 'Spectrum'))
        if legend is not None:
            ax.legend()
        return fig

    elif backend == 'plotly':
        fig = go.Figure()
        i = 0
        for yarray in Y:
            if isinstance(xarray, list):
                for w, a, i, l in zip(xarray, yarray, range(len(xarray)), legend or [None]*len(xarray)):
                    fig.add_trace(go.Scatter(x=conv(w), y=a, name=l, line=dict(color=palette[np.mod(i, len(palette))], width=3)))
            else:
                label = legend[i] if legend is not None and i < len(legend) else None
                fig.add_trace(go.Scatter(x=conv(xarray), y=yarray, name=label, line=dict(color=palette[np.mod(i, len(palette))], width=3)))
            i += 1
        fig.update_xaxes(title_text=x_axis_label)
        fig.update_yaxes(title_text=kwargs.get('y_axis_label', 'Spectrum'))
        height = kwargs.get('height', kwargs.get('plot_height', 400))
        width = kwargs.get('width', kwargs.get('plot_width', 1000))
        fig.update_layout(height=height, width=width)
        return fig


# ==================== ANALYZER CLASS ====================

class Analyzer:
    def __init__(self, full_output, backend='plotly'):
        self.full_output = full_output
        self.backend = backend

    def mixing_ratio(self, limit=50, ng=None, nt=None, molecules=None, **kwargs):
        """Returns plot of mixing ratios"""
        if ((ng == None) & (nt == None)):
            pressure = self.full_output['layer']['pressure']
            mixingratios = self.full_output['layer']['mixingratios']
        else: 
            pressure = self.full_output['layer']['pressure'][:, ng, nt]
            mixingratios = pd.DataFrame(self.full_output['layer']['mixingratios'][:, :, ng, nt], columns=molecules)

        if isinstance(molecules, type(None)):
            to_plot = [mol for mol in mixingratios.keys() if mol not in ['pressure', 'temperature', 'kz']][:limit]
        elif isinstance(molecules, str):
            to_plot = [molecules]
        elif isinstance(molecules, list):
            to_plot = molecules

        molecules = to_plot

        if self.backend == 'bokeh':
            kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 300))
            kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 400))
            if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
            if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
            kwargs['y_axis_label'] = kwargs.get('y_axis_label', 'Pressure(Bars)')
            kwargs['x_axis_label'] = kwargs.get('x_axis_label', 'Mixing Ratio(v/v)')
            kwargs['y_axis_type'] = kwargs.get('y_axis_type', 'log')
            kwargs['x_axis_type'] = kwargs.get('x_axis_type', 'log') 
            kwargs['y_range'] = kwargs.get('y_range', [np.max(pressure), np.min(pressure)])
            kwargs['x_range'] = kwargs.get('x_range', [1e-20, 5])

            fig = figure(**kwargs)
            if len(molecules) < 3: ncol = 5
            else: ncol = len(molecules)
            if limit < 3: 
                cols = pals.magma(5)
            else: 
                cols = pals.magma(min([ncol, limit]))
            legend_it = []    
            for mol, c in zip(to_plot, cols):
                f = fig.line(mixingratios[mol], pressure, color=c, line_width=3,
                            muted_color=c, muted_alpha=0.2)
                legend_it.append((mol, [f]))
            
            legend = Legend(items=legend_it, location=(0, -20),
                            **_get_legend_cols(len(legend_it), kwargs.get('height')))
            legend.click_policy="mute"
            fig.add_layout(legend, 'left')
            return fig

        elif self.backend == 'matplotlib':
            kwargs.setdefault('height', kwargs.get('plot_height', kwargs.get('height', 5)))
            kwargs.setdefault('width', kwargs.get('plot_width', kwargs.get('width', 7)))
            kwargs.pop('plot_width', None)
            kwargs.pop('plot_height', None)
            kwargs.setdefault('y_axis_label', 'Pressure(Bars)')
            kwargs.setdefault('x_axis_label', 'Mixing Ratio(v/v)')
            kwargs.setdefault('y_axis_type', 'log')
            kwargs.setdefault('x_axis_type', 'log')
            kwargs.setdefault('y_range', [np.max(pressure), np.min(pressure)])
            kwargs.setdefault('x_range', [1e-25, 5])

            fig = plt.figure(figsize=(kwargs['width'], kwargs['height']))
            axes = fig.add_subplot(1, 1, 1)
            for mol in to_plot:
                axes.plot(mixingratios[mol], pressure, label=mol)
            axes.set_xlim(kwargs['x_range'])
            axes.set_ylim(kwargs['y_range'])
            axes.legend()
            axes.set_xlabel(kwargs['x_axis_label'])
            axes.set_ylabel(kwargs['y_axis_label'])
            axes.set_yscale('log')
            axes.set_xscale('log')
            axes.invert_yaxis()
            return fig

        elif self.backend == 'plotly':
            width = kwargs.get('plot_width', kwargs.get('width', 600))
            height = kwargs.get('plot_height', kwargs.get('height', 400))
            x_label = kwargs.get('x_axis_label', 'Mixing Ratio(v/v)')
            y_label = kwargs.get('y_axis_label', 'Pressure(Bars)')

            fig = go.Figure()
            for mol in to_plot:
                fig.add_trace(go.Scatter(x=mixingratios[mol], y=pressure, mode='lines', name=mol, line=dict(width=3)))
            
            fig.update_xaxes(type="log", title_text=x_label, range=[np.log10(1e-20), np.log10(5)])
            fig.update_yaxes(type="log", title_text=y_label, range=[np.log10(np.max(pressure)), np.log10(np.min(pressure))])
            fig.update_layout(width=width, height=height, title=kwargs.get('title', 'Mixing Ratios'))
            return fig

    def pt(self, ng=None, nt=None, **kwargs):
        """Returns plot of pressure temperature profile"""
        if ((ng == None) & (nt == None)):
            pressure = self.full_output['layer']['pressure']
            temperature = self.full_output['layer']['temperature']
        else: 
            pressure = self.full_output['layer']['pressure'][:, ng, nt]
            temperature = self.full_output['layer']['temperature'][:, ng, nt]

        kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 300))
        kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 400))
        if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
        if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
        kwargs['title'] = kwargs.get('title', 'Pressure-Temperature Profile')
        kwargs['y_axis_label'] = kwargs.get('y_axis_label', 'Pressure(Bars)')
        kwargs['x_axis_label'] = kwargs.get('x_axis_label', 'Temperature (K)')
        kwargs['y_axis_type'] = kwargs.get('y_axis_type', 'log')
        kwargs['y_range'] = kwargs.get('y_range', [np.max(pressure), np.min(pressure)])

        if self.backend == 'bokeh':
            fig = figure(**kwargs)
            fig.line(temperature, pressure, line_width=3) 
            plot_format(fig)
            return fig

        elif self.backend == 'matplotlib':
            fig = plt.figure(figsize=(kwargs['width']/100, kwargs['height']/100))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(temperature, pressure, linewidth=3)
            ax.set_yscale('log')
            ax.set_ylim(kwargs['y_range'])
            ax.set_xlabel(kwargs['x_axis_label'])
            ax.set_ylabel(kwargs['y_axis_label'])
            ax.set_title(kwargs['title'])
            return fig

        elif self.backend == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=temperature, y=pressure, mode='lines', line=dict(width=3)))
            fig.update_xaxes(title_text=kwargs['x_axis_label'])
            fig.update_yaxes(type="log", title_text=kwargs['y_axis_label'], range=[np.log10(np.max(pressure)), np.log10(np.min(pressure))])
            fig.update_layout(title=kwargs['title'], width=kwargs['width'], height=kwargs['height'])
            return fig

    def photon_attenuation(self, at_tau=0.5, return_output=False, igauss=0, **kwargs):
        wave = 1e4/self.full_output['wavenumber']

        dtaugas = self.full_output['taugas'][:, :, igauss]
        dtaucld = self.full_output['taucld'][:, :, igauss]*self.full_output['layer']['cloud']['w0']
        dtauray = self.full_output['tauray'][:, :, igauss]
        shape = dtauray.shape
        taugas = np.zeros((shape[0]+1, shape[1]))
        taucld = np.zeros((shape[0]+1, shape[1]))
        tauray = np.zeros((shape[0]+1, shape[1]))

        taugas[1:, :] = numba_cumsum(dtaugas)
        taucld[1:, :] = numba_cumsum(dtaucld)
        tauray[1:, :] = numba_cumsum(dtauray)

        pressure = self.full_output['level']['pressure']

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

        gas_dominate_ind = np.where((at_pressures_gas<at_pressures_cld) & (at_pressures_gas<at_pressures_ray))[0]
        cld_dominate_ind = np.where((at_pressures_cld<at_pressures_gas) & (at_pressures_cld<at_pressures_ray))[0]
        ray_dominate_ind = np.where((at_pressures_ray<at_pressures_cld) & (at_pressures_ray<at_pressures_gas))[0]

        gas_dominate = np.zeros(shape[1]) + 1e-8
        cld_dominate = np.zeros(shape[1]) + 1e-8
        ray_dominate = np.zeros(shape[1])+ 1e-8

        gas_dominate[gas_dominate_ind] = at_pressures_gas[gas_dominate_ind]
        cld_dominate[cld_dominate_ind] = at_pressures_cld[cld_dominate_ind]
        ray_dominate[ray_dominate_ind] = at_pressures_ray[ray_dominate_ind]

        kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 345))
        kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 1000))
        if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
        if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
        kwargs['title'] = kwargs.get('title', 'Pressure at 𝞽 =' + str(at_tau))
        kwargs['y_axis_label'] = kwargs.get('y_axis_label', 'Pressure(Bars)')
        kwargs['x_axis_label'] = kwargs.get('x_axis_label', 'Wavelength [μm]')
        kwargs['y_axis_type'] = kwargs.get('y_axis_type', 'log')
        kwargs['y_range'] = kwargs.get('y_range', [np.max(pressure), 1e-2])

        if self.backend == 'bokeh':
            fig = figure(**kwargs)
            legend_it = []
            f = fig.line(wave, at_pressures_gas, line_width=3, color=Colorblind8[0]) 
            legend_it.append(('Gas Opacity', [f]))
            f = fig.line(wave, at_pressures_cld, line_width=3, color=Colorblind8[3]) 
            legend_it.append(('Cloud Opacity', [f]))
            f = fig.line(wave, at_pressures_ray, line_width=3, color=Colorblind8[6]) 
            legend_it.append(('Rayleigh Opacity', [f]))

            legend = Legend(items=legend_it, location=(0, -20),
                                **_get_legend_cols(len(legend_it), kwargs.get('height')))
            legend.click_policy="mute"
            fig.add_layout(legend, 'right')   

            if len(gas_dominate) > 0:
                band_x = np.append(np.array(wave), np.array(wave[::-1]))
                band_y = np.append(np.array(gas_dominate), np.array(gas_dominate)[::-1]*0+1e-8)
                fig.patch(band_x, band_y, color=Colorblind8[0], alpha=0.3)
            if len(cld_dominate) > 0:
                band_x = np.append(np.array(wave), np.array(wave[::-1]))
                band_y = np.append(np.array(cld_dominate), np.array(cld_dominate)[::-1]*0+1e-8)
                fig.patch(band_x, band_y, color=Colorblind8[3], alpha=0.3)
            if len(ray_dominate) > 0:
                band_x = np.append(np.array(wave), np.array(wave[::-1]))
                band_y = np.append(np.array(ray_dominate), np.array(ray_dominate)[::-1]*0+1e-8)
                fig.patch(band_x, band_y, color=Colorblind8[6], alpha=0.3)

            plot_format(fig)

        elif self.backend == 'matplotlib':
            fig = plt.figure(figsize=(kwargs['width']/100, kwargs['height']/100))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(wave, at_pressures_gas, label='Gas Opacity', color=Colorblind8[0], linewidth=3)
            ax.plot(wave, at_pressures_cld, label='Cloud Opacity', color=Colorblind8[3], linewidth=3)
            ax.plot(wave, at_pressures_ray, label='Rayleigh Opacity', color=Colorblind8[6], linewidth=3)
            
            ax.fill_between(wave, 1e-8, gas_dominate, color=Colorblind8[0], alpha=0.3)
            ax.fill_between(wave, 1e-8, cld_dominate, color=Colorblind8[3], alpha=0.3)
            ax.fill_between(wave, 1e-8, ray_dominate, color=Colorblind8[6], alpha=0.3)
            
            ax.set_yscale('log')
            ax.set_ylim(kwargs['y_range'][0], kwargs['y_range'][1])
            ax.set_xlabel(kwargs['x_axis_label'])
            ax.set_ylabel(kwargs['y_axis_label'])
            ax.set_title(kwargs['title'])
            ax.legend()

        elif self.backend == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wave, y=at_pressures_gas, name='Gas Opacity', line=dict(color=Colorblind8[0], width=3)))
            fig.add_trace(go.Scatter(x=wave, y=at_pressures_cld, name='Cloud Opacity', line=dict(color=Colorblind8[3], width=3)))
            fig.add_trace(go.Scatter(x=wave, y=at_pressures_ray, name='Rayleigh Opacity', line=dict(color=Colorblind8[6], width=3)))
            
            if len(gas_dominate) > 0:
                band_x = np.append(np.array(wave), np.array(wave[::-1]))
                band_y = np.append(np.array(gas_dominate), np.array(gas_dominate)[::-1]*0+1e-8)
                fig.add_trace(go.Scatter(x=band_x, y=band_y, fill='toself', fillcolor=Colorblind8[0], opacity=0.3, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            if len(cld_dominate) > 0:
                band_x = np.append(np.array(wave), np.array(wave[::-1]))
                band_y = np.append(np.array(cld_dominate), np.array(cld_dominate)[::-1]*0+1e-8)
                fig.add_trace(go.Scatter(x=band_x, y=band_y, fill='toself', fillcolor=Colorblind8[3], opacity=0.3, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            if len(ray_dominate) > 0:
                band_x = np.append(np.array(wave), np.array(wave[::-1]))
                band_y = np.append(np.array(ray_dominate), np.array(ray_dominate)[::-1]*0+1e-8)
                fig.add_trace(go.Scatter(x=band_x, y=band_y, fill='toself', fillcolor=Colorblind8[6], opacity=0.3, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                
            fig.update_xaxes(title_text=kwargs['x_axis_label'])
            fig.update_yaxes(type="log", title_text=kwargs['y_axis_label'], range=[np.log10(kwargs['y_range'][0]), np.log10(kwargs['y_range'][1])])
            fig.update_layout(title=kwargs['title'], width=kwargs['width'], height=kwargs['height'])

        if return_output:
            return fig, wave, at_pressures_gas, at_pressures_cld, at_pressures_ray
        else:
            return fig

    def cloud(self):
        dat01 = self.full_output['layer']['cloud']
        pressure = self.full_output['layer']['pressure']
        wave = 1e4 / self.full_output['wavenumber']
        
        w0 = dat01['w0'] 
        opd = dat01['opd'] + 1e-60
        g0 = dat01['g0'] 

        if self.backend == 'matplotlib':
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            im1 = ax1.pcolormesh(wave, pressure, w0, cmap='magma', vmin=0, vmax=1)
            ax1.set_title("Single Scattering Albedo")
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.pcolormesh(wave, pressure, opd, cmap='viridis_r', norm=colors.LogNorm(vmin=1e-3, vmax=10))
            ax2.set_title("Cloud Optical Depth Per Layer")
            fig.colorbar(im2, ax=ax2)

            im3 = ax3.pcolormesh(wave, pressure, g0, cmap='gray_r', vmin=0, vmax=1)
            ax3.set_title("Asymmetry Parameter")
            fig.colorbar(im3, ax=ax3)

            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel('Wavelength (micron)')
                ax.set_ylabel('Pressure (bar)')
                ax.set_yscale('log')
                ax.invert_yaxis()
            return fig

        elif self.backend == 'bokeh':
            fig1 = figure(title="Single Scattering Albedo", x_axis_label="Wavelength (micron)", y_axis_label="Pressure (bar)", y_axis_type="log", y_range=[pressure.max(), pressure.min()])
            fig1.image(image=[w0], x=wave.min(), y=pressure.min(), dw=wave.max()-wave.min(), dh=pressure.max()-pressure.min(), palette="Magma256")
            
            fig2 = figure(title="Cloud Optical Depth Per Layer", x_axis_label="Wavelength (micron)", y_axis_label="Pressure (bar)", y_axis_type="log", y_range=[pressure.max(), pressure.min()])
            fig2.image(image=[np.log10(opd)], x=wave.min(), y=pressure.min(), dw=wave.max()-wave.min(), dh=pressure.max()-pressure.min(), palette="Viridis256")
            
            fig3 = figure(title="Asymmetry Parameter", x_axis_label="Wavelength (micron)", y_axis_label="Pressure (bar)", y_axis_type="log", y_range=[pressure.max(), pressure.min()])
            fig3.image(image=[g0], x=wave.min(), y=pressure.min(), dw=wave.max()-wave.min(), dh=pressure.max()-pressure.min(), palette="Greys256")
            
            return row(fig1, fig2, fig3)

        elif self.backend == 'plotly':
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Single Scattering Albedo", "Cloud Optical Depth Per Layer", "Asymmetry Parameter"))
            fig.add_trace(go.Heatmap(z=w0, x=wave, y=pressure, colorscale='Magma', zmin=0, zmax=1), row=1, col=1)
            fig.add_trace(go.Heatmap(z=np.log10(opd), x=wave, y=pressure, colorscale='Viridis', zmin=-3, zmax=1), row=1, col=2)
            fig.add_trace(go.Heatmap(z=g0, x=wave, y=pressure, colorscale='Greys', zmin=0, zmax=1), row=1, col=3)
            fig.update_yaxes(type="log", autorange="reversed")
            fig.update_layout(height=500, width=1500)
            return fig

    def disco(self, wavelength, calculation='reflected'):
        if calculation == 'reflected': to_plot = 'albedo_3d'
        elif calculation == 'thermal': to_plot = 'thermal_3d'

        if isinstance(wavelength, (float, int)): wavelength = [wavelength]

        wave = 1e4/self.full_output['wavenumber']
        latitude = self.full_output['latitude']  
        longitude = self.full_output['longitude'] 
        u, v = np.meshgrid(longitude, latitude)
        x, y, z = lon_lat_to_cartesian(u, v)

        if self.backend == 'matplotlib':
            nrow = int(np.ceil(len(wavelength)/3))
            ncol = int(np.min([3, len(wavelength)]))
            fig = plt.figure(figsize=(6*ncol, 4*nrow))
            for i, w in zip(range(len(wavelength)), wavelength):
                ax = fig.add_subplot(nrow, ncol, i+1, projection='3d')
                indw = find_nearest_1d(wave, w)
                xint_at_top = self.full_output[to_plot][:, :, indw]

                cmap_cm = plt.get_cmap('plasma')
                ax.plot_wireframe(x, y, z, color="gray")
                sc = ax.scatter(x, y, z, c=xint_at_top.T.ravel(), cmap=cmap_cm, s=150)
                fig.colorbar(sc)
                ax.set_zlim3d(-1, 1)                    
                ax.set_ylim3d(-1, 1)                    
                ax.set_xlim3d(-1, 1)
                ax.view_init(0, 0)
                ax.set_title(str(wave[indw])+' Microns')
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.axis('off')
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            return fig

        elif self.backend == 'plotly':
            fig = make_subplots(rows=1, cols=len(wavelength), specs=[[{'type': 'scene'}]*len(wavelength)])
            for i, w in enumerate(wavelength):
                indw = find_nearest_1d(wave, w)
                xint_at_top = self.full_output[to_plot][:, :, indw]
                fig.add_trace(go.Scatter3d(
                    x=x.ravel(), y=y.ravel(), z=z.ravel(),
                    mode='markers',
                    marker=dict(size=6, color=xint_at_top.T.ravel(), colorscale='Plasma', showscale=True),
                    name=f'{wave[indw]:.2f} um'
                ), row=1, col=i+1)
            fig.update_layout(height=500, width=500*len(wavelength))
            return fig

        elif self.backend == 'bokeh':
            figs = []
            for w in wavelength:
                indw = find_nearest_1d(wave, w)
                xint_at_top = self.full_output[to_plot][:, :, indw]
                p = figure(title=f'{wave[indw]:.2f} um', x_axis_label='Longitude', y_axis_label='Latitude')
                p.image(image=[xint_at_top.T], x=longitude.min()*180/np.pi, y=latitude.min()*180/np.pi,
                        dw=(longitude.max()-longitude.min())*180/np.pi, dh=(latitude.max()-latitude.min())*180/np.pi,
                        palette="Plasma256")
                figs.append(p)
            return row(*figs)

    def map(self, pressure=[0.1], plot='temperature', wavelength=None, igauss=0):
        to_plot = explore(self.full_output, plot)

        if isinstance(to_plot, np.ndarray):
            if len(to_plot.shape) < 3: 
                raise Exception("The key you are search for is not a 3D matrix.")
            elif len(to_plot.shape) == 4: 
                wave = 1e4/self.full_output['wavenumber']
                indw = find_nearest_1d(wave, wavelength)
                to_plot = to_plot[:, indw, :, :]
            elif len(to_plot.shape) == 5: 
                wave = 1e4/self.full_output['wavenumber']
                indw = find_nearest_1d(wave, wavelength)
                to_plot = to_plot[:, indw, :, :, igauss]
        else:
            raise Exception("The key you are search for is not an np.ndarray.")

        latitude = self.full_output['latitude']  
        longitude = self.full_output['longitude'] 
        u, v = np.meshgrid(longitude, latitude)
        x, y, z = lon_lat_to_cartesian(u, v)

        pressure_grid = self.full_output['layer']['pressure'][:, 0, 0]

        if self.backend == 'matplotlib':
            nrow = int(np.ceil(len(pressure)/3))
            ncol = int(np.min([3, len(pressure)]))
            fig = plt.figure(figsize=(6*ncol, 4*nrow))
            for i, p in zip(range(len(pressure)), pressure):
                ax = fig.add_subplot(nrow, ncol, i+1, projection='3d')
                indp = find_nearest_1d(np.log10(pressure_grid), np.log10(p))
                to_map = to_plot[indp, :, :]

                cmap_cm = plt.get_cmap('plasma')
                ax.plot_wireframe(x, y, z, color="gray")
                sc = ax.scatter(x, y, z, c=to_map.T.ravel(), cmap=cmap_cm, s=150)
                fig.colorbar(sc)
                ax.set_zlim3d(-1, 1)                    
                ax.set_ylim3d(-1, 1)                    
                ax.set_xlim3d(-1, 1)
                ax.view_init(0, 0)
                ax.set_title(str(p)+' bars')
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.axis('off')
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            return fig

        elif self.backend == 'plotly':
            fig = make_subplots(rows=1, cols=len(pressure), specs=[[{'type': 'scene'}]*len(pressure)])
            for i, p in enumerate(pressure):
                indp = find_nearest_1d(np.log10(pressure_grid), np.log10(p))
                to_map = to_plot[indp, :, :]
                fig.add_trace(go.Scatter3d(
                    x=x.ravel(), y=y.ravel(), z=z.ravel(),
                    mode='markers',
                    marker=dict(size=6, color=to_map.T.ravel(), colorscale='Plasma', showscale=True),
                    name=f'{p} bars'
                ), row=1, col=i+1)
            fig.update_layout(height=500, width=500*len(pressure))
            return fig

        elif self.backend == 'bokeh':
            figs = []
            for p in pressure:
                indp = find_nearest_1d(np.log10(pressure_grid), np.log10(p))
                to_map = to_plot[indp, :, :]
                fig_b = figure(title=f'{p} bars', x_axis_label='Longitude', y_axis_label='Latitude')
                fig_b.image(image=[to_map.T], x=longitude.min()*180/np.pi, y=latitude.min()*180/np.pi,
                        dw=(longitude.max()-longitude.min())*180/np.pi, dh=(latitude.max()-latitude.min())*180/np.pi,
                        palette="Plasma256")
                figs.append(fig_b)
            return row(*figs)

    def flux_at_top(self, plot_bb=True, R=None, pressures=[1e-1, 1e-2, 1e-3], ng=None, nt=None, **kwargs):
        if ((ng == None) & (nt == None)):
            pressure_all = self.full_output['full_output']['layer']['pressure']
            temperature_all = self.full_output['full_output']['layer']['temperature']
        else: 
            pressure_all = self.full_output['full_output']['layer']['pressure'][:, ng, nt]
            temperature_all = self.full_output['full_output']['layer']['temperature'][:, ng, nt]

        if not isinstance(pressures, (np.ndarray, list)): 
            raise Exception('check pressure input. It must be list or array.')

        kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 300))
        kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 400))
        if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
        if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
        kwargs['title'] = kwargs.get('title', 'Outgoing Thermal Radiation')
        kwargs['y_axis_label'] = kwargs.get('y_axis_label', 'Flux (erg/s/cm^3)')
        kwargs['x_axis_label'] = kwargs.get('x_axis_label', 'Wavelength [μm]')
        kwargs['y_axis_type'] = kwargs.get('y_axis_type', 'log')
        kwargs['x_axis_type'] = kwargs.get('x_axis_type', 'log') 

        wno = self.full_output['wavenumber']
        if isinstance(R, (int, float)): 
            wno, thermal = mean_regrid(wno, self.full_output['thermal'], R=R)
        else: 
            thermal = self.full_output['thermal']

        cols = pals.magma(max(3, len(pressures)))

        if self.backend == 'bokeh':
            fig = figure(**kwargs)
            fig.line(1e4/wno, thermal, color='black', line_width=4)

            for p, c in zip(pressures, cols): 
                ip = find_nearest_1d(pressure_all, p)
                t = temperature_all[ip]
                intensity = blackbody(t, 1/wno)[0] 
                flux = np.pi * intensity
                fig.line(1e4/wno, flux, color=c, alpha=0.5, legend_label=str(int(t))+' K at '+str(p)+' bars', line_width=4)
            return fig

        elif self.backend == 'matplotlib':
            fig = plt.figure(figsize=(kwargs['width']/100, kwargs['height']/100))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(1e4/wno, thermal, color='black', linewidth=4, label='Thermal')
            for p, c in zip(pressures, cols):
                ip = find_nearest_1d(pressure_all, p)
                t = temperature_all[ip]
                intensity = blackbody(t, 1/wno)[0] 
                flux = np.pi * intensity
                ax.plot(1e4/wno, flux, color=c, alpha=0.5, linewidth=4, label=f'{int(t)} K at {p} bars')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(kwargs['x_axis_label'])
            ax.set_ylabel(kwargs['y_axis_label'])
            ax.set_title(kwargs['title'])
            ax.legend()
            return fig

        elif self.backend == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=1e4/wno, y=thermal, name='Thermal', line=dict(color='black', width=4)))
            for p, c in zip(pressures, cols):
                ip = find_nearest_1d(pressure_all, p)
                t = temperature_all[ip]
                intensity = blackbody(t, 1/wno)[0] 
                flux = np.pi * intensity
                fig.add_trace(go.Scatter(x=1e4/wno, y=flux, name=f'{int(t)} K at {p} bars', line=dict(color=c, width=4), opacity=0.5))
            fig.update_xaxes(type="log", title_text=kwargs['x_axis_label'])
            fig.update_yaxes(type="log", title_text=kwargs['y_axis_label'])
            fig.update_layout(title=kwargs['title'], width=kwargs['width'], height=kwargs['height'])
            return fig

    def taumap(self, at_tau=1, wavelength=1, igauss=0):
        all_dtau_gas = self.full_output['taugas'][:, :, :, :, igauss]
        all_dtau_cld = self.full_output['taucld'][:, :, :, :, igauss]*self.full_output['layer']['cloud']['w0'][:, :, :, :]
        all_dtau_ray = self.full_output['tauray'][:, :, :, :, igauss]

        ng = all_dtau_gas.shape[2]
        nt = all_dtau_gas.shape[3]

        map_gas = np.zeros((ng, nt))
        map_cld = np.zeros((ng, nt))
        map_ray = np.zeros((ng, nt))

        wave = 1e4/self.full_output['wavenumber']
        iw = find_nearest_1d(wave, wavelength)

        for ig in range(ng):
            for it in range(nt):
                dtaugas = all_dtau_gas[:, iw, ig, it]
                dtaucld = all_dtau_cld[:, iw, ig, it]
                dtauray = all_dtau_ray[:, iw, ig, it]
                
                shape = len(dtaugas)
                taugas = np.zeros(shape+1)
                taucld = np.zeros(shape+1)
                tauray = np.zeros(shape+1)

                taugas[1:] = np.cumsum(dtaugas)
                taucld[1:] = np.cumsum(dtaucld)
                tauray[1:] = np.cumsum(dtauray)

                pressure = self.full_output['level']['pressure'][:, ig, it]

                ind_gas = find_nearest_1d(taugas, at_tau)
                ind_cld = find_nearest_1d(taucld, at_tau)
                ind_ray = find_nearest_1d(tauray, at_tau)

                if (len(taucld[taucld == 0]) == len(taucld.shape)): 
                    ind_cld = ind_cld*0 + shape

                map_gas[ig, it] = pressure[ind_gas]
                map_cld[ig, it] = pressure[ind_cld]
                map_ray[ig, it] = pressure[ind_ray]

        all_maps = [map_gas, map_cld, map_ray]
        labels = ['Molecular Opacity', 'Cloud Opacity', 'Rayleigh Opacity']

        latitude = self.full_output['latitude']  
        longitude = self.full_output['longitude'] 
        u, v = np.meshgrid(longitude, latitude)
        x, y, z = lon_lat_to_cartesian(u, v)

        if self.backend == 'matplotlib':
            fig = plt.figure(figsize=(18, 4))
            for i, w, l in zip(range(len(all_maps)), all_maps, labels):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                cmap_cm = plt.get_cmap('plasma')
                ax.plot_wireframe(x, y, z, color="gray")
                sc = ax.scatter(x, y, z, c=w.T.ravel(), cmap=cmap_cm, s=150)
                fig.colorbar(sc)
                ax.set_zlim3d(-1, 1)                    
                ax.set_ylim3d(-1, 1)                    
                ax.set_xlim3d(-1, 1)
                ax.view_init(0, 0)
                ax.set_title(l)
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.axis('off')
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            return fig

        elif self.backend == 'plotly':
            fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}]*3])
            for i, w, l in zip(range(len(all_maps)), all_maps, labels):
                fig.add_trace(go.Scatter3d(
                    x=x.ravel(), y=y.ravel(), z=z.ravel(),
                    mode='markers',
                    marker=dict(size=6, color=w.T.ravel(), colorscale='Plasma', showscale=True),
                    name=l
                ), row=1, col=i+1)
            fig.update_layout(height=500, width=1500)
            return fig

        elif self.backend == 'bokeh':
            figs = []
            for w, l in zip(all_maps, labels):
                fig_b = figure(title=l, x_axis_label='Longitude', y_axis_label='Latitude')
                fig_b.image(image=[w.T], x=longitude.min()*180/np.pi, y=latitude.min()*180/np.pi,
                        dw=(longitude.max()-longitude.min())*180/np.pi, dh=(latitude.max()-latitude.min())*180/np.pi,
                        palette="Plasma256")
                figs.append(fig_b)
            return row(*figs)

    def all_optics_1d(self, wave_range, return_output=False, legend=None, ng=None, nt=None, colors=Colorblind8, **kwargs):
        kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 300))
        kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 300))
        if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
        if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
        kwargs['y_axis_type'] = kwargs.get('y_axis_type', 'log')

        full_output_list = self.full_output if isinstance(self.full_output, list) else [self.full_output]

        if ((ng == None) & (nt == None)):
            pressure = full_output_list[0]['layer']['pressure']
        else: 
            pressure = full_output_list[0]['layer']['pressure'][:, ng, nt]

        kwargs['y_range'] = kwargs.get('y_range', [max(pressure), min(pressure)])     

        all_opds, all_ssas, all_g0s = [], [], []
        for i, results in enumerate(full_output_list): 
            if ((ng == None) & (nt == None)):
                press = results['layer']['pressure']
                ssa = results['layer']['cloud']['w0']
                g0 = results['layer']['cloud']['g0']
                opd = results['layer']['cloud']['opd']
            else: 
                press = results['layer']['pressure'][:, ng, nt]
                ssa = results['layer']['cloud']['w0'][:, :, ng, nt]
                g0 = results['layer']['cloud']['g0'][:, :, ng, nt]
                opd = results['layer']['cloud']['opd'][:, :, ng, nt]

            wno = results['wavenumber']
            inds = np.where((1e4/wno > wave_range[0]) & (1e4/wno < wave_range[1]))

            opd_mean = np.mean(opd[:, inds], axis=2)[:, 0]
            g0_mean = np.mean(g0[:, inds], axis=2)[:, 0]
            ssa_mean = np.mean(ssa[:, inds], axis=2)[:, 0]

            all_opds.append(opd_mean)
            all_ssas.append(ssa_mean)
            all_g0s.append(g0_mean)

        if self.backend == 'bokeh':
            fssa = figure(x_axis_label='Single Scattering Albedo', **kwargs)
            fg0 = figure(x_axis_label='Asymmetry', **kwargs)
            fopd = figure(x_axis_label='Optical Depth', y_axis_label='Pressure (bars)', x_axis_type='log', **kwargs)

            for i in range(len(full_output_list)):
                fopd.line(all_opds[i], pressure, color=colors[np.mod(i, len(colors))], line_width=3)
                fg0.line(all_g0s[i], pressure, color=colors[np.mod(i, len(colors))], line_width=3)
                
                if isinstance(legend, type(None)):
                    fssa.line(all_ssas[i], pressure, color=colors[np.mod(i, len(colors))], line_width=3)
                else:
                    fssa.line(all_ssas[i], pressure, color=colors[np.mod(i, len(colors))], line_width=3, legend_label=legend[i])
                    fssa.legend.location = 'top_left'

            fig = gridplot([[fopd, fssa, fg0]])

        elif self.backend == 'matplotlib':
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10, 5))
            for i in range(len(full_output_list)):
                l = legend[i] if legend is not None else None
                ax1.plot(all_opds[i], pressure, color=colors[np.mod(i, len(colors))], linewidth=3)
                ax2.plot(all_ssas[i], pressure, color=colors[np.mod(i, len(colors))], linewidth=3, label=l)
                ax3.plot(all_g0s[i], pressure, color=colors[np.mod(i, len(colors))], linewidth=3)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(kwargs['y_range'][0], kwargs['y_range'][1])
            ax1.set_ylabel('Pressure (bars)')
            ax1.set_xlabel('Optical Depth')
            ax2.set_xlabel('Single Scattering Albedo')
            ax3.set_xlabel('Asymmetry')
            if legend is not None:
                ax2.legend()
            fig.tight_layout()

        elif self.backend == 'plotly':
            fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=("Optical Depth", "Single Scattering Albedo", "Asymmetry"))
            for i in range(len(full_output_list)):
                l = legend[i] if legend is not None else f'Trace {i}'
                fig.add_trace(go.Scatter(x=all_opds[i], y=pressure, name=l, line=dict(color=colors[np.mod(i, len(colors))], width=3)), row=1, col=1)
                fig.add_trace(go.Scatter(x=all_ssas[i], y=pressure, name=l, line=dict(color=colors[np.mod(i, len(colors))], width=3), showlegend=False), row=1, col=2)
                fig.add_trace(go.Scatter(x=all_g0s[i], y=pressure, name=l, line=dict(color=colors[np.mod(i, len(colors))], width=3), showlegend=False), row=1, col=3)
            fig.update_xaxes(type="log", row=1, col=1)
            fig.update_yaxes(type="log", autorange="reversed")
            fig.update_layout(height=400, width=900)

        if return_output:
            return fig, [all_opds, all_ssas, all_g0s]
        else:
            return fig

    def thermal_contribution(self, tau_max=1.0, R=100, **kwargs):
        kwargs['norm'] = kwargs.get('norm', colors.LogNorm())
        kwargs['shading'] = kwargs.get('shading', 'auto')

        all_taus = np.squeeze(self.full_output['taugas'] + self.full_output['taucld'] + self.full_output['tauray'])
        if len(all_taus.shape) > 2:
            all_taus = all_taus[:, :, 0] if len(all_taus.shape) == 3 else all_taus[:, :, 0, 0]
        
        all_taus[all_taus > tau_max] = tau_max
        sum_taus = np.cumsum(all_taus, axis=0)

        pressure = self.full_output['layer']['pressure']
        if len(pressure.shape) > 1:
            pressure = pressure[:, 0] if len(pressure.shape) == 2 else pressure[:, 0, 0]

        press2D = np.transpose(np.repeat(pressure[np.newaxis], np.shape(sum_taus)[1], axis=0))

        temperatures = self.full_output['layer']['temperature']
        if len(temperatures.shape) > 1:
            temperatures = temperatures[:, 0] if len(temperatures.shape) == 2 else temperatures[:, 0, 0]

        bb = np.ones(np.shape(press2D))
        for i, temp in enumerate(temperatures):
            for j, wave in enumerate(1/self.full_output['wavenumber']):
                bb[i, j] = blackbody(temp, wave)[0][0]
        CF = bb[0:-1, :] * np.exp(-sum_taus[0:-1, :]) * all_taus[0:-1, :] / np.diff(np.log(press2D), axis=0)

        if not isinstance(R, type(None)):
            wno, _ = mean_regrid(self.full_output['wavenumber'], self.full_output['wavenumber'], R=R)
            CF_bin = np.zeros((len(pressure)-1, len(wno)))
            for i in range(len(pressure)-1):
                _, CF_bin[i, :] = mean_regrid(self.full_output['wavenumber'], CF[i, :], newx=wno)
        else: 
            CF_bin = CF
            wno = self.full_output['wavenumber']

        wave = 1e4/wno

        if self.backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=(15, 10))
            smap = ax.pcolormesh(wave, pressure[0:-1], CF_bin, **kwargs)
            ax.set_ylim(np.max(pressure), np.min(pressure))
            ax.set_yscale('log')
            ax.set_ylabel('Pressure (bar)', fontsize=20)
            ax.set_xlabel(r'Wavelength ($\mu$m)', fontdict={'fontsize':20})
            cm = plt.colorbar(smap)
            cm.ax.set_ylabel('Emission Contribution Function', fontdict={'fontsize':18})
            for l in cm.ax.yaxis.get_ticklabels():
                l.set_fontsize(16)
            ax.set_yticks(np.logspace(-5, 1, 7), minor=False)
            ax.set_yticklabels(np.logspace(-5, 1, 7), fontdict={'fontsize':18})
            return fig, ax, CF_bin

        elif self.backend == 'bokeh':
            fig = figure(x_axis_label='Wavelength (μm)', y_axis_label='Pressure (bar)', y_axis_type='log', y_range=[np.max(pressure), np.min(pressure)])
            fig.image(image=[CF_bin], x=np.min(wave), y=np.min(pressure), dw=np.max(wave)-np.min(wave), dh=np.max(pressure)-np.min(pressure), palette="Viridis256")
            return fig, None, CF_bin

        elif self.backend == 'plotly':
            fig = go.Figure(data=go.Heatmap(z=CF_bin, x=wave, y=pressure[0:-1], colorscale='Viridis'))
            fig.update_yaxes(type="log", autorange="reversed")
            fig.update_layout(title="Emission Contribution Function", xaxis_title="Wavelength (μm)", yaxis_title="Pressure (bar)")
            return fig, None, CF_bin

    def transmission_contribution(self, R=None, **kwargs):
        DTAU = (self.full_output['taugas'][:, :, 0] + 
                self.full_output['taucld'][:, :, 0] +  
                self.full_output['tauray'][:, :, 0])
        z = self.full_output['level']['z']
        dz = self.full_output['level']['dz']
        nlevel = len(dz)
        nwno = DTAU.shape[1]
        rstar = 1
        k_b = 1
        amu = 1
        player = self.full_output['layer']['pressure']
        tlayer = self.full_output['layer']['temperature']
        colden = self.full_output['layer']['column_density']
        mmw = self.full_output['layer']['mmw']

        zs = []
        for i in range(DTAU.shape[0]):
            dtau_copy = copy.deepcopy(DTAU)
            dtau_copy[i, :] = 0 
            if i == 0: 
                norm = get_transit_1d(z, dz, nlevel, nwno, rstar, mmw, k_b, amu,
                                player, tlayer, colden, DTAU)
            zs += [get_transit_1d(z, dz, nlevel, nwno, rstar, mmw, k_b, amu,
                                player, tlayer, colden, dtau_copy)]
        
        CF = (norm - np.array(zs)) / np.sum(norm - np.array(zs), axis=0)
        
        if not isinstance(R, type(None)):
            wno, _ = mean_regrid(self.full_output['wavenumber'], self.full_output['wavenumber'], R=R)
            CF_bin = np.zeros((len(self.full_output['layer']['pressure']), len(wno)))
            for i in range(len(self.full_output['layer']['pressure'])):
                _, CF_bin[i, :] = mean_regrid(self.full_output['wavenumber'], CF[i, :], newx=wno)
        else: 
            CF_bin = CF
            wno = self.full_output['wavenumber']

        wave = 1e4/wno

        if self.backend == 'matplotlib':
            kwargs['norm'] = kwargs.get('norm', colors.LogNorm())
            kwargs['shading'] = kwargs.get('shading', 'auto')

            fig, ax = plt.subplots()
            smap = ax.pcolormesh(wave, self.full_output['layer']['pressure'], CF_bin, **kwargs)
            ax.set_ylim(np.max(self.full_output['layer']['pressure']), np.min(self.full_output['layer']['pressure']))
            ax.set_yscale('log')
            ax.set_ylabel('Pressure (bar)')
            ax.set_xlabel(r'Wavelength ($\mu$m)')
            plt.colorbar(smap, label='Transmission CF')
            return fig, ax, wave, CF_bin

        elif self.backend == 'bokeh':
            fig = figure(x_axis_label='Wavelength (μm)', y_axis_label='Pressure (bar)', y_axis_type='log', y_range=[np.max(self.full_output['layer']['pressure']), np.min(self.full_output['layer']['pressure'])])
            fig.image(image=[CF_bin], x=np.min(wave), y=np.min(self.full_output['layer']['pressure']), dw=np.max(wave)-np.min(wave), dh=np.max(self.full_output['layer']['pressure'])-np.min(self.full_output['layer']['pressure']), palette="Viridis256")
            return fig, None, wave, CF_bin

        elif self.backend == 'plotly':
            fig = go.Figure(data=go.Heatmap(z=CF_bin, x=wave, y=self.full_output['layer']['pressure'], colorscale='Viridis'))
            fig.update_yaxes(type="log", autorange="reversed")
            fig.update_layout(title="Transmission Contribution Function", xaxis_title="Wavelength (μm)", yaxis_title="Pressure (bar)")
            return fig, None, wave, CF_bin


# ==================== REMAINING ORIGINAL FUNCTIONS ====================

def plot_cld_input(nwno, nlayer, filename=None, df=None, pressure=None, wavelength=None, **pd_kwargs):
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

    if not isinstance(filename, type(None)):
        dat01 = pd.read_csv(filename, **pd_kwargs)
    elif not isinstance(df, type(None)):
        dat01 = df
        
    w0 = np.reshape(dat01['w0'].astype(float).values, (nlayer, nwno))
    opd = np.reshape(dat01['opd'].astype(float).values, (nlayer, nwno)) + 1e-60
    g0 = np.reshape(dat01['g0'].astype(float).values, (nlayer, nwno))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot W0
    im1 = ax1.pcolormesh(wave, pressure, w0, cmap='magma', vmin=0, vmax=1)
    ax1.set_title("Single Scattering Albedo")
    fig.colorbar(im1, ax=ax1)

    # Plot OPD
    im2 = ax2.pcolormesh(wave, pressure, opd, cmap='viridis_r', norm=colors.LogNorm(vmin=1e-3, vmax=10))
    ax2.set_title("Cloud Optical Depth Per Layer")
    fig.colorbar(im2, ax=ax2)

    # Plot G0
    im3 = ax3.pcolormesh(wave, pressure, g0, cmap='gray_r', vmin=0, vmax=1)
    ax3.set_title("Asymmetry Parameter")
    fig.colorbar(im3, ax=ax3)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(wavelength_label)
        ax.set_ylabel(pressure_label)
        if yaxis == 'log': ax.set_yscale('log')
        ax.set_ylim(pressure.max(), pressure.min())

    return fig

def spectrum_hires(wno, alb, legend=None, **kwargs):
    import holoviews as hv
    from holoviews.operation.datashader import datashade

    hv.extension('bokeh')

    kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 345))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 1000))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label', 'Albedo')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label', 'Wavelength [μm]')
    kwargs['y_range'] = kwargs.get('y_range', [0, 1.2])
    kwargs['x_range'] = kwargs.get('x_range', [0.3, 1])

    points_og = datashade(hv.Curve((1e4/wno, alb)))
    return points_og

def plot_evolution(evo, y="Teff", **kwargs):
    kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 400))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 500))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['title'] = kwargs.get('title', 'Thermal Evolution')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label', y)
    kwargs['x_axis_label'] = kwargs.get('x_axis_label', 'Age(years)')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type', 'log') 

    f = figure(**kwargs)

    lp = len(y)
    evo_hot = evo['hot']
    evo_cold = evo['cold']
    source_hot = ColumnDataSource(data=dict(evo_hot))
    source_cold = ColumnDataSource(data=dict(evo_cold))

    colors_list = pals.viridis(10)
    for i, ikey in enumerate(list(evo_hot.keys())[1:]):
        if y in ikey:
            mass = int(ikey[ikey.rfind(y[-1])+1:ikey.find('M')])
            icolor = mass - 1
            f1 = f.line(x='age_years', y=ikey, line_width=2,
                   color=colors_list[icolor],
                   legend_label='Hot Start', source=source_hot)
            f.add_tools(HoverTool(renderers=[f1], tooltips=[('Teff', f'@Teff{ikey[lp:]} K'),
                                                            ('Age', '@age_years Yrs'),
                                                            ('Gravity', f'@grav_cgs{ikey[lp:]} cm/s2'),
                                                           ('Mass', str(mass)+" Mj")]
                                  ))
            f2 = f.line('age_years', ikey, line_width=2,
                   color=colors_list[icolor],
                  line_dash='dashed', legend_label='Cold Start', source=source_cold)
            f.add_tools(HoverTool(renderers=[f2], tooltips=[('Teff', f'@Teff{ikey[lp:]} K'),
                                                            ('Age', '@age_years Yrs'),
                                                            ('Gravity', f'@grav_cgs{ikey[lp:]} cm/s2'),
                                                           ('Mass', str(mass)+" Mj")]))

    color_bar = ColorBar(title='Mass (Mj)',
        color_mapper=LinearColorMapper(palette="Viridis256", low=1, high=10), 
        label_standoff=12, location=(0, 0))

    f.add_layout(color_bar, 'right')
    return f

def heatmap_taus(out, R=0):
    nrow = 1
    ncol = 3 
    fig = plt.figure(figsize=(6*ncol, 4*nrow))
    for it, itau in enumerate(['taugas', 'taucld', 'tauray']):
        ax = fig.add_subplot(nrow, ncol, it+1)
        tau_bin = []
        for i in range(out['full_output'][itau].shape[0]):
            if R == 0: 
                x, y = out['wavenumber'], out['full_output'][itau][i, :, 0]
            else: 
                x, y = mean_regrid(out['wavenumber'],
                                  out['full_output'][itau][i, :, 0], R=R)
            tau_bin += [[y]]
        tau_bin = np.array(tau_bin)
        tau_bin[tau_bin == 0] = 1e-100
        tau_bin = np.log10(tau_bin)[:, 0, :]
        X, Y = np.meshgrid(1e4/x, out['full_output']['layer']['pressure'])
        Z = tau_bin
        pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap='RdBu_r')
        cbar = fig.colorbar(pcm, ax=ax)
        pcm.set_clim(-3.0, 3.0)
        ax.set_title(itau)
        ax.set_yscale('log')
        ax.set_ylim([np.max(out['full_output']['layer']['pressure']), np.min(out['full_output']['layer']['pressure'])])
        ax.set_ylabel('Pressure(bars)')
        ax.set_xlabel('Wavelength(um)')
        cbar.set_label('log Opacity')
    return ax

def phase_snaps(allout, x='longitude', y='pressure', z='temperature', palette='RdBu_r',
    y_log=True, x_log=False, z_log=False, col_wrap=3, collapse='np.mean', igauss=0):
    allowed_xy = ['longitude', 'latitude', 'pressure']
    if x not in allowed_xy:
        raise Exception(f'Allowable x options are {allowed_xy}')
    if y not in allowed_xy:
        raise Exception(f'Allowable y options are {allowed_xy}')

    allowed_z = ['temperature', 'taugas', 'taucld', 'tauray', 'w0', 'g0', 'opd']
    if z not in allowed_z:
        raise Exception(f'Allowable z options are {allowed_z}')

    phases = list(allout.keys())
    nrows = int(np.ceil(len(phases) / col_wrap))
    fig = plt.figure(figsize=(4*nrows, 3*col_wrap), dpi=80)

    for ind in range(len(phases)):
        iphase = phases[ind]
        full_output = allout[iphase]['full_output']

        xd = explore(full_output, x)
        yd = explore(full_output, y)
        if len(xd.shape) == 1:
            x_1d = xd*180/np.pi
        else: 
            x_1d = xd[:, 0, 0]
        if len(yd.shape) == 1:
            y_1d = yd*180/np.pi  
        else: 
            y_1d = yd[:, 0, 0]

        x_mesh, y_mesh = np.meshgrid(x_1d, y_1d)

        zd = explore(full_output, z)
        len_zd = len(zd.shape)
        if len_zd == 3:
            to_collapse = [i for i, key in enumerate(['pressure', 'longitude', 'latitude']) if key not in [x, y]]
        elif len_zd == 4:
            to_collapse = [i for i, key in enumerate(['pressure', 'wavelength', 'longitude', 'latitude']) if key not in [x, y]]
        elif len_zd == 5:
            zd = zd[:, :, :, :, igauss]
            to_collapse = [i for i, key in enumerate(['pressure', 'wavelength', 'longitude', 'latitude']) if key not in [x, y]]

        allowed_collapse = ['np.mean', 'np.max', 'np.min', 'np.median']
        if ((len(to_collapse) >= 1) & (not isinstance(collapse, list))): 
            collapse = [collapse]*len(to_collapse)

        count = 0
        for i, method in zip(to_collapse, collapse): 
            if ((isinstance(method, str)) & (method in allowed_collapse)):
                foo = eval(method)
                zd = foo(zd, axis=i-count); count += 1
            elif isinstance(method, int):
                select = [':']*len(zd.shape)
                select[i-count] = str(method); count += 1
                zd = eval('zd['+','.join(select)+']')
            else: 
                raise Exception(f'Collapse not allowed.')

        minmax = {  'z': [zd.min(), zd.max()],
                    'x': [x_mesh.min(), x_mesh.max()],
                    'y': [y_mesh.min(), y_mesh.max()]}
        if x == 'pressure': minmax['x'] = minmax['x'][::-1]
        if y == 'pressure': minmax['y'] = minmax['y'][::-1]
        
        ax = fig.add_subplot(col_wrap, nrows, ind+1)
        if z_log: 
            c = ax.pcolormesh(x_mesh, y_mesh, zd, cmap=palette, 
                          norm=colors.LogNorm(vmin=minmax['z'][0], vmax=minmax['z'][1]))
        else: 
            c = ax.pcolormesh(x_mesh, y_mesh, zd, cmap=palette, 
                          vmin=minmax['z'][0], vmax=minmax['z'][1])

        ax.set_title(f'Phase={int(iphase*180/np.pi)}')
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
    palette=pals.Spectral11, verbose=True, 
    reorder_output=False, **kwargs):
    kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 400))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 600))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['title'] = kwargs.get('title', 'Phase Curves')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label', to_plot)
    kwargs['x_axis_label'] = kwargs.get('x_axis_label', 'Orbital Phase')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type', 'linear')
    kwargs['x_axis_type'] = kwargs.get('x_axis_type', 'linear') 

    fig = figure(**kwargs)

    if (isinstance(collapse, (float, int)) or isinstance(collapse, str)):
        collapse = [collapse]
    elif isinstance(collapse, list): 
        for i in collapse : assert isinstance(i, (float, int))
    else: 
        raise Exception('Collapse must either be float, str or list')
    if len(collapse) > len(palette): 
        palette = pals.magma(len(collapse))

    all_curves = np.zeros((len(allout.keys()), len(collapse)))
    all_ws = np.zeros(len(collapse))
    phases = np.array(list(allout.keys()))
    
    for i, iphase in enumerate(phases):
        for j, icol in enumerate(collapse): 
            if icol in ['np.mean', 'np.sum']:
                w, f = eval(icol)(allout[iphase]['wavenumber']), eval(icol)(allout[iphase][to_plot])
                all_curves[i, j] = f 
                all_ws[j] = w
            else: 
                w, f = mean_regrid(allout[iphase]['wavenumber'],
                                   allout[iphase][to_plot], R=R)
                iw = np.argmin(abs(1e4/w-icol)) 
                w, f = w[iw], f[iw]
                all_curves[i, j] = f
                all_ws[j] = w
    legend_it = []  
    for i in range(len(collapse)): 
        f = fig.line(phases*180/np.pi, all_curves[:, i], line_width=3, color=palette[i])
        legend_it.append((str(int(1e4/all_ws[i]*100)/100)+'um', [f]))

    legend = Legend(items=legend_it, location=(0, -20),
                        **_get_legend_cols(len(legend_it), kwargs.get('height')))
    legend.click_policy = "mute"
    fig.add_layout(legend, 'left') 
    
    fig.xgrid.grid_line_alpha = 0
    fig.ygrid.grid_line_alpha = 0
    plot_format(fig)

    front_half_phases = phases[:len(phases)//2]
    back_half_phases = phases[len(phases)//2:] - (2*np.pi)
    reorder_phases = np.concatenate((back_half_phases, front_half_phases))

    front_half_all_curves = all_curves[:len(all_curves)//2]
    back_half_all_curves = all_curves[len(all_curves)//2:]
    reorder_all_curves = np.concatenate((back_half_all_curves, front_half_all_curves))

    if to_plot == "fpfs_reflected" or to_plot == "albedo":
        fig2 = figure(**kwargs)
        for i in range(len(collapse)): 
            fig2.line(reorder_phases, reorder_all_curves[:, i], line_width=3, color=palette[i])
    
        legend2 = Legend(items=legend_it, location=(0, -20),
                        **_get_legend_cols(len(legend_it), kwargs.get('height')))
        legend2.click_policy = "mute"
        fig2.add_layout(legend2, 'left')

        fig2.xgrid.grid_line_alpha = 0
        fig2.ygrid.grid_line_alpha = 0
        plot_format(fig2)
    
    if reorder_output:
        return reorder_phases, reorder_all_curves, all_ws, fig
    
    return phases, all_curves, all_ws, fig

def molecule_contribution(contribution_out, opa, min_pressure=4.5, R=100, **kwargs):
    kwargs['height'] = kwargs.get('plot_height', kwargs.get('height', 400))
    kwargs['width'] = kwargs.get('plot_width', kwargs.get('width', 500))
    if 'plot_width' in kwargs.keys() : kwargs.pop('plot_width')
    if 'plot_height' in kwargs.keys() : kwargs.pop('plot_height')
    kwargs['y_axis_label'] = kwargs.get('y_axis_label', 'Tau Pressure (bars)')
    kwargs['x_axis_label'] = kwargs.get('x_axis_label', 'Wavelength')
    kwargs['y_axis_type'] = kwargs.get('y_axis_type', 'log')
    kwargs['y_range'] = kwargs.get('y_range', [1e2, 1e-4])
    kwargs['title'] = kwargs.get('title', 'User Input Tau Pressure Surface')

    tau_p_surface = contribution_out['tau_p_surface']
    wno = []
    spec = []
    labels = []
    for j in tau_p_surface.keys(): 
        x, y = mean_regrid(opa.wno, tau_p_surface[j], R=R) 
        if np.min(y) < min_pressure: 
            wno += [x]
            spec += [y]
            labels += [j]
    fig = spectrum(wno, spec, legend=labels, **kwargs)
    return fig

def brightness_temperature(out_dict, plot=True, R=None, with_guide=True): 
    flux = out_dict['thermal']/np.pi*1e-7
    wno = out_dict['wavenumber']
    lam = 1e4/wno
    a = 1.43877735e-2  
    hc2 = 2*5.95521476e-17   
    flux = flux*1e6
    lam = lam*1e-6

    T_B  = (a/lam)/np.log(1+(hc2/flux/lam**5))

    if not isinstance(R, type(None)):
        wno, T_B = mean_regrid(wno, T_B, R=R)

    if plot: 
        if with_guide: t_eq = out_dict['full_output']['layer']['temperature']
        f = plt.figure(figsize=(15, 8))
        plt.xlabel("Wavelength [microns]", fontsize=20)
        plt.ylabel("Brightness Temperature [K]", fontsize=20)
        plt.xlim(min(1e4/wno), max(1e4/wno))
        if with_guide: plt.ylim(np.min(t_eq)-0.1*np.min(t_eq), np.max(t_eq)+0.1*np.min(t_eq))

        plt.semilogx(1e4/wno, T_B, color='k', label="Brightness Temperature")
        if with_guide: plt.axhline(np.min(t_eq), linewidth=5, color="blue", label="Minimum Temperature")
        if with_guide: plt.axhline(np.max(t_eq), linewidth=5, color="red", label="Maximum Temperature")
        plt.legend(fontsize=10)        
        return T_B, f
    else: 
        return T_B

def animate_convergence(clima_out, picaso_bundle, opacity, calculation='thermal',
    wave_range=[0.3, 6],
    molecules=['H2O', 'CH4', 'CO', 'NH3']):
    map_calc = {'thermal': 'thermal', 'reflected': 'albedo', 'transmission': 'transit_depth'}
    t_eq, p_eq, all_profiles_eq = (
                np.copy(clima_out['temperature']), 
                np.copy(clima_out['pressure']), 
                np.copy(clima_out['all_profiles']))
    
    if 'cld_output_final' in clima_out:
        all_opd = np.copy(clima_out['all_opd'])
        cld_p = np.copy(clima_out['cld_output_picaso']['pressure'][0::196])
    
    nlevel = len(t_eq)
    nstep = int(all_profiles_eq.shape[0]/nlevel)
    split_profiles = np.array_split(all_profiles_eq, nstep)

    mols_to_plot = {i: np.zeros(all_profiles_eq.size) for i in molecules}
    spec = np.zeros(shape=(nstep, opacity.nwno))
    
    for i in range(nstep):
        picaso_bundle.add_pt(split_profiles[i], p_eq)
        picaso_bundle.premix_atmosphere(opacity)

        if 'cld_output_picaso' in clima_out:
            picaso_bundle.clouds(df=clima_out['cld_output_picaso'])

        df_spec = picaso_bundle.spectrum(opacity, calculation=calculation, full_output=True)
        spec[i, :] = df_spec[map_calc[calculation]]
        for imol in molecules:
            mols_to_plot[imol][i*nlevel:(i+1)*nlevel] = picaso_bundle.inputs['atmosphere']['profile'][imol]
    
    wh = np.where((1e4/df_spec['wavenumber'] > wave_range[0]) & (1e4/df_spec['wavenumber'] < wave_range[1]))
    wv = 1e4/df_spec['wavenumber'][wh]    

    writergif = animation.PillowWriter(fps=3) 
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig = plt.figure(figsize=(50, 10))

    if 'cld_output_picaso' in clima_out:
        x = '''
        AA.BB.CC.DD
        '''
        ax = fig.subplot_mosaic(x, gridspec_kw={
            "height_ratios": [1],
            "width_ratios": [1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1]})
    else:
        x = '''
        AA.BB.CC
        '''
        ax = fig.subplot_mosaic(x, gridspec_kw={
            "height_ratios": [1],
            "width_ratios": [1, 1, 0.1, 1, 1, 0.1, 1, 1]})

    temp = split_profiles[0]
    lines = {}
    for imol, col in zip(molecules, Colorblind8):
        lines[imol], = ax['B'].loglog(mols_to_plot[imol][0:nlevel], p_eq, linewidth=3, color=col, label=imol)

    lines['temp'], = ax['A'].semilogy(temp, p_eq, linewidth=3, color='k')
    if calculation == 'thermal':
        lines['spec'], = ax['C'].semilogy(1e4/df_spec['wavenumber'], spec[0, :], linewidth=3, color="k")
    else: 
        lines['spec'], = ax['C'].plot(1e4/df_spec['wavenumber'], spec[0, :], linewidth=3, color="k")
    
    if 'cld_output_picaso' in clima_out:
        lines['opd'], = ax['D'].loglog(all_opd[:91], np.append(cld_p, 0), linewidth=3, color='k')

    def init():
        ax['A'].set_xlabel('Temperature [K]', fontsize=30)
        ax['A'].set_ylabel('Pressure [Bars]', fontsize=30)
        ax['A'].set_xlim(0, max(t_eq))
        ax['A'].set_ylim(max(p_eq), min(p_eq))
        ax['B'].set_xlabel('Abundance [V/V]', fontsize=30)
        ax['B'].set_ylabel('Pressure [Bars]', fontsize=30)
        ax['B'].set_xlim(1e-6, 1e-2)
        ax['B'].set_ylim(max(p_eq), min(p_eq))
        ax['B'].legend(fontsize=20)
        ax['C'].set_xlabel(r'Wavelength [$\mu$m]', fontsize=30)
        ax['C'].set_ylabel('Spectrum', fontsize=30)
        ax['C'].set_xlim(0, 6)
        ax['A'].minorticks_on()
        ax['A'].tick_params(axis='both', which='major', length=30, width=2, direction='in', labelsize=30)
        ax['A'].tick_params(axis='both', which='minor', length=10, width=2, direction='in', labelsize=30)
        ax['B'].minorticks_on()
        ax['B'].tick_params(axis='both', which='major', length=30, width=2, direction='in', labelsize=30)
        ax['B'].tick_params(axis='both', which='minor', length=10, width=2, direction='in', labelsize=30)
        ax['C'].minorticks_on()
        ax['C'].tick_params(axis='both', which='major', length=30, width=2, direction='in', labelsize=30)
        ax['C'].tick_params(axis='both', which='minor', length=10, width=2, direction='in', labelsize=30)

        if 'cld_output_picaso' in clima_out:
            ax['D'].set_xlabel('Optical Depth', fontsize=30)
            ax['D'].set_ylabel('Pressure [Bars]', fontsize=30)
            ax['D'].set_xlim(1e-7, 1e3)
            ax['D'].set_ylim(max(p_eq), min(p_eq))
            ax['D'].minorticks_on()
            ax['D'].tick_params(axis='both', which='major', length=30, width=2, direction='in', labelsize=30)
            ax['D'].tick_params(axis='both', which='minor', length=10, width=2, direction='in', labelsize=30)

        for ikey in molecules+['temp']:
            lines[ikey].set_ydata(p_eq)
        lines['spec'].set_xdata(wv)

        if 'cld_output_picaso' in clima_out:
            lines['opd'].set_ydata(np.append(cld_p, max(cld_p)+1))
        return lines
    
    def animate(i):                       
        lines['temp'].set_xdata(split_profiles[i])
        for imol in molecules:
            lines[imol].set_xdata(mols_to_plot[imol][i*nlevel:(i+1)*nlevel])
        lines['spec'].set_ydata(spec[i, :][wh])
        if 'cld_output_picaso' in clima_out:
            lines['opd'].set_xdata(np.append(all_opd[i*90:(i+1)*90], 1e-50))
        return lines

    ani = animation.FuncAnimation(fig, animate, frames=nstep, init_func=init, interval=50, blit=False)
    plt.close()
    return ani

def create_heat_map(data, rayleigh=True, extend=False, plot_height=300, plot_width=300, font_size="12px"):
    reverse = True
    data.columns.name = 'w0' 
    data.index.name = 'g0' 
    data.index = data.index.astype(str)
    data = data.rename(index={"-1.0": "Ray"})
    if not rayleigh:
        data = data.drop(["Ray"])  
    for w in data.columns[0:]:
        if pd.isnull(data.loc['0.0'][w]):
            data = data.drop(columns=[w])
            reverse = False

    x_range = list(data.index)
    if reverse:
        y_range = list(reversed(data.columns))
    else:
        y_range = list(data.columns)

    df = pd.DataFrame(data.stack(), columns=['albedo']).reset_index()

    colors_list = RdGy[11]
    bd = max(abs(df.albedo.min()), abs(df.albedo.max()))
    mapper = LinearColorMapper(palette=colors_list, low=-bd, high=bd)

    color_bar_height = plot_height + 11
    color_bar_width = int(plot_width * 0.26)

    p = figure(height=plot_height, width=plot_width,
       y_range=y_range, x_range=x_range,
       x_axis_location="above"
        )

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = font_size
    p.axis.major_label_standoff = 20
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect(x="g0", y="w0", width=1, height=1,
       source=df,
       fill_color={'field': 'albedo', 'transform': mapper},
       line_color='black')

    color_bar = ColorBar(color_mapper=mapper,
                    major_label_text_font_size=font_size,
                    ticker=BasicTicker(desired_num_ticks=len(colors_list)),
                    label_standoff=12, border_line_color=None, location=(0, 10))

    color_bar_plot = figure(
                        title_location="right", 
                        height=color_bar_height, width=color_bar_width, 
                        min_border=0, 
                        outline_line_color=None
                        )

    color_bar_plot.add_layout(color_bar, 'right')
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = '24px'
    
    p.axis.major_label_text_font_size = font_size
    layout = row(p, color_bar_plot)
    return layout

def rt_heatmap(data, figure_kwargs={}, cmap_kwargs={}):
    reverse = True
    data.columns.name = 'w0' 
    data.index.name = 'g0' 
    data.index = data.index.astype(str)
    x_range = list(data.index)
    if reverse:
        y_range = list(reversed(data.columns))
    else:
        y_range = list(data.columns)

    df = pd.DataFrame(data.stack(), columns=['albedo']).reset_index()
    bd = max(abs(df.albedo.min()), abs(df.albedo.max()))

    cmap_kwargs['palette'] = cmap_kwargs.get('palette', pals.RdGy[11])
    cmap_kwargs['low'] = cmap_kwargs.get('low', -bd)
    cmap_kwargs['high'] = cmap_kwargs.get('high', bd)    

    mapper = LinearColorMapper(**cmap_kwargs)
    colors_list = cmap_kwargs['palette']
    
    figure_kwargs['height'] = figure_kwargs.get('height', 400)
    figure_kwargs['width'] = figure_kwargs.get('width', 300)
    figure_kwargs['x_axis_location'] = figure_kwargs.get('x_axis_location', 'above')
    figure_kwargs['tools'] = figure_kwargs.get('tools', "hover,save,pan,box_zoom,reset,wheel_zoom")
    figure_kwargs['toolbar_location'] = figure_kwargs.get('toolbar_location', 'below') 
    figure_kwargs['y_range'] = figure_kwargs.get('y_range', y_range) 
    figure_kwargs['x_range'] = figure_kwargs.get('x_range', x_range) 
    figure_kwargs['y_axis_label'] = figure_kwargs.get('y_axis_label', 'Single Scattering Albedo') 
    figure_kwargs['x_axis_label'] = figure_kwargs.get('x_axis_label', 'Asymmetry')
    figure_kwargs['title'] = figure_kwargs.get('title', '% Diff')
    
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
                     ticker=BasicTicker(desired_num_ticks=len(colors_list)),
                     label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'below')
    p.axis.major_label_text_font_size = '12px'
    return p

def pt_adiabat(clima_out, input_class, opacityclass, plot=True):
    t_table = input_class.inputs['climate']['t_table']
    p_table = input_class.inputs['climate']['p_table']
    grad = input_class.inputs['climate']['grad']
    cp = input_class.inputs['climate']['cp']
    moist = input_class.inputs['climate']['moistgrad']
    AdiabatBundle = namedtuple('AdiabatBundle', ['t_table', 'p_table', 'grad', 'cp'])
    AdiabatBundle = AdiabatBundle(t_table, p_table, grad, cp)

    Atmosphere = calculate_atm(input_class, opacityclass, only_atmosphere=True)
    layer_p = clima_out['spectrum_output']['full_output']['layer']['pressure']
    
    grad, cp = convec(clima_out['temperature'], clima_out['pressure'],
                      AdiabatBundle, Atmosphere, moist=moist)
                      
    plt.semilogy(clima_out['dtdp'], layer_p)
    plt.semilogy(grad, layer_p) 
    plt.ylim([1e4, 1e-4])
    plt.xlabel('dT/dP vs adiabat')
    plt.ylabel('Pressure(bars)')
    return cp, grad, clima_out['dtdp'], layer_p
