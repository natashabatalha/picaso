from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Colorblind8
import numpy as np
import pandas as pd
from bokeh.layouts import column,row
import numpy as np
from bokeh.palettes import magma as colfun1
from bokeh.palettes import viridis as colfun2
from bokeh.palettes import gray as colfun3
from bokeh.palettes import Spectral11
from bokeh.models import HoverTool
from bokeh.models import LinearColorMapper, LogTicker,BasicTicker, ColorBar,LogColorMapper,Legend
import os 
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews.operation.datashader import datashade
hv.extension('bokeh')

def mixing_ratio(full_output,**kwargs):
	"""Returns plot of mixing ratios 

	Parameters
	----------
	full_output : class
		picaso.atmsetup.ATMSETUP
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


	fig = figure(**kwargs)
	cols = colfun1(len(molecules))
	legend_it=[]	
	for mol , c in zip(molecules,cols):
		ind = np.where(mol==np.array(molecules))[0][0]
		f = fig.line(full_output['layer']['mixingratios'][mol],pressure, color=c, line_width=3,
					muted_color=c, muted_alpha=0.2)
		legend_it.append((mol, [f]))

	legend = Legend(items=legend_it, location=(0, -20))
	legend.click_policy="mute"
	fig.add_layout(legend, 'left')  

	return fig
	
def pt(full_output,**kwargs):
	"""Returns plot of pressure temperature profile

	Parameters
	----------
	full_output : class
		picaso.atmsetup.ATMSETUP
	**kwargs : dict 
		Any key word argument for bokeh.figure() 
	"""
	#set plot defaults
	pressure = full_output['layer']['pressure']
	temperature = full_output['layer']['temperature']

	kwargs['plot_height'] = kwargs.get('plot_height',300)
	kwargs['plot_width'] = kwargs.get('plot_width',400)
	kwargs['title'] = kwargs.get('title','Pressure-Temperature Profile')
	kwargs['y_axis_label'] = kwargs.get('y_axis_label','Pressure(Bars)')
	kwargs['x_axis_label'] = kwargs.get('x_axis_label','Temperature (K)')
	kwargs['y_axis_type'] = kwargs.get('y_axis_type','log')
	kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')	
	kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure),np.min(pressure)])

	fig = figure(**kwargs)

	f = fig.line(temperature,pressure,line_width=3) 
	plot_format(fig)
	return fig

def spectrum(wno, alb,legend=None, **kwargs):
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
		Any key word argument for bokeh.plotting.figure()

	Returns
	-------
	bokeh plot
	"""
	kwargs['plot_height'] = kwargs.get('plot_height',345)
	kwargs['plot_width'] = kwargs.get('plot_width',1000)
	kwargs['y_axis_label'] = kwargs.get('y_axis_label','Albedo')
	kwargs['x_axis_label'] = kwargs.get('x_axis_label','Wavelength [μm]')
	kwargs['y_range'] = kwargs.get('y_range',[0,1.2])
	kwargs['x_range'] = kwargs.get('x_range',[0.3,1])

	fig = figure(**kwargs)

	if isinstance(wno, list):
		if legend == None: legend=[None]*len(wno) 
		for w, a,i,l in zip(wno, alb, range(len(wno)), legend):
			fig.line(1e4/w, a, legend=l, color=Colorblind8[np.mod(i, len(Colorblind8))], line_width=3)
	else: 
		fig.line(1e4/wno, alb, legend=legend, color=Colorblind8[0], line_width=3)

	plot_format(fig)
	return fig


def photon_attenuation(full_output, at_tau=0.5,**kwargs):
	"""
	Plot breakdown of gas opacity, cloud opacity, 
	Rayleigh scattering opacity at a specified pressure level. 
	
	Parameters
	----------
	full_output : class 
		picaso.atmsetup.ATMSETUP
	at_tau : float 
		Opacity at which to plot the cumulative opacity. 
		Default =1 bar. 
	**kwargs : dict 
		Any key word argument for bokeh.plotting.figure()

	Returns
	-------
	bokeh plot
	"""
	wave = 1e4/full_output['wavenumber']

	dtaugas = full_output['taugas']
	dtaucld = full_output['taucld']*full_output['layer']['cloud']['w0']
	dtauray = full_output['tauray']
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

	ind_gas = find_nearest(taugas, at_tau)
	ind_cld = find_nearest(taucld, at_tau)
	ind_ray = find_nearest(tauray, at_tau)

	if (len(taucld[taucld == 0]) == taucld.shape[0]*taucld.shape[1]) : 
		ind_cld = ind_cld*0 + shape[0]

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
	return fig #,wave,at_pressures_gas,at_pressures_cld,at_pressures_ray

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


def plot_cld_input(nwno, nlayer, filename=None,df=None,**pd_kwargs):
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
	file : str 
		(Optional)Path to cloud input file
	df : str 
		(Optional)Dataframe of cloud input file
	pd_kwargs : kwargs
		Pandas key word arguments for `pandas.read_csv`

	Returns
	-------
	Three bokeh plots with the single scattering, optical depth, and assymetry maps
	"""
	cols = colfun1(200)
	color_mapper = LinearColorMapper(palette=cols, low=0, high=1)

	if not isinstance(filename,type(None)):
		dat01 = pd.read_csv(filename, **pd_kwargs)
	elif not isinstance(df,type(None)):
		dat01=df

	#PLOT W0
	scat01 = np.flip(np.reshape(dat01['w0'].values,(nlayer,nwno)),0)
	xr, yr = scat01.shape
	f01a = figure(x_range=[0, yr], y_range=[0,xr],
						   x_axis_label='Wavenumber Grid', y_axis_label='Pressure Grid, TOA ->',
						   title="Single Scattering Albedo",
						  plot_width=300, plot_height=300)


	f01a.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw =yr )

	color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
					   label_standoff=12, border_line_color=None, location=(0,0))

	f01a.add_layout(color_bar, 'left')


	#PLOT OPD
	scat01 = np.flip(np.reshape(dat01['opd'].values,(nlayer,nwno)),0)

	xr, yr = scat01.shape
	cols = colfun2(200)[::-1]
	color_mapper = LogColorMapper(palette=cols, low=1e-3, high=10)


	f01 = figure(x_range=[0, yr], y_range=[0,xr],
						   x_axis_label='Wavenumber Grid', y_axis_label='Pressure Grid, TOA ->',
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
						   x_axis_label='Wavenumber Grid', y_axis_label='Pressure Grid, TOA ->',
						   title="Assymetry Parameter",
						  plot_width=300, plot_height=300)

	f01b.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw =yr )

	color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
					   label_standoff=12, border_line_color=None, location=(0,0))
	f01b.add_layout(color_bar, 'left')
	return column(row(f01a, f01, row(f01b)))

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
	cols = colfun1(200)
	color_mapper = LinearColorMapper(palette=cols, low=0, high=1)

	dat01 = full_output['layer']['cloud']

	#PLOT W0
	scat01 = np.flip(dat01['w0'],0)#[0:10,:]
	xr, yr = scat01.shape
	f01a = figure(x_range=[0, yr], y_range=[0,xr],
						   x_axis_label='Wavenumber Grid', y_axis_label='Pressure Grid, TOA ->',
						   title="Single Scattering Albedo",
						  plot_width=300, plot_height=300)


	f01a.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)

	color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
					   label_standoff=12, border_line_color=None, location=(0,0))

	f01a.add_layout(color_bar, 'left')


	#PLOT OPD
	scat01 = np.flip(dat01['opd']+1e-60,0)

	xr, yr = scat01.shape
	cols = colfun2(200)[::-1]
	color_mapper = LogColorMapper(palette=cols, low=1e-3, high=10)


	f01 = figure(x_range=[0, yr], y_range=[0,xr],
						   x_axis_label='Wavenumber Grid', y_axis_label='Pressure Grid, TOA ->',
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
						   x_axis_label='Wavenumber Grid', y_axis_label='Pressure Grid, TOA ->',
						   title="Assymetry Parameter",
						  plot_width=300, plot_height=300)

	f01b.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)

	color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
					   label_standoff=12, border_line_color=None, location=(0,0))
	f01b.add_layout(color_bar, 'left')
	return column(row(f01a, f01, row(f01b)))

def lon_lat_to_cartesian(lon_r, lat_r, R = 1):
	"""
	calculates lon, lat coordinates of a point on a sphere with
	radius R
	"""
	x =  R * np.cos(lat_r) * np.cos(lon_r)
	y = R * np.cos(lat_r) * np.sin(lon_r)
	z = R * np.sin(lat_r)
	return x,y,z

def disco(full_output,wavelength=[0.3]):
	"""
	Plot disco ball with facets. Bokeh is not good with 3D things. So this is in matplotlib

	Parameters
	----------
	full_output : class 
		Full output from picaso

	wavelength : list 
		Where to plot 3d facets. Can input as many wavelengths as wanted. 
		Must be a list, must be in microns. 
	"""

	nrow = int(np.ceil(len(wavelength)/3))
	ncol = int(np.min([3,len(wavelength)])) #at most 3 columns
	fig = plt.figure(figsize=(6*ncol,4*nrow))
	for i,w in zip(range(len(wavelength)),wavelength):
		ax = fig.add_subplot(nrow,ncol,i+1, projection='3d')
		#else:ax = fig.gca(projection='3d')
		wave = 1e4/full_output['wavenumber']
		indw = find_nearest(wave,w)
		#[umg, numt, nwno] this is xint_at_top
		xint_at_top = full_output['albedo_3d'][:,:,indw]

		latitude = full_output['latitude']  #tangle
		longitude = full_output['longitude'] #gangle

		cm = plt.cm.get_cmap('plasma')
		u, v = np.meshgrid(longitude, latitude)
		
		x,y,z = lon_lat_to_cartesian(u, v)

		ax.plot_wireframe(x, y, z, color="gray")

		sc = ax.scatter(x,y,z, c = xint_at_top.T.ravel(),cmap=cm,s=150)

		fig.colorbar(sc)
		ax.set_zlim3d(-1, 1)					# viewrange for z-axis should be [-4,4]
		ax.set_ylim3d(-1, 1)					# viewrange for y-axis should be [-2,2]
		ax.set_xlim3d(-1, 1)
		ax.view_init(0, 0)
		ax.set_title(str(w)+' Microns')
		# Hide grid lines
		ax.grid(False)

		# Hide axes ticks
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])
		plt.axis('off')

	plt.subplots_adjust(wspace=0.3, hspace=0.3)
	plt.show()

#@jit(nopython=True, cache=True)
def find_nearest(array,value):
	#small program to find the nearest neighbor in a matrix
	idx = (np.abs(array-value)).argmin(axis=0)
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
	kwargs['plot_height'] = kwargs.get('plot_height',345)
	kwargs['plot_width'] = kwargs.get('plot_width',1000)
	kwargs['y_axis_label'] = kwargs.get('y_axis_label','Albedo')
	kwargs['x_axis_label'] = kwargs.get('x_axis_label','Wavelength [μm]')
	kwargs['y_range'] = kwargs.get('y_range',[0,1.2])
	kwargs['x_range'] = kwargs.get('x_range',[0.3,1])

	points_og = datashade(hv.Curve((1e4/wno, alb)))

	return points_og


	