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

from .fluxes import blackbody

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
	if len(molecules) < 3: ncol = 5
	else: ncol = len(molecules)
	cols = colfun1(ncol)
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
	kwargs['x_axis_type'] = kwargs.get('x_axis_type','log')	
	kwargs['y_range'] = kwargs.get('y_range',[np.max(pressure),np.min(pressure)])

	fig = figure(**kwargs)

	f = fig.line(temperature,pressure,line_width=3) 
	plot_format(fig)
	return fig

def spectrum(xarray, yarray,legend=None,wno_to_micron=True, **kwargs):
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
	**kwargs : dict 	
		Any key word argument for bokeh.plotting.figure()

	Returns
	-------
	bokeh plot
	"""
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

	if isinstance(xarray, list):
		if legend==None: legend=[None]*len(xarray[0])
		for w, a,i,l in zip(xarray, yarray, range(len(xarray)), legend):
			if l == None: 
				fig.line(conv(w),  a,  color=Colorblind8[np.mod(i, len(Colorblind8))], line_width=3)
			else:
				fig.line(conv(w), a, legend_label=l, color=Colorblind8[np.mod(i, len(Colorblind8))], line_width=3)
	else: 
		if legend ==None:
			fig.line(conv(xarray), yarray,  color=Colorblind8[0], line_width=3)
		else:
			fig.line(conv(xarray), yarray, legend_label=legend, color=Colorblind8[0], line_width=3)
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
		Default = 0.5. 
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

	ind_gas = find_nearest_2d(taugas, at_tau)
	ind_cld = find_nearest_2d(taucld, at_tau)
	ind_ray = find_nearest_2d(tauray, at_tau)

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
	cols = colfun1(200)
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
	cols = colfun2(200)[::-1]
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

def disco(full_output,wavelength=[0.3],calc_type='reflected'):
	"""
	Plot disco ball with facets. Bokeh is not good with 3D things. So this is in matplotlib

	Parameters
	----------
	full_output : class 
		Full output from picaso

	wavelength : list 
		Where to plot 3d facets. Can input as many wavelengths as wanted. 
		Must be a list, must be in microns. 
	type : str, optional 
		Default is to plot 'reflected' light but can also switch to 'thermal' if it has been computed
	"""
	if calc_type=='reflected':to_plot='albedo_3d'
	elif calc_type=='thermal':to_plot='flux_planet_3d'

	nrow = int(np.ceil(len(wavelength)/3))
	ncol = int(np.min([3,len(wavelength)])) #at most 3 columns
	fig = plt.figure(figsize=(6*ncol,4*nrow))
	for i,w in zip(range(len(wavelength)),wavelength):
		ax = fig.add_subplot(nrow,ncol,i+1, projection='3d')
		#else:ax = fig.gca(projection='3d')
		wave = 1e4/full_output['wavenumber']
		indw = find_nearest_1d(wave,w)
		#[umg, numt, nwno] this is xint_at_top
		xint_at_top = full_output[calc_type][to_plot][:,:,indw]

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

def map(full_output,pressure=[0.1], plot='temperature', wavelength = None):
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
		
	"""
	
	to_plot = explore(full_output, plot)

	if isinstance(to_plot,np.ndarray):
		if len(to_plot.shape) < 3: 
			raise Exception("The key you are search for is not a 3D matrix. This function \
				is used to plot out a map of a matrix that is [nlayer, nlong, nlat] or \
				[nlayer, nwave, nlong, nlat, ].")
		elif len(to_plot.shape) == 4: 
			wave = 1e4/full_output['wavenumber']
			indw = find_nearest_1d(wave,wavelength)
			to_plot= to_plot[:,indw,:,:]
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
		ax.set_zlim3d(-1, 1)					# viewrange for z-axis should be [-4,4]
		ax.set_ylim3d(-1, 1)					# viewrange for y-axis should be [-2,2]
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

def flux_at_top(full_output, plot_bb = True, pressures = [1e-1,1e-2,1e-3],ng=None, nt=None, **kwargs):
	"""
	Routine to plot the OLR with overlaying black bodies. 

	Flux units are CGS = erg/s/cm^3 
	
	Parameters
	----------

	full_output : class
		picaso.atmsetup.ATMSETUP
	plot_bb : bool , optional 
		Default is to plot black bodies for three pressures specified by `pressures`
	pressures : list, optional 
		Default is a list of three pressures (in bars) =  [1e-1,1e-2,1e-3]
	**kwargs : dict 
		Any key word argument for bokeh.figure() 
	"""
	if ((ng==None) & (nt ==None)):
		pressure_all = full_output['layer']['pressure']
		temperature_all = full_output['layer']['temperature']
	else: 
		pressure_all = full_output['layer']['pressure'][:,ng,nt]
		temperature_all = full_output['layer']['temperature'][:,ng,nt]

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
	cols = colfun1(ncol)

	wno = full_output['wavenumber']
	fig.line(1e4/wno, full_output['thermal']['flux_planet'], color='black', line_width=4)

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

def taumap(full_output, at_tau=1, wavelength=1):
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
	**kwargs : dict 
		Any key word argument for bokeh.plotting.figure()

	Returns
	-------
	bokeh plot
	"""	
	all_dtau_gas = full_output['taugas']
	all_dtau_cld = full_output['taucld']*full_output['layer']['cloud']['w0']
	all_dtau_ray = full_output['tauray']

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
		ax.set_zlim3d(-1, 1)					# viewrange for z-axis should be [-4,4]
		ax.set_ylim3d(-1, 1)					# viewrange for y-axis should be [-2,2]
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
	