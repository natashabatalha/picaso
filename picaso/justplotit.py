from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Colorblind8
from bokeh.layouts import column 
import numpy as np

def spectrum(wno, alb,legend=None):
	"""Plot formated albedo spectrum

	Parameters
	----------
	wno : float array, list of arrays
		wavenumber 
	alb : float array, list of arrays 
		albedo 
	legend : list of str 
		legends for plotting 


	Returns
	-------
	bokeh plot
	"""
	fig = figure(x_range=[0.3,1],y_range=[0,1.2], width=2000, height=690,x_axis_label='Wavelength [μm]', y_axis_label='Geometric Albedo',)

	if isinstance(wno, list):
		if legend == None: legend=[None]*len(wno) 
		for w, a,i,l in zip(wno, alb, range(len(wno)), legend):
			fig.line(1e4/w, a, legend=l, color=Colorblind8[np.mod(i, len(Colorblind8))], line_width=3)
	else: 
		fig.line(1e4/wno, alb, legend=legend, color=Colorblind8[0], line_width=3)

	plot_format(fig)
	return fig




def plot_format(df):
	"""Function to reformat plots"""
	df.xaxis.axis_label_text_font='times'
	df.yaxis.axis_label_text_font='times'
	df.xaxis.major_label_text_font_size='45pt'
	df.yaxis.major_label_text_font_size='45pt'
	df.xaxis.axis_label_text_font_size='45pt'
	df.yaxis.axis_label_text_font_size='45pt'
	df.xaxis.major_label_text_font='times'
	df.yaxis.major_label_text_font='times'
	df.xaxis.axis_label_text_font_style = 'bold'
	df.yaxis.axis_label_text_font_style = 'bold'