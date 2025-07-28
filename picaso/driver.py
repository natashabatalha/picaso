from .justdoit import *
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import tomllib 
from bokeh.layouts import column, row
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, Tabs, Panel, ColumnDataSource
import pandas as pd #this is already in the jdi?

def run(driver_file):

    with open(driver_file, "rb") as f:
        config = tomllib.load(f)
    
    if config['calc_type'] =='spectrum':
        out = spectrum(config)
    
    ### I made these # because they stopped the run fucntion from doing return out and wouldn't let me use my plot PT fucntion
    elif config['calc_type']=='climate':
        raise Exception('WIP not ready yet')
        out = climate(config)
    
    elif config['calc_type']=='retrieval':
        raise Exception('WIP not ready yet')

    return out 

def spectrum(config):

    OPA = opannection(
        filename_db=config['OpticalProperties']['opacity_files'], #database(s)
        method=config['OpticalProperties']['opacity_method'], #resampled, preweighted, resortrebin
        **config['OpticalProperties']['opacity_kwargs'] #additonal inputs 
        ) #opanecction connects to the opacity database
    irradiated = config['irradiated']
    if not irradiated: 
        A = inputs(calculation='browndwarf',climate=False) #if it isn't irradiated we are calculating a browndwarf
    else: 
        A = inputs(calculation='planet',climate=False) #if irradiated we are calculating a planet 
    
    #WIP TODO A.approx()

    phase = config['geometry'].get('phase', {}).get('value',None)
    phase_unit = config['geometry'].get('phase', {}).get('unit',None)
    rad = (phase * u.Unit(phase_unit)).to(u.rad).value
    A.phase_angle(rad) #input the radian angle of the event/geometry of browndwarf/planet

    A.gravity(gravity     = config['object'].get('gravity', {}).get('value',None), 
              gravity_unit= u.Unit(config['object'].get('gravity', {}).get('unit',None)), 
              radius      = config['object'].get('radius', {}).get('value',None), 
              radius_unit = u.Unit(config['object'].get('radius', {}).get('unit',None)), 
              mass        = config['object'].get('mass', {}).get('value',None), 
              mass_unit   = u.Unit(config['object'].get('mass', {}).get('unit',None)))
    #gravity parameters for a planet/browndwarf

    
    if irradiated: #calculating spectrum for a planet by defining star properties

        #check if userfile is requested
        filename = config['star'].get('userfile',{}).get('filename',None)
        if os.path.exists(str(filename)): #file with wavelength and flux 
            w_unit=config['star']['userfile'].get('w_unit')
            f_unit=config['star']['userfile'].get('f_unit')
        else: #properties of star 
            w_unit=None
            f_unit=None
            temp= config['star'].get('grid',{}).get('teff',None) #temperature of star
            metal= config['star'].get('grid',{}).get('feh',None) #metallicity of star
            logg= config['star'].get('grid',{}).get('logg',None) #log gravity of star
            database= config['star'].get('grid',{}).get('database',None) #specify database

        A.star(OPA,
               temp=temp, 
               metal=metal, 
               logg=logg ,
               database=database,
               radius = config['star'].get('radius', {}).get('value',None), 
               radius_unit= u.Unit(config['star'].get('radius', {}).get('unit',None)),
               semi_major=config['star'].get('semi_major', {}).get('value',None), 
               semi_major_unit = u.Unit(config['star'].get('semi_major', {}).get('unit',None)), 
               filename=filename, 
               w_unit=w_unit, 
               f_unit=f_unit
               ) 
        #star parameters for a planet/browndwarf
    
    # tempreature 
    pt = config['temperature']
    df_pt = PT_handler(pt, A) #datafile for pressure temperature profile

    # chemistry
    chem_config = config['chemistry']
    if chem_config['method']=='free': #user defined chemistry per molecule
        df=chem_free(df_pt,chem_config)
        A.atmosphere(df=df)
    elif chem_config['method']=='visscher': #chemistry inputs manually
        log_mh = chem_config['method']['visscher'].get('logmh',None) 
        cto = chem_config['method']['visscher'].get('cto',{}).get('value') 
        cto_unit = chem_config['method']['visscher'].get('cto',{}).get('unit')
        if cto_unit=='relative': #make cto value relative to asplunds
            cto_relative=cto 
            cto_absolute=cto_relative*0.55
        else: #leave cto value as is
            cto_relative=None
            cto_absolute=cto
        A.add_pt(T=df_pt['temperature'],P=['pressure'])
        A.chemeq_visscher_2121(cto_absolute, log_mh)
    else: 
        raise Exception('Only options are free and visscher so far')


    #WIP TODO: A.surface_reflect()
    
    
    #WIP TODO:if clouds: 
    cloud_config = config.get('clouds',None)
    if isinstance(cloud_config , dict):
        do_clouds=True
    else: 
        do_clouds=False
    
    if do_clouds == True: 
        cld_type = cloud_config['cloud1_type']

        if cld_type=='deck-grey':

            cloud_config['cloud1'][cld_type]['p']=cloud_config['cloud1'][cld_type]['p']['value']#pressure level log
            cloud_config['cloud1'][cld_type]['dp']=cloud_config['cloud1'][cld_type]['dp']['value']#cloud thickness log
            for ikey in cloud_config['cloud1'][cld_type].keys(): 
                val = cloud_config['cloud1'][cld_type][ikey]
                if isinstance(val, (float,int)):
                    val=[val]
                    cloud_config['cloud1'][cld_type][ikey]=val

            A.clouds( **cloud_config['cloud1'][cld_type])


    #WIP TODO:    A.clouds()
    #WIP TODO:    A.virga()

    out = A.spectrum(OPA, full_output=True, calculation = config['observation_type']) 
    out['pt_profile']=df_pt
    return out

def PT_handler(pt_config, A): 
    type = pt_config['profile']

    #check if supplied file for pt profile
    if type == 'userfile': 
        filename = pt_config['userfile']
        kwargs = pt_config.get('panda_kwargs', {})
        df = pd.read_csv(filename, **kwargs)
 
    elif type == 'guillot':
        #guillot analytical pt profile
        #extract guillot-specific parameters from toml
        params = pt_config.get('guillot', {})
        #call picaso method on atmosphere instance 'A'
        df = A.guillot_pt(**params)

    elif type == 'sonora_bobcat':
        #sonora bobcat grid pt profile from picaso-data
        params = pt_config.get('sonora_bobcat', {})
        #call picaso's sonora function with parameters
        A.sonora(**params)
        #the resulting pt profile is stored inside a.inputs['atmosphere']['profile']
        df = A.inputs['atmosphere']['profile']

    else: #build pt profile using param tools built in to param_tools?
        P_config = pt_config['pressure']
        minp = P_config.get('min', {}).get('value')  #minimum pressure
        minp_unit = P_config.get('min', {}).get('unit')
        maxp = P_config.get('max', {}).get('value')  #max pressure
        maxp_unit = P_config.get('max', {}).get('unit')
        spacing = P_config.get('spacing')
        nlevel = P_config.get('nlevel')
        minp_bar = ((minp * u.Unit(minp_unit)).to(u.bar)).value
        maxp_bar = ((maxp * u.Unit(maxp_unit)).to(u.bar)).value

        if spacing == 'log':
            pressure = np.logspace(np.log10(minp_bar), np.log10(maxp_bar), nlevel)
        else:
            pressure = np.linspace(minp_bar, maxp_bar, nlevel)

        #use param_tools method matching the profile type to generate temperature array
        temperature_function = getattr(param_tools, type)
        temperature = temperature_function(pressure, **pt_config[type])

        #create dataframe with pressure and temperature
        df = pd.DataFrame({'temperature': temperature, 'pressure': pressure})

    return df


#WIP TODO REPLACE THIS WITH THE PARAMTOOLS BEING BUILT
class param_tools: 
    def isothermal(pressure, T):
        return pressure*0+T 
    def knots(pressure,foo1,foo2):
        return temperature
    def madhu_seager_09(pressure,foo1,foo2):
        return temperature
     
    

def chem_free(pt_df, chem_config):
    """
    Creates dataframe based on user input free chem 
    """
    free = chem_config['free']
    pressure_grid = pt_df['pressure'].values 
    total_sum_of_gases = 0*pressure_grid
    for i in free.keys(): 
        #make sure its not the background
        if i !='background':
            value = free[i].get('value',None)
            #easy case where there is just one well-mixed value 
            if value is not None: #abundance of the chemistry input per molecule
                pt_df[i] = value
                
            else: #each molecule input manually
                values =  free[i].get('values') 
                pressures = free[i].get('pressures') 
                pressure_unit= free[i].get('pressure_unit') 
                pressure_bar = (np.array(pressures)*u.Unit(pressure_unit)).to(u.bar).value 
                
                #make sure its in ascending pressure order 
                first = pressure_bar[0] 
                last = pressure_bar[-1] 

                #flip if the ordering has been input incorrectly
                if first > last : 
                    pressure_bar=pressure_bar[::-1]
                    values=values[::-1]

                vmr = values[0] + 0*pressure_grid
                # need to finish the ability to input free chem here 
                for ii,ivmr in enumerate(values[1:]):
                    vmr[pressure_grid>=pressure_bar[ii]] = ivmr 
                
                #add to dataframe 
                pt_df[i]=vmr

            total_sum_of_gases += pt_df[i].values
    #add background gas if it is requested
    if 'background' in free.keys():
        total_sum_of_background = 1-total_sum_of_gases
        if len(free['background']['gases'])==2: #2 background gasses
            gas1_name = free['background']['gases'][0]
            gas2_name = free['background']['gases'][1]
            fraction = free['background']['fraction']
            gas2_absolute_value = total_sum_of_background / (fraction + 1)
            gas1_absolute_value = fraction * gas2_absolute_value
            pt_df[gas1_name] = gas1_absolute_value
            pt_df[gas2_name] = gas2_absolute_value
        if len(free['background']['gases'])==1: #1 background gas
            pt_df[free['background']['gases'][0]] = total_sum_of_background
    return pt_df

def viz(picaso_output): 
    spectrum_plot_list = []

    if isinstance(picaso_output.get('transit_depth', jpi.np.nan), jpi.np.ndarray):
        spectrum_plot_list += [jpi.spectrum(picaso_output['wavenumber'], picaso_output['transit_depth'], title='Transit Depth Spectrum')]

    if isinstance(picaso_output.get('albedo', jpi.np.nan), jpi.np.ndarray):
        spectrum_plot_list += [jpi.spectrum(picaso_output['wavenumber'], picaso_output['albedo'], title='Albedo Spectrum')]

    if isinstance(picaso_output.get('thermal', jpi.np.nan), jpi.np.ndarray):
        spectrum_plot_list += [jpi.spectrum(picaso_output['wavenumber'], picaso_output['thermal'], title='Thermal Emission Spectrum')]

    if isinstance(picaso_output.get('fpfs_reflected', jpi.np.nan), jpi.np.ndarray):
        spectrum_plot_list += [jpi.spectrum(picaso_output['wavenumber'], picaso_output['fpfs_reflected'], title='Reflected Light Spectrum')]

    if isinstance(picaso_output.get('fpfs_thermal', jpi.np.nan), jpi.np.ndarray):
        spectrum_plot_list += [jpi.spectrum(picaso_output['wavenumber'], picaso_output['fpfs_thermal'], title='Relative Thermal Emission Spectrum')]

    if isinstance(picaso_output.get('fpfs_total', jpi.np.nan), jpi.np.ndarray):
        spectrum_plot_list += [jpi.spectrum(picaso_output['wavenumber'], picaso_output['fpfs_total'], title='Relative Full Spectrum')]

    output_file("spectrum_output.html")
    show(column(children=spectrum_plot_list, sizing_mode="scale_width"))
    
    return spectrum_plot_list



def plot_pt_profile(full_output, **kwargs):
    fig = jpi.pt(full_output, **kwargs)
    show(fig)
    return fig
