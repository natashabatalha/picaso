from .justdoit import *
import tomllib 

def run(driver_file):

    with open(driver_file, "rb") as f:
        config = tomllib.load(f)
    
    if config['calc_type'] =='spectrum':
        out = spectrum(config)
    
    elif config['calc_type']=='climate':
        raise Exception('WIP not ready yet')
        out = climate(config)
    
    elif config['calc_type']=='retrieval':
        raise Exception('WIP not ready yet')

    return out 

def spectrum(config):

    OPA = opannection(
        filename_db=config['OpticalProperties']['opacity_files'],
        method=config['OpticalProperties']['opacity_method'],
        **config['OpticalProperties']['opacity_kwargs']
        )
    irradiated = config['irradiated']
    if not irradiated: 
        A = inputs(calculation='browndwarf',climate=False)
    else: 
        A = inputs(calculation='planet',climate=False)
    
    #WIP TODO A.approx()

    phase = config['geometry'].get('phase', {}).get('value',None)
    phase_unit = config['geometry'].get('phase', {}).get('unit',None)
    rad = (phase * u.Unit(phase_unit)).to(u.rad).value
    A.phase_angle(rad)

    A.gravity(gravity     = config['object'].get('gravity', {}).get('value',None), 
              gravity_unit= u.Unit(config['object'].get('gravity', {}).get('unit',None)), 
              radius      = config['object'].get('radius', {}).get('value',None), 
              radius_unit = u.Unit(config['object'].get('radius', {}).get('unit',None)), 
              mass        = config['object'].get('mass', {}).get('value',None), 
              mass_unit   = u.Unit(config['object'].get('mass', {}).get('unit',None)))

    
    if irradiated: 

        #check if userfile is requested
        filename = config['star'].get('userfile',{}).get('filename',None)
        if os.path.exists(str(filename)):
            w_unit=config['star']['userfile'].get('w_unit')
            f_unit=config['star']['userfile'].get('f_unit')
        else: 
            w_unit=None
            f_unit=None
            temp= config['star'].get('grid',{}).get('teff',None)
            metal= config['star'].get('grid',{}).get('feh',None) 
            logg= config['star'].get('grid',{}).get('logg',None)
            database= config['star'].get('grid',{}).get('database',None)

        A.star(OPA,
               temp=temp, 
               metal=metal, 
               logg=logg ,
               database=database,
               radius = config['star'].get('radius', {}).get('value',None), 
               radius_unit= u.Unit(config['star'].get('radius', {}).get('unit',None)),
               semi_major=config['star'].get('semi_major', {}).get('value',None), 
               semi_major_unit = u.Unit(config['star'].get('semi_major', {}).get('unit',None)), 
               filename=None, 
               w_unit=w_unit, 
               f_unit=f_unit
               )
    
    # tempreature 
    pt = config['temperature']
    df_pt = PT_handler(pt)

    # chemistry
    chem_config = config['chemistry']
    if chem_config['method']=='free':
        df=chem_free(df_pt,chem_config)
        A.atmosphere(df=df)
    elif chem_config['method']=='visscher':
        log_mh = chem_config['method']['visscher'].get('logmh',None)
        cto = chem_config['method']['visscher'].get('cto',{}).get('value')
        cto_unit = chem_config['method']['visscher'].get('cto',{}).get('unit')
        if cto_unit=='relative':
            cto_relative=cto 
            cto_absolute=cto_relative*0.55
        else:
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

            cloud_config['cloud1'][cld_type]['p']=cloud_config['cloud1'][cld_type]['p']['value']
            cloud_config['cloud1'][cld_type]['dp']=cloud_config['cloud1'][cld_type]['dp']['value']
            for ikey in cloud_config['cloud1'][cld_type].keys(): 
                val = cloud_config['cloud1'][cld_type][ikey]
                if isinstance(val, (float,int)):
                    val=[val]
                    cloud_config['cloud1'][cld_type][ikey]=val

            A.clouds( **cloud_config['cloud1'][cld_type])


    #WIP TODO:    A.clouds()
    #WIP TODO:    A.virga()

    out = A.spectrum(OPA, full_output=True, calculation = config['observation_type'])

    return out 

def PT_handler(pt_config): 
    type = pt_config['profile']

    #check if supplied file for pt profile
    if type == 'userfile': 
        filename = pt_config['userfile']
        kwargs = pt_config['panda_kwargs']
        df = pd.read_csv(filename, **kwargs)
    else:
        #first setup pressure grid 
        P_config = pt_config['pressure']
        minp = P_config.get('min',{}).get('value')
        minp_unit  = P_config.get('min',{}).get('unit')
        minp_bar = ((minp*u.Unit(minp_unit)).to(u.bar)).value
        maxp = P_config.get('max',{}).get('value')
        maxp_unit  = P_config.get('max',{}).get('unit')
        maxp_bar = ((maxp*u.Unit(maxp_unit)).to(u.bar)).value
        spacing = pt_config['pressure']['spacing']
        nlevel = pt_config['pressure']['nlevel']
        if spacing == 'log':
            pressure = np.logspace(np.log10(minp_bar),np.log10(maxp_bar),nlevel)
        else: 
            pressure = np.linespace(minp_bar,maxp_bar,nlevel)
        
        #get the temperature as a function of pressure 
        temperature_function = getattr(param_tools, type)
        temperature_inputs = pt_config[type]
        temperature = temperature_function(pressure,**temperature_inputs)
        df = pd.DataFrame({'temperature':temperature,'pressure':pressure})
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
            if value is not None: 
                pt_df[i] = value
                
            else: 
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
        if len(free['background']['gases'])==2:
            gas1_name = free['background']['gases'][0]
            gas2_name = free['background']['gases'][1]
            fraction = free['background']['fraction']
            gas2_absolute_value = total_sum_of_background / (fraction + 1)
            gas1_absolute_value = fraction * gas2_absolute_value
            pt_df[gas1_name] = gas1_absolute_value
            pt_df[gas2_name] = gas2_absolute_value
        if len(free['background']['gases'])==1:
            pt_df[free['background']['gases'][0]] = total_sum_of_background
    return pt_df
            