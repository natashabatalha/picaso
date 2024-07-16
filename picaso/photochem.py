#def run_photochem(temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop,filename = '111',stfilename='111',network=None,network_ct=None,first=True,pc=None,user_psurf=True,user_psurf_add=3):
#    raise Exception("Photochemistry not yet availble")


import numpy as np
from matplotlib import pyplot as plt
import cantera as ct
from scipy import constants as const
from scipy import integrate
from astropy import constants
import pickle
#import utils
#import planets
from photochem.utils._format import  MyDumper, Loader, yaml, FormatReactions_main
from photochem.utils import photochem2cantera
from photochem import Atmosphere, EvoAtmosphere, PhotoException
from .deq_chem import quench_level
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

class flowmap( dict ): pass
def flowmap_rep(dumper, data):
    return dumper.represent_mapping( u'tag:yaml.org,2002:map', data, flow_style=True)

class blockseqtrue( list ): pass
def blockseqtrue_rep(dumper, data):
    return dumper.represent_sequence( u'tag:yaml.org,2002:seq', data, flow_style=True )
yaml.add_representer(blockseqtrue, blockseqtrue_rep)
yaml.add_representer(flowmap, flowmap_rep)



class TempPress:

    def __init__(self, P, T):
        self.log10P = np.log10(P)[::-1]
        self.T = T[::-1]

    def temperature(self, P):
        return np.interp(np.log10(P), self.log10P, self.T)

def gravity(radius, mass, z):
    G_grav = 6.67430e-11
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

def rhs(P, u, mubar, radius, mass, pt):

    z = u[0]
    grav = gravity(radius, mass, z)
    T = pt.temperature(P)
    k_boltz = const.Boltzmann*1e7

    dz_dP = -(k_boltz*T*const.Avogadro)/(mubar*grav*P)

    return np.array([dz_dP])

def run_photochem(temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop,filename = '111',stfilename='111',network=None,network_ct=None,first=True,pc=None,user_psurf=True,user_psurf_add=3):
    # somehow all photochemical models want stuff flipped
    # dont worry we flip back before exiting the function
    
    temp,pressure = np.flip(temp), np.flip(pressure)
    nlevelmin1=len(temp)-1
    if np.logical_and(first == False, pc == None):
        raise ValueError('If this is not the first run, pc should be recycled')

    print(logMH,cto)    
    # mass and radius in cm
    CtoO = 0.458*cto
    gas = ct.Solution(network_ct)

    # Composition from VULCAN file (see vulcan_files/cfg_wasp39b_10Xsolar_evening20deg.py)
    O_H = 5.37E-4*(10.0**logMH)*(0.8)
    C_H = 2.95E-4*(10.0**logMH)
    N_H = 7.08E-5*(10.0**logMH)
    S_H = 1.41E-5*(10.0**logMH)
    He_H = 0.0838

    y = (C_H+O_H)/(CtoO*O_H + O_H)
    x =y* (CtoO  *O_H)/C_H
    
    C_H,O_H =  x*C_H,y*O_H

    comp = {}
    comp['H'] = 1.0
    comp['O'] = O_H
    comp['C'] = C_H
    comp['N'] = N_H
    comp['S'] = S_H
    comp['He'] = He_H

    # compute chemical equilibrium at every atmospheric pressure
    sol = {}
    for key in gas.species_names:
        sol[key] = np.empty(pressure.shape[0])

    # compute equilibrium
    for i in range(pressure.shape[0]):
        gas.TPX = temp[i], pressure[i]*1e5, comp
        gas.equilibrate('TP')

        for j,sp in enumerate(gas.species_names):
            sol[sp][i] = gas.X[j]
    
    setting_template_text = \
        '''
        atmosphere-grid:
            bottom: 0.0
            top: 15000.0e5
            number-of-layers: 100

        photolysis-grid:
            regular-grid: true
            lower-wavelength: 92.5
            upper-wavelength: 855.0
            number-of-bins: 200

        planet:
#            background-gas: H2
#            surface-pressure: 1.047
            planet-mass: 5.314748872540942e+29
            planet-radius: 9079484000.0
            surface-albedo: 0.0
            solar-zenith-angle: 83.0
            hydrogen-escape:
                type: none
            default-gas-lower-boundary: Moses
            water:
                fix-water-in-troposphere: false
                gas-rainout: false
                water-condensation: true
                condensation-rate: {A: 1.0e-8, rhc: 1, rh0: 1.05}
                
        boundary-conditions:
        - name: O1D
          type: short lived
        - name: N2D
          type: short lived
        '''
    setting_template = yaml.safe_load(setting_template_text)
    
    
    
    dist = np.abs(np.log10(pressure)-np.log10(pressure_surf))
    wh= np.where(dist == np.min(dist))
    ind_surf = wh[0][0]
    print('User surface pressure in bar =',pressure[ind_surf])
    if pressure[ind_surf] == np.min(pressure):
        print('You are running Equilibrium Chemistry through photochem.')
        eq_chem=True
    else:
        print('You are running Disequilibrium Chemistry through photochem.')
        eq_chem=False
    grav_quench = 6.67430e-11 *((float(mass.value))/1.0e3) / ((float(radius.value))/1.0e2)**2.0
    if first==True:
        quench_levels = quench_level(np.flip(pressure), np.flip(temp), np.flip(kzz) ,pressure*0+2.34, grav_quench, return_mix_timescale= False)
        print("Quench time max pressure is ",np.flip(pressure)[np.max(quench_levels)])
        if user_psurf == False:
            pressure_surf = np.flip(pressure)[np.max(quench_levels)+user_psurf_add]
            dist = np.abs(np.log10(pressure)-np.log10(pressure_surf))
            wh= np.where(dist == np.min(dist))
            ind_surf = wh[0][0]
            print('Overwriting user surface pressure (as user_psurf is True) in bar =',pressure[ind_surf])

        if np.logical_and(np.flip(pressure)[np.max(quench_levels)] >=  pressure[ind_surf],eq_chem==False):
            raise Exception("You need a deeper Surface Pressure. Surface is ", pressure[ind_surf], " max quench is ",np.flip(pressure)[np.max(quench_levels)] )
        
        new_quench = nlevelmin1-quench_levels
    
    # get anundances at the surface
    surf = np.empty(len(gas.species_names))
    if eq_chem == False:
        deeper_surf = np.zeros(shape=(len(gas.species_names),len(pressure[:ind_surf])))
    else:
        deeper_surf = np.zeros(shape=(len(gas.species_names),len(pressure[:ind_surf+1])))

    species_cantera = gas.species_names
    for i,sp in enumerate(gas.species_names):
        surf[i] = sol[sp][ind_surf]
        if eq_chem == False:
            deeper_surf[i,:] = sol[sp][:ind_surf]
        else:
            deeper_surf[i,:] = sol[sp][:ind_surf+1]
    if eq_chem==False:
        bc_list = []
        for i,sp in enumerate(gas.species_names):
            
            if surf[i] > 1e-6 :
                if sp != 'O1D':
                    if sp != 'N2D':
                        
                        lb = {"type": "press", "press": float(surf[i]*pressure[ind_surf]*1e6)} # in dyne/cm^2
                        ub = {"type": "veff", "veff": 0.0}
                        entry = {}
                        entry['name'] = sp
                        entry['lower-boundary'] = lb
                        entry['upper-boundary'] = ub
                        
                        bc_list.append(entry)
                else:
                    pass

        # add these boundary conditions to the settings file
        for bc in bc_list:
            setting_template['boundary-conditions'].append(bc)

        pt = TempPress(pressure, temp)
        radius = float(radius.value) #planets.WASP39b.radius*(constants.R_jup.value)*1e2
        mass = float(mass.value)#planets.WASP39b.mass*(constants.M_jup.value)*1e3
        
        # Here, we must guess the mean molecular weight
        mubar = 4*surf[gas.species_names.index('He')] + 2*(1.0-surf[gas.species_names.index('He')])
        P0 = pressure[ind_surf]
        
        z0 = 0.0
        args = (mubar, radius, mass, pt)

        # Integrate the hydrostatic equation to get T(z)
        out = integrate.solve_ivp(rhs, [P0, 1e-8], np.array([z0]), t_eval=pressure[ind_surf:], args=args)

        press_subset = pressure[ind_surf:]
        temp_subset = temp[ind_surf:]

        alt = out.y[0]/1e5
        den = (press_subset*1e6)/(const.Boltzmann*1e7*temp_subset)

        # eddy diffusion. From the SO2 paper
        eddy = kzz[ind_surf:]

        fmt = '{:25}'
        if first==False:
            density_pc = pc.wrk.density
        with open(filename+'_init.txt', 'w') as f:
            f.write(fmt.format('alt'))
            f.write(fmt.format('press'))
            f.write(fmt.format('den'))
            f.write(fmt.format('temp'))
            f.write(fmt.format('eddy'))

            for key in sol:
                f.write(fmt.format(key))
            
            f.write('\n')

            for i in range(press_subset.shape[0]):
                f.write(fmt.format('%e'%alt[i]))
                f.write(fmt.format('%e'%press_subset[i]))
                f.write(fmt.format('%e'%den[i]))
                f.write(fmt.format('%e'%temp_subset[i]))
                f.write(fmt.format('%e'%eddy[i]))

                # We use chemical equilibrium as an initial condition
                if first == True:
                    for key in sol:
                        if key == 'CH4':
                            sol[key][new_quench[0]:] = sol[key][new_quench[0]:]*0 + sol[key][new_quench[0]]    
                        if key == 'CO':
                            sol[key][new_quench[0]:] = sol[key][new_quench[0]:]*0 + sol[key][new_quench[0]]
                        if key == 'H2O':
                            sol[key][new_quench[0]:] = sol[key][new_quench[0]:]*0 + sol[key][new_quench[0]]
                        if key == 'CO2':
                            sol[key][new_quench[1]:] = sol[key][new_quench[1]:]*0 + sol[key][new_quench[1]]

                        if key == 'NH3':
                            sol[key][new_quench[2]:] = sol[key][new_quench[2]:]*0 + sol[key][new_quench[2]]
                        if key == 'HCN':
                            sol[key][new_quench[3]:] = sol[key][new_quench[3]:]*0 + sol[key][new_quench[3]]
                            
                        f.write(fmt.format('%e'%sol[key][i+ind_surf]))
                else:
                    for key in sol:
                        
                        ind = pc.dat.species_names.index(key)
                        abund_write = np.abs(pc.wrk.densities[ind,i]/density_pc[i])
                        f.write(fmt.format('%e'%abund_write))
                        
                f.write('\n')

        setting_template['atmosphere-grid']['top'] = 'atmospherefile'#float(alt[-1]*1e5)
        setting_template['atmosphere-grid']['number-of-layers'] = len(alt)
        #setting_template['planet']['surface-pressure'] = float(P0)
        setting_template['planet']['planet-mass'] = mass
        setting_template['planet']['planet-radius'] = radius
        print(setting_template['planet'])
        setting_template = FormatSettings_main(setting_template)
        with open(filename+'_settings.yaml','w') as f:
            yaml.dump(setting_template,f,Dumper=MyDumper,sort_keys=False)
        
        
        
        pc_new = EvoAtmosphere(network,\
                    filename+"_settings.yaml",\
                    stfilename,\
                    filename+"_init.txt")
        pc_new.var.autodiff = True
        pc_new.var.atol = 1e-13
        pc_new.var.rtol = 1e-3
        pc_new.var.verbose = 0
        pc_new.var.conv_min_mix = 1e-10
        pc_new.var.conv_longdy = 0.05 
        pc_new.var.conv_longdydt=1e-4    

        #usol = pc_new.wrk.usol.copy()
        # usol[pc.dat.species_names.index('CH4'),:] = 1.0e-6
        #pc_new.initialize_stepper(usol)
        #tn = 0


        P_photochem_copy = pc_new.wrk.pressure
        T_photochem_copy = pc_new.var.temperature
        edd_photochem_copy = pc_new.var.edd
    
    # Flip order an get log10 for interpolation purposes
        log10P_interp = np.log10(P_photochem_copy.copy()[::-1])
        T_interp = T_photochem_copy.copy()[::-1]
        log10edd_interp = np.log10(edd_photochem_copy.copy()[::-1])

    # initialize
        pc_new.initialize_stepper(pc_new.wrk.usol)
        atol_counter = 0
        nerrors = 0
        max_dT_tol = 10 # Kelvin
        max_dlog10edd_tol = 0.2 # log10 units

       
        TOA_pressure_avg = np.min(pressure) # bars
        TOA_pressure_min = 1e-1*TOA_pressure_avg # bars
        TOA_pressure_max = 2*TOA_pressure_avg # bars
        atol_min = 1e-14
        atol_max= 1e-12
        retry_counter = 0    
        while True:
            #sol = mole_fraction_dict(pc_new)
            try:
                tn = pc_new.step()
                atol_counter += 1
            except PhotoException as e:
                # If there is an error, lets reinitialize, but get rid of any
                # negative numbers
                usol = np.clip(pc_new.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                pc_new.initialize_stepper(usol)
                nerrors += 1

                if nerrors > 10:
                    raise Exception('Repeated errors in photochemical integration!')

                continue

        # convergence checking
            converged = pc_new.check_for_convergence()

            # Compute the max difference between the P-T profile in photochemical model
            # and the desired P-T profile
            T_p_now = np.interp(np.log10(pc_new.wrk.pressure.copy()[::-1]), log10P_interp, T_interp)
            T_p_now = T_p_now.copy()[::-1]
            max_dT = np.max(np.abs(T_p_now - pc_new.var.temperature))

            # Compute the max difference between the P-edd profile in photochemical model
            # and the desired P-edd profile
            log10edd_p_now = np.interp(np.log10(pc_new.wrk.pressure.copy()[::-1]), log10P_interp, log10edd_interp)
            log10edd_p_now = log10edd_p_now.copy()[::-1]
            max_dlog10edd = np.max(np.abs(log10edd_p_now - np.log10(pc_new.var.edd)))

            TOA_pressure = pc_new.wrk.pressure[-1]/1e6 # bars

#            if converged and pc_new.wrk.nsteps > 200 and max_dT < max_dT_tol and max_dlog10edd < max_dlog10edd_tol and TOA_pressure_min < TOA_pressure < TOA_pressure_max:
                # converged!
#                print("WOOHOO ! PHOTOCHEM HAS CONVERGED !")
#                break
            if converged and pc_new.wrk.nsteps > 500:
                print("WOOHOO ! PHOTOCHEM HAS CONVERGED !")
                break


#            if atol_counter > 5000:
                # Convergence has not happened after 10000 steps, so we try a new atol
#                print("After 10000 atol iterations our code has not converged.")
               
#                pc_new.var.atol = 10.0**np.random.uniform(low=np.log10(atol_min),high=np.log10(atol_max))
#                pc_new.initialize_stepper(pc_new.wrk.usol)
#                atol_counter = 0
#                print("Restarting with atol=",pc_new.var.atol)
#                retry_counter+=1
#                continue
        
#            if not (pc_new.wrk.nsteps % 1000):
                # After 1000 steps, lets update P,T, edd and vertical grid
#                try:
#                    print('Updating TP grid')
#                    pc_new.set_press_temp_edd(P_photochem_copy,T_photochem_copy,edd_photochem_copy,hydro_pressure=False)
#                    pc_new.update_vertical_grid(TOA_pressure=TOA_pressure_avg*1e6)
#                    pc_new.initialize_stepper(pc_new.wrk.usol)
#                    print("Update TP grid Complete")
#                except PhotoException as e:
#                    print("Updating TP led to error, so hoping for the best and continuing on....")
#                    continue
            if not (pc_new.wrk.nsteps % 100):
                print('Steps=', pc_new.wrk.nsteps)
                print('longdy = %.1e,  max_dT = %.1e,  max_dlog10edd = %.1e,  TOA_pressure = %.1e,  '% \
                    (pc_new.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure))
                
    #        if retry_counter >= 3:
    #            break
                    #raise Exception('This is not working after 10 tries with different atol, try changing surface')
            
        
            
            
        
    #pc.out2atmosphere_txt(filename="WASP39b/"+filename+"_init.txt", overwrite=True, clip=True)
    #species = ['CH4','CO','CO2','H2O','HCN','NH3','H','H2','He','C2H2','C2H4','C2H6','SO2','H2S']
    species = species_cantera
    output_array = np.zeros(shape=(len(species),len(pressure)))
    if eq_chem==False:
        for i,sp in enumerate(species):
                
                ind = pc_new.dat.species_names.index(sp)
                ind_cantera = species_cantera.index(sp)
                
                
                tmp = pc_new.wrk.densities[ind,:]/pc_new.wrk.density
                #ax.plot(tmp,pc.wrk.pressure/1e6,label=sp,color=color[i])
                #ax.plot(deeper_surf[ind_cantera,:],pressure[:ind_surf],linestyle="--",color=color[i])
                output_array[i,:] = np.flip(np.concatenate([deeper_surf[ind_cantera,:],tmp]))
    else:
        for i,sp in enumerate(species):
                
                
                ind_cantera = species_cantera.index(sp)
                
                
                
                #ax.plot(tmp,pc.wrk.pressure/1e6,label=sp,color=color[i])
                #ax.plot(deeper_surf[ind_cantera,:],pressure[:ind_surf],linestyle="--",color=color[i])
                output_array[i,:] = np.flip(deeper_surf[ind_cantera,:])
        pc_new = {'eq_chem'}
    if pc_new == {'eq_chem'}:
        p_output=np.flip(pressure)
    else:
        p_output= np.concatenate([np.flip(pc_new.wrk.pressure/1e6),np.flip(pressure[:ind_surf])])
    return pc_new,output_array,species,p_output


def mole_fraction_dict(pc):
    sol = {}
    for i,sp in enumerate(pc.dat.species_names[:-2]):
        ind = pc.dat.species_names.index(sp)
        mix = pc.wrk.densities[ind,:]/pc.wrk.density
        sol[sp] = mix
    sol['alt'] = pc.var.z/1e5 # km
    sol['temp'] = pc.var.temperature # K
    sol['pressure'] = pc.wrk.pressure # dynes/cm^2
    sol['density'] = pc.wrk.density # molecules/cm^3
    return sol     

def FormatSettings_main(data):
    
    if 'planet' in data:
        if "rainout-species" in data['planet']['water'].keys():
            data['planet']['water']['rainout-species'] = blockseqtrue(data['planet']['water']['rainout-species'])

        if "condensation-rate" in data['planet']['water'].keys():
            data['planet']['water']['condensation-rate'] = flowmap(data['planet']['water']['condensation-rate'])
    
    #if 'particles' in data:
    #    for i in range(len(data['particles'])):
    #        if "condensation-rate" in data['particles'][i]:
    #            data['particles'][i]["condensation-rate"] = \
    #            flowmap(data['particles'][i]["condensation-rate"])

    if 'boundary-conditions' in data:
        for i in range(len(data['boundary-conditions'])):
            if "lower-boundary" in data['boundary-conditions'][i]:
                order = ['type','vdep','mix','flux','height','press']
                copy = data['boundary-conditions'][i]['lower-boundary'].copy()
                data['boundary-conditions'][i]['lower-boundary'].clear()
                for key in order:
                    if key in copy.keys():
                        data['boundary-conditions'][i]['lower-boundary'][key] = copy[key]

                data['boundary-conditions'][i]['lower-boundary'] = flowmap(data['boundary-conditions'][i]['lower-boundary'])

                order = ['type','veff','flux']
                copy = data['boundary-conditions'][i]['upper-boundary'].copy()
                data['boundary-conditions'][i]['upper-boundary'].clear()
                for key in order:
                    if key in copy.keys():
                        data['boundary-conditions'][i]['upper-boundary'][key] = copy[key]

                data['boundary-conditions'][i]['upper-boundary'] = flowmap(data['boundary-conditions'][i]['upper-boundary'])
        
    return data 
    
