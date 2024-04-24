def run_photochem(temp,pressure,logMH, cto,pressure_surf,mass,radius,kzz,tstop,filename = '111',stfilename='111',network=None,network_ct=None,first=True,pc=None,user_psurf=True,user_psurf_add=3):
    raise Exception("Photochemistry not yet availble")

"""
import numpy as np
from matplotlib import pyplot as plt
import cantera as ct
from scipy import constants as const
from scipy import integrate
from astropy import constants
import pickle
#import utils
#import planets
from photochem.utils._format import FormatSettings_main, MyDumper, Loader, yaml, FormatReactions_main
from photochem.utils import photochem2cantera
from photochem import Atmosphere
from .deq_chem import quench_level





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
            background-gas: H2
            surface-pressure: 1.047
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
            if surf[i] > 1e-6 and sp not in ['O1D','N2D','H2']:
                lb = {"type": "mix", "mix": float(surf[i])}
                ub = {"type": "veff", "veff": 0.0}
                entry = {}
                entry['name'] = sp
                entry['lower-boundary'] = lb
                entry['upper-boundary'] = ub
                bc_list.append(entry)

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

        setting_template['atmosphere-grid']['top'] = float(alt[-1]*1e5)
        setting_template['atmosphere-grid']['number-of-layers'] = len(alt)
        setting_template['planet']['surface-pressure'] = float(P0)
        setting_template['planet']['planet-mass'] = mass
        setting_template['planet']['planet-radius'] = radius
        print(setting_template['planet'])
        setting_template = FormatSettings_main(setting_template)
        with open(filename+'_settings.yaml','w') as f:
            yaml.dump(setting_template,f,Dumper=MyDumper,sort_keys=False)
        
        
        
        pc_new = Atmosphere(network,\
                    filename+"_settings.yaml",\
                    stfilename,\
                    filename+"_init.txt")
        
        pc_new.var.atol = 1e-25
        pc_new.var.rtol = 1e-5
        pc_new.var.verbose = 0
        usol = pc_new.wrk.usol.copy()
        # usol[pc.dat.species_names.index('CH4'),:] = 1.0e-6
        pc_new.initialize_stepper(usol)
        tn = 0
        #tstop = 1e3
        #color=['r','b','k','green','pink','cyan','purple','yellow','olive']
        

        # np.flip(ch4),np.flip(co),np.flip(co2),np.flip(h2o),np.flip(hcn),np.flip(nh3), np.flip(h),np.flip(h2),np.flip(he),np.flip(c2h2),np.flip(c2h4),np.flip(c2h6),np.flip(so2),np.flip(h2s)
        while tn<tstop:
            
            #clear_output(wait=True)
            #plt.rcParams.update({'font.size': 15})
            #fig,ax = plt.subplots(1,1,figsize=[5,5])
            #fig.patch.set_facecolor("w")
        
            sol = pc_new.mole_fraction_dict()

            sp = 'CH4'
            ind = pc_new.dat.species_names.index(sp)
            if tn > 0:
                diff_ch4 = np.abs((pc_new.wrk.densities[ind,:]/pc_new.wrk.density) - old_meth)
                
                wh = np.where(diff_ch4  == np.max(diff_ch4))
                change_ch4 = diff_ch4[wh[0][0]]/old_meth[wh[0][0]]
                print("Methane relative Change ",change_ch4," tn ", tn)
            old_meth = (pc_new.wrk.densities[ind,:]/pc_new.wrk.density).copy()
            sp = 'NH3'
            ind = pc_new.dat.species_names.index(sp)
            
            if tn > 0:
                diff_nh3 = np.abs((pc_new.wrk.densities[ind,:]/pc_new.wrk.density) - old_nh3)
                
                wh = np.where(diff_nh3  == np.max(diff_nh3))
                change_nh3 = diff_nh3[wh[0][0]]/old_nh3[wh[0][0]]
                
                print("Amonnia relative Change ",change_nh3," tn ", tn)
            old_nh3 = (pc_new.wrk.densities[ind,:]/pc_new.wrk.density).copy()
            sp = 'SO2'
            ind = pc_new.dat.species_names.index(sp)
            
            if tn > 0:
                diff_so2 = np.abs((pc_new.wrk.densities[ind,:]/pc_new.wrk.density) - old_so2)
                
                wh = np.where(diff_so2  == np.max(diff_so2))
                change_so2 = diff_so2[wh[0][0]]/old_so2[wh[0][0]]
                print("SO2 relative Change ",change_so2," tn ", tn)
            old_so2 = (pc_new.wrk.densities[ind,:]/pc_new.wrk.density).copy()
            
            sp = 'CO2'
            ind = pc_new.dat.species_names.index(sp)
            if tn > 0:
                diff_co2 = np.abs((pc_new.wrk.densities[ind,:]/pc_new.wrk.density) - old_co2)
                
                wh = np.where(diff_co2  == np.max(diff_co2))
                change_co2 = diff_co2[wh[0][0]]/old_co2[wh[0][0]]
                print("CO2 relative Change ",change_co2," tn ", tn)
            old_co2 = (pc_new.wrk.densities[ind,:]/pc_new.wrk.density).copy()

            sp = 'H2O'
            ind = pc_new.dat.species_names.index(sp)
            if tn > 0:
                diff_h2o = np.abs((pc_new.wrk.densities[ind,:]/pc_new.wrk.density) - old_h2o)
                
                wh = np.where(diff_h2o  == np.max(diff_h2o))
                change_h2o = diff_h2o[wh[0][0]]/old_h2o[wh[0][0]]
                print("H2O relative Change ",change_h2o," tn ", tn)
            old_h2o = (pc_new.wrk.densities[ind,:]/pc_new.wrk.density).copy()

            sp = 'CO'
            ind = pc_new.dat.species_names.index(sp)
            if tn > 0:
                diff_co = np.abs((pc_new.wrk.densities[ind,:]/pc_new.wrk.density) - old_co)
                
                wh = np.where(diff_co  == np.max(diff_co))
                change_co = diff_co[wh[0][0]]/old_co[wh[0][0]]
                print("CO relative Change ",change_co," tn ", tn)
            old_co = (pc_new.wrk.densities[ind,:]/pc_new.wrk.density).copy()

            sp = 'HCN'
            ind = pc_new.dat.species_names.index(sp)
            if tn > 0:
                diff_hcn = np.abs((pc_new.wrk.densities[ind,:]/pc_new.wrk.density) - old_hcn)
                
                wh = np.where(diff_hcn  == np.max(diff_hcn))
                change_hcn = diff_hcn[wh[0][0]]/old_hcn[wh[0][0]]
                print("HCN relative Change ",change_hcn," tn ", tn)
            old_hcn = (pc_new.wrk.densities[ind,:]/pc_new.wrk.density).copy()

            
        
            #species = ['CO','CO2','CH4','H','H2','NH3','C2H2','H2S','SO2']
            #for i,sp in enumerate(species):
                
            #    ind = pc.dat.species_names.index(sp)
            #    ind_cantera = species_cantera.index(sp)
                
                
            #    tmp = pc.wrk.densities[ind,:]/pc.wrk.density
            #    ax.plot(tmp,pc.wrk.pressure/1e6,label=sp,color=color[i])
            #    ax.plot(deeper_surf[ind_cantera,:],pressure[:ind_surf],linestyle="--",color=color[i])
                
            
            #ax.set_xlim(1e-8,1)
            #ax.set_ylim(1e2,1e-7)
            #ax.set_xscale('log')
            #ax.set_yscale('log')
            #ax.text(0.02, 1.04, 't = '+'%e'%tn, \
            #    size = 15,ha='left', va='bottom',transform=ax.transAxes)
            #ax.grid()
            #ax.legend(ncol=2,bbox_to_anchor=(1,1.0),loc='upper left')
            #ax.set_xlabel('Mixing Ratio')
            #ax.set_ylabel('Pressure (mbar)')
            
            #plt.show()
            if tn>1e5:
                if change_ch4 <= 1e-3:
                    if change_nh3 <= 1e-3:
                        if change_so2 <= 1e-3:
                            if change_co2 <= 1e-3:
                                if change_h2o <= 1e-3:
                                    if change_co <= 1e-3:
                                        if change_hcn <= 1e-3:
                                            print("Stopping because Relative changes in CH4, NH3, SO2, H2O, CO, CO2, HCN are ", change_ch4," ",change_nh3," ",change_so2," ",change_h2o," ",change_co," ",change_co2," ",change_hcn)
                                            break
            elif np.logical_and(tn < 1e5,tn>0):
                if change_ch4 <= 1e-13:
                    if change_nh3 <= 1e-13:
                        if change_so2 <= 1e-13:
                            if change_co2 <= 1e-13:
                                if change_h2o <= 1e-13:
                                    if change_co <= 1e-13:
                                        if change_hcn <= 1e-13:
                                            print("Stopping early because Relative changes in CH4, NH3, SO2, H2O, CO, CO2, HCN are ", change_ch4," ",change_nh3," ",change_so2," ",change_h2o," ",change_co," ",change_co2," ",change_hcn)
                                            break

            tn_prev = tn
            for i in range(100):
                tn = pc_new.step()
                if tn > tstop:
                    break
            
        
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

    return pc_new,output_array,species,np.flip(pressure)


"""
      
    
