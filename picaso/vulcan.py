import numpy as np
import os
import pysynphot as psyn

import pickle
# this function here runs the vulcan script on the fly to do real photochemical calculations.
# this routine is very much hard coded so be extremely careful with this.
# paths had to added to many functions in the vulcan_whole folder to make them work properly.
def run_vulcan_chem(pressure,temp,kz,grav, first = False, photochem = False, r_star=None,semi_major=None,r_planet=None):
    # first save profile for vulcan to read from
    #print(pressure*1e6,temp,kz)
    np.savetxt("/home/sagnick/Disequilibrium-Picaso/picaso/vulcan_whole/atm/tpkzz.txt",np.transpose([pressure*1e6,temp,kz]))
    # first modify the vulcan_cfg file
    if first == True:
        

        cfg_path = "/home/sagnick/Disequilibrium-Picaso/picaso/vulcan_whole/vulcan_cfg.py"
        f = open(cfg_path, 'r')    # pass an appropriate path of the required file
        lines = f.readlines()
        lines[79-1] = "gs = "+str(grav*100)+"\n" #cgs    # n is the line number you want to edit; subtract 1 as indexing of list starts from 0
        lines[33-1] = "ini_mix = 'EQ' \n"
        lines[68-1] = "nz = "+str(len(pressure))+" \n"
        lines[69-1] = "P_b = "+str(np.max(pressure*1e6))+" \n"
        lines[70-1] = "P_t = "+str(np.min(pressure)*1e6)+" \n"

        if photochem == True :
            lines[9-1] = "network = path+'thermo/NCHO_photo_network.txt'"+" \n"
            lines[15-1] = "sflux_file = path+'atm/stellar_flux/starfile.txt'"+" \n"
            lines[39-1] = "use_photo = True \n"
            lines[41-1] = "r_star = "+str(r_star)+" \n"
            lines[42-1] = "Rp = "+str(r_planet)+"*7.1492E9"+ " \n"
            lines[43-1] = "orbit_radius = " + str(semi_major)+" \n"
        else :
            lines[9-1] = "network = path+'thermo/NCHO_thermo_network.txt'"+" \n"
            lines[39-1] = "use_photo = False \n"
        f.close()   # close the file and reopen in write mode to enable writing to file; you can also open in append mode and use "seek", but you will have some unwanted old data if the new data is shorter in length.
        f = open(cfg_path, 'w')
        f.writelines(lines)
        # do the remaining operations on the file
        f.close()

        os.system("python /home/sagnick/Disequilibrium-Picaso/picaso/vulcan_whole/vulcan.py")

    else :
        cfg_path = "/home/sagnick/Disequilibrium-Picaso/picaso/vulcan_whole/vulcan_cfg.py"
        f = open(cfg_path, 'r')    # pass an appropriate path of the required file
        lines = f.readlines()
        #lines[79-1] = "gs = "+str(grav*100)+"\n" #cgs    # n is the line number you want to edit; subtract 1 as indexing of list starts from 0
        lines[33-1] = "ini_mix = 'vulcan_ini'  \n"
        f.close()   # close the file and reopen in write mode to enable writing to file; you can also open in append mode and use "seek", but you will have some unwanted old data if the new data is shorter in length.
        f = open(cfg_path, 'w')
        f.writelines(lines)
        # do the remaining operations on the file
        f.close()

        os.system("python /home/sagnick/Disequilibrium-Picaso/picaso/vulcan_whole/vulcan.py -n")

    #if changing chemical network file then remove the -n at last
    #os.system("python /Users/sagnickmukherjee/Documents/GitHub/Disequilibrium-Picaso/picaso/vulcan_whole/vulcan.py")

    output = open('/home/sagnick/Disequilibrium-Picaso/picaso/vulcan_whole/output/mix_table/test.txt', "w")

    plot_EQ = False
    vul = '/home/sagnick/Disequilibrium-Picaso/picaso/vulcan_whole/output/profile.vul'
    with open(vul, 'rb') as handle:
        vul = pickle.load(handle)
    
    species = vul['variable']['species'] 
    out_species = ['CH4', 'CO', 'CO2', 'C2H2', 'H2', 'H', 'H2O', 'HCN', 'He', 'NH3', 'O2', 'NO', 'OH']

    ost = '{:<8s}'.format('(dyn/cm2)')  + '{:>9s}'.format('(K)') + '{:>9s}'.format('(cm)') + '\n'
    ost += '{:<8s}'.format('Pressure')  + '{:>9s}'.format('Temp')+ '{:>9s}'.format('Hight')
    for sp in out_species: ost += '{:>10s}'.format(sp) 
    ost +='\n'
    
    for n, p in enumerate(vul['atm']['pco']):
        ost += '{:<8.3E}'.format(p)  + '{:>8.1f}'.format(vul['atm']['Tco'][n])  + '{:>10.2E}'.format(vul['atm']['zco'][n])
        for sp in out_species:
            if plot_EQ == True:
                ost += '{:>10.2E}'.format(vul['variable']['y_ini'][n,species.index(sp)]/vul['atm']['n_0'][n])
            else: 
                ost += '{:>10.2E}'.format(vul['variable']['ymix'][n,species.index(sp)])
        ost += '\n'

    ost = ost[:-1]
    output.write(ost)
    output.close()

    ch4,co,co2,h2o,hcn,nh3,h=np.loadtxt("/home/sagnick/Disequilibrium-Picaso/picaso/vulcan_whole/output/mix_table/test.txt",usecols=[3,4,5,9,10,12,8],unpack=True, skiprows=2)

    return np.flip(ch4),np.flip(co),np.flip(co2),np.flip(h2o),np.flip(hcn),np.flip(nh3), np.flip(h)




    

