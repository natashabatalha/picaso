from pickle import STOP
from astropy.io import fits
import pandas as pd
import numpy as np
import json
import os
from numba import jit
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import inferno
import io 
import sqlite3
import math

class InterpolateCIAs():
    """
    This will be the class to retrieve correlated-k tables from the database. 
    Right now this is in beta mode and is retrieving the direct heritage 
    196 grid files. 

    Parameters
    ----------
    ck_dir : str 
        Directory of the pre-mixed correlated k table data. Check that you are pointing to a 
        directory with the files "ascii_data" and "full_abunds". 
    cont_dir : str 
        This should be in the references directory on Github. This has all the continuum opacity 
        that you need (e.g. H2H2). It also has some extra cross sections that you may want to add 
        in like Karkoschka methane and the optical ozone band. 
    wave_range : list 
        NOT FUNCTIONAL YET. 
        Wavelength range to compuate in the format [min micron, max micron]
    """
    def __init__(self, ck_dir, cont_dir, wave_range=None):
        self.ck_filename = ck_dir
        self.db_filename = cont_dir
        self.get_legacy_data(wave_range) #wave_range not used yet
        self.get_available_continuum()
        
        self.run_cia_spline()
        return

    def get_legacy_data(self,wave_range):
        

        path = '/Users/sagnickmukherjee/Documents/GitHub/Disequilibrium-Picaso/reference/climate_INPUTS/'
        wvno_new,dwni_new = np.loadtxt(path+"wvno_661",usecols=[0,1],unpack=True)
        self.wno = wvno_new
        self.delta_wno = dwni_new
        self.nwno = len(wvno_new) 

        
    
    
    def get_available_continuum(self):
        """Get the pressures and temperatures that are available for the continuum and molecules
        """
        #open connection 
        cur, conn = self.open_local()

        #get temps
        cur.execute('SELECT temperature FROM continuum')
        self.cia_temps = np.unique(cur.fetchall())
    
    def run_cia_spline(self):
        
        cur, conn = self.open_local()
        
        temps = self.cia_temps
        

        
        query_temp = 'AND temperature in '+str(tuple(temps) )
                
        cia_mol = [['H2', 'H2'], ['H2', 'He'], ['H2', 'N2'], ['H2', 'H'], ['H2', 'CH4'], ['H-', 'bf'], ['H-', 'ff'], ['H2-', '']]
        cia_names = {key[0]+key[1]  for key in cia_mol}
        query_mol = 'WHERE molecule in '+str(tuple(cia_names) )       

        cur.execute("""SELECT molecule,temperature,opacity 
                    FROM continuum 
                    {} 
                    {}""".format(query_mol, query_temp))

        data = cur.fetchall()
        data = dict((x+'_'+str(y), dat) for x, y,dat in data)
        

        cia_names = list(key[0]+key[1]  for key in cia_mol)
        
        
        #for i in cia_names:
        mol1 = np.zeros(shape = (len(temps),self.nwno))
        mol_name_1 = cia_names[0]
        for j,ind in zip(temps,range(len(temps))):
                atT_array = data[mol_name_1+'_'+str(j)]
                mol1[ind,:] = atT_array[:]
        
        mol2 = np.zeros(shape = (len(temps),self.nwno))
        mol_name_2 = cia_names[1]
        for j,ind in zip(temps,range(len(temps))):
                atT_array = data[mol_name_2+'_'+str(j)]
                mol2[ind,:] = atT_array[:]
        
        mol3 = np.zeros(shape = (len(temps),self.nwno))
        mol_name_3 = cia_names[2]
        for j,ind in zip(temps,range(len(temps))):
                atT_array = data[mol_name_3+'_'+str(j)]
                mol3[ind,:] = atT_array[:]
        
        mol4 = np.zeros(shape = (len(temps),self.nwno))
        mol_name_4 = cia_names[3]
        for j,ind in zip(temps,range(len(temps))):
                atT_array = data[mol_name_4+'_'+str(j)]
                mol4[ind,:] = atT_array[:]
        
        mol5 = np.zeros(shape = (len(temps),self.nwno))
        mol_name_5 = cia_names[4]
        for j,ind in zip(temps,range(len(temps))):
                atT_array = data[mol_name_5+'_'+str(j)]
                mol5[ind,:] = atT_array[:]
        
        mol6 = np.zeros(shape = (len(temps),self.nwno))
        mol_name_6 = cia_names[5]
        for j,ind in zip(temps,range(len(temps))):
                atT_array = data[mol_name_6+'_'+str(j)]
                mol6[ind,:] = atT_array[:]
        
        mol7 = np.zeros(shape = (len(temps),self.nwno))
        mol_name_7 = cia_names[6]
        for j,ind in zip(temps,range(len(temps))):
                atT_array = data[mol_name_7+'_'+str(j)]
                mol7[ind,:] = atT_array[:]
        
        mol8 = np.zeros(shape = (len(temps),self.nwno))
        mol_name_8 = cia_names[7]
        for j,ind in zip(temps,range(len(temps))):
                atT_array = data[mol_name_8+'_'+str(j)]
                mol8[ind,:] = atT_array[:]
        
        y_mol1 = np.zeros(shape=(len(temps),self.nwno))
        y_mol2 = np.zeros(shape=(len(temps),self.nwno))
        y_mol3 = np.zeros(shape=(len(temps),self.nwno))
        y_mol4 = np.zeros(shape=(len(temps),self.nwno))
        y_mol5 = np.zeros(shape=(len(temps),self.nwno))
        y_mol6 = np.zeros(shape=(len(temps),self.nwno))
        y_mol7 = np.zeros(shape=(len(temps),self.nwno))
        y_mol8 = np.zeros(shape=(len(temps),self.nwno))

        dts0 = temps[1]-temps[0]
        dts1 = temps[-1]-temps[-2]
        for iw in range(self.nwno):
            # for mol1
            yp0 = (mol1[1,iw]-mol1[2,iw])/dts0
            ypn = (mol1[-1,iw]-mol1[-2,iw])/dts1
            y_mol1[:,iw] = spline(temps,mol1[:,iw],len(temps),yp0,ypn)

            yp0 = (mol2[1,iw]-mol2[2,iw])/dts0
            ypn = (mol2[-1,iw]-mol2[-2,iw])/dts1
            y_mol2[:,iw] = spline(temps,mol2[:,iw],len(temps),yp0,ypn)

            yp0 = (mol3[1,iw]-mol3[2,iw])/dts0
            ypn = (mol3[-1,iw]-mol3[-2,iw])/dts1
            y_mol3[:,iw] = spline(temps,mol3[:,iw],len(temps),yp0,ypn)

            yp0 = (mol4[1,iw]-mol4[2,iw])/dts0
            ypn = (mol4[-1,iw]-mol4[-2,iw])/dts1
            y_mol4[:,iw] = spline(temps,mol4[:,iw],len(temps),yp0,ypn)

            yp0 = (mol5[1,iw]-mol5[2,iw])/dts0
            ypn = (mol5[-1,iw]-mol5[-2,iw])/dts1
            y_mol5[:,iw] = spline(temps,mol5[:,iw],len(temps),yp0,ypn)

            yp0 = (mol6[1,iw]-mol6[2,iw])/dts0
            ypn = (mol6[-1,iw]-mol6[-2,iw])/dts1
            y_mol6[:,iw] = spline(temps,mol6[:,iw],len(temps),yp0,ypn)

            yp0 = (mol7[1,iw]-mol7[2,iw])/dts0
            ypn = (mol7[-1,iw]-mol7[-2,iw])/dts1
            y_mol7[:,iw] = spline(temps,mol7[:,iw],len(temps),yp0,ypn)

            yp0 = (mol8[1,iw]-mol8[2,iw])/dts0
            ypn = (mol8[-1,iw]-mol8[-2,iw])/dts1
            y_mol8[:,iw] = spline(temps,mol8[:,iw],len(temps),yp0,ypn)

        
        
        hdu = fits.PrimaryHDU(y_mol1)
        hdul = fits.HDUList([hdu])
        hdul.writeto("INPUTS/"+cia_names[0]+'661.fits',overwrite=True)
        hdul.close()
        

        hdu = fits.PrimaryHDU(y_mol2)
        hdul = fits.HDUList([hdu])
        hdul.writeto("INPUTS/"+cia_names[1]+'661.fits',overwrite=True)
        hdul.close()
        

        hdu = fits.PrimaryHDU(y_mol3)
        hdul = fits.HDUList([hdu])
        hdul.writeto("INPUTS/"+cia_names[2]+'661.fits',overwrite=True)
        hdul.close()

        hdu = fits.PrimaryHDU(y_mol4)
        hdul = fits.HDUList([hdu])
        hdul.writeto("INPUTS/"+cia_names[3]+'661.fits',overwrite=True)
        hdul.close()

        hdu = fits.PrimaryHDU(y_mol5)
        hdul = fits.HDUList([hdu])
        hdul.writeto("INPUTS/"+cia_names[4]+'661.fits',overwrite=True)
        hdul.close()
        
        hdu = fits.PrimaryHDU(y_mol6)
        hdul = fits.HDUList([hdu])
        hdul.writeto("INPUTS/"+cia_names[5]+'661.fits',overwrite=True)
        hdul.close()

        hdu = fits.PrimaryHDU(y_mol7)
        hdul = fits.HDUList([hdu])
        hdul.writeto("INPUTS/"+cia_names[6]+'661.fits',overwrite=True)
        hdul.close()

        hdu = fits.PrimaryHDU(y_mol8)
        hdul = fits.HDUList([hdu])
        hdul.writeto("INPUTS/"+cia_names[7]+'661.fits',overwrite=True)
        hdul.close()
    
    def open_local(self):
        """Code needed to open up local database, interpret arrays from bytes and return cursor"""
        conn = sqlite3.connect(self.db_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        #tell sqlite what to do with an array
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)
        cur = conn.cursor()
        return cur,conn

    def get_opacities(self, atmosphere):
        self.get_continuum(atmosphere)
        self.get_pre_mix_ck(atmosphere)

    def adapt_array(arr):
        """needed to interpret bytes to array"""
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(clas, text):
        """needed to interpret bytes to array"""
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

def spline(x , y, n, yp0, ypn):
    
    u=np.zeros(shape=(n))
    y2 = np.zeros(shape=(n))

    if yp0 > 0.99 :
        y2[0] = 0.0
        u[0] =0.0
    else:
        y2[0]=-0.5
        u[0] = (3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp0)

    for i in range(1,n-1):
        sig=(x[i]-x[i-1])/(x[i+1]-x[i-1])
        p=sig*y2[i-1]+2.
        y2[i]=(sig-1.)/p
        u[i]=(6.0*((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]))/(x[i+1]-x[i-1])-sig*u[i-1])/p

    if ypn > 0.99 :
        qn = 0.0
        un = 0.0
    else:
        qn =0.5
        un = (3.0/(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
    
    y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2]+1.0)

    for k in range(n-2, -1, -1):
        y2[k] = y2[k] * y2[k+1] +u[k]
    
    return y2

filename_db="/Users/sagnickmukherjee/Documents/GitHub/Disequilibrium-Picaso/ck_cx_cont_opacities_661.db"
ck_db='/Users/sagnickmukherjee/Documents/software/picaso-dev/reference/opacities/ck_db/m+0.5_co1.0.data.196'
opacityclass=InterpolateCIAs(
                    ck_db, 
                    filename_db, 
                    wave_range = [0,1])

#hdul = fits.open('H2H2.fits')
#data= hdul[0].data
#print(data)