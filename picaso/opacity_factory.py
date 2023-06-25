import numpy as np
import os
import json
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import scipy.signal as sig
import sqlite3
import io
from astropy.io import fits
import math
from scipy.io import FortranFile
import glob
from scipy.stats import binned_statistic


__refdata__ = os.environ.get('picaso_refdata')


def restruct_continuum(original_file,colnames, new_wno,new_db, overwrite):
    """
    The continuum factory takes the CIA opacity file and adds in extra sources of 
    opacity from other references to fill in empty bands. It assumes that the original file is 
    structured as following,with the first column wavenumber and the rest molecules: 
    1000 198
    75.
    0.0  -33.0000  -33.0000  -33.0000  -33.0000  -33.0000
    20.0   -7.4572   -7.4518   -6.8038   -6.0928   -5.9806
    40.0   -6.9547   -6.9765   -6.6322   -5.7934   -5.4823
    ... ... ... etc 

    Where 1000 corresponds to the number of wavelengths, 198 corresponds to the number of temperatures
    and 75 is the first temperature. If this structure changes, we will need to restructure this top 
    level __init__ function that reads it in the original opacity file. 
    
    Parameters
    ----------
    original_file : str
        Filepath that points to original opacity file (see description above)
    colnames : list 
        defines the sources of opacity in the original file. For the example file above, 
        colnames would be ['wno','H2H2','H2He','H2H','H2CH4','H2N2']
    new_wno : numpy.ndarray, list 
        wavenumber grid to interpolate onto (units of inverse cm)
    new_db : str 
        New database name 
    overwrite : bool 
        Default is set to False as to not overwrite any existing files. This parameter controls overwriting 
        cia database 
    """
    og_opacity, temperatures, old_wno, molecules = get_original_data(original_file,
        colnames, overwrite=overwrite,new_db=new_db)

    ntemp = len(temperatures)

    #restructure and insert to database 
    restructure_opacity(new_db,ntemp,temperatures,molecules,og_opacity,old_wno,new_wno)

def insert_hitran_cia(original_file, molname, new_db, new_wno):
    """
    This continuum function takes a HTIRAN CIA file and adds it as an extra 
    source of opacity

    Parameters
    ----------
    original_file : str
        Filepath that points to HITRAN file (e.g. H2-CH4_eq_2011.cia)
    new_db : str 
        New database name 
    new_wno : numpy.ndarray, list 
        wavenumber grid to interpolate onto (units of inverse cm)
    overwrite : bool 
        Default is set to False as to not overwrite any existing files. This parameter controls overwriting 
        cia database 
    """
    #get original H2- CIA that is already inserted
    cur, conn = open_local(new_db)
    try:
        cur.execute('SELECT temperature FROM continuum')
        current_cia_temps = np.unique(cur.fetchall())
        conn.close()
    except: 
        raise Exception('No temperatures found in continuum table. Usually we insert the H2 continuum first since it exists on a larger H2 Temperature grid. Then, we insert other CIA. Please run restruct_continuum first to insert H/H2 based opacity first.')

    #https://hitran.org/data/CIA/CIA_Readme.pdf
    hitran_header = {'chemical':[0,20],
                'wavenumber':[20,40],
                'num_pts':[40,47],
                'temp':[47,54]}

    #cureated grid based on analyses of each molecule
    #this tells you what tgrid to use for each molecule, and what 
    #opacities to ignore from hitran
    hitran_choices = {'N2N2':{'ignore_tag':['0-10atm'],
                          'tgrid':[70.,  80.,  90., 100., 110., 120., 130., 140., 150., 160., 170.,
       180., 190., 200., 210., 220., 230., 240., 250., 260., 270., 280.,
       290., 300., 310., 320., 330., 340., 350., 360., 370., 380., 390.,
       400.],
                         }} 

    cia = pd.read_csv(original_file,names=['wno','cm5/molecule2'], header=None,usecols=[0,1],
                delim_whitespace=True)
    header = pd.read_csv(original_file,header=None,names=['header'])


    idx_data =[]
    idx_head = []
    for i in cia.index: 
        try: 
            float(cia.loc[i,'wno'])
            idx_data += [i]
        except:
            idx_head += [i]
            
    data_header = header.loc[idx_head,:]
    data_cia = cia.loc[idx_data,:].astype(float)
    #dont care about zero opacity grid points
    data_cia = data_cia.loc[data_cia['cm5/molecule2']>0]

    temperatures = [float(i[hitran_header['temp'][0]:hitran_header['temp'][1]]) 
         for i in data_header['header']]

    #log the index of each data bundle beginning
    starts = list(data_header.index+1)+[data_cia.index[-1]+1]

    #loop over each data bundle and interpolate all on the same wavelength grid requested 
    #by the user
    #also check the ignore tag to see if we should be skipping
    #skip data if specified
    wno_arrays=[]
    cx_arrays =[]
    temp_array=[]
    wno_range_array=[]
    for i,iss in enumerate(starts[0:-1]):
        ignore = False
        if molname in hitran_choices.keys():
            if 'ignore_tag' in hitran_choices[molname]: 
                for it in hitran_choices[molname]['ignore_tag']:
                    if it in data_header.loc[iss-1,'header']:
                        ignore=True
        if ignore: 
            continue
        
        #cx = data_cia.loc[starts[i]:starts[i+1]-1,'cm2.mol']#']
        #cia_array[:,i]=cx
        wno = data_cia.loc[starts[i]:starts[i+1]-1,'wno'].values #'])
        #wno_arrays += [wno.values]
        
        cx =  data_cia.loc[starts[i]:starts[i+1]-1,'cm5/molecule2']
        cx_arrays += [10**np.interp(new_wno, wno, np.log10(cx.values),right=-100,left=-100)]
        #cia_array[:,i]=10**(np.interp(all_wnos, wno, np.log10(cx),right=-100,left=-100))
        
        #get temp array and wno ranges just so we can double check for no overlap
        temp_array += [temperatures[i]]
        wno_range_array += [[float(ir) for ir in data_header.loc[iss-1,'header'][
            hitran_header['wavenumber'][0]:hitran_header['wavenumber'][1]].split(' ') if len(ir)>0]]
    
    #now start to deal with overlapping temperatures 
    #double check for overlapping wavelengths within grid and raise error if it exists
    temp_array = np.array(temp_array)
    segs = np.diff(temp_array)
    #if there are overlapping temperature regions need to get on common tgrid
    if len(segs[segs<0])>0:
        iranges = []
        tranges =[]
        wranges =[]
        cx_segments = []
        
        inds = [0]+list(np.where(segs<0)[0]+1)+[len(temp_array)]
        
        for i in range(len(inds)-1):
            iranges += [[inds[i],inds[i+1]]]
            tranges += [temp_array[inds[i]:inds[i+1]]]
            wranges += [[np.min(wno_range_array[inds[i]:inds[i+1]]),np.max(wno_range_array[inds[i]:inds[i+1]])]]
        if molname in hitran_choices.keys(): 
            tgrid = hitran_choices[molname]['tgrid']
        else:
            raise Exception('Overlapping Temperature Grid, need to provide one in hitran choices',tranges)
        #check for overlapping wavelengths 
        w_sets = np.diff(np.argsort(np.concatenate(wranges)))
        if len(w_sets[w_sets<0])>0:
            raise Exception('Overlapping Wavelength Grid within Temp, need to provide one in hitran choices',wranges)
    #otherwise just use the full temperature array
    else: 
        tranges = [temp_array]
        tgrid = temp_array
        iranges = [[0,len(temp_array)]]


    #finally, get everything on the same temperature grid for each individual wavelength
    summed_cx = np.zeros(( len(current_cia_temps),len(new_wno)))
    ii = 0 
    for og_t_grid,inds in zip(tranges,iranges): 
        if not np.array_equal(og_t_grid, tgrid):
            #for each temp
            cx = np.array(cx_arrays)[inds[0]:inds[1],:]
            for iw in range(len(new_wno)): 
                #get onto new tgrid
                #NOTE this seems repetetive but in this interpolation
                #we are extrapolating the bands to a uniform grid 
                #in the next interpolation, we are zeroing the bands outside the T range
                ycx= 10**np.interp(tgrid, og_t_grid, np.log10(cx[:,iw]))#purposefully leaveing as default left, right
                #now interpolate to the full CIA t grid
                #notice the addition of left and right bounds
                ycx= 10**np.interp(current_cia_temps, tgrid, np.log10(ycx),
                        left = -100, right=-100)
                summed_cx[:,iw] += ycx

        else: 
            cx = np.array(cx_arrays)[inds[0]:inds[1],:]
            for iw in range(len(new_wno)): 
                #interp to master cia temp grid 
                summed_cx[:,iw] += 10**np.interp(current_cia_temps,tgrid,np.log10(cx[:, iw]),
                            left = -100, right=-100)


    cur, conn = open_local(new_db)
    for it in range(len(current_cia_temps)):
        #CONVERT FROM cm5/molecule2 to amagat-2 cm-1
        #Eqn 3: https://www.sciencedirect.com/science/article/pii/S0019103518306997
        final_bundle = summed_cx[it, :]/1.385277e-39
        insert(cur,conn,molname, current_cia_temps[it], final_bundle)
    conn.commit()
    conn.close()
    return 

def get_original_data(original_file,colnames,new_db, overwrite=False):
    """
    The continuum factory takes the CIA opacity file and adds in extra sources of 
    opacity from other references to fill in empty bands. It assumes that the original file is 
    structured as following,with the first column wavenumber and the rest molecules: 
    1000 198
    75.
    0.0  -33.0000  -33.0000  -33.0000  -33.0000  -33.0000
    20.0   -7.4572   -7.4518   -6.8038   -6.0928   -5.9806
    40.0   -6.9547   -6.9765   -6.6322   -5.7934   -5.4823
    ... ... ... etc 

    Where 1000 corresponds to the number of wavelengths, 198 corresponds to the number of temperatures
    and 75 is the first temperature. If this structure changes, we will need to restructure this top 
    level __init__ function that reads it in the original opacity file. 
    
    Parameters
    ----------
    original_file : str
        Filepath that points to original opacity file (see description above)
    colnames : list 
        defines the sources of opacity in the original file. For the example file above, 
        colnames would be ['wno','H2H2','H2He','H2H','H2CH4','H2N2']
    new_db : str 
        New database name 
    overwrite : bool 
        Default is set to False as to not overwrite any existing files. This parameter controls overwriting 
        cia database 
   """
    og_opacity = pd.read_csv(original_file,delim_whitespace=True,names=colnames)
    
    temperatures = og_opacity['wno'].loc[np.isnan(og_opacity[colnames[1]])].values

    og_opacity = og_opacity.dropna()
    old_wno = og_opacity['wno'].unique()
    #define units
    w_unit = 'cm-1'
    opacity_unit = 'cm-1 amagat^-2'
    molecules = colnames[1:]
    temperature_unit = 'K'
    
    #create database file
    if os.path.exists(new_db):
        if overwrite:
            raise Exception("Overwrite is set to false to save db's from being overwritten.")

    return og_opacity, temperatures, old_wno, molecules

def insert(cur,conn,mol,T,opacity):
    """Insert into """
    cur.execute('INSERT INTO continuum (molecule, temperature, opacity) values (?,?,?)', (mol,float(T), opacity))

def restructure_opacity(new_db,ntemp,temperatures,molecules,og_opacity,old_wno,new_wno):
    #start by opening connection 
    dw = new_wno[1] - new_wno[0] 
    kernel_size = (10050 - 9960)/dw
    kernel_size= int(np.ceil(kernel_size) // 2 * 2 + 1)
    cur, conn = open_local(new_db)
    nwno = len(old_wno)
    #needed for any empty arrays (e.g. h2 minus below 600)
    zero_bundle  = np.zeros(len(new_wno)) + 1e-33
    hminbf_run = True
    for i in range(ntemp): 
        for m in molecules:
            opa_bundle = og_opacity.iloc[ i*nwno : (i+1) * nwno][m].values
            new_bundle = 10**(np.interp(new_wno,  old_wno, opa_bundle,right=-33,left=-33))
            #now for anywhere that doesn't have opacity (-33) replace with linsky
            if m=='H2H2':
                #first add h2h2 overtone band 
                h2h2, loc, add =h2h2_overtone(temperatures[i],new_wno)
                if add:
                    new_bundle[loc] = h2h2

                #add last linsky hack in between h2h2 overtone and original data 
                loc_33 = np.where((new_bundle==1e-33) & (new_wno>=1000))
                
                new_bundle_w_lin = fit_linsky(temperatures[i],new_wno[loc_33])
                new_bundle[loc_33]  = new_bundle_w_lin 

                if len(loc_33[0])!=0:
                    #now fix discontinuity (if they exist near 1 micron)
                    if max(new_wno[loc_33] <12000):
                        loc_smooth = np.where((new_wno>9950) & (new_wno<11200))
                        if len(loc_smooth[0]) != 0: 
                            new_bundle_smooth = sig.medfilt(np.array(new_bundle[loc_smooth]), kernel_size=kernel_size)
                            new_bundle[loc_smooth] = new_bundle_smooth

                #this is to smooth the discontinuous parts 
            insert(cur,conn,m, temperatures[i], new_bundle)

        #NOW H2- for temperatures greater than 600 
        if temperatures[i]<600.0:
            bundle = zero_bundle 
        else:
            bundle = get_h2minus(temperatures[i],new_wno)
        insert(cur,conn,'H2-', temperatures[i], bundle)

        #NOW H-bf for temperatures greater than 600 
        if temperatures[i]<800.0:
            bundle = zero_bundle
            insert(cur,conn,'H-bf', temperatures[i], bundle)
        else:
            if hminbf_run: 
                #make sure to only run once since there is no temp dependence 
                hminusbf = get_hminusbf(new_wno)
                hminbf_run = False
            insert(cur,conn,'H-bf', temperatures[i], hminusbf)

        #NOW H-FF
        if temperatures[i]<800.0:
            bundle = zero_bundle*1e-30
        else: 
            bundle = get_hminusff(temperatures[i], new_wno)
        insert(cur,conn,'H-ff', temperatures[i], bundle)  

    conn.commit()
    conn.close()


def h2h2_overtone(t, wno):
    """
    Add in special CIA h2h2 band at 0.8 microns

    Parameters
    ---------- 
    t : float
        Temperature 
    wno : numpy.array
        wave number

    Returns
    -------
    H2-H2 absorption in cm-1 amagat-2       
    """
    fname = os.path.join(__refdata__, 'opacities','H2H2_ov2_eq.tbl')
    df = pd.read_csv(fname, delim_whitespace=True).set_index('wavenumber').apply(np.log10)      
    temps = [ float(i) for i in df.keys()]

    if t > max(temps):
        return np.nan, np.nan, False

    it = find_nearest(temps, t)
    placeholder_temp = temps[it]
    loc = np.where((wno>=df.index.min()) & (wno<=df.index.max()))
    new_opa = 10**(np.interp(wno[loc],  np.array(df.index), df[list(df.keys())[it]].values,right=-33,left=-33))
    return new_opa, loc, True

def fit_linsky(t, wno, va=3):
    """
    Adds H2-H2 opacity from Linsky (1969) and Lenzuni et al. (1991) 
    to values that were set to -33 as place holders. 

    Parameters
    ---------- 
    t : float
        Temperature 
    wno : numpy.array
        wave number
    va : int or float
        (Optional) 1,2 or 3 (depending on what overtone to compute 

    Returns
    -------
    H2-H2 absorption in cm-1 amagat-2 
    """

    #these numbers are hard coded from Lenuzi et al 1991 Table 8. 
    sig0 = np.array([4162.043,8274.650,12017.753]) #applicable sections in wavelength 

    d1 = np.array([1.2750e5,1.32e6,1.32e6])
    d2 = np.array([2760.,2760.,2760.])
    d3 = np.array([0.40,0.40,0.40])

    a1 = np.array([-7.661,-9.70,-11.32])
    a2 = np.array([0.5725,0.5725,0.5725])

    b1 = np.array([0.9376,0.9376,0.9376])
    b2 = np.array([0.5616,0.5616,0.5616])
    va = va-1
    w = sig0[va]

    d=d3[va]*np.sqrt(d1[va]+d2[va]*t)
    a=10**(a1[va]+a2[va]*np.log10(t))
    b=10**(b1[va]+b2[va]*np.log10(t))
    aa=4.0/13.0*a/d*np.exp(1.5*d/b)
    kappa = aa*wno*np.exp(-(wno-w)/b)
    smaller = np.where(wno<w)

    if len(smaller)>0:
        kappa[smaller] = a*d*wno[smaller]*np.exp((wno[smaller]-w)/0.6952/t)/((wno[smaller]-w)**2+d*d)
    even_smaller = np.where(wno<w+1.5*d)
    if len(even_smaller)>0:
        kappa[even_smaller]=a*d*wno[even_smaller]/((wno[even_smaller]-w)**2+d*d)

    return kappa

def get_h2minus(t, new_wno):
    """
    This results gives the H2 minus opacity, needed for temperatures greater than 600 K. 
    K L Bell 1980 J. Phys. B: At. Mol. Phys. 13 1859, Table 1 
    theta=5040/T(K)

    The result is given in cm4/dyn, which is why will will  multiply by nh2*ne*k*T
    where:
    nh2: number of h2 molecules/cm3
    ne: number of electrons/cm3
    This will happen when we go to sum opacities. For now returns will be in cm4/dyn
    
    Parameters
    ----------
    t : numpy.ndarray
        temperature in K
    wno : numpy.ndarray
        wave number cm-1

    Returns
    -------
    H2 minus opacity in units of cm4/dyn
    """
    fname = os.path.join(os.environ.get('picaso_refdata'), 'opacities','h2minus.csv')
    df = pd.read_csv(fname, skiprows=5, header=0).set_index('theta').apply(np.log10)

    #Bell+1980 wavenumber
    wno_bell = 1e8/df.columns.astype(float).values

    new_theta = 5040.0/t
    itheta = find_nearest_1d(df.index.values,new_theta)

    #grab what we want to interp
    kappa_bell = (10**df.values[itheta,:])*1e-26 

    kappa_new = np.interp(new_wno, wno_bell,kappa_bell,left=1e-33, right=1e-33)

    return kappa_new

def get_hminusbf(wno):
    """
    H- bound free opacity, which is only dependent on wavelength. From John 1988 http://adsabs.harvard.edu/abs/1988A%26A...193..189J

    Parameters
    ----------
    wno :  numpy.ndarray 
        Wavenumber in cm-1

    Returns 
    -------
    array of floats 
        Absorption coefficient in units of cm2
    """
    coeff = np.array([152.519,49.534,-118.858,92.536, -34.194,4.982])[::-1]
    lambda_0 = 1.6419
    wave =1e4/wno
    nonzero = np.where(wno > 1e4/lambda_0)
    f = np.zeros(np.size(wave))
    x = np.zeros(np.size(wave))
    result = np.zeros(np.size(wave)) + 1e-33

    x[nonzero]=np.sqrt(1.0/wave[nonzero]-1.0/lambda_0)
    
    for i in coeff: 
        f[nonzero] = f[nonzero]*x[nonzero] + i 
    result[nonzero] = (wave[nonzero]*x[nonzero])**3*f[nonzero]*1e-18
    return result
        
def get_hminusff(t, wno):
    """
    H- free free opacity, which is both wavelength and temperature dependent. 
    From Bell & Berrington (1987)
    Also includes factor for simulated emission 
    
    Parameters
    ----------
    t :  float
        Temperature in K 
    wno :  numpy.ndarray
        Wavenumber in cm-1

    Returns
    -------
    array of float
        Gives cross section in cm^5
    """
    AJ1= [0.e0, 2483.346, -3449.889, 2200.040, -696.271, 88.283]
    BJ1= [0.e0, 285.827, -1158.382, 2427.719, -1841.400, 444.517]
    CJ1= [0.e0, -2054.291, 8746.523, -13651.105, 8624.970, -1863.864]
    DJ1= [0.e0, 2827.776, -11485.632, 16755.524, -10051.530, 2095.288]
    EJ1= [0.e0, -1341.537, 5303.609, -7510.494, 4400.067,  -901.788]
    FJ1= [0.e0, 208.952, -812.939, 1132.738, -655.020, 132.985]
    AJ2= [518.1021, 473.2636, -482.2089, 115.5291, 0.0,0.0]
    BJ2= [-734.8666, 1443.4137, -737.1616, 169.6374, 0.0,0.0]
    CJ2= [1021.1775, -1977.3395, 1096.8827, -245.649, 0.0,0.0]
    DJ2= [-479.0721, 922.3575, -521.1341, 114.243, 0.0,0.0]
    EJ2= [93.1373, -178.9275, 101.7963, -21.9972,0.0,0.0]
    FJ2= [-6.4285, 12.3600, -7.0571, 1.5097, 0.0,0.0    ]

    wave = 1e4/wno 

    nwave = np.size(wave)

    if t<800 : 
        return np.zeros(nwave ) + 1e-60

    t_coeff = 5040.0/t



    hj = np.zeros((6, nwave))
    longw = np.where(wave>0.3645)
    midw = np.where((wave<=0.3645))
    shortw = np.where(wave<0.1823)
    wave[shortw] = 0.1823
    for i in range(6):
        hj[i,longw] = 1e-29*(wave[longw]*wave[longw]*AJ1[i] + BJ1[i] + (CJ1[i] + (
                        DJ1[i] + (EJ1[i] + FJ1[i]/wave[longw])/wave[longw])/wave[longw])/wave[longw])
        hj[i,midw] = 1e-29*(wave[midw]*wave[midw]*AJ2[i] + BJ2[i] + (
                        CJ2[i] + (DJ2[i] +(EJ2[i] + FJ2[i]/wave[midw])/wave[midw])/wave[midw])/wave[midw])

    hm_cx = np.zeros(nwave)
    for i in range(6):
        hm_cx  += t_coeff**((i+1)/2.0)*hj[i, :]

    #this parameterization is not valid past 20 micron..
    past20 = np.where(wave>20.0)

    if np.size(past20) > 0 :
        hm_cx[past20] = np.zeros(np.size(past20)) 

    return  hm_cx * 1.380658e-16 * t 



#these functions are so that you can store your float arrays as bytes to minimize storage
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def open_local(db_f):
    """Code needed to open up local database, interpret arrays from bytes and return cursor"""
    conn = sqlite3.connect(db_f, detect_types=sqlite3.PARSE_DECLTYPES)
    #tell sqlite what to do with an array
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    cur = conn.cursor()
    return cur,conn

#class MolecularFactory():

#these functions are so that you can store your float arrays as bytes to minimize storage
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def open_local(db_f):
    """Code needed to open up local database, interpret arrays from bytes and return cursor"""
    conn = sqlite3.connect(db_f, detect_types=sqlite3.PARSE_DECLTYPES)
    #tell sqlite what to do with an array
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    cur = conn.cursor()
    return cur,conn


def build_skeleton(db_f):
    """This functionb builds a skeleton sqlite3 database with three tables:
        1) header
        2) molecular
        3) continuum
    """
    cur, conn = open_local(db_f)
    #header
    command="""DROP TABLE IF EXISTS header;
    CREATE TABLE header (
        id INTEGER PRIMARY KEY,
        pressure_unit VARCHAR,
        temperature_unit VARCHAR,
        wavenumber_grid array,
        continuum_unit VARCHAR,
        molecular_unit VARCHAR
        );"""

    cur.executescript(command)
    #molecular data table, note the existence of PTID which will be very important
    command = """DROP TABLE IF EXISTS molecular;
    CREATE TABLE molecular (
        id INTEGER PRIMARY KEY,
        ptid INTEGER,
        molecule VARCHAR ,
        pressure FLOAT,
        temperature FLOAT,
        opacity array);"""

    cur.executescript(command)
    #continuum data table
    command = """DROP TABLE IF EXISTS continuum;
    CREATE TABLE continuum (
        id INTEGER PRIMARY KEY,
        molecule VARCHAR ,
        temperature FLOAT,
        opacity array);"""

    cur.executescript(command)
    
    conn.commit() #this commits the changes to the database
    conn.close()



def insert_wno_grid(wno_grid, cur, con):
    """
    Inserts basics into the header file. This puts all the units to the database.
    """
    cur.execute('INSERT INTO header (pressure_unit, temperature_unit, wavenumber_grid, continuum_unit,molecular_unit) values (?,?,?,?,?)', 
                ('bar','kelvin', np.array(wno_grid), 'cm-1 amagat-2', 'cm2/molecule'))
    con.commit()

    con.close()

def create_grid_minR(min_wavelength, max_wavelength, minimum_R):
    """Simple function to create a wavelength grid defined with a minimum R. 
    This does not create a "constant" resolution grid. Rather, it defines the R
    at the minimum R so that the wavelength grid satisfies all_Rs>R
    """
    #final new grid 
    dwvno = 1e4/(min_wavelength**2)*(min_wavelength/minimum_R)
    new_wvno_grid = np.arange(1e4/max_wavelength, 1e4/min_wavelength, dwvno)
    return new_wvno_grid,dwvno

def create_grid(min_wavelength, max_wavelength, constant_R):
    """Simple function to create a wavelength grid defined with a constant R.

    Parameters
    ----------
    min_wavelength : float 
        Minimum wavelength in microns
    max_wavelength : float 
        Maximum wavelength in microns
    constant_R : float 
        Constant R spacing

    Returns
    -------
    wavenumber grid defined at constant Resolution
    """
    spacing = (2.*constant_R+1.)/(2.*constant_R-1.)
    
    npts = np.log(max_wavelength/min_wavelength)/np.log(spacing)
    
    wsize = int(np.ceil(npts))+1
    newwl = np.zeros(wsize)
    newwl[0] = min_wavelength
    
    for j in range(1,wsize):
        newwl[j] = newwl[j-1]*spacing
    
    return 1e4/newwl[::-1]
    
def insert_molecular_1060(molecule, min_wavelength, max_wavelength, new_R, 
            og_directory, new_db,dir_kark_ch4=None, dir_optical_o3=None):
    """
    Function to resample 1060 grid data onto lower resolution grid. The general procedure 
    in this function is to interpolate original 1060 data onto very high resolution 
    grid (R=1e6). Then, determine number of bins to take given input 'new_R'. The final 
    opacity grid will be : original_opacity[::BINS]

    NOTE: From several tests "new_R" should be at least 100x higher than the ultimate 
    planet spectrum you want to bin down to. 

    Parameters 
    ----------
    molecule : str 
        Name of molecule (should match a directory with 1060 files)
    min_wavelength : float 
        Minimum wavelength in database in units of micron 
    max_wavelength : float 
        Maximum wavelength in database in units of micron 
    new_R : float 
        New R to regrid to. If new_R=None, it will retain the original 1e6 million resolution 
        of the lbl grid. 
    """
    #open database connection 
    ngrid = 1060
    old_R = 1e6 #hard coding this initial resolution to roughly match's wvno gird
    #min_1060_grid = 0.3 #hard coding this also to match 1060 grid
    #dwvno_old = 1e4/(min_1060_grid**2)*(min_1060_grid/old_R)
    interp_wvno_grid = create_grid(min_wavelength, max_wavelength, old_R)

    cur,conn = open_local(new_db)

    #min_wno = 1e4/max_wavelength
    #max_wno = 1e4/min_wavelength

    if isinstance(new_R,type(None)):
        new_R = 1e6

    #BINS = int(dwvno_new/dwvno_old)
    BINS = int(old_R/new_R)

    #new wave grid 
    new_wvno_grid = interp_wvno_grid[::BINS]

    #insert to database 
    cur.execute('INSERT INTO header (pressure_unit, temperature_unit, wavenumber_grid, continuum_unit,molecular_unit) values (?,?,?,?,?)', 
                ('bar','kelvin', np.array(new_wvno_grid), 'cm-1 amagat-2', 'cm2/molecule'))
    conn.commit()

    #EHSAN, YOU MIGHT NOT NEED THIS SO YOU WILL HAVE TO CHANGE ELSE TO ELIF 
    if molecule in ['Cs','K','Li','Na','Rb']:
        numw = [200000]*ngrid
        delwn = [(33340 - 200)/(numw[0]-1)]*ngrid
        start = [200]*ngrid 
    elif molecule not in ['CH3D']: 
        # Get Richard's READ ME information
        f = os.path.join(og_directory,molecule,'readomni.fits')
        hdulist = fits.open(f)
        sfits = hdulist[1].data
        numw = sfits['Valid rows'] #defines number of wavelength points for each 1060 layer
        delwn = sfits['Delta Wavenum'] #defines constant delta wavenumber for each 1060 layer
        start = sfits['Start Wavenum'] #defines starting wave number for each 1060 layer

    s1060 = pd.read_csv(os.path.join(og_directory,'grid1060.csv'),dtype=str)
    #all pressures 
    pres=s1060['pressure_bar'].values.astype(float)
    #all temperatures 
    temp=s1060['temperature_K'].values.astype(float)
    #file_num
    ifile=s1060['file_number'].values.astype(int)
    for i,p,t in zip(ifile,pres,temp):  
        #path to richard's data
        fdata = os.path.join(og_directory,molecule,'p_'+str(int(i)))


        #Grab 1060 in various format data
        if molecule in ['Cs','K','Li','Na','Rb']:
            openf=FortranFile(fdata,'r')
            dset = openf.read_ints(dtype=float)
            og_wvno_grid=np.arange(numw[i-1])*delwn[i-1]+start[i-1] 
        elif molecule =='CH3D':
            df = pd.read_csv(os.path.join(og_directory,molecule,'fort.{0}.bz2'.format(int(i)))
                             ,delim_whitespace=True, skiprows=23,header=None)
            dset=df[1].values
            og_wvno_grid=df[0].values
        else: 
            dset = np.fromfile(fdata, dtype=float) 
            og_wvno_grid=np.arange(numw[i-1])*delwn[i-1]+start[i-1] 

        #interp on high res grid
        #basic interpolation here onto a new wavegrid that 
        dset = np.interp(interp_wvno_grid,og_wvno_grid, dset,right=1e-50, left=1e-50)

        #resample evenly
        y = dset[::BINS]


        if ((molecule == 'CH4') & (isinstance(dir_kark_ch4, str)) & (t<500)):
            opa_k,loc = get_kark_CH4(dir_kark_ch4,new_wvno_grid, t)
            y[loc] = opa_k
        if ((molecule == 'O3') & (isinstance(dir_optical_o3, str)) & (t<500)):
            opa_o3 = get_optical_o3(dir_optical_o3,new_wvno_grid)
            y = y + opa_o3     
        cur.execute('INSERT INTO molecular (ptid, molecule, temperature, pressure,opacity) values (?,?,?,?,?)', (int(i),molecule,float(t),float(p), y))
    conn.commit()
    conn.close()
    return new_wvno_grid
#import spectres
def insert_molecular_1460(molecule, min_wavelength, max_wavelength,og_directory, new_db,
                          new_R=None,new_dwno=None, 
                          old_R=1e6, old_dwno=0.0035,
                        alkali_dir='alkalis', dir_kark_ch4=None, dir_optical_o3=None,
                        insert_direct=False):
    """
    DEVELOPER USE ONLY. 
    Function to resample 1060 grid data onto lower resolution grid. The general procedure 
    in this function is to interpolate original 1060 data onto very high resolution 
    grid (R=1e6). Then, determine number of bins to take given input 'new_R'. The final 
    opacity grid will be : original_opacity[::BINS]

    NOTE: From several tests "new_R" should be at least 100x higher than the ultimate 
    planet spectrum you want to bin down to. 

    Parameters 
    ----------
    molecule : str 
        Name of molecule (should match a directory with 1060 files)
    min_wavelength : float 
        Minimum wavelength in database in units of micron 
    max_wavelength : float 
        Maximum wavelength in database in units of micron 
    new_R : int,float , optional
        Optional, new R to regrid to. This will create a new wavelength solution that is 
        constant in R. The other option (new_dwno is to do constant wavenumber bin) 
        This must be smaller than old_R, which is default to 1e6. 
    old_R : int,float, optional
        This is set to match the approximate cross sections that are computed LBL. If you do not 
        want any resampling, then you need to set new_R=old_R
    new_dwno : float, optional 
        Optional, new constant wavenumber bin to create wavelength solution. This must be bigger 
        than old_dwno. 
    old_dwno : float, optional 
        This is set to 0.0035 to match the smallest wavenumber than we compute in our grid. 
        If you do not want any resampling, then you need to set new_dwno=old_dwno
    og_directory : str 
        Directory of all the cross sections that include folders e.g. "H2O", "CH4"
    alkali_dir : str 
        Alakalis directory 
    new_db : str 
        New database name 
    dir_kark_ch4 : str 
        Karkoschka methane to hack in 
    dir_optical_o3 : str 
        optical ozone to hack in 
    insert_direct : bool 
        this ignores all other inputs except min and max wavelength and 
        inserts opacities into the database diretly as is 
    """
    #open database connection 
    ngrid = 1460
    cur,conn = open_local(new_db)
    
    if isinstance(new_R,(float, int)):
        interp_wvno_grid = create_grid(min_wavelength, max_wavelength, old_R)
        BINS = int(old_R/new_R)
    elif isinstance(new_dwno,(float, int)):
        interp_wvno_grid = np.arange(1e4/max_wavelength,1e4/min_wavelength,  old_dwno)          
        BINS = int(new_dwno/old_dwno)
    elif insert_direct: 
        interp_wvno_grid = None 
        BINS = 1
    else: 
        raise Exception('Need to either input a new constant R (new_R) or constant delta wno (new_dwno)')
    #new wave grid 
    if not insert_direct: 
        new_wvno_grid = interp_wvno_grid[::BINS]

    grid_file = os.path.join(og_directory,'grid1460.csv')
    if not os.path.exists(grid_file): 
        grid_file = os.path.join(os.environ['picaso_refdata'],'opacities','grid1460.csv')
    if not os.path.exists(grid_file): 
        raise Exception('cannot find grid1460 file. its possible your reference folder is out of date. please check the Github reference folder')
    s1460 = pd.read_csv(grid_file,dtype=str)
    #all pressures 
    pres=s1460['pressure_bar'].values.astype(float)
    #all temperatures 
    temp=s1460['temperature_K'].values.astype(float)
    
    #file_num
    ifile=s1460['file_number'].values.astype(int)
    
    #alkalis are created using the sep.alkali from a fortran file 
    alks = ['Na','K','Rb','Cs','Li']
    if molecule in alks: 
        if alkali_dir == 'alkalis':
            mol_dir = os.path.join(og_directory,alkali_dir)
        elif alkali_dir == 'individual_file':
            mol_dir = os.path.join(og_directory,molecule)
        else: 
            mol_dir = alkali_dir
    else:
        mol_dir = os.path.join(og_directory,molecule)
    
    #determine file type
    find_p_files = glob.glob(os.path.join(mol_dir,'*p_*'))
    find_npy_files = glob.glob(os.path.join(mol_dir,'*npy*'))
    find_txt_files =  glob.glob(os.path.join(mol_dir,'*txt*'))
    find_fort_files =  glob.glob(os.path.join(mol_dir,'fort.*'))

    if len(find_p_files)>1000:
        ftype = 'fortran_binary'
    elif len(find_npy_files)>1000: 
        ftype = 'python'
    elif len(find_txt_files)>1000:
        ftype='lupu_txt'
    elif len(find_fort_files)>1000:
        ftype='rfree_fort'
    else: 
        raise Exception('Could not find npy or p_ files. npy are assumed to be read via np.load, where as p_ files are assumed to be unformatted binary or alkali files')
    read_fits = os.path.join(mol_dir,'readomni.fits' )
    lupu_wave= os.path.join(mol_dir,'wavelengths.txt' )
    if os.path.exists(read_fits):
        # Get Richard's READ ME information
        hdulist = fits.open(read_fits)
        sfits = hdulist[1].data
        numw = sfits['Valid rows'] #defines number of wavelength points for each 1060 layer
        delwn = sfits['Delta Wavenum'] #defines constant delta wavenumber for each 1060 layer
        start = sfits['Start Wavenum'] #defines starting wave number for each 1060 layer
    elif os.path.exists(lupu_wave):
        numw,delwn,start=np.nan,np.nan,np.nan
    else: 
        #ehsan makes his opacities on uniform 
        numw = s1460['number_wave_pts'].values.astype(int)
        delwn = s1460['delta_wavenumber'].values.astype(float)
        start = s1460['start_wavenumber'].values.astype(float)
        

    
    for i,p,t in zip(ifile,pres,temp):  
        #path to data
        if 'fortran' in ftype:
            fdata = os.path.join(mol_dir, 'p_'+str(int(i)))
        elif 'rfree_fort' in ftype: 
            fdata = os.path.join(mol_dir, 'fort.'+str(int(i)))
        elif 'python' in ftype: 
            fdata = os.path.join(mol_dir, str(int(i))+'.npy')
        elif 'lupu' in ftype: 
            mbar = p*1e3
            fdata = os.path.join(mol_dir,f'{molecule}_{mbar:.2e}mbar_{t:.0f}K.txt') 
        
        #Grab 1460 in various format data
        if 'lupu' in ftype: 
            dset =  pd.read_csv(fdata,skiprows=2,header=None).values[:,0]
            og_wvno_grid = 1e4/pd.read_csv(lupu_wave).iloc[:,0].values
        elif molecule in alks:
            dset = pd.read_csv(fdata)
            og_wvno_grid = dset['wno'].values.astype(float)
            dset = dset[molecule].values.astype(float)
        elif 'fortran' in ftype: 
            dset = np.fromfile(fdata, dtype=float) 
            og_wvno_grid=np.arange(numw[i-1])*delwn[i-1]+start[i-1]
        elif 'python' in ftype: 
            dset = np.load(open(fdata,'rb'))
            og_wvno_grid=np.arange(numw[i-1])*delwn[i-1]+start[i-1]  
        elif 'rfree_fort' in ftype: 
            df = pd.read_csv(fdata,delim_whitespace=True, skiprows=27, header=None, names=['wno','cx'])
            dset=df.loc[:,'cx'].values
            og_wvno_grid=df.loc[:,'wno'].values           

        if not insert_direct:
            #interp on high res grid
            #basic interpolation here onto a new wavegrid that 
            dset = np.interp(interp_wvno_grid,og_wvno_grid, dset,right=1e-50, left=1e-50)
            dset[dset<1e-200] = 1e-200 
            #resample evenly
            y = dset[::BINS]
            #y = 10**spectres.spectres(interp_wvno_grid[::BINS],interp_wvno_grid,np.log10(dset), verbose=False,fill=np.nan)
            #x,y=regrid(interp_wvno_grid, dset, newx=interp_wvno_grid[::BINS], statistic='median')
        else: 
            new_wvno_grid = og_wvno_grid[np.where(((1e4/og_wvno_grid>min_wavelength) & (1e4/og_wvno_grid<max_wavelength)))]
            dset[dset<1e-200] = 1e-200
            y = dset[np.where(((1e4/og_wvno_grid>min_wavelength) & (1e4/og_wvno_grid<max_wavelength)))]

        if ((molecule == 'CH4') & (isinstance(dir_kark_ch4, str)) & (t<500)):
            opa_k,loc = get_kark_CH4(dir_kark_ch4,new_wvno_grid, t,y)#dset)
            y[loc] = opa_k
        if ((molecule == 'O3') & (isinstance(dir_optical_o3, str)) & (t<500)):
            opa_o3 = get_optical_o3(dir_optical_o3,new_wvno_grid)
            y = y + opa_o3     
        cur.execute('INSERT INTO molecular (ptid, molecule, temperature, pressure,opacity) values (?,?,?,?,?)', (int(i),molecule,float(t),float(p), y))
    
    #insert to database 
    conn.commit()
    cur.execute('SELECT pressure_unit from header')
    p_find = cur.fetchall()
    if len(p_find)==0:
        cur.execute('INSERT INTO header (pressure_unit, temperature_unit, wavenumber_grid, continuum_unit,molecular_unit) values (?,?,?,?,?)',
                ('bar','kelvin', np.array(new_wvno_grid), 'cm-1 amagat-2', 'cm2/molecule'))
        conn.commit()
    conn.close()

    return new_wvno_grid

def insert_molecular_1460_old(molecule, min_wavelength, max_wavelength, new_R, 
            og_directory, new_db,dir_kark_ch4=None, dir_optical_o3=None):
    """
    Function to resample Ehsan's 1460 grid data onto lower resolution grid, 1060 grid. The general procedure 
    in this function is to interpolate original 1060 data onto very high resolution 
    grid (R=1e6). Then, determine number of bins to take given input 'new_R'. The final 
    opacity grid will be : original_opacity[::BINS]

    NOTE: From several tests "new_R" should be at least 100x higher than the ultimate 
    planet spectrum you want to bin down to. 

    Parameters 
    ----------
    molecule : str 
        Name of molecule (should match a directory with 1060 files)
    min_wavelength : float 
        Minimum wavelength in database in units of micron 
    max_wavelength : float 
        Maximum wavelength in database in units of micron 
    new_R : float 
        New R to regrid to. If new_R=None, it will retain the original 1e6 million resolution 
        of the lbl grid. 
    """
    #open database connection 
    ngrid = 1060
    old_R = 1e6 #hard coding this initial resolution to roughly match's wvno gird
    #min_1060_grid = 0.3 #hard coding this also to match 1060 grid
    #dwvno_old = 1e4/(min_1060_grid**2)*(min_1060_grid/old_R)
    interp_wvno_grid = create_grid(min_wavelength, max_wavelength, old_R)

    cur,conn = open_local(new_db)

    #min_wno = 1e4/max_wavelength
    #max_wno = 1e4/min_wavelength

    if isinstance(new_R,type(None)):
        new_R = 1e6

    #BINS = int(dwvno_new/dwvno_old)
    BINS = int(old_R/new_R)

    #new wave grid 
    new_wvno_grid = interp_wvno_grid[::BINS]


    s1060 = pd.read_csv(os.path.join(og_directory,'grid1060.csv'))
    s1460 = pd.read_csv(os.path.join(og_directory,'grid1460.csv'))

    #all pressures 
    pres=s1060['pressure_bar'].values.astype(float)
    #all temperatures 
    temp=s1060['temperature_K'].values.astype(float)
    #file_num
    ifile=s1060['file_number'].values.astype(int)
    for i1060,p,t in zip(ifile, pres,temp):
        idf = s1460.loc[(s1460['pressure_bar']==p)].reset_index()

        i = int(idf.loc[(idf['temperature_K']-t).abs().argsort()[0],'file_number'])
        numw = idf.loc[(idf['temperature_K']-t).abs().argsort()[0],'number_wave_pts']
        delwn = idf.loc[(idf['temperature_K']-t).abs().argsort()[0],'delta_wavenumber']
        start = idf.loc[(idf['temperature_K']-t).abs().argsort()[0],'start_wavenumber']

        fdata = os.path.join(og_directory,molecule,'p_'+str(int(i)))
        dset = np.fromfile(fdata, dtype=float) 
        #get original grid 
        og_wvno_grid=np.arange(int(numw))*float(delwn)+float(start)   

        #interp on high res grid
        dset = np.interp(interp_wvno_grid,og_wvno_grid, dset,right=1e-50, left=1e-50)

        #resample evenly
        y = dset[::BINS]


        if ((molecule == 'CH4') & (isinstance(dir_kark_ch4, str)) & (t<500)):
            opa_k,loc = get_kark_CH4(dir_kark_ch4,new_wvno_grid, t)
            y[loc] = opa_k
        if ((molecule == 'O3') & (isinstance(dir_optical_o3, str)) & (t<500)):
            opa_o3 = get_optical_o3(dir_optical_o3,new_wvno_grid)
            y = y + opa_o3     

        cur.execute('INSERT INTO molecular (ptid, molecule, temperature, pressure,opacity) values (?,?,?,?,?)', (int(i1060),molecule,float(t),float(p), y))
    conn.commit()
    conn.close()
    return new_wvno_grid


def get_kark_CH4_noTdependence(kark_dir,new_wave, temperature):
    """
    Files from kark+2010 paper

    Returns 
    -------
    opacity in cm2/species
    """

    new_beers = pd.read_csv(os.path.join(kark_dir, 'kark_beers.csv'),delim_whitespace=True)
    two_term = pd.read_csv(os.path.join(kark_dir, 'kark_two_term.csv'),delim_whitespace=True)
    four_term = pd.read_csv(os.path.join(kark_dir, 'kark_four_term.csv'),delim_whitespace=True)
    wts = pd.read_csv(os.path.join(kark_dir, 'kark_gauss_weights.csv'),delim_whitespace=True)
    wts4 = wts.loc[wts['number']==4,[str(i) for i in range(1,5)]].values
    wts2 = wts.loc[wts['number']==2,[str(i) for i in range(1,3)]].values
    wave = []
    kappa = []
    for r in new_beers.index:
        for c in ['0','2','4','6','8']:
            iwave = float(new_beers.loc[r,'wavelength(nm)'])+float(c)
            wave += [iwave]
            try:
                kappa += [float(new_beers.loc[r,c])]
            except: 
                symbol = new_beers.loc[r,c]
                if symbol == '=':
                    coefs = four_term.loc[four_term['wavelength(nm)'] == iwave,['coef1','coef2','coef3','coef4']].values
                    sum_kap =np.sum(coefs*wts4)
                    kappa += [sum_kap] #four term 
                else: 
                    coefs = two_term.loc[two_term['wavelength(nm)'] == iwave,['coef1','coef2']].values
                    sum_kap =np.sum(coefs*wts2)
                    kappa += [sum_kap] #two term 
    #km-am to cm2/g to cm2/molecule 
    kappa = np.array(kappa)/71.80*1.6726219e-24*16 
    kappa = kappa[::-1] #need to flip because everything else is increase wavenumber
    wvno_kark = (1e4/(np.array(wave)*1e-3))[::-1]
    opacity = np.interp(new_wave,wvno_kark,kappa,left=1e-33, right=1e-33)
    return opacity

def get_kark_CH4(kark_file, new_wno , T,dset):
    kappa = pd.read_csv(kark_file,delim_whitespace=True,skiprows=2,
                           header=None, names = ['nu','nm','100','198','296','del/al'])

    kappa = kappa.loc[kappa['nm']<1000]
    z = (T-198.0)/98.9
    logKT = 10.0**(0.5*z*(z-1.0)*np.log10(kappa['100'].values) + 
             (1-z**2.0)*np.log10(kappa['198'].values) + 
             0.5*z*(z+1)*np.log10(kappa['296'].values))
    #km-am to cm2/g to cm2/molecule 
    logKT = logKT/71.80*1.6726219e-24*16 
    #only less than 1 micron!
    loc = np.where(((1e4/new_wno < 1.0) & (dset==1e-200)))
    logKT = np.interp(new_wno[loc],kappa['nu'].values,logKT)
    return logKT, loc    

def get_optical_o3(file_o3,new_wvno_grid):
    """
    This hacked ozone is from here: 
    http://satellite.mpic.de/spectral_atlas/cross_sections/Ozone/O3.spc
    """
    df1 = pd.read_csv(file_o3,delim_whitespace=True,names=['nm','cx'])
    wno_old = 1e4/(df1['nm']*1e-3).values[::-1]
    opa = df1['cx'].values[::-1]
    o3 = np.interp(new_wvno_grid, wno_old ,opa, left=1e-33, right=1e-33)
    return o3

def vectorize_rebin_median(bins,Fp):
    lf = len(Fp)
    mod = np.mod(lf,bins)
    rows = int(np.floor(lf/bins)) + 1
    add_row = (bins - mod)
    if add_row != bins: Fp = list(Fp) + [0]*add_row
    med = np.reshape(Fp,(rows,bins))
    final = np.median(med, axis=1)
    if add_row != bins: 
        final[-1] = np.median(Fp[-bins:-add_row])
    return final
def vectorize_rebin_mean(bins, Fp):
    lf = len(Fp)
    mod = np.mod(lf,bins)
    rows = int(np.floor(lf/bins)) + 1
    add_row = (bins - mod)
    if add_row != bins: Fp = list(Fp) + [0]*add_row
    med = np.reshape(Fp,(rows,bins))
    final = np.mean(med, axis=1)
    if add_row != bins: 
        final[-1] = np.mean(Fp[-bins:-add_row])
    return final

def vresample_and_insert_molecular(molecule, min_wavelength, max_wavelength, new_R, 
            og_directory, new_db):
    """This function is identical to resample_and_insert_molecular but it was 
    written to determine if taking the median was better than taking every BIN'th 
    data point. It uses a special  vectorize function to speed up the median.
    Results are about the same but this is much slower to run. 
    So dont bother using this unless there is something specific to try. 
    """
    #open database connection 
    cur,conn = open_local(new_db)


    min_wno = 1e4/max_wavelength
    max_wno = 1e4/min_wavelength
    dlambda = (max_wavelength)/new_R
    dwvno_new = 1e4/(max_wavelength**2)*(max_wavelength/new_R)

    #DEFINE UNIFORM HI-RES GRID THAT EVRYTHING WILL BE INTERPOLATED
    #ON BEFORE RESAMPLING
    #trying without this first 
    old_R = 6e6 #hard coding this initial resolution to roughly match's wvno gird
    min_1060_grid = 0.3 #hard coding this also to roughly match 1060 grid
    dlambda = (min_1060_grid)/old_R
    dwvno_old = 1e4/(min_1060_grid**2)*(min_1060_grid/old_R)
    interp_wvno_grid = np.arange(min_wno, max_wno, dwvno_old)

    #define how many bins to take
    BINS = int(dwvno_new/dwvno_old)

    #new wave grid 
    new_wvno_grid = vectorize_rebin_mean(BINS, interp_wvno_grid)
    #insert to database 
    cur.execute('INSERT INTO header (pressure_unit, temperature_unit, wavenumber_grid, continuum_unit,molecular_unit) values (?,?,?,?,?)', 
                ('bar','kelvin', np.array(new_wvno_grid), 'cm-1 amagat-2', 'cm2/molecule'))

    # Get Richard's READ ME information
    f = os.path.join(og_directory,molecule,'readomni.fits')
    hdulist = fits.open(f)
    sfits = hdulist[1].data

    numw = sfits['Valid rows']
    delwn = sfits['Delta Wavenum']
    start = sfits['Start Wavenum']

    s = pd.read_csv(os.path.join(og_directory,'PTgrid1060.txt'),delim_whitespace=True,skiprows=1,
                        header=None, names=['i','pressure','temperature'],dtype=str)
    #all pressures 
    pres=s['pressure'].values.astype(float)
    #all temperatures 
    temp=s['temperature'].values.astype(float)

    for i,p,t in zip(list(range(1,1061)),pres,temp):  
        #path to richard's data
        fdata = os.path.join(og_directory,molecule,'p_'+str(int(i)))

        #Grab 1060 format data
        dset = np.fromfile(fdata, dtype=float)   #openf.read_ints(dtype=float)
        og_wvno_grid=np.arange(numw[i-1])*delwn[i-1]+start[i-1]      

        #interp on high res grid
        dset = np.interp(interp_wvno_grid,og_wvno_grid, dset,right=1e-33, left=1e-33)

        #vectorize grab median from sample
        y = vectorize_rebin_median(BINS,dset)

        cur.execute('INSERT INTO molecular (ptid, molecule, temperature, pressure,opacity) values (?,?,?,?,?)', (i,molecule,float(t),float(p), y))
    
    conn.commit()
    conn.close()
    return new_wvno_grid


def continuum_avail(db_file):
    cur, conn = open_local(db_file)
    #what molecules inside db exist?
    cur.execute('SELECT molecule FROM continuum')
    molecules = list(np.unique(cur.fetchall()))
    cur.execute('SELECT temperature FROM continuum')
    cia_temperatures = list(np.unique(cur.fetchall()))
    conn.close()
    return molecules, cia_temperatures

def molecular_avail(db_file):
    cur, conn = open_local(db_file)
    cur.execute('SELECT ptid, pressure, temperature FROM molecular')
    data= cur.fetchall()
    pt_pairs = sorted(list(set(data)),key=lambda x: (x[0]) )

    cur.execute('SELECT molecule FROM molecular')
    molecules = np.unique(cur.fetchall())
    return list(molecules), pt_pairs
def find_nearest_1d(array,value):
    #small program to find the nearest neighbor in a matrix
    ar , iar ,ic = np.unique(array,return_index=True,return_counts=True)
    idx = (np.abs(ar-value)).argmin(axis=0)
    if ic[idx]>1: 
        idx = iar[idx] + (ic[idx]-1)
    else: 
        idx = iar[idx]
    return idx
def get_continuum(db_file, species, temperature):
    """
    Grab continuum opacity from sqlite database 

    db_file : str 
        sqlite3 database filename 
    species : list of str 
        Single species name. you can run continuum_avail to see what species are present.
    temperature : list of float
        Temperature to grab. can grab available temperatures from continuum_avail
    """
    cur, conn = open_local(db_file)

    if not isinstance(temperature,list):
        raise Exception('Make Temperature a list, even if it is single valued')
    if not isinstance(species,list):
        raise Exception('Make Species Input a list, even if it is single valued')
    cur.execute('SELECT temperature FROM continuum')
    cia_temperatures = np.unique(cur.fetchall())

    temp_nearest = [cia_temperatures[find_nearest_1d(cia_temperatures,t)] for t in temperature]

    #cur.execute("""SELECT molecule,temperature,opacity
    #            FROM continuum
    #            WHERE molecule in {}
    #            AND temperature in {}""".format(str(tuple(species)), str(tuple(temp_nearest))))

    if ((len(species) ==1 )& (len(temp_nearest) >1)):
        cur.execute("""SELECT molecule,temperature,opacity
                FROM continuum
                WHERE molecule = ?
                AND temperature in {}""".format(str(tuple(temp_nearest))),( species[0],))
    elif ((len(species) >1) & (len(temp_nearest) ==1)):
        cur.execute("""SELECT molecule,temperature,opacity
                FROM continuum
                WHERE molecule in {}
                AND temperature = ?""".format(str(tuple(species))),( temp_nearest[0],))
    elif ((len(species) ==1) & (len(temp_nearest) ==1)):
        cur.execute("""SELECT molecule,temperature,opacity
                    FROM continuum
                    WHERE molecule = ?
                    AND temperature = ?""",(species[0],temp_nearest[0]))        
    else: 
        cur.execute("""SELECT molecule,temperature,opacity
                    FROM continuum
                    WHERE molecule in {}
                    AND temperature in {}""".format(str(tuple(species)), str(tuple(temp_nearest))))

    
    data= cur.fetchall()
    restruct = {i:{} for i in species}

    for im, it, dat in data : restruct[im][it] = dat

    cur.execute('SELECT wavenumber_grid FROM header')
    wave_grid = cur.fetchone()[0]

    conn.close()
    restruct['wavenumber'] = wave_grid

    return restruct

def get_molecular(db_file, species, temperature,pressure):
    """
    Grab molecular opacity from sqlite database 

    db_file : str 
        sqlite3 database filename 
    species : list of str 
        Single species name. you can run molecular_avail to see what species are present.
    temperature : list of flaot,int
        Temperature to grab. This will grab the nearest pair neighbor so you dont 
        have to be exact.
        This should be used in conjuction with pressure. For example, if you want 
        one temperature at two pressures, then this should be input as 
        t = [100,100] (and pressure would be p=[0.1,1] for instance)
    pressure : list of flaot,int
        Pressure to grab. This will grab the nearest pair neighbor so you dont 
        have to be exact.
        This should be used in conjuction with pressure. For example, if you want 
        one temperature at two pressures, then this should be input as 
        t = [100,100] (and pressure would be p=[0.1,1] for instance)
    """
    if not isinstance(temperature,list):
        raise Exception('Make temperature a list, even if it is single valued')
    if not isinstance(pressure,list):
        raise Exception('Make pressure a list, even if it is single valued')
    if not isinstance(species,list):
        raise Exception('Make Species a list, even if it is single valued')
    if len(temperature) != len(pressure):
        raise Exception('Temperature and Pressure must be the same size because these \
            are treated as pairs. t=[500,600], p=[1.0, 0.01] will grab [500,1.0 ] and [600,0.01]')
    
    cur, conn = open_local(db_file)
    #get pt pairs
    cur.execute('SELECT ptid, pressure, temperature FROM molecular')
    data= cur.fetchall()    
    pt_pairs = sorted(list(set(data)),key=lambda x: (x[0]) )
    #here's a little code to get out the correct pair (so we dont have to worry about getting the exact number right)
    ind_pt = [min(pt_pairs, key=lambda c: math.hypot(c[1]- coordinate[0], c[2]-coordinate[1]))[0]
              for coordinate in  zip(pressure,temperature)]
    if ((len(species) ==1 )& (len(ind_pt) >1)):
        cur.execute("""SELECT molecule,ptid,pressure,temperature,opacity
                FROM molecular
                WHERE molecule = ?
                AND ptid in {}""".format(str(tuple(ind_pt))),( species[0],))
    elif ((len(species) >1) & (len(ind_pt) ==1)):
        cur.execute("""SELECT molecule,ptid,pressure,temperature,opacity
                FROM molecular
                WHERE molecule in {}
                AND ptid = ?""".format(str(tuple(species))),( ind_pt[0],))
    elif ((len(species) ==1) & (len(ind_pt) ==1)):
        cur.execute("""SELECT molecule,ptid,pressure,temperature,opacity
                    FROM molecular
                    WHERE molecule = ?
                    AND ptid = ?""",(species[0],ind_pt[0]))        
    else: 
        cur.execute("""SELECT molecule,ptid,pressure,temperature,opacity
                    FROM molecular
                    WHERE molecule in {}
                    AND ptid in {}""".format(str(tuple(species)), str(tuple(ind_pt))))


    data= cur.fetchall()

    temp_nearest = [pt_pairs[i-1][2] for i in ind_pt]
    pres_nearest = [pt_pairs[i-1][1] for i in ind_pt]
    restruct = {i:{} for i in species}
    for i in restruct.keys():
        for t in temp_nearest:
            restruct[i][t] = {}
    for im, iid,ip,it, dat in data : restruct[im][it][ip] = dat

    cur.execute('SELECT wavenumber_grid FROM header')
    wave_grid = cur.fetchone()[0]

    conn.close()
    restruct['wavenumber'] = wave_grid
    return restruct

def delete_molecule(mol, db_filename):
    """
    Delete a record from the database

    Parameters
    ----------
    mol : str 
        molecule to delete 
    db_filename : str
        database to delete record from
    """
    try:
        sqliteConnection = sqlite3.connect(db_filename)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sql_update_query = """DELETE from molecular where molecule = ?"""
        cursor.execute(sql_update_query, (mol,))
        sqliteConnection.commit()
        print("Record deleted successfully")

        cursor.close()

    except sqlite3.Error as error:
        print("Failed to delete reocord from a sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("sqlite connection is closed")

# Correlated-K functions
# get guass points for 2 Gauss method
def g_w_2gauss(order=4,gfrac=0.95):
    """
    Gets gauss and weight poitns for correlated-k function 
    Specifically for the 2-point gauss method 

    Parameters
    ----------
    order : int
        Gauss order 
    gfrac : float
        Adjustable parameter which sets the division in the values of G which will
        be sampled by the first and second set of Gauss points.
    Returns 
    -------
    gauss, weights
    """
    g,w=np.polynomial.legendre.leggauss(order)

    #see equation 25 in https://arxiv.org/pdf/2208.07836.pdf 
    wnew1 = gfrac*w*0.5
    gnew1= gfrac*0.5*(g + 1.)

    wnew2 =(1. - gfrac)*w*0.5
    gnew2=gfrac + (1. - gfrac)*0.5*(g + 1.)
    
    wfin = np.concatenate((wnew1,wnew2))
    gfin = np.concatenate((gnew1,gnew2))
    
    return gfin,wfin

def get_wvno_grid(filename, min_wavelength=None, max_wavelength=None, R=None):
    if not isinstance(filename, type(None)):
        wvno_new,dwni_new = np.loadtxt(filename,usecols=[0,1],unpack=True)
        wvno_low = 0.5*(2*wvno_new - dwni_new)
        wvno_high = 0.5*(2*wvno_new + dwni_new)
    else: 
        wvno_new = create_grid(min_wavelength, max_wavelength, R)
        dwni_new = list(np.diff(wvno_new))
        dwni_new = np.array([dwni_new[0]] + dwni_new)
        wvno_low = 0.5*(2*wvno_new - dwni_new)
        wvno_high = 0.5*(2*wvno_new + dwni_new)
    return wvno_low,wvno_high

def compute_ck_molecular(molecule,og_directory,wv_file_name=None,
    order=4,gfrac=0.95,dir_kark_ch4=None,alkali_dir=None,
    min_wavelength=None, max_wavelength=None, R=None, 
    verbose=True):
    """
    Function to generate correlated-K tables for each individual gas
    
    Parameters
    ----------
    molecule : str 
        Name of molecule 
    og_directory : str 
        Directory of all the cross sections that include folders e.g. "H2O", "CH4"
    alkali_dir : str 
        Alakalis directory
    wv_file_name : str 
        (optional) file name with wavelength. First column wavenumber, second column delta wavenumber 
        Must supply this OR combo of min, max wavelength and R
    order : int     
        (Optional) Gauss Legendre order which by default is set to 4, with the double gauss method 
    gfrac : int     
        (Optional) Double-gauss method of splitting up the gauss points, by default we use 0.95 
    dir_kark_ch4 : str 
        (Optional) directory of additional karkochka methane 
    alkali_dir : str 
        (Optional) Directory of alkalis or "individual_file" which is the method to use for Roxana Lupus data. 
    verbose: bool 
        (Optional) prints out status of which p,t, point the code is at 

    """
    grid_file = os.path.join(og_directory,'grid1460.csv')
    npres = 20 
    ntemp = 73 
    ngauss = order*2 

    s1460 = pd.read_csv(grid_file,dtype=str)
    #all pressures
    pres=s1460['pressure_bar'].values.astype(float)
    #all temperatures
    temp=s1460['temperature_K'].values.astype(float)

    #file_num
    ifile=s1460['file_number'].values.astype(int)

    #alkalis are created using the sep.alkali from a fortran file 
    alks = ['Na','K','Rb','Cs','Li']
    if molecule in alks: 
        if alkali_dir == 'alkalis':
            mol_dir = os.path.join(og_directory,alkali_dir)
        elif alkali_dir == 'individual_file':
            mol_dir = os.path.join(og_directory,molecule)
        else: 
            mol_dir = alkali_dir
    else:
        mol_dir = os.path.join(og_directory,molecule)


    #determine file type    
    find_p_files = glob.glob(os.path.join(mol_dir,'*p_*'))
    find_npy_files = glob.glob(os.path.join(mol_dir,'*npy*'))
    find_txt_files =  glob.glob(os.path.join(mol_dir,'*txt*'))

    if len(find_p_files)>1000:
        ftype = 'fortran_binary'
    elif len(find_npy_files)>1000:
        ftype = 'python'
    elif len(find_txt_files)>1000:
        ftype='lupu_txt'
    else:
        raise Exception('Could not find npy or p_ files. npy are assumed to be read via np.load, where as p_ files are assumed to be unformatted binary or alkali files')


    read_fits = os.path.join(mol_dir,'readomni.fits' )
    lupu_wave= os.path.join(mol_dir,'wavelengths.txt' )
    if os.path.exists(read_fits):
        # Get Richard's READ ME information
        hdulist = fits.open(read_fits)
        sfits = hdulist[1].data
        numw = sfits['Valid rows'] #defines number of wavelength points for each 1060 layer
        delwn = sfits['Delta Wavenum'] #defines constant delta wavenumber for each 1060 layer
        start = sfits['Start Wavenum'] #defines starting wave number for each 1060 layer
    elif os.path.exists(lupu_wave):
        og_wvno_grid = 1e4/pd.read_csv(lupu_wave).values[:,0]
        numw,delwn,start=np.nan,np.nan,np.nan
    else: 
        #ehsan makes his opacities on uniform 
        numw = s1460['number_wave_pts'].values.astype(int)
        delwn = s1460['delta_wavenumber'].values.astype(float)
        start = s1460['start_wavenumber'].values.astype(float)
    
    gi,wi = g_w_2gauss(order,gfrac)
    
    if not isinstance(wv_file_name,type(None)):
        wvno_low,wvno_high = get_wvno_grid(wv_file_name)
    else: 
        wvno_low,wvno_high = get_wvno_grid(None, min_wavelength, max_wavelength, R)
    k_coeff_arr = np.zeros(shape=(npres,ntemp,len(wvno_low),ngauss))
    ctp,ctt = 0,0
    for i,p,t in zip(ifile,pres,temp):  
        #path to data
        if 'fortran' in ftype:
            fdata = os.path.join(mol_dir, 'p_'+str(int(i)))
        elif 'python' in ftype: 
            fdata = os.path.join(mol_dir, str(int(i))+'.npy')
        elif 'lupu' in ftype: 
            mbar = pres*1e3
            fdata = os.path.join(mol_dir,f'{molecule}_{mbar:.2e}mbar_{temp:.0f}K.txt') 
        
        #Grab 1460 in various format data
        if 'lupu' in ftype: 
            dset =  pd.read_csv(fdata,skiprows=2).values[:,0]
        elif molecule in alks:
            dset = pd.read_csv(fdata)
            og_wvno_grid = dset['wno'].values.astype(float)
            dset = dset[molecule].values.astype(float)
        elif 'fortran' in ftype: 
            dset = np.fromfile(fdata, dtype=float) 
            og_wvno_grid=np.arange(numw[i-1])*delwn[i-1]+start[i-1]
        elif 'python' in ftype: 
            dset = np.load(open(fdata,'rb'))
            og_wvno_grid=np.arange(numw[i-1])*delwn[i-1]+start[i-1]      

        
    
            
        for iwvbin in range(len(wvno_low)):
            wvlow_temp = wvno_low[iwvbin]
            wvhigh_temp = wvno_high[iwvbin]
            
            wh = np.where(np.logical_and((og_wvno_grid > wvlow_temp),(og_wvno_grid <=  wvhigh_temp)))
            
            
            
            linelist = dset[wh]
            linelist += 1e-200
            wh2 = np.where(linelist < 0.0)
            if len(linelist[wh2]) > 0:
                
                for icorr in range(len(linelist)):
                    if linelist[icorr] < 0:
                        linelist[icorr] = 1e-200
                
            if len(linelist) == 0:
                data = np.sort(np.zeros(10)-250.0)
                
            else:
                data = np.sort(np.log(linelist))
                        
            wvno_now = 0.5*(wvlow_temp+wvhigh_temp)

            if ((molecule == 'CH4') & (isinstance(dir_kark_ch4, str)) & (t<500)) & (1e4/wvno_now < 1.0): # short of 1 microns
                opa_k,loc = get_kark_CH4(dir_kark_ch4,wvno_now, t)
                
                data = np.sort(np.log(opa_k*(1+np.zeros(100))))
            if len(data) > 1:
                x = np.arange(len(data))/(len(data)-1.)

                k_coeff=np.interp(gi,x, data)
        
                k_coeff_arr[ctp,ctt,iwvbin,:] = k_coeff[:]
            else :
                k_coeff_arr[ctp,ctt,iwvbin,:] += -250.00

        ctp+=1
        
        if ctp == 20:
            ctp =0
            ctt +=1
        
        

        if verbose: print(i,p,t)
    return k_coeff_arr

#keeping old name here for Sagnick
func_read_gas = compute_ck_molecular
          
def find_nearest(array,value):
    #small program to find the nearest neighbor in temperature  
    idx = (np.abs(array-value)).argmin()
    return idx

def listdir(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def regrid(x, y, newx=None, R=None,statistic='mean'):
    """
    Rebin the spectrum at a minimum R or on a fixed grid 

    Parameters
    ----------
    x : array 
        Wavenumbers
    y : array 
        Anything (e.g. albedo, flux)
    newx : array 
        new array to regrid on. 
    R : float 
        create grid with constant R

    Returns
    -------
    final x, and final y
    """
    if (isinstance(newx, type(None)) & (not isinstance(R, type(None)))) :
        newx = create_grid(1e4/max(x), 1e4/min(x), R)
    elif (not isinstance(newx, type(None)) & (isinstance(R, type(None)))) :
        d = np.diff(newx)
        binedges = np.array([newx[0]-d[0]/2] + list(newx[0:-1]+d/2.0) + [newx[-1]+d[-1]/2])
        newx = binedges
    else:
        raise Exception('Please either enter a newx or a R')
    y, edges, binnum = binned_statistic(x,y,bins=newx, statistic=statistic)
    newx = (edges[0:-1]+edges[1:])/2.0

    return newx, y


