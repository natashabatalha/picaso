from .disco import compress_disco
import os
import pickle as pk
import numpy as np
import pandas as pd
import json


__refdata__ = os.environ.get('picaso_refdata')

def disort_albedos(output_dir=None, rayleigh=False, phase=True):
    print('DID YOU RUN DISORT CODE TO GET UP-TO-DATE DATA?')
    #real_answer = pd.read_csv(os.path.join(__refdata__,'base_cases', 'DLUGACH_TEST.csv'))
    real_answer = pd.read_csv('/Users/crooney/Documents/codes/picaso/docs/notebooks/SH4.csv')
    real_answer = real_answer.set_index('Unnamed: 0')
    perror = real_answer.copy()
    directory = '/Users/crooney/Documents/codes/picaso/picaso/cdisort_comparison/'
    disort_dir = directory + 'cdisort_data/'
    picaso_dir = directory + 'picaso_data/'

    # Rayleigh
    if rayleigh:
        for w in real_answer.keys():
            if float(w) ==1.000:
                w0 = 0.999999
            else: 
                w0 = float(w)

            disort_filename = disort_dir + 'data_rayleigh_%.3f.pk' % w0
            picaso_filename = picaso_dir + 'data_rayleigh_%.3f.pk' % w0
            disort = pk.load(open(disort_filename,'rb'), encoding = 'bytes')
            picaso = pk.load(open(picaso_filename,'rb'), encoding = 'bytes')
        
            #for reflected light use compress_disco routine
            albedo = compress_disco(disort[b'nwno'], disort[b'cos_theta'], disort[b'xint_at_top'],
            #albedo = compress_disco(1, disort[b'cos_theta'], disort[b'xint_at_top'],
                    disort[b'gweight'], disort[b'tweight'], disort[b'F0PI'])
                                
            perror.loc[-1][w] = albedo[-1]

    if phase:
        for g0 in real_answer.index[2:]:
            for w in real_answer.keys():
                if float(w) ==1.000:
                    w0 = 0.999999
                else: 
                    w0 = float(w)

                disort_filename = disort_dir + 'data_%.3f_%.3f.pk' % (g0, w0)
                picaso_filename = picaso_dir + 'data_%.3f_%.3f.pk' % (g0, w0)
                disort = pk.load(open(disort_filename,'rb'), encoding = 'bytes')
                picaso = pk.load(open(picaso_filename,'rb'), encoding = 'bytes')
        
                #for reflected light use compress_disco routine
                albedo = compress_disco(disort[b'nwno'], disort[b'cos_theta'], disort[b'xint_at_top'],
                #albedo = compress_disco(1, disort[b'cos_theta'], disort[b'xint_at_top'],
                        disort[b'gweight'], disort[b'tweight'], disort[b'F0PI'])
                                    
                perror.loc[g0][w] = albedo[-1]
    
    if output_dir!=None: perror.to_csv(os.path.join(output_dir))
    return perror
