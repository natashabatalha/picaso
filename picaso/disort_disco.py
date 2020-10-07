from .disco import compress_disco
import os
import pickle as pk
import numpy as np
import pandas as pd
import json


__refdata__ = os.environ.get('picaso_refdata')

def disort_albedos(output_dir=None, rayleigh=True, phase=True):
    print('DID YOU RUN DISORT CODE TO GET UP-TO-DATE DATA?')
    real_answer = pd.read_csv(os.path.join(__refdata__,'base_cases', 'DLUGACH_TEST.csv'))
    real_answer = real_answer.set_index('Unnamed: 0')
    perror = real_answer.copy()
    input_dir = '/Users/crooney/Documents/codes/picaso/docs/notebooks/disort_data/'

    # Rayleigh
    if rayleigh:
        for w in real_answer.keys():
            if float(w) ==1.000:
                w0 = 0.999999
            else: 
                w0 = float(w)

            input_filename = input_dir + 'data_rayleigh_%.3f.pk' % w0
            inputs = pk.load(open(input_filename,'rb'), encoding = 'bytes')
        
            #for reflected light use compress_disco routine
            #albedo = compress_disco(inputs[b'nwno'], inputs[b'cos_theta'], inputs[b'xint_at_top'],
            albedo = compress_disco(1, inputs[b'cos_theta'], inputs[b'xint_at_top'],
                    inputs[b'gweight'], inputs[b'tweight'], inputs[b'F0PI'])
                                
            perror.loc[-1][w] = albedo[-1]

    if phase:
        for g0 in real_answer.index[1:]:
            for w in real_answer.keys():
                if float(w) ==1.000:
                    w0 = 0.999999
                else: 
                    w0 = float(w)

            input_filename = input_dir + 'data_%.3f_%.3f.pk' % (g0, w0)
            inputs = pk.load(open(input_filename,'rb'), encoding = 'bytes')
        
            #for reflected light use compress_disco routine
            #albedo = compress_disco(inputs[b'nwno'], inputs[b'cos_theta'], inputs[b'xint_at_top'],
            albedo = compress_disco(1, inputs[b'cos_theta'], inputs[b'xint_at_top'],
                    inputs[b'gweight'], inputs[b'tweight'], inputs[b'F0PI'])
                                
            perror.loc[-1][w] = albedo[-1]
    
    if output_dir!=None: perror.to_csv(os.path.join(output_dir))
    return perror
