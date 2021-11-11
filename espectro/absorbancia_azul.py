#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:17:19 2021

@author: dina
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

data_ref = np.loadtxt(open('data/dia 3/lampara_halogena_sin_nada_dia_3.csv').readlines()[:-1],
                               skiprows = 33, delimiter = ';')

long_ref = data_ref[:-50, 0]
int_ref = data_ref[:-50, 1]

lon_az_5 = np.loadtxt(open('data/dia 3/5seg_azul.csv').readlines()[:-1],
                                skiprows = 33, delimiter = ';')[:-50, 0]
int_az_5 = np.loadtxt(open('data/dia 3/5seg_azul.csv').readlines()[:-1],
                                skiprows = 33, delimiter = ';')[:-50, 1]

ref_5 = np.loadtxt(open('data/dia 3/05_ll.csv').readlines()[:-1],
                                skiprows = 33, delimiter = ';')[:-50, 1]
#%% moving cosdas

def moving_average(x, n):
    return np.convolve(x, np.ones(n), mode="same")/n


def moving_std(x, n):
    MA = moving_average(x, n)
    MSTD = []
    for i, mean in enumerate(MA):
        if i - n//2 < 0:
            MSTD.append(np.std(x[0:i + n//2]))
        elif i + n//2 > len(x):
            MSTD.append(np.std(x[i - n//2: -1]))
        else:
            MSTD.append(np.std(x[i-n//2:i + n//2]))
    return MSTD

#%% 

fig, ax = plt.subplots(figsize=(10,10))

absorbancia_az_5 = moving_average(-np.log10(int_az_5/int_ref), 30) 
abs_ref_5 = moving_average(-np.log10(ref_5/int_ref), 30)

ax.plot(lon_az_5, absorbancia_az_5)
ax.plot(lon_az_5, abs_ref_5)
plt.show()
