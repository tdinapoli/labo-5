# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:02:25 2021

@author: Ale
"""

from importar_datos import importar, calc_diferencias, calc_modulos, calc_msd, \
                             msd_retrasos, calc_msd_vec, msd_retrasos_vec, chisq, \
                                 rval, pvalor    
import matplotlib.pyplot as plt
import numpy as np

#%%

data_num = np.arange(0, 40)
x = []
y = []
t = []
for i in data_num:
    tp, xp, yp = np.loadtxt('Data/Brow/brown' + str(i) + '.txt', skiprows = 2).T
    x.append(xp)
    y.append(yp)
    t.append(yp)
    
    
X = []
Y = []
T = []

for i in range(len(x)):
    if len(x[i]) > 700:
        X.append(x[i][:706])
        Y.append(y[i][:706])
        T.append(t[i][:706])

#%% Autocorrelacion diff

plt.figure()
autocorr = []
for i in range(len(X)):
    autocorr.append(np.correlate(np.diff(X[i]), np.diff(X[i]), mode = 'full')/np.correlate(np.diff(X[i]),np.diff(X[i])))    
    plt.plot(np.correlate(np.diff(X[i]), np.diff(X[i]), mode = 'full')/np.correlate(np.diff(X[i]),np.diff(X[i])))
plt.plot(np.mean(autocorr, axis = 0), color = 'k', linewidth = 3)

#%% Autocorrelacion Data Sin sentido
plt.figure()
autocorr = []
for i in range(len(X)):
    autocorr.append(np.correlate(X[i], X[i], mode = 'full')/np.correlate(X[i],X[i]))    
    plt.plot(np.correlate(X[i], X[i], mode = 'full')/np.correlate(X[i],X[i]))
plt.plot(np.mean(autocorr, axis = 0), color = 'k', linewidth = 3)