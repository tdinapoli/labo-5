#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:45:57 2021

@author: dina
"""
import numpy as np
import matplotlib.pyplot as plt
from importar_datos import calc_msd, msd_retrasos

reps = 218
X = []
Y = []
for _ in range(reps):
    n_pasos = 150
    x = np.zeros(150)
    y = np.zeros(150)
    amp = 2e-8
    
    for i in range(n_pasos):
        signo = np.random.random()
        if signo < 0.6:
            x[i] += x[i-1] + np.random.random()*amp
            y[i] += y[i-1] + np.random.random()*amp
        else:
            x[i] += x[i-1] - np.random.random()*amp
            y[i] += y[i-1] - np.random.random()*amp
    
    X.append(x)
    Y.append(y)

plt.figure(figsize=(8,5))
t = np.arange(0,150,1)/10
for i in range(len(X)):
    x = X[i]
    y = Y[i]
    plt.plot(t, x)
    plt.plot(t,y)
plt.xlim([0,15])
plt.show()


msd = msd_retrasos(X)
plt.plot(np.arange(150), msd)
plt.show()
