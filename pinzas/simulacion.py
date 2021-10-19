#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:45:57 2021

@author: dina
"""
import numpy as np
import matplotlib.pyplot as plt

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
        if signo < 0.5:
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


MSDS = []
for trayectoria in range(len(X)):
    x = X[trayectoria]
    y = Y[trayectoria]
    r = np.sqrt(x**2 + y**2)
    N = len(x)
    msd = np.zeros(N)
    
    for retraso in range(N):
        for i in range(N - retraso - 1):
            msd[retraso] += (r[i+retraso] - r[i])**2
        msd[retraso] = msd[retraso] / (N - (retraso + 1))
    
    MSDS.append(msd[:-1])
    plt.plot(np.arange(N), msd)
MSDS = np.array(MSDS)
plt.plot(np.arange(N-1),np.nanmean(MSDS, axis=0), color="k", linewidth=2)
plt.ylim(0, 0.4e-13)
plt.show()
