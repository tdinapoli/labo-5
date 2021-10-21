#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 14:44:59 2021

@author: dina
"""

from importar_datos import importar, calc_diferencias, calc_modulos, calc_msd, msd_retrasos, calc_msd_vec, msd_retrasos_vec
import matplotlib.pyplot as plt
import numpy as np
#%%

#importo los datos

T, X, Y , TA, XA, YA= importar()

Xdif, Ydif = calc_diferencias(X,Y)
XAdif, YAdif = calc_diferencias(XA, YA)

#filtro los valores absurdos
Xdif = Xdif[np.abs(Xdif)<0.5e-7]
Ydif = Ydif[np.abs(Ydif)<0.5e-7]

modulos = calc_modulos(Xdif, Ydif)
modulosA = calc_modulos(XAdif, YAdif)

#%%

#Desplazamiento vs tiempo tanto para x como para y para partículas no atrapadas

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13,5))

for i, t in enumerate(T):
    x = X[i]
    y = Y[i]
    ax1.plot(t, x)
    ax2.plot(t, y)

fig.tight_layout()
ax1.set_title("Desplazamiento en x")
ax2.set_title("Desplazamiento en y")
ax1.set_ylabel("Desplazamiento (m)")
ax1.set_xlabel("Tiempo (s)")
ax2.set_xlabel("Tiempo (s)")
ax2.set_xlim([0,15])
ax1.set_xlim([0,15])
ax1.grid(alpha=0.5)
ax2.grid(alpha=0.5)


plt.show()
    

#%%

#25 trayectorias de partículas

fix, axs = plt.subplots(5,5, figsize=(10,10), sharex=True, sharey=True)

cuenta = 0
filtro = len(X) == 150
X_tray = X[filtro]
Y_tray = Y[filtro]
for ii in range(5):
    for jj in range(5):
        ax = axs[ii,jj]
        ax.set_xticks([])
        ax.set_yticks([])
        x = X_tray[cuenta]
        y = Y_tray[cuenta]
        ax.plot(x[0], y[0], 'ok')
        ax.plot(x, y)
        cuenta += 1
plt.show()

#%%

#Cuatro histogramas de la magnitud de los pasos en X y en Y para atrapadas y no atrapadas

fig, axs = plt.subplots(2,2, figsize=(10,10))
ax1, ax2 = axs[0]
ax3, ax4 = axs[1]


ax1.hist(Xdif, bins=100)
ax1.set_xlim([-0.2e-7,0.2e-7])

ax2.hist(Ydif, bins=100)
ax2.set_xlim([-0.2e-7,0.2e-7])

ax3.hist(YAdif, bins=100)
ax3.set_xlim([-0.1e-7,0.1e-7])

ax4.hist(XAdif, bins=100)
ax4.set_xlim([-0.1e-7,0.1e-7])
plt.show()

#%%

#Histograma del módulo del desplazamiento tanto para atrapadas como para no atrapadas

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12,8))

fig.tight_layout()
ax1.hist(modulos, bins=260)
ax2.hist(modulosA, bins=120)
ax1.set_xlim([0, 0.2e-7])
ax2.set_xlim([0, 0.2e-7])
ax2.set_xlabel("Módulo de dr")
plt.show()

#%%

#Histograma en función del paso tanto para x como para y, y para atrapadas y no atrapadas
#(Ver tésis página 10)


pasos = [10, 15, 30, 60, 100, 150]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
fig.tight_layout()

bins_tot = []
for paso in pasos:
    x_paso = []
    y_paso = []

    for i, t in enumerate(T):
        x = X[i]
        y = Y[i]
        
        try:
            x_paso.append(x[paso])
            y_paso.append(y[paso])
        except:
            pass
    nx, binsx, patchx = ax1.hist(x_paso, histtype="step", bins=50, density=False,
                                 label=str(paso), linewidth=2)
    ny, binsy, patchy = ax2.hist(y_paso, histtype="step", bins=50, density=False,
                                 label=str(paso), linewidth=2)

ax1.set_xlim([-2e-7, 2e-7])
ax2.set_xlim([-2e-7, 2e-7])
plt.legend()
plt.show()
    
#%%

#MSD
X_msd = np.array(X)
X_msd = X[len(X[:,])] 
plt.show()

#%% Autocorrelacion
plt.figure()
autocorr = []
for i in range(len(X)):
    if len(X[i])== 150:
        autocorr.append(np.correlate(X[i], X[i], mode = 'same')/np.correlate(X[i],X[i]))    
        #plt.plot(np.correlate(X[i], X[i], mode = 'same')/np.correlate(X[i],X[i]))
plt.plot(np.mean(autocorr, axis = 0), color = 'k', linewidth = 3)

#%% Correlación Cruzada

corr_cruzada = []
for i in range(len(X)):
    for j in range(len(X)):
        if len(X[i]) == 150 and len(X[j])== 150:
            corr_cruzada.append(np.correlate(X[i], X[j], mode = 'same')/np.correlate(X[i],X[j]))
            #plt.plot(np.correlate(X[i], X[j], mode = 'same')/np.correlate(X[i],X[j]))
    print(i)
plt.plot(np.mean(corr_cruzada, axis = 0), color = 'k', linewidth = 3)

#%%

X_150 = []
Y_150 = []
for i in range(len(X)):
    if len(X[i]) == 150:
        X_150.append(X[i])
        Y_150.append(Y[i])

msd = msd_retrasos_vec(X_150, Y_150)
