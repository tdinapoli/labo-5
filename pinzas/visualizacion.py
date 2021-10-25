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

T_, X_, Y_ , TA, XA, YA= importar()

X = []
Y = []
T = []
for i in range(len(X_)):
    if len(X_[i]) == 150:
        X.append(X_[i])
        Y.append(Y_[i])
        T.append(T_[i])

X = np.array(X)
Y = np.array(Y)
T = np.array(T)


filtro_x = np.std(X, axis = 1) >= 1e-8
filtro_y = np.std(Y, axis = 1) >= 1e-8

X = X[filtro_x & filtro_y]
Y = Y[filtro_x & filtro_y]
T = T[filtro_x & filtro_y]

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
    ax1.plot(t, x, alpha =0.2) 
    ax2.plot(t, y, alpha = 0.2)

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

#%% Grafico de las desviaciones standar, hay que importar los datos sin el std

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13,5))

long = np.arange(len(X))
x = X[i]
y = Y[i]
ax1.plot(long, np.std(np.array(X), axis  = 1),'o')
ax2.plot(long, np.std(np.array(Y), axis = 1),'o')

fig.tight_layout()
ax1.set_title("Desplazamiento en x")
ax2.set_title("Desplazamiento en y")
ax1.set_ylabel("Desplazamiento (m)")
ax1.set_xlabel("Tiempo (s)")
ax2.set_xlabel("Tiempo (s)")

ax1.grid(alpha=0.5)
ax2.grid(alpha=0.5)

#plt.savefig('tray.pdf')
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
    

#%% Autocorrelacion
plt.figure()
autocorr = []
for i in range(len(X)):
    autocorr.append(np.correlate(np.diff(X[i]), np.diff(X[i]), mode = 'same')/np.correlate(np.diff(X[i]),np.diff(X[i])))    
    plt.plot(np.correlate(np.diff(X[i]), np.diff(X[i]), mode = 'same')/np.correlate(np.diff(X[i]),np.diff(X[i])))
plt.plot(np.mean(autocorr, axis = 0), color = 'k', linewidth = 3)

#%% Correlación Cruzada

corr_cruzada = []
for i in range(len(X)):
    for j in range(len(X)):
        corr_cruzada.append(np.correlate(np.diff(X[i]), np.diff(X[j]), mode = 'same')/np.correlate(np.diff(X[i]),np.diff(X[j])))
        plt.plot(np.correlate(X[i], X[j], mode = 'same')/np.correlate(X[i],X[j]))
    print(i)
plt.plot(np.mean(corr_cruzada, axis = 0), color = 'k', linewidth = 3)

#%% Correlación cruzada parte 2

plt.figure()
corr_cruzada = []
for i in range(len(X)):
    for j in range(len(X)):
        corr_cruzada.append(np.correlate(np.diff(X[i]) -np.mean(np.diff(X[i]))  , np.diff(Y[j]) -np.mean(np.diff(Y[j])), mode = 'same'))#/np.correlate(np.diff(X_150[i]),np.diff(X_150[j])))
        plt.plot(np.correlate(np.diff(X[i]) -np.mean(np.diff(X[i]))  , np.diff(Y[j]) -np.mean(np.diff(Y[j])), mode = 'same'))
    print(i)
plt.plot(np.mean(corr_cruzada, axis = 0), color = 'k', linewidth = 3)


#%% MSD
from scipy.optimize import curve_fit
def lineal(x,a,b):
    return x*a+b
def cuad(x,a,b,c):
    return a*x**2+b*x+c
def exp(x,a):
    a*np.exp(x)

retraso = np.arange(38)
msd_tot = []
plt.figure()
for j in range(len(X)):
    msd_tray = []
    for i in retraso:
        msd_tray.append(calc_msd_vec(i, X[j], Y[j]))
    plt.plot(msd_tray, alpha = 0.6)
    msd_tot.append(msd_tray)

msd_prom = np.mean(msd_tot, axis = 0)

popt, pcov = curve_fit(lineal, retraso[:-5], msd_prom[:-5])
popt2, pcov2 = curve_fit(cuad, retraso[:-5], msd_prom[-5], p0 = [1e-20, 1e-25, 0])
#popt3, pcov3 = curve_fit(exp, rastro[:-5], msd_prom[-5], p0 = [1e-80])

plt.plot(msd_prom[:-1], 'k-', linewidth = 3)
#plt.plot(rastro, lineal(rastro, *popt), 'r-', linewidth = 3)
#plt.plot(rastro, cuad(rastro, *popt2), 'b-', linewidth = 3)
#plt.plot(rastro, exp(rastro, 1e-80))
#plt.plot(rastro, exp(rastro, *popt3), 'g-', linewidth = 3)
#plt.xlim([0,140])
plt.show()
#msd = msd_retrasos_vec(X_150, Y_150)

#%% Valor medio y varianza


def val_med(N, x):
    valores_medios = []
    for i in range(len(x)-1):
        valores_medios.append(x[i][N])
    val_medi = np.mean(valores_medios)
    std = np.std(valores_medios)
    return val_medi, std

rangos = np.arange(150)
plt.figure()
for i in rangos:
    plt.plot(i, val_med(i,X)[0],'ok')
plt.show()

plt.figure()
for i in rangos:
    plt.plot(i, val_med(i,X)[1],'ok')
    plt.plot(i, val_med(i,Y)[1],'ro')
plt.show()
#%% Grafico de las trayectorias y sus valores medios


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13,5))

for i, t in enumerate(T):
    x = X[i]
    y = Y[i]
    ax1.plot(t, x, alpha =0.2) 
    ax2.plot(t, y, alpha = 0.2)

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


#fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13,5))

for i in rangos:
    ax1.errorbar(T[2][i], val_med(i, X)[0], yerr = val_med(i, X)[1],fmt = 'ok',alpha = 1)

    ax2.errorbar(T[2][i], val_med(i, Y)[0], yerr = val_med(i, Y)[1],fmt = 'ok', alpha = 1)

plt.show()
