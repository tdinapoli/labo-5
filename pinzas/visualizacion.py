#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 14:44:59 2021

@author: dina
"""

from importar_datos import importar, calc_diferencias, calc_modulos, calc_msd, \
                             msd_retrasos, calc_msd_vec, msd_retrasos_vec, chisq, \
                                 rval, pvalor    
from simulacion import simular2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem
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


X_sin_filtro = np.copy(X)
Y_sin_filtro = np.copy(Y)
T_sin_filtro = np.copy(T)


filtro_x = np.std(X, axis = 1) >= 1e-8
filtro_y = np.std(Y, axis = 1) >= 1e-8

X = X[filtro_x & filtro_y]
Y = Y[filtro_x & filtro_y]
T = T[filtro_x & filtro_y]

Xdif, Ydif = calc_diferencias(X,Y)
XAdif, YAdif = calc_diferencias(XA, YA)

Xsim, Ysim = simular2()

Xsim_dif, Ysim_dif = calc_diferencias(Xsim, Ysim)

#filtro los valores absurdos
Xdif = Xdif[np.abs(Xdif)<0.5e-7]
Ydif = Ydif[np.abs(Ydif)<0.5e-7]

modulos = calc_modulos(Xdif, Ydif)
modulosA = calc_modulos(XAdif, YAdif)



retraso = np.arange(38)

msd_tot_x = []
msd_tot_y = []
msd_tot_x_sim = []
msd_tot_y_sim = []
for j in range(len(X)):
    msd_tray_x = []
    msd_tray_y = []
    msd_tray_x_sim = []
    msd_tray_y_sim = []
    for i in retraso:
        msd_tray_x.append(calc_msd(i, X[j]))
        msd_tray_y.append(calc_msd(i, Y[j]))
        msd_tray_x_sim.append(calc_msd(i, Xsim[j]))
        msd_tray_y_sim.append(calc_msd(i, Ysim[j]))
    msd_tot_x.append(msd_tray_x)
    msd_tot_y.append(msd_tray_y)
    msd_tot_x_sim.append(msd_tray_x_sim)
    msd_tot_y_sim.append(msd_tray_y_sim)
    
msd_x_prom = np.mean(msd_tot_x, axis=0)
msd_y_prom = np.mean(msd_tot_y, axis=0)
msd_x_prom_sim = np.mean(msd_tot_x_sim, axis=0)
msd_y_prom_sim = np.mean(msd_tot_y_sim, axis=0)



#%%

#Desplazamiento vs tiempo tanto para x como para y para partículas no atrapadas

labels = {'fontname' : 'Times New Roman',
          'fontsize' : 20}
ticks = {'fontname' : 'Times New Roman',
         'fontsize' : 15}
legends = {'fontname' : 'Times New Roman',
           'fontsize' : 10}


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))

cmap = plt.get_cmap("plasma")
n_plots = len(T)
colores = [cmap(i) for i in np.linspace(0.25, 0.75, n_plots)]
#np.random.shuffle(colores)

for i, color in enumerate(colores):
    t = T[i]
    x = X[i]
    y = Y[i]
    xsim = Xsim[i]
    ysim = Ysim[i]
    ax1.plot(t, x*1e9, '-', linewidth=5, color=color, alpha=((1-(i/len(colores))**0.95))) 
    #ax1.plot(t, xsim, color="red", alpha=0.5)
    #ax2.plot(t, ysim, color="red", alpha=0.5)
    ax2.plot(t, y*1e9, '-', linewidth=5, color=color, alpha=((1- (i/(len(colores)))**0.95)))

t_raiz = np.arange(150)
# ax1.plot(t_raiz, np.sqrt(t)*1e-7, '--k')
# ax1.plot(t_raiz, -np.sqrt(t)*1e-7, '--k')
# ax2.plot(t_raiz, -np.sqrt(t)*1e-7, '--k')
# ax2.plot(t_raiz, np.sqrt(t)*1e-7, '--k')


fig.tight_layout()
ax1.set_ylabel("Desplazamiento [nm]", **labels)
ax1.set_xlabel("Tiempo [s]", **labels)
ax2.set_xlabel("Tiempo [s]", **labels)
ax1.set_ylim([-3e2, 3e2])
ax1.tick_params(labelsize=15)
ax2.tick_params(labelsize=15)

ax2.set_xlim([0,15])
ax1.set_xlim([0,15])
ax1.grid(alpha=0.5)
ax2.grid(alpha=0.5)
plt.gcf().subplots_adjust(bottom=0.15, left=0.1)

plt.show()

#%% Grafico de las desviaciones standar, hay que importar los datos sin el std

fig, ax = plt.subplots(sharey=True, figsize=(5,5))

labels = {'fontname' : 'Times New Roman',
          'fontsize' : 20}
ticks = {'fontname' : 'Times New Roman',
         'fontsize' : 10}
legends = {'fontname' : 'Times New Roman',
           'fontsize' : 10}


cmap = plt.get_cmap("plasma")
color1, color2 = [cmap(0.25), cmap(0.75)]

long = np.arange(len(X_sin_filtro))
ax.plot(long, np.std(np.array(X_sin_filtro)*1e9, axis  = 1),'o',
        alpha=0.5, color=color1)
ax.plot(long, np.std(np.array(Y_sin_filtro)*1e9, axis = 1),'o',
        alpha=0.5, color=color2)

fig.tight_layout()
ax.grid(alpha=0.5)
ax.set_ylabel("STD [nm]", **labels)
ax.set_xlabel("Medición", **labels)
L = ax.legend(["Trayectorias en $\hat{x}$", "Trayectorias en $\hat{y}$"])
plt.setp(L.texts, family="Times New Roman", fontsize=15)
ax.grid(alpha=0.5)

plt.gcf().subplots_adjust(bottom=0.1, left=0.15)
plt.savefig('../../imagenes_informe/std_dispares.pdf')
plt.show()
        


#%%

#25 trayectorias de partículas

fix, axs = plt.subplots(5,5, figsize=(10,10), sharex=True, sharey=True)

cuenta = 0
for ii in range(5):
    for jj in range(5):
        ax = axs[ii,jj]
        x = X[cuenta]
        y = Y[cuenta]
        ax.plot(x[0], y[0], 'ok')
        ax.plot(x, y)
        cuenta += 1
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))

for ii in range(5):
    for jj in range(5):
        x = X[cuenta]
        y = Y[cuenta]
        offsetx = ii*3e-7
        offsety = jj*3e-7
        ax.plot(x + offsetx, y + offsety)
        ax.plot(x[0] + offsetx, y[0] + offsety, 'ok')
        ax.plot(x[-1] + offsetx, y[-1] + offsety, '*k', markersize=10)
        cuenta += 1
plt.show()


#%% Todos (para cada set de datos) los histogramas de la magnitud de los pasos 


for i in range(len(X)):

    Xdif_tmp = np.diff(X[i])
    Ydif_tmp = np.diff(Y[i])

    fig, axs = plt.subplots(1,2, figsize=(10,5))
    ax1, ax2 = axs
    
    print(i)
    ax1.hist(Xdif_tmp, bins=20, density=False)
    ax1.set_xlim([-0.2e-7,0.2e-7])
    
    ax2.hist(Ydif_tmp, bins=20, density=False)
    ax2.set_xlim([-0.2e-7,0.2e-7])
    
    plt.show()
    
#%% #Cuatro histogramas de la magnitud de los pasos en X y en Y para 
#   atrapadas y no atrapadas


fig, axs = plt.subplots(2,2, figsize=(10,5))
ax1, ax2 = axs[0]
ax3, ax4 = axs[1]

ax1.hist(Xdif, bins=100, density=True)
ax1.set_xlim([-0.2e-7,0.2e-7])

ax2.hist(Ydif, bins=100, density=True)
ax2.set_xlim([-0.2e-7,0.2e-7])

ax3.hist(YAdif, bins=100, density=True)
ax3.set_xlim([-0.1e-7,0.1e-7])

ax4.hist(XAdif, bins=100, density=True)
ax4.set_xlim([-0.1e-7,0.1e-7])
plt.show()

fig, axs = plt.subplots(1,2, figsize=(10,5))
ax1, ax2 = axs

ax1.hist(Xsim_dif, bins=50, density=True)
ax1.set_xlim([-0.2e-7,0.2e-7])

ax2.hist(Ysim_dif, bins=50, density=True)
ax2.set_xlim([-0.2e-7,0.2e-7])

plt.show()


#%%

#Histograma del módulo del desplazamiento tanto para atrapadas como para no atrapadas

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12,8))

fig.tight_layout()
ax1.hist(modulos, bins=260, density=True)
ax2.hist(modulosA, bins=120, density=True)
ax1.set_xlim([0, 0.2e-7])
ax2.set_xlim([0, 0.2e-7])
ax2.set_xlabel("Módulo de dr")
plt.show()

#%%

#Histograma en función del paso tanto para x como para y, y para atrapadas y no atrapadas
#(Ver tésis página 10)


pasos = [10, 60, 149]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
fig.tight_layout()

bins_tot = []
for paso in pasos:
    x_paso = []
    y_paso = []

    for i, t in enumerate(T):
        x = X[i]
        y = Y[i]
        x_paso.append(x[paso])
        y_paso.append(y[paso])
        x_paso.append(x[paso])
        y_paso.append(y[paso])

    nx, binsx, patchx = ax1.hist(x_paso, histtype="step", bins=10, density=True,
                                 label=str(paso), linewidth=2)
    ny, binsy, patchy = ax2.hist(y_paso, histtype="step", bins=10, density=True,
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


#%% MSD fit
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

popt, pcov = curve_fit(lineal, retraso, msd_prom)
popt2, pcov2 = curve_fit(cuad, retraso, msd_prom, p0 = [1e-28, 1e-19, 0])
#popt3, pcov3 = curve_fit(exp, rastro[:-5], msd_prom[-5], p0 = [1e-80])

plt.plot(msd_prom[:-1], 'k-', linewidth = 3)
plt.plot(retraso, lineal(retraso, *popt), 'r-', linewidth = 3)
plt.plot(retraso, cuad(retraso, *popt2), 'b-', linewidth = 3)
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


#%% MSD para x e y por separado



retraso = np.arange(38)
msd_tot = []
plt.figure(figsize=(12,5))
for j in range(len(X)):
    msd_tray = []
    for i in retraso:
        msd_tray.append(calc_msd(i, X[j]))
    plt.plot(msd_tray, alpha = 0.2)
    msd_tot.append(msd_tray)

msd_prom = np.mean(msd_tot, axis = 0)

semx = sem(msd_tot, axis=0)

popt, pcov = curve_fit(lineal, retraso, msd_prom)
popt2, pcov2 = curve_fit(cuad, retraso, msd_prom, p0 = [1e-28, 1e-19, 0])
print("poptx", popt)
print("pcovx", np.sqrt(np.diag(pcov)))

chix = chisq(msd_prom, lineal(retraso, *popt), semx)
rx, kx = rval(msd_prom, chix, 2)
pvalx =  pvalor(chix, kx)

plt.title("X")
plt.ylim(0, 3e-17)
plt.errorbar(retraso, msd_prom, yerr=semx,fmt='k-', linewidth = 3)
plt.plot(retraso, lineal(retraso, *popt), 'r--', linewidth = 2)
plt.plot(retraso, cuad(retraso, *popt2), 'b--', linewidth = 2)
plt.show()

retraso = np.arange(38)
msd_tot = []
plt.figure(figsize=(12,5))
for j in range(len(Y)):
    msd_tray = []
    for i in retraso:
        msd_tray.append(calc_msd(i, Y[j]))
    plt.plot(msd_tray, alpha = 0.3)
    msd_tot.append(msd_tray)

msd_prom = np.mean(msd_tot, axis = 0)

semy = sem(msd_tot, axis=0)

popt, pcov = curve_fit(lineal, retraso, msd_prom)
popt2, pcov2 = curve_fit(cuad, retraso, msd_prom, p0=[1e-28, 1e-19, 0])
print("popty", popt)
print("pcovy", pcov)

chiy = chisq(msd_prom, cuad(retraso, *popt2), semy)
ry, ky = rval(msd_prom, chiy, 3)
pvaly =  pvalor(chiy, ky)

plt.title("MSD en Y para series de 150 datos")
plt.ylim(0, 3e-17)
#plt.plot(msd_prom, 'k-', linewidth = 3)
plt.ylabel("MSD en Y")
plt.xlabel("Retraso")
plt.errorbar(retraso, msd_prom, yerr= semy,fmt='k-', linewidth = 3, alpha=1, zorder=0)
plt.plot(retraso, lineal(retraso, *popt), 'r--', linewidth = 3)
plt.plot(retraso, cuad(retraso, *popt2), '--', color="lime", linewidth = 3, alpha=1)
plt.show()


#%% MSD para X y para Y

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15,10))
ax1, ax2, ax3, ax4 = axs

retrasos = np.arange(38)

for ax in axs:
    ax.set_xscale("log")
    ax.set_yscale("log")

for j in range(len(X)):
    msdx = msd_tot_x[j]
    msdy = msd_tot_y[j]
    msdx_sim = msd_tot_x_sim[j]
    msdy_sim = msd_tot_y_sim[j]
    ax1.plot(retrasos, msdx, alpha=0.3)    
    ax2.plot(retrasos, msdy, alpha=0.3)
    ax3.plot(retrasos, msdx_sim, alpha=0.3)
    ax4.plot(retrasos, msdy_sim, alpha=0.3)

ax1.plot(retrasos, msd_x_prom, '--k')
ax2.plot(retrasos, msd_y_prom, '--k')
ax3.plot(retrasos, msd_x_prom_sim, '--k')
ax4.plot(retrasos, msd_y_prom_sim, '--k')

semx = sem(np.log(msd_tot_x))[1:]
semy = sem(np.log(msd_tot_y))[1:]
semxsim = sem(np.log(msd_tot_x_sim))[1:]
semysim = sem(np.log(msd_tot_y_sim))[1:]

poptx, pcovx = curve_fit(lineal, np.log(retraso)[1:], np.log(msd_x_prom)[1:],
                         sigma=semx, absolute_sigma=True)
popty, pcovy = curve_fit(lineal, np.log(retraso)[1:], np.log(msd_y_prom)[1:],
                         sigma=semy, absolute_sigma=True)

poptx_sim, pcovx_sim = curve_fit(lineal, np.log(retraso)[1:],
                                 np.log(msd_x_prom_sim)[1:], sigma=semxsim,
                                 absolute_sigma=True)
popty_sim, pcovy_sim = curve_fit(lineal, np.log(retraso)[1:],
                                 np.log(msd_y_prom_sim)[1:], sigma=semysim,
                                 absolute_sigma=True)


ax1.plot(retraso, np.exp(lineal(np.log(retraso), *poptx)), "-b", linewidth=2)
ax2.plot(retraso, np.exp(lineal(np.log(retraso), *popty)), "-b", linewidth=2)
ax3.plot(retraso, np.exp(lineal(np.log(retraso), *poptx_sim)), "-b",
         linewidth=2)
ax4.plot(retraso, np.exp(lineal(np.log(retraso), *popty_sim)), "-b",
         linewidth=2)


#%%

t = T[0]
varx = np.var(X, axis=0)
vary = np.var(Y, axis=0)
varxsim = np.var(Xsim, axis=0)
varysim = np.var(Ysim, axis=0)
plt.plot(t, varx, 'o')
plt.plot(t, vary, 'o')
plt.plot(t, varxsim, 'o')
plt.plot(t, varysim, 'o')
plt.show()


