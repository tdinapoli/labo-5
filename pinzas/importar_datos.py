#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 14:57:56 2021

@author: dina
"""
import numpy as np
from scipy.stats import chi2


def importar():
    data_num = np.arange(0,40)
    
    t = []
    x = []
    y = []
    
    for i in data_num:
        if i < 22:
            tp, xp, yp = np.loadtxt('Data/Brow/brown' + str(i) + '.txt', skiprows = 2).T #p = parcial
            t.append(tp-tp[0])
            x.append(xp-xp[0])
            y.append(yp-yp[0])
        if i>=22:
            if i == 26 or i == 27:
                tp, xp, yp = np.loadtxt('Data/Brow/brown' + str(i) + '.txt', skiprows = 2).T #p = parcial
                t.append(tp[0:150]-tp[0])
                x.append(xp[0:150]-xp[0])
                y.append(yp[0:150]-yp[0])
                t.append(tp[-150:]-tp[-150])
                x.append(xp[-150:]-xp[-150])
                y.append(yp[-150:]-yp[-150])
            elif i == 24 or  i==28:
                l = np.arange(0,6)
                for j in range(len(l)-1):
                    tp, xp, yp = np.loadtxt('Data/Brow/brown' + str(i) + '.txt', skiprows = 2).T #p = parcial
                    t.append(tp[l[j]*150:l[j+1]*150]-tp[l[j]*150])
                    x.append(xp[l[j]*150:l[j+1]*150]-xp[l[j]*150])
                    y.append(yp[l[j]*150:l[j+1]*150]-yp[l[j]*150])
            else:
                l = np.arange(0,8)
                for j in range(len(l)-1):
                    tp, xp, yp = np.loadtxt('Data/Brow/brown' + str(i) + '.txt', skiprows = 2).T #p = parcial
                    t.append(tp[l[j]*150:l[j+1]*150]-tp[l[j]*150])
                    x.append(xp[l[j]*150:l[j+1]*150]-xp[l[j]*150])
                    y.append(yp[l[j]*150:l[j+1]*150]-yp[l[j]*150])
    t_atr = []
    x_atr = []
    y_atr = []
    
    lista = np.arange(0,6)
    dif_brow = np.arange(0, 2)
    
    for i in lista:
        tp, xp, yp = np.loadtxt('Data/atrapada_2/atrapada_' + str(i) + '.txt', skiprows = 2).T #p = parcial
        t_atr.append(tp - tp[0])
        x_atr.append(xp - xp[0])
        y_atr.append(yp - yp[0])
        for k in dif_brow:
            tp, xp, yp = np.loadtxt('Data/atrapada_2/brow' + str(k) + '_' + str(i) + '.txt', skiprows = 2).T #p = parcial
            l = np.arange(0,8)
            for j in range(len(l)-1):
                t.append(tp[l[j]*150:l[j+1]*150]-tp[l[j]*150])
                x.append(xp[l[j]*150:l[j+1]*150]-xp[l[j]*150])
                y.append(yp[l[j]*150:l[j+1]*150]-yp[l[j]*150])
                
    for index in range(len(x)):
        try:
            x[index] = x[index][:150]
            y[index] = y[index][:150]
            t[index] = t[index][:150]
        except:
            pass
    return t, x, y, t_atr, x_atr, y_atr

def calc_diferencias(X, Y):
    X_nuevo = []
    Y_nuevo = []
    for x in X:
        X_nuevo += np.diff(x).tolist()
    for y in Y:
        Y_nuevo += np.diff(y).tolist()
    return np.array(X_nuevo), np.array(Y_nuevo)

def calc_modulos(x, y):
    n_datos = len(x)
    modulos = np.zeros(n_datos)
    for i in range(n_datos):
        x_tmp, y_tmp = x[i], y[i]
        modulos[i] += np.sqrt(x_tmp**2 + y_tmp**2)
    return modulos


def calc_msd(retraso, x):
	N = len(x)
	msd = np.zeros(N-retraso)
	for i in range(N-retraso):
		msd[i] += (x[i+retraso] - x[i])**2 / (N - retraso - 1)
	return np.mean(msd)

def msd_retrasos(X):
    n = len(X[0])
    msd_por_retraso = np.zeros(n)
    for retraso in range(int(n/4)):
        msd_tmp = 0
        for x in X:		
            msd_tmp += calc_msd(retraso, x)
        msd_tmp = msd_tmp / n
        msd_por_retraso[retraso] += msd_tmp
    return msd_por_retraso


def calc_msd_vec(retraso, x, y):
    N = len(x)
    msd = np.zeros(N-retraso-1)
    for i in range(N-retraso-1):
        msd[i] += ((x[i+retraso] - x[i])**2 + (y[i+retraso] - y[i])**2) / (N - retraso - 1)
    return np.mean(msd)


def msd_retrasos_vec(X, Y):
	n = len(X[0])
	msd_por_retraso = np.zeros(n)	
	for retraso in range(n):
		msd_tmp = 0
		for j in range(len(X)):
			msd_tmp += calc_msd_vec(retraso, X[j], Y[j])
		msd_tmp = msd_tmp / n
		msd_por_retraso[retraso] += msd_tmp
	return msd_por_retraso


def chisq(ydata, ymodelo,sd):
    chisq = sum(((ydata-ymodelo)/sd)**2)
    return chisq
    
def rval( ydata, chi, gl):  #reduce chi square   gl = numero de parametros de la funcion es bueno q este cerca de 1
    k=len(ydata)-1-gl
    rval=chi/k   
    return rval, k

def pvalor(chi, k):         #integral de Ã±a distribucion de chisq es bueno q este cerca de 0
    p  = 1- chi2.cdf(chi, k)
    return p