#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 14:57:56 2021

@author: dina
"""
import numpy as np

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