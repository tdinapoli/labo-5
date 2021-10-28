#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:45:57 2021

@author: dina
"""
import numpy as np
import matplotlib.pyplot as plt
from importar_datos import calc_msd, msd_retrasos

#%%


def simular(reps=80, n_pasos=150, probx=0.5, proby=0.5, ampx=4e-9, ampy=4e-9):
    X = []
    Y = []
    for _ in range(reps):
        x = np.zeros(n_pasos)
        y = np.zeros(n_pasos)
        
        for i in range(n_pasos):
            signox = np.random.random()
            signoy = np.random.random()
            if signox < probx:
                x[i] += x[i-1] + ampx
            else:
                x[i] += x[i-1] - ampx
            if signoy < proby:
                y[i] += y[i-1] + ampy
            else:
                y[i] += y[i-1] - ampy
        
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return [X, Y]

def simular2(reps=80, n_pasos=150, probx=0.5, proby=0.5, amp=6.38e-9):
    X = []
    Y = []
    for _ in range(reps):
        x = np.zeros(n_pasos)
        y = np.zeros(n_pasos)
        
        for i in range(n_pasos):
            tita = np.random.random()*np.pi * 2
            
            pasox = amp * np.cos(tita)
            pasoy = amp * np.sin(tita)
                        
            x[i] += x[i-1] + pasox
            y[i] += y[i-1] + pasoy

        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return [X, Y]
