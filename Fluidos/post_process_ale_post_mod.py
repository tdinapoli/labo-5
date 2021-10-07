from openpiv.piv import simple_piv
from openpiv import tools, pyprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cv2
from os import listdir
from scipy.optimize import curve_fit


def ran(x,o,c): #o == omega c= c
    y = []
    for i in range(len(x)):
        if x[i]<= c:
            y.append(o*x[i])
        else:
            y.append(o*c**2/x[i])
            
    return np.array(y)

def bur(x, g, nu,a): #g==gamma; a == alpha
    y  = np.zeros(len(x))
    for i in range(len(x)):
        y[i] += g*(1-np.e**(-a*x[i]**2 / (2*nu)))/(2*np.pi*x[i])
    return y

def bur_vr(r, c, nu):
    vr = -2*nu*r/c**2


path = "28-09/Data/todo_g"
def analisador(glicerina, vaso, velocidad, calibracion_pos = 1, calibracion_vel = 1, grafico_centro = False):
    '''
    

    Parameters
    ----------
    glicerina : String
        Seleccionar concentracion de glicerina, 36 o 50.
    vaso : String
        Seleccionar el tamaño del vaso 10cm o 15cm.
    velocidad : String
        Seleccionar cual de las velocidades se quiere mostrar, algunos no tienen todas las velocidades posibles.
    calibracion_pos : Float
        Conversion de px a metros.
     calibracion_vel: Float
         conversion de px/fr a m/s.
     grafico_centro : Boolean
         grafico de la variacion de la posicion del circulo en x y en y.

    '''
    importar = path+glicerina + '_' + velocidad + '_' + vaso + '.txt'
    x, y, u, v, vr, vt, dist, xc, yc= np.loadtxt(importar).T
    
    dists_plot = np.arange(0, 405, 5)
    vts_plot = []
    vrs_plot = []
    
    for i in range(1, len(dists_plot)):
        dist_min = dists_plot[i-1]
        dist_max = dists_plot[i]
        filtro1 = dist > dist_min
        filtro2 = dist < dist_max
        filtro = filtro1 & filtro2
        filtro = filtro.reshape(vt.shape)
        
        vts_plot.append(np.nanmean(vt[filtro]))
        vrs_plot.append(np.nanmean(vr[filtro]))



    vtitas = np.array(vts_plot)*calibracion_vel    
    vrs = np.array(vrs_plot)*calibracion_vel    
    dists = dists_plot*calibracion_pos
    
    popt, pcov = curve_fit(ran, dists[1:-1], np.abs(vtitas[1:]), p0 =[1000,0.01])
    popt2, pcov2 = curve_fit(bur, dists[1:-1], np.abs(vtitas[1:]), p0 =[1,1,100000 ])
    
    vts_plot = np.array(vts_plot)
    plt.figure(figsize = (10,6))
    plt.plot(dists[:-1], np.abs(vtitas),'.', label ='Datos')
    plt.plot(dists[:-1], ran(dists[:-1],*popt), label = 'Ajuste Rankie')
    plt.plot(dists[:-1], bur(dists[:-1],*popt2), label ='Ajuste Burguers')
    plt.plot(dists[:-1], np.abs(np.array(vrs)),'--k', label ='vr')
    plt.legend()
    plt.grid(alpha = 0.7)
    if calibracion_vel != 1 and calibracion_pos != 1:
        plt.xlabel("distancia [m]")
        plt.ylabel(r"Velocidad $\theta$ [m/s]")
    else:
        plt.xlabel("distancia [px]")
        plt.ylabel(r"Velocidad $\theta$ [px/fr]")
    plt.show()
    
    
    if grafico_centro == True:
        fig, ax = plt.subplots(2,1, figsize=(10,6))
        ax[0].plot(xc)
        ax[0].set_title('pos x')
        ax[1].plot(yc)
        ax[1].set_title('pos y')
    else:
        pass
    return vts_plot,popt, popt2, calibracion_pos, calibracion_vel

#%%    

# g 36 vaso 10cm v1 cal_p = 0.00011 pos, cal_v = 0.00804 vel
# g 36 vaso 10cm v2 cal_p = 0.00011 pos, cal_v = 0.00796 vel

# g 36 vaso 15cm v2 cal_p = 0.00015 pos, cal_v = 0.0105 vel

# g 50 vaso 10cm v1 cal_p = 0.00012 pos, cal_v = 0.00895 vel
# g 50 vaso 10cm v2 cal_p = 0.00012 pos, cal_v = 0.00895 vel
# g 50 vaso 10cm v3 cal_p = 0.00012 pos, cal_v = 0.0087 vel

# g 50 vaso 15cm v2 cal_p = 0.00015 pos, cal_v = 0.01084 vel
# g 50 vaso 15cm v3 cal_p = 0.00015 pos, cal_v = 0.01084 vel

analisador('50', '15cm', 'v2', 0.00012, 0.000895)
#%%

vt_g50 ,popt_g50, popt2_g50, cal_pos_g50, cal_vel_g50 = analisador('50', '10cm', 'v2', 0.00012, 0.00895)
vt_g36 ,popt_g36, popt2_g36, cal_pos_g36, cal_vel_g36  = analisador('36', '10cm', 'v2', 0.00011, 0.00796)

#%%

dists_plot = np.arange(0, 405, 5) * cal_pos_g50
plt.plot(dists_plot[:-1], np.abs(vt_g50),'.', label ='Datos G50')
plt.plot(dists_plot[:-1], ran(dists_plot[:-1],*popt_g50), label = 'Ajuste Rankie')
plt.plot(dists_plot[:-1], bur(dists_plot[:-1],*popt2_g50), label ='Ajuste Burguers')

plt.plot(dists_plot[:-1], np.abs(vt_g36),'.', label ='Datos G36')
plt.plot(dists_plot[:-1], ran(dists_plot[:-1],*popt_g36), label = 'Ajuste Rankie')
plt.plot(dists_plot[:-1], bur(dists_plot[:-1],*popt2_g36), label ='Ajuste Burguers')
plt.xlabel('Distancia [cm]')
plt.ylabel('Velocidad [cm/s]')
plt.grid(0.7)
plt.legend()

#%%
vt_v1 ,popt_v1, popt2_v1, cal_pos_v1, cal_vel_v1 = analisador('50', '10cm', 'v1', 0.00012*100, 0.00895*100)
vt_v2 ,popt_v2, popt2_v2, cal_pos_v2, cal_vel_v2  = analisador('50', '10cm', 'v2', 0.00012*100, 0.00895*100)
vt_v3 ,popt_v3, popt2_v3, cal_pos_v3, cal_vel_v3  = analisador('50', '10cm', 'v3', 0.00012*100, 0.0087*100)

#%%
dists_plot = np.arange(0, 405, 5)
plt.plot(dists_plot[:-1]*cal_pos_v1, np.abs(vt_v1),'.', label ='Datos v1')
# plt.plot(dists_plot[:-1]*cal_pos_v1*100, ran(dists_plot[:-1],*popt_v1)*cal_vel_v1, label = 'Ajuste Rankie')
# plt.plot(dists_plot[:-1]*cal_pos_v1*100, bur(dists_plot[:-1],*popt2_v1)*cal_vel_v1, label ='Ajuste Burguers')

plt.plot(dists_plot[:-1]*cal_pos_v2, np.abs(vt_v2),'.', label ='Datos v2')
# plt.plot(dists_plot[:-1]*cal_pos_v2*100, ran(dists_plot[:-1],*popt_v2)*cal_vel_v2, label = 'Ajuste Rankie')
# plt.plot(dists_plot[:-1]*cal_pos_v2*100, bur(dists_plot[:-1],*popt2_v2)*cal_vel_v2, label ='Ajuste Burguers')

plt.plot(dists_plot[:-1]*cal_pos_v3, np.abs(vt_v3),'.', label ='Datos v3')
# plt.plot(dists_plot[:-1]*cal_pos_v3*100, ran(dists_plot[:-1],*popt_v3)*cal_vel_v3, label = 'Ajuste Rankie')
# plt.plot(dists_plot[:-1]*cal_pos_v3*100, bur(dists_plot[:-1],*popt2_v3)*cal_vel_v3, label ='Ajuste Burguers')


plt.xlabel('Distancia [cm]')
plt.ylabel('Velocidad [cm/s]')
plt.grid(0.7)
plt.legend()