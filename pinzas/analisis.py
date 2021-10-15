import matplotlib.pyplot as plt
import numpy as np

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
            t.append(tp[150:]-tp[150])
            x.append(xp[150:]-xp[150])
            y.append(yp[150:]-yp[150])
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
    

plt.figure()
for i in range(len(t)):
    plt.plot(t[i],x[i], '-')
plt.xlim([0,15])
plt.show()

#%%
def val_med(N, x):
    valores_medios = []
    for i in range(len(x)-1):
        valores_medios.append(x[i][N])
    val_medi = np.mean(valores_medios)
    return val_medi

rangos = np.arange(0,100)
plt.figure()
for i in rangos:
    plt.plot(i, val_med(i,y),'ok')
plt.show()

#%% Atrapada

t_atr = []
x_atr = []
y_atr = []

t_brow = []
x_brow = []
y_brow = []

lista = np.arange(0,6)
dif_brow = np.arange(0, 2)

for i in lista:
    tp, xp, yp = np.loadtxt('Data/atrapada_2/atrapada_' + str(i) + '.txt', skiprows = 2).T #p = parcial
    t_atr.append(tp-tp[0])
    x_atr.append(xp-xp[0])
    y_atr.append(yp-yp[0])
    for j in dif_brow:
        tp, xp, yp = np.loadtxt('Data/atrapada_2/brow' + str(j) + '_' + str(i) + '.txt', skiprows = 2).T #p = parcial
        t_brow.append(tp-tp[0])
        x_brow.append(xp-xp[0])
        y_brow.append(yp-yp[0])

plt.figure()
for i in range(len(x_atr)):
    plt.plot(x_atr[i], y_atr[i], label = 'intensidad' + str(i+1))
plt.legend()
plt.show()

