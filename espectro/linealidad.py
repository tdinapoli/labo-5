import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from scipy.signal import find_peaks as fp 

#%%
def importar_datos(lista, path = str(os.getcwd())):
    longitudes_de_onda = []
    intensidades       = []
    for i in lista:
        data = np.loadtxt(open(path+'hi_'+i+'.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
        longitudes_de_onda.append(data[:,0])
        intensidades.append(data[:,1])
    
    return longitudes_de_onda, intensidades
#%%
tiempo = ['0500', '1050', '1150', '1250', '1350', '1500', '2500', '3500', '4500', '5500', '6500', '7500', '8500', '9500']
intensidad = []
longitudes = []
path = 'data/dia 1/Linealidad/'
longitudes, intensidades = importar_datos(tiempo, path)


lon_max = []
int_max = []
for i in range(len(tiempo)):
    find_peks = fp(intensidades[i][1350:1400], height = 0.02)
    lon_max.append(longitudes[i][find_peks[0]+1350])
    int_max.append(intensidades[i][find_peks[0]+1350])
    
for i in range(len(tiempo)):
    plt.figure()
    plt.plot(longitudes[i], intensidades[i])
    plt.plot(lon_max[i], int_max[i],'o')

    plt.show()

lon_max[0] = np.array(lon_max[0][0])
int_max[0] = np.array(int_max[0][0])

#%%
tiempos = [0.000500, 0.010500, 0.011500, 0.012500, 0.01350, 0.001500, 0.002500, 0.003500,  0.004500, 0.005500, 0.006500, 0.007500, 0.008500, 0.009500]

plt.plot(tiempos[5:], int_max[5:],  'ko')
plt.plot(tiempos[:5], int_max[:5], 'ro')
plt.grid(alpha = 0.5)
