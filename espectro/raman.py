import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

#%%
def importar_datos(lista, path):
    longitudes_de_onda = []
    intensidades       = []
    for i in lista:
        data = np.loadtxt(open(path+i+'.csv').readlines()[:-1], skiprows = 33, delimiter = ',')
        longitudes_de_onda.append(data[:,0])
        intensidades.append(data[:,1])
    
    return longitudes_de_onda, intensidades
#%%



datos_raman = ['laser_raman', '00_raman', '05_raman', '07_raman', '10_raman']
path = 'data/dia 4/'
longitudes, intensidades = importar_datos(datos_raman, path)


import matplotlib as mpl
COLOR = '#494949'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
font = {'family' : 'arial'}
rc('font', **font)

for i in range(len(intensidades)):
    if i == 2:
        plt.figure()
        plt.plot(longitudes[i], intensidades[i]- intensidades[0], label = datos_raman[i], color = COLOR)
        plt.ylim([-0.02, 1])
        plt.grid(0.5)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlabel('Longitud de onda [nm]', fontsize = 14)
        plt.ylabel('Intensidad relativa', fontsize = 14)
        #plt.legend(fontsize = 12, loc = 'upper left')
plt.savefig('intensidad_relativa.png', transparent = True, dpi = 1000)

#%%
import matplotlib as mpl
COLOR = '#494949'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
font = {'family' : 'arial'}
rc('font', **font)


fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
ax1.plot(longitudes[0], intensidades[0],  color = COLOR)
ax1.set_ylim([-0.02,1])
ax2.plot(longitudes[0], intensidades[2],  color = COLOR)
ax1.set_ylim([-0.04,1])
plt.ylim([-0.04, 1])
plt.grid(0.5)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Longitud de onda [nm]', fontsize = 14)
ax1.set_ylabel('Intensidad relativa', fontsize = 14)
ax2.set_ylabel('Intensidad relativa', fontsize = 14)
#plt.legend(fontsize = 12, loc = 'upper left')
plt.savefig('int_laser_raman.png', transparent = True, dpi = 1000)