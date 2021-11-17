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

for i in range(len(intensidades)):
    if i > 1:
        
        plt.figure()
        plt.plot(longitudes[i], intensidades[i]-intensidades[0], label = datos_raman[i], color = 'k')
        plt.legend()
        plt.show()