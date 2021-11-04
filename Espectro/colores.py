import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as fp 
#%%
data_lamapra = np.loadtxt(open('lampara_halogena_t_2100.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
int_lampara = data_lamapra[:,1]
lon_lampara = data_lamapra[:,0]


colores_celofan = ['verde', 'amarillo', 'naranja', 'rojo', 'azul']
int_papel = []
lon_papel = []

for i in colores_celofan:
    data =  np.loadtxt(open('f_'+i+ '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
    lon_papel.append(data[:,0])
    int_papel.append(data[:,1])

RGB= ['R', 'G', 'B']
int_RGB = []
lon_RGB = []

for i in RGB:
    data =  np.loadtxt(open('RGB_'+i+ '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
    lon_RGB.append(data[:,0])
    int_RGB.append(data[:,1])


#%%
colors = ['green', 'yellow', 'orange', 'red', 'blue']
plt.figure()
#plt.plot(lon_lampara, int_lampara, label = 'lampara')
for i in range(len(colores_celofan)):
    plt.plot(lon_papel[i], -np.log10(int_papel[i]/int_lampara),color= colors[i] ,label = colores_celofan[i])
plt.legend()
plt.show()

#%%

plt.figure()
colors = ['red', 'green', 'blue']
for i in range(len(RGB)):
    plt.plot(lon_RGB[i], int_RGB[i], color= colors[i],label = RGB[i])
plt.show()