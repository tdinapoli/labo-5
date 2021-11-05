import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as fp 
#%%
data_lamapra = np.loadtxt(open('data/dia 2/lampara_halogena_t_2100.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
int_lampara = data_lamapra[:,1]
lon_lampara = data_lamapra[:,0]


colores_celofan = ['verde', 'amarillo', 'naranja', 'rojo', 'azul']
int_papel = []
lon_papel = []

for i in colores_celofan:
    data =  np.loadtxt(open('data/dia 2/f_'+i+ '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
    lon_papel.append(data[:,0])
    int_papel.append(data[:,1])

RGB= ['R', 'G', 'B']
int_RGB = []
lon_RGB = []

for i in RGB:
    data =  np.loadtxt(open('data/dia 2/RGB_'+i+ '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
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

#%%
import os

from colour.plotting import *
import colour as colour
colour_style();

#%%
inte = []
long_onda = []
inte_lamp = []
lon = np.arange(193, 1200)
for j in lon:
    for i in range(len(lon_papel[0])-1):
        if lon_papel[0][i+1]>j and lon_papel[0][i]<j:
            #print(lon_papel[0][i])
            long_onda.append(int(lon_papel[0][i]))
            inte.append(int_papel[4][i])
            inte_lamp.append(int_lampara[i])
#%%
data_dic = dict(zip(long_onda,np.array(inte)/np.array(inte_lamp)))
sdm     = ( colour.SpectralDistribution(data_dic)).normalise()
XYZ = colour.sd_to_XYZ(sdm)
RGB = colour.XYZ_to_sRGB(XYZ)
plt.style.use({'figure.figsize': (5, 5)})
plot_single_colour_swatch(ColourSwatch('Mediodía', RGB/np.max(RGB)),text_parameters={'size': 'x-large'});            
