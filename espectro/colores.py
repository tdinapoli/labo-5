import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as fp 
from matplotlib import rc
import matplotlib.font_manager as font_manager
import matplotlib.font_manager as fm

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
import matplotlib as mpl

colors = ['#0A9396', '#EE9B00', '#CA6702', '#AE2012', '#005F73']
colores_celofan = ['Verde', 'Amarillo', 'Naranja', 'Rojo', 'Azul']

COLOR = '#494949'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
font = {'family' : 'arial'}

plt.figure(facecolor = 'w')
#plt.plot(lon_lampara, int_lampara, label = 'lampara')
for i in range(len(colores_celofan)):
    plt.plot(lon_papel[i]-1.24, -np.log10(int_papel[i]/int_lampara),color= colors[i] ,label = colores_celofan[i], linewidth = 1.5)
plt.xlim([200, 1000])
plt.ylim([-0.2, 1.2])
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Longitud de onda [nm]', fontsize = 14)
plt.ylabel('Absorbancia', fontsize = 14)
plt.legend(fontsize = 12)
plt.savefig('celofan.png', transparent = True, dpi = 1000)
plt.show()

#%%
#csfont = {'fontname':'Roboto Condensed'}
#prop = fm.FontProperties(fname="Roboto Condensed")
import matplotlib as mpl
COLOR = '#494949'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
font = {'family' : 'arial'}

rc('font', **font)
#font = font_manager.FontProperties(family='Roboto Condensed', size=14)
plt.figure(facecolor = 'w')
colores_RGB = ['Rojo', 'Verde', 'Azul']
colors = ['#e63946', '#90be6d', '#0077B6']
colores_prefijos = [630, 532, 465]
for i in range(len(RGB)):
    plt.plot(lon_RGB[i]-1.24, int_RGB[i], color= colors[i],label = colores_RGB[i], linewidth = 1.5)
    plt.vlines(colores_prefijos[i], ymin = 0, ymax = 1, color = colors[i], alpha = 0.3, linestyle = 'dashed')
    
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Longitud de onda [nm]', fontsize = 14)
plt.ylabel('Intensidad normalizada', fontsize = 14)
plt.legend(fontsize = 12)
plt.savefig('RGB.png', transparent = True, dpi = 1000)
plt.show()

#%%

COLOR = '#494949'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
font = {'family' : 'arial'}

plt.figure(facecolor = 'w')
plt.plot(lon_lampara, int_lampara, color = COLOR)
# plt.xlim([200, 1000])
# plt.ylim([-0.2, 1.2])
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Longitud de onda [nm]', fontsize = 14)
plt.ylabel('Intensidad normalizada', fontsize = 14)
#plt.legend(fontsize = 12)
plt.savefig('lampara.png', transparent = True, dpi = 1000)
plt.show()


#%%
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
            inte.append(int_papel[0][i])
            inte_lamp.append(int_lampara[i])
#%%
data_dic = dict(zip(long_onda,np.array(inte)))
sdm     = ( colour.SpectralDistribution(data_dic)).normalise()
XYZ = colour.sd_to_XYZ(sdm)
RGB = colour.XYZ_to_sRGB(XYZ)
plt.style.use({'figure.figsize': (5, 5)})
plot_single_colour_swatch(ColourSwatch('Verde', RGB/np.max(RGB)),text_parameters={'size': 'x-large'});            
