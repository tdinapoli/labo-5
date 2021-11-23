import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os


def importar_datos(lista, path = str(os.getcwd())):
    '''
    Importar siempre primero el dato de la lampara halogena y despues los datos de los distintos cubre objetos

    Parameters
    ----------
    lista : lista con los nombres de los datos a importar siempre con la lampara halogena al principio
    pe: [l_00, t_05, t_10]
    
    path : path donde se encuentran los csv
    
    Returns
    -------
    Listas con las longitudes de onda e intensidades medidas por el espectrometro
    '''
    longitudes_de_onda = []
    intensidades       = []
    for i in lista:
        data = np.loadtxt(open(path+i+'.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
        longitudes_de_onda.append(data[:,0])
        intensidades.append(data[:,1])
    
    return longitudes_de_onda, intensidades
    

def importar_datos2(lista, path = str(os.getcwd())):
    '''
    Importar siempre primero el dato de la lampara halogena y despues los datos de los distintos cubre objetos

    Parameters
    ----------
    lista : lista con los nombres de los datos a importar siempre con la lampara halogena al principio
    pe: [l_00, t_05, t_10]
    
    path : path donde se encuentran los csv
    
    Returns
    -------
    Listas con las longitudes de onda e intensidades medidas por el espectrometro
    '''
    longitudes_de_onda = []
    intensidades       = []
    for i in lista:
        data = np.loadtxt(open(path+i+'.csv').readlines()[:-1], skiprows = 33, delimiter = ',')
        longitudes_de_onda.append(data[:,0])
        intensidades.append(data[:,1])
    
    return longitudes_de_onda, intensidades
    


def calc_absorbancia(intensidades):
    '''
    Mismos parametros que el caso anterior, devuelve la absorbancia de cada
    uno de los vidrios, siempre teniendo en cuenta que el primer nombre de la
    lista corresponde a la lampara halogena o con un cubre sin nada.
    '''
    
    absorbancia = []
    for i in range(len(intensidades)-1):
        absorbancia.append(-np.log10(intensidades[i+1]/intensidades[0]))
    
    return absorbancia

def moving_average(x, n):
    return np.convolve(x, np.ones(n), mode="same")/n


def moving_std(x, n):
    MA = moving_average(x, n)
    MSTD = []
    for i, mean in enumerate(MA):
        if i - n//2 < 0:
            MSTD.append(np.std(x[0:i + n//2]))
        elif i + n//2 > len(x):
            MSTD.append(np.std(x[i - n//2: -1]))
        else:
            MSTD.append(np.std(x[i-n//2:i + n//2]))
    return 

def moving_avrege_y_std(absorbancias):
    MA = []
    MSTD = []
    for i in absorbancias:
        MA.append(moving_average(i, 30))
        MSTD.append(moving_std(i, 30))
        
    return MA, MSTD

def grafico(lista, MA, MSTD):
    colores = ["#c29b92","#d29182","#da8267","#e37750","#ea6f3a","#f36828","#b45124","#773518","#3b180d","#010005"]
    labels = {"fontname":"Times New Roman", "fontsize":15}
    todo = {"family": "sans-serif", "sans-serif":["Times New Roman"]}
    rc("font", **todo)
    fig, ax = plt.subplots(1, figsize=(6,10))
    for i, ma in enumerate(MA):
        color = colores[-i-1]
        ax.plot(longitudes[i+1], ma, color = color, label = lista[i+1] + 's')
        #ax.fill_between(longitudes[i+1], MA[i]+MSTD[i], MA[i]-MSTD[i],
         #               color= color,  alpha = 0.2, linewidth = 0)
    ax.set_yticks(np.arange(0, 0.55, 0.05))
    ax.set_ylim([0, 0.5])
    ax.grid(alpha=0.5)
    ax.set_xlabel("Longitud de onda [nm]", **labels)
    ax.set_ylabel("Absorbancia", **labels)
    ax.set_xlim([400, 1000])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=15, loc="upper left", framealpha=1)
    ax.tick_params(labelsize=12)
    plt.show()
    
    return
#%% Importo los datos del dia 2 y del dia 4
lista1 = ['lampara_halogena_t_2100']
laminado = ['05', '20', '35', '50', '85']
for i in laminado:
    lista1.append('s2_e'+ i+ '_t_2100')

path1 = 'data/dia 2/'

lista2 = ['lampara', '05_le_t_38', '07_le_t_38', '10_le_t_38']
path2 = 'data/dia 4/'


lista_azul = ['lampara_azul', '00_azul', '05_azul', '07_azul', '10_azul']
path3 = 'data/dia 4/'

longitudes1, intensidades1 = importar_datos(lista1, path1)
abosrbancias1 = calc_absorbancia(intensidades1)
MA1, MSTD1 = moving_avrege_y_std(abosrbancias1)


longitudes2, intensidades2 = importar_datos(lista2, path2)
abosrbancias2 = calc_absorbancia(intensidades2)
MA2, MSTD2 = moving_avrege_y_std(abosrbancias2)

longitudes3, intensidades3 = importar_datos2(lista_azul, path3)
abosrbancias3 = calc_absorbancia(intensidades3)
MA3, MSTD3 = moving_avrege_y_std(abosrbancias3)

#%% Hago dos figuras, una para los datos de la absorbancia del dia 2 y dia 4 de las laminas de oro, y otra para los de azul de metileno
MA = [MA2 + MA1[1:], MA3]
#MA = [MA1, MA2, MA3]
#MSTD = [MSTD1, MSTD2, MSTD3]
# lista = [lista1, lista2, lista_azul]
# lista = [['lampara', '05 dia2', '20', '35', '50', '85'],['lampara2', '05 dia4', '07', '10'], ['lamapra_azul', '00_azul', '05_azul','07_azul', '10_azul']]
# longitudes = [longitudes1, longitudes2, longitudes3]

import matplotlib as mpl
COLOR = '#494949'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
font = {'family' : 'arial'}
rc('font', **font)

colores = ["#c29b92","#d29182","#da8267","#e37750","#ea6f3a","#f36828","#b45124","#773518","#3b180d","#010005"]
#colores = ["#e79194","#ed9598","#f3989c","#fe9fa3","#d88496","#c57790","#b16989"]

longitudes = longitudes1[2:]+ longitudes2[1:]
lista = ['5s', '7s', '10s', '20s', '35s', '50s', '85s']
plt.figure(facecolor = 'w', figsize=(6,10))
for i in range(len(MA[0])):
    plt.plot(longitudes[i]-1.24, MA[0][i],  color = colores[-i-1], label = lista[i])
    print('hola')
plt.vlines(532, ymin = 0, ymax = 0.5, linestyle = 'dashed', label = '532 nm')
plt.grid(0.5)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Longitud de onda [nm]', fontsize = 14)
plt.ylabel('Absorbancia', fontsize = 14)
plt.legend(fontsize = 12, loc = 'upper left')
plt.xlim([400, 900])
#plt.savefig('oro_zoom.png', transparent = True, dpi = 1000)
plt.show()
#%%

import matplotlib as mpl
COLOR = '#494949'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
font = {'family' : 'arial'}
rc('font', **font)

lognitudes = longitudes3
plt.figure(facecolor = 'w', figsize=(15,8))
plt.plot(longitudes[i]-1.24, MA[1][0],  color = '#4ea0db')
plt.xlabel('Longitud de onda [nm]', fontsize = 14)
plt.ylabel('Absorbancia', fontsize = 14)
plt.grid(alpha = 0.5)
plt.xlim([400,1000])

#plt.savefig('azul.png', transparent = True, dpi = 1000)

#%%
longitudes_notch, intensidadesnotch = importar_datos2(['filtro_notcho_luz_natural'], 'data/dia 4/')


import matplotlib as mpl
COLOR = '#494949'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
font = {'family' : 'arial'}
rc('font', **font)
plt.figure(facecolor = 'w', figsize =(15,8))
plt.plot(longitudes_notch[0]-1.24, intensidadesnotch[0], color = COLOR)
plt.xlabel('Longitud de onda [nm]', fontsize = 14)
plt.ylabel('Intensidad normalizada', fontsize = 14)
plt.vlines(532, ymin = 0, ymax = 1, linestyle = 'dashed', color = '#ff744d')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha = 0.5)
plt.savefig('notch.png', transparent = True, dpi = 1000)



#%%
fig, ax = plt.subplots(1, figsize=(6,10))

for j in range(len(MA)):
    if j<2:
        colores = ["#c29b92","#d29182","#da8267","#e37750","#ea6f3a","#f36828","#b45124","#773518","#3b180d","#010005"]
        labels = {"fontname":"Times New Roman", "fontsize":15}
        todo = {"family": "sans-serif", "sans-serif":["Times New Roman"]}
        rc("font", **todo)
    #    fig, ax = plt.subplots(1, figsize=(6,10)
        for i, ma in enumerate(MA[j]):
            if j == 0:
                color = colores[-i-1]
                ax.plot(longitudes[j][i+1], ma,  color = color, label = lista[j][i+1] + 's')
            else:
                color = colores[-i-1 -len(MA[0])]
                ax.plot(longitudes[j][i+1], ma,  color = color, label = lista[j][i+1] + 's')
                

                #ax.fill_between(longitudes[i+1], MA[j][i]+MSTD[j][i], MA[j][i]-MSTD[j][i],
                 #               color= color,  alpha = 0.2, linewidth = 0)
        ax.axvline(533, linestyle = 'dashed')
        ax.set_yticks(np.arange(0, 0.55, 0.05))
        ax.set_ylim([0, 0.5])
        ax.grid(alpha=0.5)
        ax.set_xlabel("Longitud de onda [nm]", **labels)
        ax.set_ylabel("Absorbancia", **labels)
        ax.set_xlim([400, 1000])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize=15, loc="upper left", framealpha=1)
        ax.tick_params(labelsize=12)
        plt.show()

        
    else:
        
        fig, ax = plt.subplots(1, figsize=(6,10))
        colores = ["#c29b92","#d29182","#da8267","#e37750","#ea6f3a","#f36828","#b45124","#773518","#3b180d","#010005"]
        labels = {"fontname":"Times New Roman", "fontsize":15}
        todo = {"family": "sans-serif", "sans-serif":["Times New Roman"]}
        rc("font", **todo)
    #    fig, ax = plt.subplots(1, figsize=(6,10))
        for i, ma in enumerate(MA[j]):
            color = colores[-i-1]
            ax.plot(longitudes[j][i+1], ma,  label = lista[j][i+1] + 's')
            #ax.fill_between(longitudes[i+1], MA[i]+MSTD[i], MA[i]-MSTD[i],
             #               color= color,  alpha = 0.2, linewidth = 0)
        ax.set_yticks(np.arange(0, 0.55, 0.05))
        ax.set_ylim([0, 0.5])
        ax.grid(alpha=0.5)
        ax.set_xlabel("Longitud de onda [nm]", **labels)
        ax.set_ylabel("Absorbancia", **labels)
        ax.set_xlim([400, 1000])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize=15, loc="upper left", framealpha=1)
        ax.tick_params(labelsize=12)
        plt.show()
      
