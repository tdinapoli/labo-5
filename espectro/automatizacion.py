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
#%%
# lista = ['00', '05', '20', '35', '50', '85']
# path  = 'data/dia 2/s1_e'
#lista = [nombre_lampara, nombre, cubrimiento_oro1, ...]
#path = D:/...
longitudes, intensidades = importar_datos(lista, path)
abosrbancias = calc_absorbancia(intensidades)
MA, MSTD = moving_avrege_y_std(abosrbancias)
grafico(lista, MA, MSTD)

