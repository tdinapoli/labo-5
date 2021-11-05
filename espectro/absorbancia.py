import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as fp 
#%%


set_up = ['s1','s2']
laminado = ['00', '05', '20', '35', '50', '85']

lon_s1 = []
int_s1 = []

lon_s2 = []
int_s2 = []


data_lamapra = np.loadtxt(open('data/dia 2/lampara_halogena_t_2100.csv').readlines()[:-1], skiprows = 33, delimiter = ';')

lon_s2.append(data_lamapra[:-50,0])
int_s2.append(data_lamapra[:-50,1])

for i in laminado:
    for j in set_up:
        if j == 's1':
            try:
                data =  np.loadtxt(open("data/dia 2/"+j + '_e' + i + '_t_07' + '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
                lon_s1.append(data[:-50,0])
                int_s1.append(data[:-50,1])
            except:
                pass
        elif j == 's2':
            try:
                data =  np.loadtxt(open("data/dia 2/"+j + '_e' + i + '_t_2100' + '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
                lon_s2.append(data[:-50,0])
                int_s2.append(data[:-50,1])
            except:
                pass
#%%

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12, 10), sharex = True)
fig.tight_layout()
for i in range(len(int_s1)):
    ax1.plot(lon_s1[i], int_s1[i], label  = i)
    ax2.plot(lon_s2[i], int_s2[i], label = i)
ax1.legend()
ax2.legend()
            
#%%

absorbancia_s1 = []
absorbancia_s2 = []


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12, 10), sharex = True)
fig.tight_layout()
for i in range(len(int_s1)-1):
    absorbancia_s1.append(-np.log10(int_s1[i+1]/int_s1[0]))
    absorbancia_s2.append(-np.log10(int_s2[i+1]/int_s2[0]))
    #ax1.plot(frecs_s1[i+1], int_s1[i+1]/int_s1[0], label = i+1)
    #ax2.plot(frecs_s2[i+1], int_s2[i+1]/int_s2[0], label = i+1)
    
    ax1.plot(lon_s1[i+1], absorbancia_s1[i], label = i+1)
    ax2.plot(lon_s2[i+1], absorbancia_s2[i], label = i+1)
    
# ax1.set_xlim([100, 900])
# ax2.set_xlim([100, 900])

ax1.legend()
ax2.legend()
plt.show()

#%% Funcion que calcula el Moving Average 

def moving_average(x, n):
    return np.convolve(x, np.ones(n), mode="same")/n

# def moving_std(x, n):
#     MA = moving_average(x)
#     MSTD = []
#     for mean in MA:
#         MSTD.append(np.std(x[]))

#%% Moving Average de la absorbancia para eliminar un poco el ruido
MA_abs_s1 = []
MA_abs_s2 = []
for i in range(len(absorbancia_s1)):
    MA_abs_s1.append(moving_average(absorbancia_s1[i], 30))
    MA_abs_s2.append(moving_average(absorbancia_s2[i], 30))

#%% Gráficos lindos de la absorbancia, tratando de copiar a merlen2009

from matplotlib import rc

colores = ["#03071e","#370617","#6a040f","#9d0208","#d00000","#dc2f02","#e85d04","#f48c06","#faa307","#ffba08"]


labels = {"fontname":"Times New Roman", "fontsize":15}
todo = {"family": "sans-serif", "sans-serif":["Times New Roman"]}
rc("font", **todo)

fig, ax = plt.subplots(1, figsize=(6,10))

for i, absorb in enumerate(MA_abs_s1):
    color = colores[2*i]
    ax.plot(lon_s1[i], absorb, color=color, label=laminado[i+1]+" s")
    
ax.set_yticks(np.arange(0, 0.55, 0.05))
ax.set_ylim([0, 0.5])
ax.grid(alpha=0.5)
ax.set_xlabel("Longitud de onda [nm]", **labels)
ax.set_ylabel("Absorbancia", **labels)
ax.set_xlim([400, 800])
ax.legend(fontsize=15)
ax.tick_params(labelsize=12)
plt.savefig("../../imagenes_informe/espectro/absorbancias")
plt.show()


