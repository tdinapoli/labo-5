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
plt.show()
            
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
    return MSTD

#%% Moving Average de la absorbancia para eliminar un poco el ruido
MA_abs_s1 = []
MA_abs_s2 = []
MSTD_abs_s1 = []
MSTD_abs_s2 = []
for i in range(len(absorbancia_s1)):
    MA_abs_s1.append(moving_average(absorbancia_s1[i], 30))
    MA_abs_s2.append(moving_average(absorbancia_s2[i], 30))
    MSTD_abs_s1.append(moving_std(absorbancia_s1[i], 30))
    MSTD_abs_s2.append(moving_std(absorbancia_s2[i], 30))

#%% Gráficos lindos de la absorbancia, tratando de copiar a merlen2009

from matplotlib import rc

#colores = ["#03071e","#370617","#6a040f","#9d0208","#d00000","#dc2f02","#e85d04","#f48c06","#faa307","#ffba08"]
#colores = ["#f72585","#b5179e","#7209b7","#560bad","#480ca8","#3a0ca3","#3f37c9","#4361ee","#4895ef","#4cc9f0"]
colores = ["#c29b92","#d29182","#da8267","#e37750","#ea6f3a","#f36828","#b45124","#773518","#3b180d","#010005"]

labels = {"fontname":"Times New Roman", "fontsize":15}
todo = {"family": "sans-serif", "sans-serif":["Times New Roman"]}
rc("font", **todo)

fig, ax = plt.subplots(1, figsize=(6,10))

for i, absorb in enumerate(MA_abs_s1):
    color = colores[-i-1]
    ax.plot(lon_s1[i], absorb, color=color, label=laminado[i+1]+" s")
    ax.fill_between(lon_s1[i], absorb + MSTD_abs_s1[i], absorb - MSTD_abs_s1[i],
                    color=color, alpha=0.2, linewidth=0)
    # ax.plot(lon_s1[i], absorb + MSTD_abs_s1[i], linestyle="solid", color=color)
    # ax.plot(lon_s1[i], absorb - MSTD_abs_s1[i], linestyle="solid", color=color)
    #break
    
ax.set_yticks(np.arange(0, 0.55, 0.05))
ax.set_ylim([0, 0.5])
ax.grid(alpha=0.5)
ax.set_xlabel("Longitud de onda [nm]", **labels)
ax.set_ylabel("Absorbancia", **labels)
ax.set_xlim([400, 1000])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=15, loc="upper left", framealpha=1)
ax.tick_params(labelsize=12)
plt.savefig("../../imagenes_informe/espectro/absorbancias")
plt.show()


#%% DIA 3

lados = ["ll", "le"]

lons = [[], []]
ints = [[], []]

lamp_sola = np.loadtxt(open("data/dia 3/lampara_halogena_sin_nada_dia_3.csv").readlines()[:-1],
                       skiprows=33, delimiter=";")
lons[0].append(lamp_sola[:-50, 0])
lons[1].append(lamp_sola[:-50, 0])
ints[0].append(lamp_sola[:-50, 1])
ints[1].append(lamp_sola[:-50, 1])

for i in laminado[1:]:
    for j, lado in enumerate(lados):
        data = np.loadtxt(open(f"data/dia 3/{i}_{lado}.csv").readlines()[:-1], skiprows=33, delimiter=";")
        lons[j].append(data[:-50, 0])
        ints[j].append(data[:-50, 1])

lon_ll, lon_le = lons
int_ll, int_le = ints

#%%

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12, 8), sharex = True)
for i in range(len(int_ll)):
    ax1.plot(lon_ll[i], int_ll[i], label  = laminado[i])
    ax2.plot(lon_le[i], int_le[i], label = laminado[i])
    
ax1.set_ylabel("Intensidad plasmones mirando\n a la lámpara", fontsize=20)
ax2.set_ylabel("Intensidad plasmones mirando\n al espectrómetro", fontsize=20)
ax2.set_xlabel("Longitud de onda [nm]", fontsize=20)
plt.subplots_adjust(hspace=0)
fig.tight_layout()
ax1.grid()
ax2.grid()
ax1.legend()
ax2.legend()
plt.show()

#%%

absorbancia_ll = []
absorbancia_le = []


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12, 10), sharex = True)
fig.tight_layout()
for i in range(len(int_ll)-1):
    absorbancia_ll.append(-np.log10(int_ll[i+1]/int_ll[0]))
    absorbancia_le.append(-np.log10(int_le[i+1]/int_le[0]))
    #ax1.plot(frecs_s1[i+1], int_s1[i+1]/int_s1[0], label = i+1)
    #ax2.plot(frecs_s2[i+1], int_s2[i+1]/int_s2[0], label = i+1)
    
    ax1.plot(lon_ll[i+1], absorbancia_ll[i], label = i+1)
    ax2.plot(lon_le[i+1], absorbancia_le[i], label = i+1)
    
# ax1.set_xlim([100, 900])
# ax2.set_xlim([100, 900])

ax1.legend()
ax2.legend()
plt.show()

#%% Moving Average de la absorbancia para eliminar un poco el ruido

MA_abs_ll = []
MA_abs_le = []
MSTD_abs_ll = []
MSTD_abs_le = []
for i in range(len(absorbancia_ll)):
    MA_abs_ll.append(moving_average(absorbancia_ll[i], 30))
    MA_abs_le.append(moving_average(absorbancia_le[i], 30))
    MSTD_abs_ll.append(moving_std(absorbancia_ll[i], 30))
    MSTD_abs_le.append(moving_std(absorbancia_le[i], 30))

#%% absorbancias promediadas para ll

labels = {"fontname":"Times New Roman", "fontsize":15}
todo = {"family": "sans-serif", "sans-serif":["Times New Roman"]}
rc("font", **todo)

fig, ax = plt.subplots(1, figsize=(6,10))

for i, absorb in enumerate(MA_abs_ll):
    color = colores[-i-1]
    ax.plot(lon_ll[i], absorb, color=color, label=laminado[i+1]+" s")
    ax.fill_between(lon_ll[i], absorb + MSTD_abs_ll[i], absorb - MSTD_abs_ll[i],
                    color=color, alpha=0.2, linewidth=0)
    # ax.plot(lon_s1[i], absorb + MSTD_abs_s1[i], linestyle="solid", color=color)
    # ax.plot(lon_s1[i], absorb - MSTD_abs_s1[i], linestyle="solid", color=color)
    #break
    
ax.set_yticks(np.arange(0, 0.85, 0.05))
ax.set_ylim([0, 0.7])
ax.grid(alpha=0.5)
ax.set_xlabel("Longitud de onda [nm]", **labels)
ax.set_ylabel("Absorbancia", **labels)
ax.set_title("Sint. Día 2, medido día 3 LL", **labels)
ax.set_xlim([400, 1000])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=15, loc="upper left", framealpha=1)
ax.tick_params(labelsize=12)
plt.savefig("../../imagenes_informe/espectro/absorbancias")
plt.show()


#%% Absorbancias promediadas para le

labels = {"fontname":"Times New Roman", "fontsize":15}
todo = {"family": "sans-serif", "sans-serif":["Times New Roman"]}
rc("font", **todo)

fig, ax = plt.subplots(1, figsize=(6,10))

for i, absorb in enumerate(MA_abs_le):
    color = colores[-i-1]
    ax.plot(lon_le[i], absorb, color=color, label=laminado[i+1]+" s")
    ax.fill_between(lon_le[i], absorb + MSTD_abs_le[i], absorb - MSTD_abs_le[i],
                    color=color, alpha=0.2, linewidth=0)
    # ax.plot(lon_s1[i], absorb + MSTD_abs_s1[i], linestyle="solid", color=color)
    # ax.plot(lon_s1[i], absorb - MSTD_abs_s1[i], linestyle="solid", color=color)
    #break
    
ax.set_yticks(np.arange(0, 0.85, 0.05))
ax.set_ylim([0, 0.7])
ax.grid(alpha=0.5)
ax.set_xlabel("Longitud de onda [nm]", **labels)
ax.set_ylabel("Absorbancia", **labels)
ax.set_title("Sint. Día 2, medido día 3 LE")
ax.set_xlim([400, 1000])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=15, loc="upper left", framealpha=1)
ax.tick_params(labelsize=12)
plt.savefig("../../imagenes_informe/espectro/absorbancias")
plt.show()


#%% ABSORBANCIA MUESTRAS DIA 3 - importo los datos

lons_d3 = []
ints_d3 = []

lamp_sola = np.loadtxt(open("data/dia 3/lampara_halogena_sin_nada_dia_3.csv").readlines()[:-1],
                       skiprows=33, delimiter=";")
lons[0].append(lamp_sola[:-50, 0])
lons[1].append(lamp_sola[:-50, 0])
ints[0].append(lamp_sola[:-50, 1])
ints[1].append(lamp_sola[:-50, 1])

archivos_d3 = ["data/dia 3/2_sint3.csv", "data/dia 3/3_sint3.csv", "data/dia 3/8_sint3.csv"]

for archivo in archivos_d3:
    datos = np.loadtxt(open(archivo).readlines()[:-1], skiprows=33, delimiter=";")
    lons_d3.append(datos[:-50, 0])
    ints_d3.append(datos[:-50, 1])

#%% Intensidades muestras sint dia 3

fig, ax = plt.subplots(figsize=(10,10))
for i in range(len(lons_d3)):
    long = lons_d3[i]
    inten = ints_d3[i]
    ax.plot(long, inten)
plt.show()


#%% Absorbancias muestras sint dia 3

fig, ax = plt.subplots(figsize=(10,10))
for i in range(len(lons_d3)):
    long = lons_d3[i]
    inten = ints_d3[i]
    ax.plot(long, -np.log10(inten/ints_d3[0]))
plt.show()