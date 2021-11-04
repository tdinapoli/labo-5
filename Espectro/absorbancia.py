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


data_lamapra = np.loadtxt(open('lampara_halogena_t_2100.csv').readlines()[:-1], skiprows = 33, delimiter = ';')

lon_s2.append(data_lamapra[:,0])
int_s2.append(data_lamapra[:,1])

for i in laminado:
    for j in set_up:
        if j == 's1':
            try:
                data =  np.loadtxt(open(j + '_e' + i + '_t_07' + '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
                lon_s1.append(data[:,0])
                int_s1.append(data[:,1])
            except:
                pass
        elif j == 's2':
            try:
                data =  np.loadtxt(open(j + '_e' + i + '_t_2100' + '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
                lon_s2.append(data[:,0])
                int_s2.append(data[:,1])
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

