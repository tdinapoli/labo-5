import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as fp 
#%% Importo los datos de frecuencia e intensidad tanto para no saturado como para si saturado

elementos = ['hi', 'me', 'kr', 'he', 'ne', 'ar', 'so']
tipo  =  ['ns', 'ss']

lon_ns = []
int_ns = []

lon_ss = []
int_ss = []

for i in elementos:
    for j in tipo:
        if j == 'ns':
            data =  np.loadtxt(open("data/dia 1/"+i + '_' + j + '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
            lon_ns.append(data[:,0])
            int_ns.append(data[:,1])
        else:
            data =  np.loadtxt(open("data/dia 1/"+i + '_' + j + '.csv').readlines()[:-1], skiprows = 33, delimiter = ';')
            lon_ss.append(data[:,0])
            int_ss.append(data[:,1])


#%% Cargo las lineas espectrales teoricas del NIST

# hi_lines = np.array([6562.8518, 6562.7110, 4861.3615, 6562.7248, 4861.2786, 4340.462])/10
# me_lines = np.array([2052.828, 2262.223, 2536.517, 2847.675, 2967.280, 3650.153, 3983.931, 4358.328, 5460.735, 5677.105, 6149.475, 7944.555])*0.1
# kr_lines = np.array([3718.02, 3778.046, 3783.095,4057.037, 4065.128, 4088.337, 4273.9694, 4292.923, 4355.477, 4739.002, 5870.9160, 8059.5048, 8104.3655,  8112.9012	, 8190.0566, 8263.2426, 8281.0522, 8298.1099, 8508.8728, 8776.7505, 8928.6934, 9293.82, 9577.52, 9751.7610, 9803.14])*0.1
# he_lines = np.array([3888.6456, 3888.6489, 4471.479, 5015.678, 5875.6148, 5875.6404, 5875.9663, 6678.1517, 7065.1771,7281.35 ])  *0.1
# ne_lines = np.array([5852.4879, 5881.8952, 6029.9969,6074.3377,  6143.0626, 6163.5939, 6217.2812, 6266.4950,6334.4278, 6382.9917, 6402.248, 6506.5281, 6598.9529, 6929.4673, 7024.0504, 7032.4131, 7173.9381, 7245.1666, 7488.8712,7535.7741, 8300.3258, 8495.3598 ])*0.1
# ar_lines = np.array([6965.431, 7067.218, 7383.980, 7503.869, 7514.652, 7635.106, 7723.761, 7724.207, 7948.176, 8006.157, 8014.786, 8103.693, 8115.311, 8408.210, 8424.648, 9122.967, 9224.499, 9657.786])*0.1
# so_lines = np.array([4455.23, 5889.950, 5895.924, 8194.824])*0.1


hi_lines = np.array([6562.8518, 4861.3615, 6562.7248, 4861.2786])/10
me_lines = np.array([4358.328, 5460.735, 5677.105, 6149.475, 7944.555])*0.1
kr_lines = np.array([5870.9160, 7601.5457,7854.8234, 8059.5048, 8104.3655,  8112.9012	, 8190.0566, 8263.2426, 8281.0522, 8298.1099, 8508.8728, 8776.7505, 8928.6934])*0.1
he_lines = np.array([3888.6456, 3888.6489, 4471.479, 5015.678, 5875.6148, 5875.6404, 5875.9663, 6678.1517, 7065.1771,7281.35 ])  *0.1
ne_lines = np.array([5852.4879, 5881.8952, 5944.8342, 6029.9969,6074.3377,  6143.0626, 6163.5939, 6217.2812, 6266.4950,6334.4278, 6382.9917, 6402.248, 6506.5281, 6598.9529, 6929.4673, 7024.0504, 7032.4131, 7173.9381, 7245.1666, 7488.8712,7535.7741, 8300.3258,8377.6080, 8495.3598 ])*0.1
ar_lines = np.array([6965.431, 7067.218, 7272.936, 7383.980, 7503.869, 7514.652, 7635.106, 7723.761, 7724.207, 7948.176, 8006.157, 8014.786, 8103.693, 8115.311,  8264.522,	8408.210, 8424.648,8521.442, 8667.944, 9122.967, 9224.499, 9657.786])*0.1
so_lines = np.array([5889.950, 5895.924, 8194.824])*0.1

teo_lines = [hi_lines, me_lines, kr_lines, he_lines, ne_lines, ar_lines, so_lines]

#%% Caculo los picos de de cada uno de los datos y los appendeo a otro lista

lon_max = []
int_max = []
for i in range(len(elementos)):
    find_peks = fp(int_ns[i], height = 0.02)
    lon_max.append(lon_ns[i][find_peks[0]])
    int_max.append(int_ns[i][find_peks[0]])
    

#%% Ploteo los resultados en 7 figuras distintas
import matplotlib as mpl

COLOR = '#494949'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
font = {'family' : 'arial'}

colores = ["#e79194","#ed9598","#f3989c","#fe9fa3","#d88496","#c57790","#b16989"]
for i in range(len(elementos)):
    if i == 3:
        plt.figure(facecolor = 'w')
        plt.plot(lon_ns[i]-1.24, int_ns[i], label=elementos[i], color = COLOR)
        plt.plot(lon_max[i]-1.24, int_max[i],'o', color = colores[5], markersize = 4)
        for j in range(len(teo_lines[i])):
            plt.vlines(teo_lines[i][j], ymin = 0, ymax = 1.05,linestyle = 'dashed', color = '#ff744d', linewidth = 1, alpha = 0.7)
        plt.legend(('Espectro medido', 'Máximos locales','Lineas NIST'), fontsize=12)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlabel('Longitud de onda [nm]', fontsize = 14)
        plt.ylabel('Intensidad normalizada', fontsize = 14)
        #plt.savefig('argon_zoom.png', transparent = True,dpi = 1000)
        plt.show()
    
#%%Doblete del sodio
data =  np.loadtxt(open("data/dia 1/sodio_doblete.csv").readlines()[:-1], skiprows = 33, delimiter = ';')
lon_so = data[:,0]
int_so = data[:,1]
find_peks = fp(int_so, height = 0.02)
lon_max_so = lon_so[find_peks[0]]
int_max_so = int_so[find_peks[0]]

plt.figure(facecolor = 'w', figsize = (15,8))
plt.plot(lon_so-1.24, int_so, label = 'so', color = COLOR)
plt.plot(lon_max_so-1.24, int_max_so,'o', color = colores[5], markersize=4)
for i in range(len(teo_lines[-1])):
    plt.vlines(teo_lines[-1][i], ymin = 0, ymax = 1.05,linestyle = 'dashed', color = '#ff744d', linewidth = 2, alpha = 0.7)
plt.legend(('Espectro medido', 'Máximos locales','Lineas NIST'), fontsize=12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Longitud de onda [nm]', fontsize = 14)
plt.ylabel('Intensidad normalizada', fontsize = 14)
plt.xlim([585,594])
plt.ylim([-0.002, 0.18])

plt.grid(alpha = 0.5)
plt.savefig('sodio_zoom.png', transparent = True,dpi = 1000)
plt.show()
