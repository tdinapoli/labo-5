from openpiv.piv import simple_piv
from openpiv import tools, pyprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cv2
from os import listdir

path_campos = '28-09/Glicerina_36/vaso_10cm/glitter_led_v2/campos_px_16/'    



def calcular_dist(x,xc,y,yc):
    dist = np.sqrt((x-xc)**2 +(y-yc)**2)
    filtro = dist<150
    return [dist[filtro], filtro]

def calcular_L(x, y, u, v):
    L = x*v - y*u
    return L

def calcular_L_muchos(txt): #txt con data se queda con algunos puntos del "centro" 
    x,y,u,v,type_ = txt.T
    
    dy = y[1]-y[0]
    dx = dy
    
    x_c = 336
    y_c = 336
    
    filtrox1 = (x > (x_c - 20*dx)) 
    filtrox2 = (x < (x_c +20*dx))
    filtroy1 = (y > (y_c - 20*dy))
    filtroy2 =  (y < (y_c +20*dy))
    
    x_rec = np.unique(x[filtrox1 & filtrox2])
    y_rec = np.unique(y[filtroy1 & filtroy2])
    
    Ls = np.zeros(x_rec.shape[0]*y_rec.shape[0])
    vel_modulo = np.zeros(x_rec.shape[0]*y_rec.shape[0])

    posx = np.zeros(x_rec.shape[0]*y_rec.shape[0])
    posy = np.zeros(x_rec.shape[0]*y_rec.shape[0])
    index_max = len(x_rec)
    for indexx,xc in enumerate(x_rec):
        for indexy, yc in enumerate(y_rec):
            dist, filtro = calcular_dist(x, xc, y, yc)
            u_sum,v_sum =  np.sum(u[filtro]), np.sum(v[filtro])
            vel_modulo[len(x_rec)*indexx + indexy] += u_sum**2 + v_sum**2
            L = calcular_L((x[filtro]-xc), (y[filtro]-yc), u[filtro], v[filtro])
            Ls[len(x_rec)*indexx + indexy] += np.mean(L)
            posx[len(x_rec)*indexx + indexy] += xc
            posy[len(x_rec)*indexx + indexy] += yc
            #print(xc, yc)
       # print(indexx, index_max, np.mean(L))
        
    return vel_modulo, Ls, posx, posy


#%%

xtotales = []
ytotales = []
utotales = []
vtotales = []
vrtotales = []
vtitatotales = []
disttotales = []
xctotales = []
yctotales = []

path_campos = "../../28-09/Glicerina_36/vaso_15cm/Med2/campos/"

for i in range(59):
    nombre = 'PIVlab_'+ str(i+1).zfill(4)+'.txt'
    path = path_campos+nombre
    data = np.loadtxt(path, delimiter = ',',skiprows = 3) 
    
    x,y,u,v,type_ = data.T
    print(v)
    
    
    vels, L, posx, posy = calcular_L_muchos(data)
    minv = np.where(vels == min(vels))
    minL = np.where(L ==min(L))

    # fig, ax = plt.subplots(1, figsize  = (10,8))
    # ax.quiver(x,y,u,v, scale =150)
    # ax.plot(posx[minv], posy[minv], 'r.')
    # ax.plot(posx[minL], posy[minL], 'b.')
    # plt.show()
    
    xc = posx[minL][0]
    yc = posy[minL][0]
    dist = np.sqrt((x-xc)**2 + (y-yc)**2)
    vr = (x-xc) *u/dist + (y-yc)*v/dist
    vt = (x-xc) *v/dist - (y-yc)*u/dist
    

    xtotales = xtotales+ x.tolist()
    ytotales = ytotales+ y.tolist()
    utotales = utotales+ u.tolist()
    vtotales = vtotales+ v.tolist()
    vrtotales = vrtotales + vr.tolist()
    vtitatotales = vtitatotales + vt.tolist()
    disttotales = disttotales + dist.tolist()
    xctotales = xctotales + [xc] * len(x)
    yctotales = yctotales + [yc] * len(x)
    
    
todo  =np.zeros((len(x)*59,9))
todo[:,0] = xtotales
todo[:,1] = ytotales
todo[:,2] =  utotales
todo[:,3] =  vtotales
todo[:,4] = vrtotales
todo[:,5] = vtitatotales
todo[:,6] = disttotales
todo[:,7] = xctotales
todo[:,8] = yctotales
np.savetxt('../../28-09/Glicerina_36/vaso_15cm/Med2/campos/todo_v2_15cm.txt', todo)

#%%

x, y, u, v, vr, vt, dist, xc, yc= np.loadtxt('../../28-09/Glicerina_36/vaso_10cm/glitter_led_v2/campos_px_16/todo_v2_10cm.txt').T

#%%

dists_plot = np.arange(0, 405, 5)
vts_plot = []
vrs_plot = []

for i in range(1, len(dists_plot)):
    dist_min = dists_plot[i-1]
    dist_max = dists_plot[i]
    filtro1 = dist > dist_min
    filtro2 = dist < dist_max
    filtro = filtro1 & filtro2
    filtro = filtro.reshape(vt.shape)
    
    vts_plot.append(np.nanmean(vt[filtro]))
    vrs_plot.append(np.nanmean(vr[filtro]))


vts_plot = np.array(vts_plot)
plt.plot(dists_plot[:-1], np.abs(vts_plot))
#plt.plot(dists_plot[:-1], np.abs(vrs_plot))



#%%
