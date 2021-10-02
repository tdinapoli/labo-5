from openpiv.piv import simple_piv
from openpiv import tools, pyprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cv2
from os import listdir

path_campos = '28-09/Glicerina_36/vaso_10cm/glitter_led_v2/campos_px_16/'    


data0 = 'PIVlab_0001.txt'

data = np.loadtxt(path_campos+data0, delimiter = ',',skiprows = 3) # x, y, u, v, type

x,y,u,v,type_ = data.T

d = np.arange(0, 0.1, 0.005)
vel_mag = np.sqrt(u**2 + v**2)

# x_c = 0.0321
# y_c = 0.041
dist = np.sqrt((x-x_c)**2+ (y-y_c)**2)

vel_mag_prom = []
dist_prom = []

for i in range(len(d)):
    filtro1 = dist > d[i-1]
    filtro2 = dist < d[i]
    filtro = filtro1 & filtro2 
    filtro = filtro.reshape(vel_mag.shape)
    
    vel_mag_prom.append(np.nanmean(vel_mag[filtro]))
    dist_prom.append(np.nanmean(dist[filtro]))

plt.plot(dist_prom, vel_mag_prom, '.')


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

vels, L, posx, posy = calcular_L_muchos(data)
minv = np.where(vels == min(vels))
minL = np.where(L ==min(L))
print('velocidades ', posx[minv], posy[minv], '\n')
print('L           ', posx[minL], posy[minL], '\n')
#tools.display_vector_field(data)#, on_img = True, image_name = '28-09/Glicerina_36/vaso_10cm/glitter_led_v2/000.jpg')
#%%

posx_c = np.zeros(60)
posy_c = np.zeros(60)
xtotales = np.array([])
ytotales = np.array([])
utotales = np.array([])
vtotales = np.array([])
vrtoatles = np.array([])
vtitatotales = np.array([])
disttotales = np.array([])
xctotales = []
yctotales = []



for i in range(59):
    nombre = 'PIVlab_'+ str(i+1).zfill(4)+'.txt'
    path = path_campos+nombre
    data = np.loadtxt(path, delimiter = ',',skiprows = 3) 
    
    x,y,u,v,type_ = data.T
    
    
    vels, L, posx, posy = calcular_L_muchos(data)
    minv = np.where(vels == min(vels))
    minL = np.where(L ==min(L))
    
    posx_c[i] +=  posx[minL]
    posy_c[i] +=  posy[minL]
    
    # fig, ax = plt.subplots(1, figsize  = (10,8))
    # ax.quiver(x,y,u,v, scale =150)
    # ax.plot(posx[minv], posy[minv], 'r.')
    # ax.plot(posx[minL], posy[minL], 'b.')
    # plt.show()
    
    xc = posx[minL]
    yc = posy[minL]
    dist = np.sqrt((x-xc)**2 + (y-yc)**2)
    vr = (x-xc) *u/dist + (y-yc)*v/dist
    vt = (x-xc) *v/dist - (y-yc)*u/dist
    
    xtotales = np.concatenate(xtotales, np.array(x))
    ytotales = np.concatenate(ytotales, np.array(y))
    utotales = np.concatenate(utotales, u) 
    vtotales = np.concatenate(vtotales, v)
    vrtoatles = np.cpmcatemate(vrtoatles, vr)
    vtitatotales = np.concatenate(vtitatotales, vt)
    disttotales = np.concatenate(disttotales, dist)
    xctotales = np.concatenate(xctotales, [xc]*len(x))
    yctotales = np.concatenate(yctotales, [yc]*len(y))
    
todo  =np.zeros((len(x)*59,9))
todo[:,0] = xtotales
todo[:,1] = ytotales
todo[:,2] =  utotales
todo[:,3] =  vtotales
todo[:,4] = vrtoatles
todo[:,5] = vtitatotales
todo[:,6] = disttotales
todo[:,7] = xctotales
todo[:,8] = yctotales
np.savetxt('28-09/Glicerina_36/vaso_10cm/glitter_led_v2/todo_v2_10cm.txt', todo)

#%%





#%%
plt.figure(figsize =(10,6))
plt.plot(posx_c[:-1], posy_c[:-1], 'bo')
plt.plot(posx_c[-1], posy_c[-1], 'ro')