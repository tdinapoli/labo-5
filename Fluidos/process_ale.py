from openpiv.piv import simple_piv
from openpiv import tools, pyprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cv2
from os import listdir


#%%

#%cd ~/facultad/labo_5/git_labo_5/fluidos

#%%
#funcion para extraer los frames de un video

def extraer_frames(video_path, frames_path, n_frames):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success and count < n_frames:
        number = str(count).zfill(3)
        cv2.imwrite(frames_path+f"{number}.jpg", image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success, f"frame{number}.jpg")
        count += 1

#%%


#%%

def process(frame_a, frame_b, winsize, searchsize, dt, overlap, save_path, percentile):
    u, v, s2n = pyprocess.extended_search_area_piv(frame_a.astype(np.int32),
                                                   frame_b.astype(np.int32),
                                                   window_size=winsize,
                                                   overlap=overlap,
                                                   dt=dt,
                                                   search_area_size=searchsize,
                                                   sig2noise_method="peak2peak")
    
    x, y = pyprocess.get_coordinates(frame_a.shape, search_area_size=searchsize, overlap=overlap)
    
    valid = s2n > np.percentile(s2n, percentile)

    tools.save(x, y, u, v, ~valid, save_path)
    
    return
   
#%%




def frames_a_campos(frames_path, save_path, winsize=16, searchsize = 16,
                overlap=8, cantidad=int(72), dt = 1/72, percentile = 15):
    '''
    Convierte todos los frames ubicados en una carpeta a archivos txt con el 
    campo de velocidades entre cada uno de los frames con nombre 000.txt para
    el primero, 001.txt para el segundo... etc.

    Parameters
    ----------
    frames_path : str
        Ubicacion de las imágenes.
    save_path : str
        Ubicación donde se guardarán los txt.
    winsize : int, optional
        Window size de openpiv. The default is 16.
    searchsize : int, optional
        Searchsize de openpiv. The default is 16.
    overlap : int, optional
        Overlap de openpiv. The default is 8.
    cantidad : int, optional
        Cantidad de frames a analizar. The default is 72.
    dt : float, optional
        dt de openpiv. The default is 1/72.
    percentile : int, optional
        Percentil para la máscara de openpiv. The default is 15.

    Returns
    -------
    None.

    '''
    frames_urls = sorted(listdir(frames_path))
    for index in range(1, cantidad):
        number = str(index).zfill(3)
        frame_a_url = frames_path+frames_urls[index-1]
        frame_b_url = frames_path+frames_urls[index]
        frame_a = tools.imread(frame_a_url)
        frame_b = tools.imread(frame_b_url)
        
        process(frame_a, frame_b, winsize, searchsize, dt, overlap, save_path+f"{number}.txt", percentile)
    
    return



#%%

def mean_vector_field(filepath):
    '''
    Genera el promedio de varios campos vectoriales y lo guarda en un archivo 
    mean.txt en el mismo directorio.
    
    Parameters
    ----------
    filepath : str
        El path donde estan guardados los archivos .txt con los valores de los vectores.

    Returns
    -------
    mean : numpy array
        Array de el nuevo campo vectorial.

    '''
    ordered_paths = sorted(listdir(filepath))
    mean = []
    
    for file in ordered_paths:
        mean.append(np.loadtxt(filepath+file))
    mean = np.array(mean)
    print(mean.shape, mean[0,:,:].shape)
    
    mean = np.nanmean(mean, axis=0)
    mean[:,-1] = np.round(mean[:,-1]).astype("bool")
    x, y, u, v, mask = mean.T
    tools.save(x,y,u,v,mask, filepath+"mean.txt")
    return mean
#%%

def encontrar_centro(img_path):
    
    return 




#%%

def dist_al_centro(imgpath, filepath, savepath):
    
    ordered_paths = sorted(listdir(filepath))
    ordered_frames = sorted(listdir(imgpath))
    
    for index, file in enumerate(ordered_paths):
        frame = ordered_frames[index]
        x_centro, y_centro = encontrar_centro(imgpath+frame)
        
        con_dist = np.loadtxt(filepath+file)
        x = con_dist[:,0]
        y = con_dist[:,1]
        u = con_dist[:,2]
        v = con_dist[:,3]
        
        con_dist = np.hstack((con_dist, np.zeros((con_dist.shape[0],7))))
        dist = np.sqrt((x - x_centro)**2 + (y - y_centro)**2)
        #vel_mag = np.sqrt(con_dist[:,2]**2 + con_dist[:,3]**2)
        vel_mag =  np.sqrt(u**2 + v**2)
        x_norm = (x-x_centro)/dist
        y_norm = (y-y_centro)/dist
        
        con_dist[:,5] += dist
        con_dist[:,6] += vel_mag
        con_dist[:,7] += x
        con_dist[:,8] += y
        
        con_dist[:,9] += (x-x_centro)*u/dist + (y-y_centro)*v/dist
        con_dist[:,10]+= (x-x_centro)*v/dist - (y-y_centro)*u/dist
        
        vr = con_dist[:,9] 
        vt = con_dist[:,10] 
        con_dist[:, 11] += np.sqrt(vr**2 + vt**2)
        #con_dist[:,9] += x_norm * u + y_norm * v
        #con_dist[:,10] += -y_norm * u + x_norm * v
        
        
        index = index + 1
        number = str(index).zfill(3)
        #print(savepath+f"{number}.txt")
        np.savetxt(savepath+f"{number}.txt", con_dist)
    return


#%%
def ran(x,a,b):
    y = []
    for i in range(len(x)):
        if x[i]<= a:
            y.append(x[i]/a**2)
        else:
            y.append(1/x[i])
            
    return np.array(y)*b    

def bur(x, c, b):
    y  = np.zeros(len(x))
    for i in range(len(x)):
        y[i] += b*(1 - np.e**(-x[i]**2 /c**2)) /x[i]
    return y

def bur_vr(r, c, nu):
    vr = -2*nu*r/c**2
    return vr

#%%

def calcular_rotor(data):
    x, y, u, v = data.T
    
    rotorX = np.zeros(len(y))
    rotorY = np.zeros(len(y))
    
    dx = x[1] - x[0]
    dy = dx
    
    for i in range(1, len(y) - 1):
        rotorX[i] = (v[i+1] - v[i-1])/(2*dx)
        rotorY[i] = (u[i+1] - u[i-1])/(2*dy)
    
    mag_rotor = np.sqrt(rotorX**2 + rotorY**2)
    indice_max_rot = np.where(mag_rotor == max(mag_rotor))
    
    
    return [x[indice_max_rot], y[indice_max_rot]]

#%%
cantidad=72
save_path = "fluidos dia 1//campo0/"
winsize = 16
searchsize = 16
overlap=8   
fps = 72
dt = 1/fps
percentile = 15


frames_a_campos(frames_path, save_path, winsize, searchsize, overlap, cantidad,dt, percentile)
#%%
imgpath = "fluidos dia 1/frames0/"
filepath = "fluidos dia 1/campo0/"
savepath = "fluidos dia 1/campo0/"
dist_al_centro(imgpath, filepath, savepath)

#%% Extracción de las distintas velocidades a partir de los txt

path = "fluidos dia 1/campo0/"

dists = np.array([])
vel_mags = np.array([])
vrs = np.array([])
vtitas = np.array([])
vel_mag_rad = np.array([])

for index, file in enumerate(sorted(listdir(path))):
    
    x, y, u, v, mask, dist, vel_mag, xc, yc, vr, vtita,vel_mag_r = np.loadtxt(path+file).T
    
    dists = np.concatenate((dists, np.array(dist)))
    vel_mags = np.concatenate((vel_mags, np.array(vel_mag)))
    vrs = np.concatenate((vrs, vr))
    vtitas = np.concatenate((vtitas, vtita))
    vel_mag_rad =  np.concatenate((vel_mag_rad, vel_mag_r))


#%% Caluclo de las velocidades promedios en anillos centrados en el vortice

import seaborn as sns

plt.figure(figsize=(13,8))
a = np.linspace(0, 450, 46*2)
vel_mag_nuevo = []
vel_r_nuevo = []
vel_tita_nuevo = []
vel_mag_rad_nuevo = []
hists = []
binn = []
for i in range(1, len(a)):
    filtro1 = dists > a[i-1]
    filtro2 = dists < a[i]
    filtro = filtro1 & filtro2 
    filtro = filtro.reshape(vel_mags.shape)
    vel_mag_nuevo.append(np.nanmedian(vel_mags[filtro]))
    vel_r_nuevo.append(np.nanmedian(vrs[filtro]))
    vel_tita_nuevo.append(np.nanmedian(vtitas[filtro]))
    vel_mag_rad_nuevo.append((np.nanmedian(vel_mag_rad[filtro])))
    # plt.figure()
    # hist = plt.hist(vtitas[filtro], bins = np.linspace(-600, 100, 50))
    # hists.append(hist)
    # binn.append(np.linspace(-600, 100, 50))
    # plt.show()
#%% Grafico de las velocidades en funcion de las distancias

#plt.plot(a[:-1], np.array(vel_mag_nuevo), '--r', linewidth=2, label = 'modulo velocidad')
#plt.plot(a[:-1], vel_mag_rad_nuevo, '.', color = 'darkgreen', linewidth = 2 ,markersize = 10,label = 'modulo calculado con vr y vo')

plt.plot(a[:-1], np.abs(np.array(vel_r_nuevo))/72, '.k', linewidth=2, label  = 'velocidad radial')
plt.plot(a[:-1],np.abs(np.array(vel_tita_nuevo))/72, '.b', linewidth=2, label = 'velocidad tita')
plt.xlabel("Distancia al vórtice [px]")
plt.ylabel("Velocidad [px/s]")
plt.legend()
plt.show()
#%% Grafico general de la velocidad en tita (sin promediar)

plt.figure(figsize = (10, 8))
#plt.plot(dists, np.abs(np.array(vrs)), '.k', linewidth=2, label  = 'velocidad radial')
plt.plot(dists, np.abs(np.array(vtitas)), '.b', linewidth=2, label = 'velocidad tita')

#%% Ajuste de la velocidad en tita por los dos modelos

from scipy.optimize import curve_fit
popt, pcov = curve_fit(ran, a[:-20], np.abs(vel_tita_nuevo[:-19]), p0 =[80,20000])
#popt2, pcov2 = curve_fit(bur, a[:-1], vel_mag_nuevo, p0 =[80,20000])

plt.figure(figsize = (10, 6))
plt.plot(a[:-1], np.abs(vel_tita_nuevo), 'b-', linewidth=2, label = 'Datos')
plt.plot(a[:-20], bur(a[:-20], *popt), color = 'lime',label = 'Burguers')
plt.plot(a[:-20], ran(a[:-20], *popt), color = 'orangered',label = 'Rankine')
plt.grid(alpha = 0.6)
plt.legend()
#%% Vel en tita una vez mas

fig, ax = plt.subplots(1, figsize=(13,8))
ax.plot(a[:-1], np.array(vel_tita_nuevo)/72, '-r', linewidth=2)
ax.set_xlim([0, 350])
ax.set_xlabel("Distancia al vórtice [px]")
ax.set_ylabel("Velocidad en tita [px/s]")
#%%

frames_path = "../../28-09/Glicerina_36/vaso_10cm/glitter_led/"
save_path = frames_path+"campos/"
frames_a_campos(frames_path, save_path)

#%%
data = np.loadtxt(save_path+"001.txt")
x_centro, y_centro = calcular_rotor(data[:,:4])
print(x_centro, y_centro)
x_centro = x_centro - 2*np.abs(x_centro - 672/2)
y_centro = y_centro + 2*np.abs(y_centro - 672/2)

fig, ax = plt.subplots(figsize=(11,11))
ax.plot(x_centro, y_centro, 'ok')

tools.display_vector_field(save_path+"020.txt", on_img=True,
                           image_name=frames_path+"019.jpg",
                           ax=ax, window_size=16)

