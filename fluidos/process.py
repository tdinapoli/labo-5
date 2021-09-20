#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 17:53:14 2021

@author: dina
"""

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
video_path = "../../14-9/videos 14-09/experimento 1/velocidad0.avi"
frames_path = "../../14-9/videos 14-09/experimento 1/frames0/"
extraer_frames(video_path, frames_path, 100)

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
    img = tools.imread(img_path)
    width, height = img.shape
    img = img[int(0.4*width):int(0.6*width),int(0.4*height):int(0.6*height)]
    
    filtro = img[:,:] > 100
    filtro = filtro.reshape(img.shape)
    img[filtro] = 1
    img[~filtro] = 0
    
    print(img==1, (img==1).shape)
    print("where", np.mean(np.where(img==1)[0]), np.mean(np.where(img==1)[1]))
    x , y = np.mean(np.where(img==1)[1]), np.mean(np.where(img==1)[0])


    
    coords = np.array([x+int(0.4*width), y + int(0.4*height)])
    print("coords",coords)
    
    plt.imshow(img, origin="upper")
    plt.plot(x, y, "ro")
    plt.show()
    return coords


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
        
        con_dist = np.hstack((con_dist, np.zeros((con_dist.shape[0],6))))
        dist = np.sqrt((con_dist[:,0] - x_centro)**2 + (con_dist[:,1] - y_centro)**2)
        vel_mag = np.sqrt(con_dist[:,2]**2 + con_dist[:,3]**2)
        x_norm = x/dist
        y_norm = y/dist
        
        con_dist[:,5] += dist
        con_dist[:,6] += vel_mag
        con_dist[:,7] += x
        con_dist[:,8] += y
        con_dist[:,9] += x_norm * u + y_norm * v
        con_dist[:,10] += -y_norm * u + x_norm * v
        
        
        index = index + 1
        number = str(index).zfill(3)
        print(savepath+f"{number}.txt")
        np.savetxt(savepath+f"{number}.txt", con_dist)
    
    return

#%%
cantidad=72
save_path = "../../14-9/videos 14-09/experimento 1/campo0/"
winsize = 16
searchsize = 16
overlap=8
fps = 72
dt = 1/fps
percentile = 15


frames_a_campos(frames_path, save_path, winsize, searchsize, overlap, cantidad,dt, percentile)
#%%
imgpath = "../../14-9/videos 14-09/experimento 1/frames0/"
filepath = "../../14-9/videos 14-09/experimento 1/campo0/"
savepath = "../../14-9/videos 14-09/experimento 1/campo0/"
dist_al_centro(imgpath, filepath, savepath)

#%%

path = "../../14-9/videos 14-09/experimento 1/campo0/"

dists = np.array([])
vel_mags = np.array([])
vrs = np.array([])
vtitas = np.array([])
for index, file in enumerate(sorted(listdir(path))):
    
    x, y, u, v, mask, dist, vel_mag, xc, yc, vr, vtita = np.loadtxt(path+file).T
    
    dists = np.concatenate((dists, np.array(dist)))
    vel_mags = np.concatenate((vel_mags, np.array(vel_mag)))
    vrs = np.concatenate((vrs, vr))
    vtitas = np.concatenate((vtitas, vtita))
    

#%%





#%%

plt.figure(figsize=(13,8))
a = np.linspace(0, 450, 46*2)
vel_mag_nuevo = []
vel_r_nuevo = []
vel_tita_nuevo = []
for i in range(1, len(a)):
    filtro1 = dists > a[i-1]
    filtro2 = dists < a[i]
    filtro = filtro1 & filtro2 
    filtro = filtro.reshape(vel_mags.shape)
    vel_mag_nuevo.append(np.nanmean(vel_mags[filtro]))
    vel_r_nuevo.append(np.nanmean(vrs[filtro]))
    vel_tita_nuevo.append(np.nanmean(vtitas[filtro]))
    
#plt.plot(a[:-1], vel_mag_nuevo, '-r', linewidth=2)
#plt.plot(a[:-1], vel_r_nuevo, color='black', linewidth=2)
plt.plot(-1*np.array(vel_tita_nuevo[30:350]), '.b', linewidth=2)
plt.plot(np.array(vel_r_nuevo)[30:350], '.r')
plt.show()
#%%

fig, ax = plt.subplots(1, figsize=(13,8))
ax.plot(a[:-1], np.array(vel_tita_nuevo)/72, '-r', linewidth=2)
ax.set_xlim([0, 350])
ax.set_xlabel("Distancia al vórtice [px]")
ax.set_ylabel("Velocidad en tita [px/s]")

