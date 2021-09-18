_#!/usr/bin/env python3
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


save_path = "../../14-9/videos 14-09/experimento 1/campo0/"
winsize = 16
searchsize = 16
overlap=8
fps = 72
dt = 1/fps
percentile = 15
frames_urls = sorted(listdir(frames_path))

def frames_a_campos(frames_path, save_path, winsize=16, searchsize = 16,
                overlap=8, cantidad=72, dt = 1/72, percentile = 15):
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
    
    for index in range(1, cantidad):
        number = str(index).zfill(3)
        save_path = f"../../14-9/videos 14-09/experimento 1/campo0/par{number}.txt"
        frame_a_url = frames_path+frames_urls[index-1]
        frame_b_url = frames_path+frames_urls[index]
        frame_a = tools.imread(frame_a_url)
        frame_b = tools.imread(frame_b_url)
        
        process(frame_a, frame_b, winsize, searchsize, dt, overlap, save_path, percentile)
    
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

def dist_al_centro(imgpath, filepath, savepath):
    
    x, y = encontrar_centro(imgpath)
    
    ordered_paths = sorted(listdir(filepath))[1:]
    
    for index, file in enumerate(ordered_paths):
        number = str(index).zfill(3)
        con_dist = np.loadtxt(filepath+file)
        print(con_dist.shape, np.zeros((con_dist.shape[0],1)).shape)
        con_dist = np.hstack((con_dist, np.zeros((con_dist.shape[0],1))))
        print(con_dist.shape)
        con_dist[:,5] = np.sqrt((con_dist[:,0] - x)**2 + (con_dist[:,1]-y)**2)
        np.savetxt(savepath+f"{number}.txt",con_dist)
        break
    
    return

imgpath = "../../14-9/videos 14-09/experimento 1/frames0/000.jpg"
filepath = '../../14-9/videos 14-09/experimento 1/campo0/'
savepath = '../../14-9/videos 14-09/experimento 1/campo0/'
dist_al_centro(imgpath, filepath, savepath)

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


    
    coords = [x, y]
    print("coords",coords)
    
    plt.imshow(img, origin="upper")
    plt.plot(x, y, "ro")
    plt.show()
    return coords

#%%

ar = np.loadtxt('../../14-9/videos 14-09/experimento 1/campo0/000.txt')
a = np.linspace(0, 800, 81)
vel_mag = np.sqrt(ar[:,2]**2 + ar[:,3]**2)
vel_mag_nuevo = []
for i in range(1, len(a)):
    filtro1 = vel_mag > a[i-1]
    filtro2 = vel_mag < a[i]
    filtro = filtro1 & filtro2
    print(filtro, a[i])
    vel_mag_nuevo.append(np.mean(vel_mag[filtro]))
    

plt.plot(a[:-1], vel_mag_nuevo, '.b')
plt.show()


#%%
tools.display_vector_field(
    '../../14-9/videos 14-09/experimento 1/campo0/mean.txt',
    scaling_factor=96.52,
    scale=6000, # scale defines here the arrow length
    width=0.0015, # width is the thickness of the arrow
    on_img=True, # overlay on the image
    image_name="../../14-9/videos 14-09/experimento 1/frames0/000.jpg",
    window_size=winsize
)




