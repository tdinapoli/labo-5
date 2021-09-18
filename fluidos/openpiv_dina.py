from openpiv.piv import simple_piv
from openpiv import tools, pyprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

frame_a = tools.imread("../../14-9/videos 14-09/experimento 1/frames0/000.jpg")
frame_b = tools.imread("../../14-9/videos 14-09/experimento 1/frames0/001.jpg")

#%%
plt.figure(figsize=(11,11))
plt.imshow(frame_a, cmap="Greys_r")
plt.show()
plt.figure(figsize=(11,11))
plt.imshow(frame_b, cmap="Greys_r")
plt.show()
#%%

x, y, u, v = simple_piv(frame_a, frame_b)
#%%


save_path = "../../14-9/videos 14-09/experimento 1/campo0/velocidad0.txt"
winsize = 16
searchsize = 16
overlap=8
fps = 72
dt = 1/fps
percentile = 10


u, v, s2n = pyprocess.extended_search_area_piv(frame_a.astype(np.int32),
                                               frame_b.astype(np.int32),
                                               window_size=winsize,
                                               overlap=overlap,
                                               dt=dt,
                                               search_area_size=searchsize,
                                               sig2noise_method="peak2peak")

x, y = pyprocess.get_coordinates(frame_a.shape, search_area_size=searchsize, overlap=overlap)

valid = s2n > np.percentile(s2n, 15)

tools.save(x, y, u, v, ~valid,"prueba.txt")

#%%

fig, ax = plt.subplots(figsize=(11,11))

rect = patches.Rectangle((0,0), width=winsize, height=winsize, edgecolor="red", facecolor="none")
ax.add_patch(rect)
tools.display_vector_field(
    'prueba.txt',
    ax=ax, scaling_factor=96.52,
    scale=7000, # scale defines here the arrow length
    width=0.0025, # width is the thickness of the arrow
    on_img=True, # overlay on the image
    image_name="../../14-9/videos 14-09/experimento 1/frames0/000.jpg",
)