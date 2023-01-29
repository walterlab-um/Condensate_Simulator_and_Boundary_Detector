from os.path import join
import numpy as np
from numpy.random import rand
import pandas as pd
from tifffile import imwrite
from tkinter import filedialog as fd

##################################
# Parameters
fovsize = 5000  # unit: nm
real_img_pxlsize = 100  # unit: nm
truth_img_pxlsize = 10  # unit: nm
condensate_r_range = (100, 2000)  # unit: nm
pad_size = 200  # push condensates back from FOV edges. unit: nm
N_condensate = 1  # number of condensates per field of view
N_fov = 10  # number of total im
folder_save = fd.askdirectory()

##################################
# Main
# generate condensate radius
condensate_r = (
    rand(N_fov * N_condensate) * (condensate_r_range[1] - condensate_r_range[0])
    + condensate_r_range[0]
)
# make sure condensates are padded from FOV edges
coor_min = condensate_r + pad_size
coor_max = fovsize - condensate_r - pad_size
# generate condensate center coordinates
center_x = []
center_y = []
for current_min, current_max in zip(coor_min, coor_max):
    center_x.append(rand() * (current_max - current_min) + current_min)
    center_y.append(rand() * (current_max - current_min) + current_min)
# generate FOV index
index = [np.repeat(x, N_condensate) for x in range(N_fov)]
index = np.hstack(index)
# Wrap up and save ground truth
df_groundtruth = pd.DataFrame(
    {
        "FOVindex": index,
        "x": center_x,
        "y": center_y,
        "r": condensate_r,
        "r_min_nm": np.repeat(condensate_r_range[0], N_fov * N_condensate),
        "r_max_nm": np.repeat(condensate_r_range[1], N_fov * N_condensate),
        "FOVsize_nm": np.repeat(fovsize, N_fov * N_condensate),
        "padding_nm": np.repeat(pad_size, N_fov * N_condensate),
    },
    dtype=object,
)
path_save = join(folder_save, "groundtruth.csv")
df_groundtruth.to_csv(path_save, index=False)
# Generate truth image
for current_fov in index:
    df_current = df_groundtruth[df_groundtruth.FOVindex == current_fov]
    pxl_x, pxl_y = np.meshgrid(
        np.arange(int(fovsize / real_img_pxlsize)),
        np.arange(int(fovsize / real_img_pxlsize)),
    )
