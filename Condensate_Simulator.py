from os.path import join
import numpy as np
from numpy.random import rand
import pandas as pd
from tifffile import imwrite
from skimage.util import img_as_uint
from tkinter import filedialog as fd
from rich.progress import track

##################################
# Parameters
fovsize = 5000  # unit: nm
real_img_pxlsize = 100  # unit: nm
truth_img_pxlsize = 10  # unit: nm
condensate_r_range = (100, 300)  # unit: nm
pad_size = 200  # push condensates back from FOV edges. unit: nm
N_condensate = 10  # number of condensates per field of view
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
        "x_nm": center_x,
        "y_nm": center_y,
        "r_nm": condensate_r,
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
for current_fov in track(index):
    df_current = df_groundtruth[df_groundtruth.FOVindex == current_fov]
    fovsize_pxl = int(fovsize / truth_img_pxlsize)
    pxl_x, pxl_y = np.meshgrid(np.arange(fovsize_pxl), np.arange(fovsize_pxl))
    center_x_pxl = df_current.x_nm.to_numpy(float) / truth_img_pxlsize
    center_y_pxl = df_current.y_nm.to_numpy(float) / truth_img_pxlsize
    r_pxl = df_current.r_nm.to_numpy(float) / truth_img_pxlsize
    # To calculate whether each pixel falls within condensate range for all pixels all at once, duplicate the center x, y, r into array stacks with (x,y) shape the same as the image and height as the number of condensates in this FOV; also duplicate pxl x, y array into the same shape
    array_center_x = np.tile(center_x_pxl.T, (fovsize_pxl, fovsize_pxl, 1))
    array_center_y = np.tile(center_y_pxl.T, (fovsize_pxl, fovsize_pxl, 1))
    array_r = np.tile(r_pxl.T, (fovsize_pxl, fovsize_pxl, 1))
    array_pxl_x = np.repeat(pxl_x[:, :, np.newaxis], N_condensate, axis=2)
    array_pxl_y = np.repeat(pxl_y[:, :, np.newaxis], N_condensate, axis=2)
    # Calculate boolean of inside condensate or not for all condensates (laysers of the array) and then stack to one image
    img_truth = np.any(
        (array_pxl_x - array_center_x) ** 2 + (array_pxl_y - array_center_y) ** 2
        < array_r**2,
        axis=2,
    )
    path_save = join(folder_save, "Truth-FOVindex-" + str(current_fov) + ".tif")
    imwrite(path_save, img_as_uint(img_truth), imagej=True)
