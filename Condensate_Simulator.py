from os.path import join, dirname, realpath
import numpy as np
from numpy.random import normal, rand
import pandas as pd
from tifffile import imwrite
from scipy.ndimage import gaussian_filter
from skimage.util import img_as_uint, random_noise
from tkinter import filedialog as fd
from rich.progress import track

##################################
folder_save = dirname(realpath(__file__))
# FOV parameters
fovsize = 5000  # unit: nm
truth_img_pxlsize = 10  # unit: nm
real_img_pxlsize = 100  # unit: nm, must be an integer multiple of truth_img_pxlsize
N_fov = 100  # number of total im
# Condensate parameters
# condensate size follows Gaussian distribution
condensate_r_ave = 500  # average size of condensates, unit: nm
condensate_r_sigma = 50
pad_size = 200  # push condensates back from FOV edges. unit: nm
N_condensate = 1  # number of condensates per field of view
C_condensed = 5000
C_dilute = 100
# Microscope parameters
Numerical_Aperature = 1.4
emission_wavelength = 488  # unit: nm
sigma_PSF = 0.21 * emission_wavelength / Numerical_Aperature


#################################################
# Step 1: Analytical ground truth
# generate condensate radius
condensate_r = normal(
    loc=condensate_r_ave, scale=condensate_r_sigma, size=N_fov * N_condensate
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
        "r_ave_nm": np.repeat(condensate_r_ave, N_fov * N_condensate),
        "r_sigma_nm": np.repeat(condensate_r_sigma, N_fov * N_condensate),
        "FOVsize_nm": np.repeat(fovsize, N_fov * N_condensate),
        "padding_nm": np.repeat(pad_size, N_fov * N_condensate),
    },
    dtype=object,
)
path_save = join(folder_save, "groundtruth.csv")
df_groundtruth.to_csv(path_save, index=False)

# Generate truth image and simulated real image
for current_fov in track(index):
    # extract condensate ground truth for current FOV
    df_current = df_groundtruth[df_groundtruth.FOVindex == current_fov]
    fovsize_pxl = int(fovsize / truth_img_pxlsize)
    pxl_x, pxl_y = np.meshgrid(np.arange(fovsize_pxl), np.arange(fovsize_pxl))
    center_x_pxl = df_current.x_nm.to_numpy(float) / truth_img_pxlsize
    center_y_pxl = df_current.y_nm.to_numpy(float) / truth_img_pxlsize
    r_pxl = df_current.r_nm.to_numpy(float) / truth_img_pxlsize

    #################################################
    # Step 2: Ground truth high-resolution image

    # To calculate whether each pixel falls within condensate range for all pixels all at once, duplicate the center x, y, r into array stacks with (x,y) shape the same as the image and height as the number of condensates in this FOV; also duplicate pxl x, y array into the same shape
    array_center_x = np.tile(center_x_pxl.T, (fovsize_pxl, fovsize_pxl, 1))
    array_center_y = np.tile(center_y_pxl.T, (fovsize_pxl, fovsize_pxl, 1))
    array_r = np.tile(r_pxl.T, (fovsize_pxl, fovsize_pxl, 1))
    array_pxl_x = np.repeat(pxl_x[:, :, np.newaxis], N_condensate, axis=2)
    array_pxl_y = np.repeat(pxl_y[:, :, np.newaxis], N_condensate, axis=2)

    # Calculate boolean of inside condensate or not for all condensates (laysers of the array) and then stack to one image
    truth_mask = np.any(
        (array_pxl_x - array_center_x) ** 2 + (array_pxl_y - array_center_y) ** 2
        < array_r**2,
        axis=2,
    )
    img_truth = truth_mask * C_condensed + (1 - truth_mask) * C_dilute

    # Save ground truth, high-resolution image
    path_save = join(folder_save, "Truth-FOVindex-" + str(current_fov) + ".tif")
    imwrite(path_save, img_as_uint(img_truth), imagej=True)

    #################################################
    # Step 3: simulated 'real' image

    # Convolution with Gaussian approximation of PSF. Interference happens irrespective of optical magnification or pixel size and thus should be performed first.
    sigma_PSF_pxl = sigma_PSF / real_img_pxlsize
    img_PSFconvolved = gaussian_filter(img_truth, sigma=sigma_PSF_pxl)
    # Magnification adjustment. Re-adjust the high-res image back to practically low-res image by integration
    fovsize_pxl = int(fovsize / real_img_pxlsize)
    pxl_x, pxl_y = np.meshgrid(np.arange(fovsize_pxl), np.arange(fovsize_pxl))
    ratio = int(real_img_pxlsize / truth_img_pxlsize)
    lst_pxl_value = []
    for xx in np.arange(fovsize_pxl):
        for yy in np.arange(fovsize_pxl):
            lst_pxl_value.append(
                np.sum(
                    img_PSFconvolved[
                        ratio * xx : ratio * (xx + 1), ratio * yy : ratio * (yy + 1)
                    ]
                )
                / (ratio**2)
            )
    img_shrinked = np.array(lst_pxl_value, dtype="uint16").reshape(
        (fovsize_pxl, fovsize_pxl)
    )
    img_real = random_noise(img_shrinked, mode="poisson")

    path_save = join(folder_save, "Test-FOVindex-" + str(current_fov) + ".tif")
    imwrite(path_save, img_as_uint(img_real), imagej=True)
