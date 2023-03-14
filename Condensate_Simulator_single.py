from os.path import join, dirname, realpath
import numpy as np
from numpy.random import normal, rand, poisson
import pandas as pd
from tifffile import imwrite
from scipy.ndimage import gaussian_filter
from scipy.stats.qmc import Sobol
from skimage.util import random_noise, img_as_uint, img_as_float
from rich.progress import track

##################################
# Parameters
# folder_save = dirname(realpath(__file__))
folder_save = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection"
## FOV parameters
fovsize = 2000  # unit: nm
truth_box_pxlsize = 10  # unit: nm
real_img_pxlsize = 100  # unit: nm, must be an integer multiple of truth_box_pxlsize
N_fov = 10  # number of total im

## Imaging system parameters
# Microscope parameters
depth_of_focus = 500  # unit, nm
Numerical_Aperature = 1.5
refractive_index = 1.515
emission_wavelength = 520  # assuming Alexa488, unit: nm
# Noise parameters
gaussian_noise_mean = 400
gaussian_noise_sigma = 5  # white noise
# PSF approximations, Ref doi: 10.1364/AO.46.001819
k_em = (2 * np.pi) / emission_wavelength
sigma_lateral = np.sqrt(2) / (k_em * Numerical_Aperature)  # x and y
sigma_axial = (2 * np.sqrt(6) * refractive_index) / (
    k_em * Numerical_Aperature**2
)  # z
# convert values to pixels
sigma_lateral = sigma_lateral / truth_box_pxlsize
sigma_axial = sigma_axial / truth_box_pxlsize
depth_of_focus = depth_of_focus / truth_box_pxlsize

## Condensate parameters
# condensate size follows Gaussian distribution
condensate_r_range = (300, 500)  # average size of condensates, unit: nm
C_condensed = 600  # N.A. unit
C_dilute = 40


#################################################
# Step 1: Analytical ground truth
# generate condensate radius with Sobol Sampling
sampler = Sobol(d=1, scramble=False)
condensate_r = condensate_r_range[0] + sampler.random(N_fov) * (
    condensate_r_range[1] - condensate_r_range[0]
)

# push condensates back from FOV edges. unit: nm
condensate_center_range = (
    condensate_r_range[1],
    fovsize - condensate_r_range[1],
)

# generate condensate center coordinates
sampler = Sobol(d=2, scramble=False)
condensate_xy = condensate_center_range[0] + sampler.random(N_fov) * (
    condensate_center_range[1] - condensate_center_range[0]
)
center_x = condensate_xy[:, 0]
center_y = condensate_xy[:, 1]

# Wrap up and save ground truth
index = np.arange(N_fov)
df_groundtruth = pd.DataFrame(
    {
        "FOVindex": index,
        "x_nm": center_x,
        "y_nm": center_y,
        "r_nm": condensate_r,
        "C_dilute": np.repeat(C_dilute, N_fov),
        "C_condensed": np.repeat(C_condensed, N_fov),
        "FOVsize_nm": np.repeat(fovsize, N_fov),
    },
    dtype=object,
)
path_save = join(folder_save, "groundtruth.csv")
df_groundtruth.to_csv(path_save, index=False)

# Generate truth image and simulated real image
for current_fov in track(index):
    #################################################
    # Step 2: Ground truth high-resolution volume "image"
    df_current = df_groundtruth[df_groundtruth.FOVindex == current_fov]
    fovsize_truth = int(fovsize / truth_box_pxlsize)

    # Make a box; minimal Z range for convolution should be r + 3*sigma_axial
    r_truth = df_current["r_nm"].squeeze() / truth_box_pxlsize
    z_min = int(fovsize_truth / 2 - r_truth - 3 * sigma_axial)
    z_max = int(fovsize_truth / 2 + r_truth + 3 * sigma_axial)
    pxl_x, pxl_y, pxl_z = np.meshgrid(
        np.arange(fovsize_truth),
        np.arange(fovsize_truth),
        np.arange(z_min, z_max),
    )

    # Make a condensate mask
    center_x_pxl = df_current.x_nm.squeeze() / truth_box_pxlsize
    center_y_pxl = df_current.y_nm.squeeze() / truth_box_pxlsize
    center_z_pxl = (fovsize / 2) / truth_box_pxlsize
    r_pxl = df_current.r_nm.squeeze() / truth_box_pxlsize
    distance_square = (
        (pxl_x - center_x_pxl) ** 2
        + (pxl_y - center_y_pxl) ** 2
        + (pxl_z - center_z_pxl) ** 2
    )
    condensate_mask = distance_square < r_pxl**2

    # Make a truth box
    truth_box = condensate_mask * C_condensed + (1 - condensate_mask) * C_dilute

    #################################################
    # Step 3: simulated 'real' image

    # Convolution with Gaussian approximation of PSF. Interference happens irrespective of optical magnification or pixel size and thus should be performed first.
    box_PSFconvolved = gaussian_filter(
        truth_box, sigma=[sigma_lateral, sigma_lateral, sigma_axial]
    )

    # A slice with the thickness of depth of focus will generate an image
    z_middle = (z_max - z_min) / 2
    img_PSFconvolved = np.sum(
        box_PSFconvolved[
            :,
            :,
            int(z_middle - depth_of_focus / 2) : int(z_middle + depth_of_focus / 2) + 1,
        ],
        axis=2,
    )

    # Magnification adjustment. Re-adjust the high-res image back to practically low-res image by integration
    fovsize_real = int(fovsize / real_img_pxlsize)
    ratio = int(real_img_pxlsize / truth_box_pxlsize)
    lst_pxl_value = []
    for xx in np.arange(fovsize_real):
        for yy in np.arange(fovsize_real):
            lst_pxl_value.append(
                np.mean(
                    img_PSFconvolved[
                        ratio * xx : ratio * (xx + 1), ratio * yy : ratio * (yy + 1)
                    ]
                )
            )
    img_shrinked = np.array(lst_pxl_value).reshape((fovsize_real, fovsize_real))
    gaussian_noise = normal(
        gaussian_noise_mean, gaussian_noise_sigma, img_shrinked.shape
    )
    img_gaussian = img_shrinked + gaussian_noise
    poisson_mask = poisson(img_gaussian)
    img_final = img_gaussian + poisson_mask

    path_save = join(folder_save, "FOVindex-" + str(current_fov) + ".tif")
    imwrite(
        path_save,
        img_final.astype("uint16"),
        imagej=True,
    )
