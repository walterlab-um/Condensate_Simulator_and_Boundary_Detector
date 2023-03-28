import os
import numpy as np
from numpy.random import normal, rand, poisson
import matplotlib.pyplot as plt
from tifffile import imwrite
from scipy.ndimage import gaussian_filter
from scipy.stats.qmc import Sobol
from rich.progress import track

##################################
# Parameters
folder_save = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials"
os.chdir(folder_save)
## FOV parameters
fovsize = 2000  # unit: nm
truth_box_pxlsize = 10  # unit: nm
real_img_pxlsize = 100  # unit: nm, must be an integer multiple of truth_box_pxlsize
# To guarantee the balance properties of Sobol sampling, N_fov = 2 ^ N_fov_base2
N_fov_base2 = 3

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


def plt_blue(img, fsave):
    plt.figure(dpi=300)
    # Contrast stretching
    vmin, vmax = np.percentile(img, (0.05, 99))
    plt.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
    plt.xlim(0, img.shape[0])
    plt.ylim(0, img.shape[1])
    plt.tight_layout()
    plt.axis("scaled")
    plt.axis("off")
    plt.savefig(fsave, format="png", bbox_inches="tight", dpi=300)
    plt.close()


#################################################
# Step 1: Analytical ground truth
x_nm = 927.6697531
y_nm = 898.5484143
r_nm = 506.25
C_condense = 9.0546875
C_dilute = 1  # N.A. unit


#################################################
# Step 2: Ground truth high-resolution volume "image"
fovsize_truth = int(fovsize / truth_box_pxlsize)

# Make a box; minimal Z range for convolution should be r + 3*sigma_axial
r_truth = r_nm / truth_box_pxlsize
z_min = int(fovsize_truth / 2 - r_truth - 3 * sigma_axial)
z_max = int(fovsize_truth / 2 + r_truth + 3 * sigma_axial)
pxl_x, pxl_y, pxl_z = np.meshgrid(
    np.arange(fovsize_truth),
    np.arange(fovsize_truth),
    np.arange(z_min, z_max),
)

# Make a condensate mask
center_x_pxl = x_nm / truth_box_pxlsize
center_y_pxl = y_nm / truth_box_pxlsize
center_z_pxl = (fovsize / 2) / truth_box_pxlsize
r_pxl = r_nm / truth_box_pxlsize
distance_square = (
    (pxl_x - center_x_pxl) ** 2
    + (pxl_y - center_y_pxl) ** 2
    + (pxl_z - center_z_pxl) ** 2
)
condensate_mask = distance_square < r_pxl**2

# Make a truth box
truth_box = condensate_mask * C_condense + (1 - condensate_mask) * C_dilute
imwrite(
    "Fig2-1-truth-box.tif",
    truth_box.astype("uint16"),
    imagej=True,
    metadata={"axes": "ZYX"},
)

#################################################
# Step 3: simulated 'real' image

# Convolution with Gaussian approximation of PSF. Interference happens irrespective of optical magnification or pixel size and thus should be performed first.
box_PSFconvolved = gaussian_filter(
    truth_box, sigma=[sigma_lateral, sigma_lateral, sigma_axial]
)
imwrite(
    "Fig2-2-3d-PSF-convolved.tif",
    box_PSFconvolved.astype("uint16"),
    imagej=True,
    metadata={"axes": "ZYX"},
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
plt_blue(img_PSFconvolved, "Fig2-3-slicing-by-DOF.png")
imwrite(
    "Fig2-3-slicing-by-DOF.tif",
    img_PSFconvolved.astype("uint16"),
    imagej=True,
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
plt_blue(img_shrinked, "Fig2-4-downsampling.png")

gaussian_noise = normal(gaussian_noise_mean, gaussian_noise_sigma, img_shrinked.shape)
img_gaussian = img_shrinked + gaussian_noise
poisson_mask = poisson(img_gaussian)
img_final = img_gaussian + poisson_mask
plt_blue(img_final, "Fig2-5-added-noise-final.png")
