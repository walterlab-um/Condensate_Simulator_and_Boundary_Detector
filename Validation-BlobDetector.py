from tifffile import imread
import os
import math
import cv2
import pandas as pd
import numpy as np
from rich.progress import track

#################################################
# Inputs
real_img_pxlsize = 100  # unit: nm, must be an integer multiple of truth_img_pxlsize
fovsize = 5000  # unit: nm
gaussian_noise_mean = 400
Nsigma = 1  # boundary will be Nsigma * sigmax/y, use 2.355 for FWHM

folder = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Simulated-4096"
)
os.chdir(folder)

path_results = "Method-3-GaussFit/GaussFit.csv"
path_groundtruth = "groundtruth.csv"


#################################################
# Functions
def when_failed():
    global success, rmsd_center, rmsd_edge, area_fold_deviation, fold_deviation_pc, fold_deviation_pc_max
    success.append(False)
    deviation_center.append(np.nan)
    rmsd_edge.append(np.nan)
    area_fold_deviation.append(np.nan)
    fold_deviation_pc.append(np.nan)
    fold_deviation_pc_max.append(np.nan)


def generate_mask(imgshape, cx, cy, r_mean):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # fill in the circle
    xx, yy = np.meshgrid(np.arange(imgshape[0]), np.arange(imgshape[1]))
    for xx in np.arange(imgshape[0]):
        for yy in np.arange(imgshape[1]):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= r_mean:
                mask[xx, yy] = 1
                # mask[yy, xx] = 1

    return mask


#################################################
# Main
df_results = pd.read_csv(path_results, dtype=float)
df_truth = pd.read_csv(path_groundtruth, dtype=float)

lst_index = df_truth.FOVindex.to_numpy(dtype=float)
lst_truth_r = []
lst_truth_pc = []  # partition coefficient
success = []
deviation_center = []
rmsd_edge = []
area_fold_deviation = []  # with sign
# difference in partition coefficient, assume outside C_dilute=1, so directly the average intensity inside, with sign
fold_deviation_pc = []
fold_deviation_pc_max = []
for index in track(lst_index):
    # retreive ground truth
    row = df_truth[df_truth.FOVindex == index]
    truth_x_nm = row["x_nm"].squeeze()
    truth_y_nm = row["y_nm"].squeeze()
    truth_r_nm = row["r_nm"].squeeze()
    truth_pc = row["C_condensed"].squeeze()
    lst_truth_r.append(truth_r_nm)
    lst_truth_pc.append(truth_pc)

    df_current = df_results[df_results["index"] == index]
    if df_current.shape[0] == 0:
        when_failed()
        continue
    elif df_current.shape[0] > 1:
        when_failed()
        continue

    # deviation of condensate center, note x and y must flip!
    cx = df_current.centery.squeeze() * real_img_pxlsize
    cy = df_current.centerx.squeeze() * real_img_pxlsize
    rx = df_current.sigmax.squeeze() * real_img_pxlsize * Nsigma
    ry = df_current.sigmay.squeeze() * real_img_pxlsize * Nsigma
    r_mean = (rx + ry) / 2
    d2center = np.sqrt((cx - truth_x_nm) ** 2 + (cy - truth_y_nm) ** 2)

    # distance to edge defined as differece to R
    sample_x = np.linspace(
        math.nextafter(cx - r_mean, cx), math.nextafter(cx + r_mean, cx), 10
    )  # bugfix! nextafter to avoid infinitely small negative number!
    sample_y = np.concatenate(
        (
            cy + np.sqrt(r_mean**2 - (sample_x - cx) ** 2),
            cy - np.sqrt(r_mean**2 - (sample_x - cx) ** 2),
        )
    )
    sample_x = np.concatenate((sample_x, sample_x))
    d2edge = (
        np.sqrt((sample_x - truth_x_nm) ** 2 + (sample_y - truth_y_nm) ** 2)
        - truth_r_nm
    )
    rmsd = np.sqrt(np.mean(d2edge**2))

    # calculate the relative deviation in area
    area = np.pi * r_mean**2

    # calculate partition coefficient
    img = imread("Simulated-FOVindex-" + str(int(index)) + ".tif")
    mask_in = generate_mask(
        img.shape,
        cx / real_img_pxlsize,
        cy / real_img_pxlsize,
        r_mean / real_img_pxlsize,
    )
    mask_out = 1 - mask_in
    partition_coefficient = (cv2.mean(img, mask=mask_in)[0] - gaussian_noise_mean) / (
        cv2.mean(img, mask=mask_out)[0] - gaussian_noise_mean
    )
    partition_coefficient_max = (
        cv2.max(img, mask=mask_in)[0] - gaussian_noise_mean
    ) / (cv2.mean(img, mask=mask_out)[0] - gaussian_noise_mean)

    # save
    success.append(True)
    deviation_center.append(d2center)
    rmsd_edge.append(rmsd)
    area_fold_deviation.append(area / (np.pi * truth_r_nm**2))
    fold_deviation_pc.append(partition_coefficient / truth_pc)
    fold_deviation_pc_max.append(partition_coefficient_max / truth_pc)


df_save = pd.DataFrame(
    {
        "index": lst_index,
        "truth_r": lst_truth_r,
        "truth_pc": lst_truth_pc,
        "success": success,
        "deviation_center": deviation_center,
        "rmsd_edge": rmsd_edge,
        "fold_deviation_area": area_fold_deviation,
        "fold_deviation_PC": fold_deviation_pc,
        "fold_deviation_PC_max": fold_deviation_pc_max,
    },
    dtype=object,
)

path_save = path_results[:-4] + "_results.csv"
df_save.to_csv(path_save, index=False)
