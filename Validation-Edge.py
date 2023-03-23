from tifffile import imread
import os
import pickle
import cv2
import pandas as pd
import numpy as np
from rich.progress import Progress

#################################################
# Inputs
real_img_pxlsize = 100  # unit: nm, must be an integer multiple of truth_img_pxlsize
fovsize = 2000  # unit: nm
folder = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Simulated-1024"
)
os.chdir(folder)
path_groundtruth = "groundtruth.csv"
path_pkl = "ilastik-Guoming/Contours_ilastik.pkl"
# lst_pkl = [
#     "Method-1-Denoise_Threshold/Contours_Denoise_Threshold.pkl",
#     "Method-2-Canny/Contours_Canny.pkl",
#     "ilastik-Guoming/Contours_ilastik.pkl",
#     "ilastik-EmilyS/Contours_ilastik.pkl",
#     "ilastik-SarahGolts/Contours_ilastik.pkl",
#     "ilastik-Sujay/Contours_ilastik.pkl",
#     "ilastik-Xiaofeng/Contours_ilastik.pkl",
#     "ilastik-Liuhan/Contours_ilastik.pkl",
#     "ilastik-Rosa/Contours_ilastik.pkl",
# ]


#################################################
# Functions
def cnt_fill(imgshape, cnt):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    cv2.fillPoly(mask, [cnt], (255))

    return mask


def when_failed():
    global success, rmsd_center, rmsd_edge, area_fold_deviation, fold_deviation_pc
    success.append(False)
    deviation_center.append(np.nan)
    rmsd_edge.append(np.nan)
    area_fold_deviation.append(np.nan)
    fold_deviation_pc.append(np.nan)


#################################################
# Main
lst_index, lst_contours = pickle.load(open(path_pkl, "rb"))
df_truth = pd.read_csv(path_groundtruth)

lst_truth_r = []
lst_truth_pc = []  # partition coefficient
success = []
deviation_center = []
rmsd_edge = []
area_fold_deviation = []  # with sign
# difference in partition coefficient, assume outside C_dilute=1, so directly the average intensity inside, with sign
fold_deviation_pc = []
with Progress() as progress:
    task = progress.add_task(path_pkl, total=len(lst_index))
    for index, contours in zip(np.array(lst_index, dtype=int), lst_contours):
        # retreive ground truth
        row = df_truth[df_truth.FOVindex == index]
        truth_x_nm = row["x_nm"].squeeze()
        truth_y_nm = row["y_nm"].squeeze()
        truth_r_nm = row["r_nm"].squeeze()
        truth_pc = row["C_condensed"].squeeze()
        lst_truth_r.append(truth_r_nm)
        lst_truth_pc.append(truth_pc)

        # fail if more than 1 condensate or no condensate was detected
        if (len(contours) > 1) | (len(contours) < 1):
            when_failed()
            continue

        detected_contour = contours[0]

        # fail if the contour is just the whole field of view
        if (
            cv2.contourArea(detected_contour) * real_img_pxlsize**2
            > 0.8 * fovsize**2
        ):
            when_failed()
            continue

        # calculate the deviation of condensate center
        M = cv2.moments(detected_contour)
        if M["m00"] == 0:
            when_failed()
            continue
        cx = int(M["m10"] / M["m00"]) * real_img_pxlsize
        cy = int(M["m01"] / M["m00"]) * real_img_pxlsize
        d2center = np.sqrt((cx - truth_x_nm) ** 2 + (cy - truth_y_nm) ** 2)
        # calculate the RMSD of detected edge to real condensate edge
        cnt_reshaped = np.reshape(
            detected_contour, (detected_contour.shape[0], detected_contour.shape[2])
        )
        d2edge_squared = []
        for cnt_x, cnt_y in cnt_reshaped:
            d2edge_squared.append(
                np.sqrt(
                    (cnt_x * real_img_pxlsize - truth_x_nm) ** 2
                    + (cnt_y * real_img_pxlsize - truth_y_nm) ** 2
                )
                - truth_r_nm
            )
        rmsd = np.sqrt(np.mean(np.array(d2edge_squared) ** 2))
        # calculate the relative deviation in area
        area = cv2.contourArea(detected_contour) * real_img_pxlsize**2
        # calculate partition coefficient
        img = imread("Simulated-FOVindex-" + str(index) + ".tif")
        mask_in = cnt_fill(img.shape, detected_contour)
        mask_out = 1 - mask_in
        partition_coefficient = (
            cv2.mean(img, mask=mask_in)[0] / cv2.mean(img, mask=mask_out)[0]
        )
        # save
        success.append(True)
        deviation_center.append(d2center)
        rmsd_edge.append(rmsd)
        area_fold_deviation.append(area / (np.pi * truth_r_nm**2))
        fold_deviation_pc.append(partition_coefficient / truth_pc)

        progress.update(task, refresh=True)


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
    },
    dtype=object,
)

path_save = path_pkl[:-4] + "_results.csv"
df_save.to_csv(path_save, index=False)
