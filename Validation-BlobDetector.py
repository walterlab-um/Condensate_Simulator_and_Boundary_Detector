from tkinter import filedialog as fd
import os
import pickle
import cv2
import pandas as pd
import numpy as np
from rich.progress import track

real_img_pxlsize = 100  # unit: nm, must be an integer multiple of truth_img_pxlsize
fovsize = 5000  # unit: nm
Nsigma = 3  # boundary will be Nsigma * sigmax/y, use 2.355 for FWHM

folder = fd.askdirectory(
    initialdir="/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/"
)
os.chdir(folder)

path_results = "GaussFit.csv"
path_groundtruth = "groundtruth.csv"

df_results = pd.read_csv(path_results, dtype=float)
df_truth = pd.read_csv(path_groundtruth, dtype=float)


def when_failed():
    global success, areas, aspect_ratio, distance2center, distance2edge, centroid, real_center
    success.append(False)
    areas.append(np.nan)
    aspect_ratio.append(np.nan)
    distance2center.append(np.nan)
    distance2edge.append(np.nan)
    centroid.append(np.nan)
    real_center.append(np.nan)


lst_index = df_truth.FOVindex.to_numpy(dtype=float)
success = []
areas = []  # area of contours
aspect_ratio = []
distance2center = []
distance2edge = []
centroid = []
real_center = []
for index in track(lst_index):
    df_current = df_results[df_results["index"] == index]
    if df_current.shape[0] == 0:
        when_failed()
        continue
    elif df_current.shape[0] > 1:
        when_failed()
        continue

    # retreive ground truth
    row = df_truth[df_truth.FOVindex == index]
    truth_x_nm = row.x_nm.squeeze()
    truth_y_nm = row.y_nm.squeeze()
    truth_r_nm = row.r_nm.squeeze()

    # distance to center and area, note x and y must flip!
    cx = df_current.centery.squeeze() * real_img_pxlsize
    cy = df_current.centerx.squeeze() * real_img_pxlsize
    rx = df_current.sigmax.squeeze() * real_img_pxlsize * Nsigma
    ry = df_current.sigmay.squeeze() * real_img_pxlsize * Nsigma
    d2center = np.sqrt((cx - truth_x_nm) ** 2 + (cy - truth_y_nm) ** 2)
    distance2center.append(d2center)
    area = np.pi * rx * ry

    # distance to edge defined as differece to R
    d2edge = np.sqrt((cx - truth_x_nm) ** 2 + (cy - truth_y_nm) ** 2) - truth_r_nm
    distance2edge.append(d2edge)

    # save
    success.append(True)
    areas.append(area)
    aspect_ratio.append(min(rx, ry) / max(rx, ry))
    centroid.append((cx, cy))
    real_center.append((truth_x_nm, truth_y_nm))


df_save = pd.DataFrame(
    {
        "index": lst_index,
        "success": success,
        "area_nm2": areas,
        "aspect_ratio": aspect_ratio,
        "distance2center": distance2center,
        "distance2edge": distance2edge,
        "centroid": centroid,
        "real_center": real_center,
    },
    dtype=object,
)

path_save = path_results[:-4] + "_validation_results.csv"
df_save.to_csv(path_save, index=False)
