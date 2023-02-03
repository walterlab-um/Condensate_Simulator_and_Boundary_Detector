from tkinter import filedialog as fd
import os
import pickle
import cv2
import pandas as pd
import numpy as np
import seaborn as sns

sns.set(color_codes=True, style="white")

folder = fd.askdirectory(
    initialdir="/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/"
)
os.chdir(folder)


path_pkl = [f for f in os.listdir(folder) if f.endswith(".pkl")][0]
path_groundtruth = [f for f in os.listdir(folder) if f.endswith("groundtruth.csv")][0]

real_img_pxlsize = 100  # unit: nm, must be an integer multiple of truth_img_pxlsize
fovsize = 5000  # unit: nm

lst_index, lst_contours = pickle.load(open(path_pkl, "rb"))
df_truth = pd.read_csv(path_groundtruth)


def when_failed():
    global success, areas, aspect_ratios, distance2center, distance2edge, centroid, real_center
    success.append(False)
    areas.append(np.nan)
    aspect_ratios.append(np.nan)
    distance2center.append(np.nan)
    distance2edge.append(np.nan)
    centroid.append(np.nan)
    real_center.append(np.nan)


success = []
areas = []  # area of contours
aspect_ratios = []
distance2center = []
distance2edge = []
centroid = []
real_center = []
for index, contours in zip(np.array(lst_index, dtype=int), lst_contours):
    if len(contours) > 1:
        when_failed()
        continue

    # retreive ground truth
    row = df_truth[df_truth.FOVindex == index]
    truth_x_nm = row.x_nm.squeeze()
    truth_y_nm = row.y_nm.squeeze()
    truth_r_nm = row.r_nm.squeeze()

    # area
    cnt = contours[0]
    area = cv2.contourArea(cnt) * real_img_pxlsize**2
    # cnt could be the whole FOV when it failed to detect spot
    if area > 0.8 * fovsize**2:
        when_failed()
        continue

    # aspect ratio
    rect = cv2.minAreaRect(cnt)
    (x, y), (width, height), angle = rect
    if max(width, height) == 0:
        when_failed()
        continue

    # distance to center
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"]) * real_img_pxlsize
    cy = int(M["m01"] / M["m00"]) * real_img_pxlsize
    d2center = np.sqrt((cx - truth_x_nm) ** 2 + (cy - truth_y_nm) ** 2)
    distance2center.append(d2center)

    # save
    success.append(True)
    areas.append(area)
    aspect_ratios.append(min(width, height) / max(width, height))
    centroid.append((cx, cy))
    real_center.append((truth_x_nm, truth_y_nm))

    # distance to edge, for each vertex
    cnt_reshaped = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
    d2edge = []
    for cnt_x, cnt_y in cnt_reshaped:
        d2edge.append(
            truth_r_nm - np.sqrt((cnt_x - truth_x_nm) ** 2 + (cnt_y - truth_y_nm) ** 2)
        )
    distance2edge.append(np.array(d2edge))


df_save = pd.DataFrame(
    {
        "index": lst_index,
        "success": success,
        "area_nm2": areas,
        "aspect_ratio": aspect_ratios,
        "distance2center": distance2center,
        "distance2edge": distance2edge,
        "centroid": centroid,
        "real_center": real_center,
    },
    dtype=object,
)

path_save = path_pkl[:-4] + "_validation_results.csv"
df_save.to_csv(path_save, index=False)
