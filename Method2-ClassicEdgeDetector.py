import os
import shutil
from os.path import join, exists
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import pickle
from rich.progress import track

####################################
# Parameters
# Canny edge detection parameters
cannythresh1 = 50
cannythresh2 = 1000
SobelSize = 5  # 3/5/7
L2gradient = True

min_intensity = 10  # filter on average intensity within a contour

dilation = False
morph_shape = cv2.MORPH_ELLIPSE
dilatation_size = 1

plow = 0.05  # imshow intensity percentile
phigh = 99

folder = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Simulated-1024"
)
os.chdir(folder)
lst_tifs = [f for f in os.listdir(folder) if f.endswith(".tif")]

switch_plot = True  # a switch to turn off plotting


####################################
# Functions
def cnt_fill(imgshape, cnt):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    cv2.fillPoly(mask, [cnt], (255))

    return mask


def pltcontours(img, contours, fsave):
    global rescale_contrast, plow, phigh, min_intensity
    plt.figure(dpi=300)
    # Contrast stretching
    vmin, vmax = np.percentile(img, (plow, phigh))
    plt.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
    for cnt in contours:
        # make mask for intensity calculations
        mask = cnt_fill(img.shape, cnt)
        if cv2.mean(img, mask=mask)[0] > min_intensity:
            x = cnt[:, 0][:, 0]
            y = cnt[:, 0][:, 1]
            plt.plot(x, y, "-", color="black", linewidth=2)
            # still the last closing line will be missing, get it below
            xlast = [x[-1], x[0]]
            ylast = [y[-1], y[0]]
            plt.plot(xlast, ylast, "-", color="black", linewidth=2)
    plt.xlim(0, img.shape[0])
    plt.ylim(0, img.shape[1])
    plt.tight_layout()
    plt.axis("scaled")
    plt.axis("off")
    plt.savefig(fsave, format="png", bbox_inches="tight", dpi=300)
    plt.close()


def cnt2mask(imgshape, contours):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    for cnt in contours:
        if cv2.contourArea(cnt) > 0:  # remove empty contours
            cv2.fillPoly(mask, [cnt], (255))
    return mask


####################################
# Main
if exists("Method-2-Canny"):
    shutil.rmtree("Method-2-Canny")
os.mkdir("Method-2-Canny")
lst_index = []
lst_contours = []
for fpath in track(lst_tifs):
    index = fpath.split("FOVindex-")[-1][:-4]
    img_raw = imread(fpath)

    # convert to uint8
    img = img_raw / img_raw.max()  # normalize to (0,1)
    img = img * 255  # re-scale to uint8
    img = img.astype(np.uint8)

    edges = cv2.Canny(
        img, cannythresh1, cannythresh2, apertureSize=SobelSize, L2gradient=L2gradient
    )

    # find contours coordinates in binary edge image. contours here is a list of np.arrays containing all coordinates of each individual edge/contour.
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Merge overlapping contours
    mask = cnt2mask(img_raw.shape, contours)
    contours_final, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    lst_index.append(index)
    lst_contours.append(contours_final)

    if switch_plot:
        fpath_img = join("Method-2-Canny", fpath[:-4] + "_Canny.png")
        pltcontours(img_raw, contours_final, fpath_img)
    else:
        continue


pickle.dump(
    [lst_index, lst_contours],
    open(join("Method-2-Canny", "Contours_Canny.pkl"), "wb"),
)
