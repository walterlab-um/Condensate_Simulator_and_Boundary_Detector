from tkinter import filedialog as fd
import os
from os.path import join, exists
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
from tifffile import imread
import pickle
from rich.progress import track

####################################
# Parameters
plow = 0.05  # imshow intensity percentile
phigh = 99
folder = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Simulated-1024"
)
os.chdir(folder)
lst_subfolders = [f for f in os.listdir(folder) if f.startswith("ilastik")]

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
    global rescale_contrast, plow, phigh
    plt.figure(dpi=300)
    # Contrast stretching
    vmin, vmax = np.percentile(img, (plow, phigh))
    plt.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
    for cnt in contours:
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
        cv2.fillPoly(mask, [cnt], (255))
    return mask


####################################
# Main
for subfolder in lst_subfolders:
    lst_tifs = [join(subfolder, f) for f in os.listdir(subfolder) if f.endswith(".tif")]
    lst_index = []
    lst_contours = []
    for fpath in track(lst_tifs):
        index = fpath.split("FOVindex-")[-1].split("Simple")[0]
        img_raw = imread("Simulated-FOVindex-" + index + ".tif")
        mask = imread(fpath)
        mask = mask > 1
        # find contours coordinates in binary edge image. contours here is a list of np.arrays containing all coordinates of each individual edge/contour.
        contours_final, _ = cv2.findContours(
            mask * 1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        lst_index.append(index)
        lst_contours.append(contours_final)

        if switch_plot:
            pltcontours(img_raw, contours_final, fpath[:-4] + "_Denoise_Threshold.png")
        else:
            continue

    pickle.dump(
        [lst_index, lst_contours],
        open(join(subfolder, "Contours_ilastik.pkl"), "wb"),
    )
