import os
from os.path import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import pickle
from rich.progress import track

####################################
# Parameters
plow = 0.05  # imshow intensity percentile
phigh = 99
folder = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Simulated-4096"
)
os.chdir(folder)
lst_subfolders = [
    "ilastik-Guoming",
    "ilastik-Liuhan",
    "ilastik-EmilyS",
    "ilastik-Rosa",
    "ilastik-SarahGolts",
    "ilastik-Sujay",
    "ilastik-Xiaofeng",
]

switch_plot = False  # a switch to turn off plotting


####################################
# Functions
def pltcontours(img, contours, fsave):
    global rescale_contrast, plow, phigh
    plt.figure(dpi=300)
    # Contrast stretching
    vmin, vmax = np.percentile(img, (plow, phigh))
    plt.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
    cnt = contours[0]
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


####################################
# Main
for subfolder in lst_subfolders:
    print("Now working on:", subfolder)
    lst_tifs = [join(subfolder, f) for f in os.listdir(subfolder) if f.endswith(".tif")]
    lst_index = []
    lst_contours = []
    for fpath in track(lst_tifs):
        index = fpath.split("FOVindex-")[-1].split("_Simple")[0]
        img_raw = imread("Simulated-FOVindex-" + index + ".tif")
        mask = imread(fpath)
        mask = 2 - mask  # background label=2, condensate label=1

        if mask.sum() == 0:  # so no condensate
            continue

        # find contours coordinates in binary edge image. contours here is a list of np.arrays containing all coordinates of each individual edge/contour.
        contours_final, _ = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        lst_index.append(index)
        lst_contours.append(contours_final)

        if switch_plot:
            pltcontours(img_raw, contours_final, fpath[:-4] + ".png")
        else:
            continue

    pickle.dump(
        [lst_index, lst_contours],
        open(join(subfolder, "Contours_ilastik.pkl"), "wb"),
    )
