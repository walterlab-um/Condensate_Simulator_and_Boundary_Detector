from tkinter import filedialog as fd
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.signal import medfilt
from tifffile import imread
import pickle
from rich.progress import track

####################################
# Parameters
med_size = 3  # pixels
threshold = 0.77  # threshold * (max - min) + min
min_intensity = 0  # filter on average intensity within a contour

dilation = True
morph_shape = cv2.MORPH_ELLIPSE
dilatation_size = 1

rescale_contrast = True
plow = 0.05  # imshow intensity percentile
phigh = 95

lst_tifs = list(fd.askopenfilenames())
# lst_tifs = [
#     "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Real-Data/forFig3-small.tif"
# ]

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
    if rescale_contrast:
        # Contrast stretching
        p1, p2 = np.percentile(img, (plow, phigh))
        img_rescale = exposure.rescale_intensity(img, in_range=(p1, p2))
        plt.imshow(img_rescale, cmap="gray")
    else:
        plt.imshow(img, cmap="gray")
    for cnt in contours:
        # make mask for intensity calculations
        mask = cnt_fill(img.shape, cnt)
        if cv2.mean(img, mask=mask)[0] > min_intensity:
            x = cnt[:, 0][:, 0]
            y = cnt[:, 0][:, 1]
            plt.plot(x, y, "-", color="firebrick", linewidth=2)
            # still the last closing line will be missing, get it below
            xlast = [x[-1], x[0]]
            ylast = [y[-1], y[0]]
            plt.plot(xlast, ylast, "-", color="firebrick", linewidth=2)
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


def mask_dilation(mask_in):
    global dilation, morph_shape, dilatation_size
    if dilation:
        element = cv2.getStructuringElement(
            morph_shape,
            (2 * dilatation_size + 1, 2 * dilatation_size + 1),
            (dilatation_size, dilatation_size),
        )
        mask_out = cv2.dilate(mask_in, element)

    else:
        mask_out = mask_in

    return mask_out


####################################
# Main
lst_index = []
lst_contours = []
for fpath in track(lst_tifs):
    index = fpath.split("FOVindex-")[-1][:-4]
    img_raw = imread(fpath)
    img_denoise = medfilt(img_raw, med_size)
    edges = (
        img_denoise
        > threshold * (img_denoise.max() - img_denoise.min()) + img_denoise.min()
    )
    edges = edges * 1
    # find contours coordinates in binary edge image. contours here is a list of np.arrays containing all coordinates of each individual edge/contour.
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Merge overlapping contours, and dilation by 1 pixel
    mask = cnt2mask(img_raw.shape, contours)
    mask_dilated = mask_dilation(mask)
    contours_final, _ = cv2.findContours(
        mask_dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    lst_index.append(index)
    lst_contours.append(contours_final)

    fpath_img = fpath[:-4] + "_Denoise_Threshold.png"
    pltcontours(img_raw, contours_final, fpath_img)

fpath_pkl = fpath[:-4] + "Contours_Denoise_Threshold.pkl"
pickle.dump([lst_index, lst_contours], open(fpath_pkl, "wb"))
