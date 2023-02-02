from tkinter import filedialog as fd
from os.path import join, dirname, basename
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.signal import medfilt
from tifffile import imread
from itertools import compress
from copy import deepcopy
import pickle
from rich.progress import track

med_size = 5  # pixels
threshold = 0.3  # threshold * (max - min) + min
dilation = True
rescale_contrast = True

# lst_tifs = list(fd.askopenfilenames())
lst_tifs = [
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Real-Data/forFig3-small.tif"
]


def cnt_fill(imgshape, cnt):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    cv2.fillPoly(mask, [cnt], (255))

    return mask


def pltcontours(img, contours, fsave):
    global rescale_contrast
    plt.figure()
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
            plt.plot(x, y, "r-", linewidth=0.2)
            # still the last closing line will be missing, get it below
            xlast = [x[-1], x[0]]
            ylast = [y[-1], y[0]]
            plt.plot(xlast, ylast, "r-", linewidth=0.2)
    plt.xlim(0, figsize[0])
    plt.ylim(0, figsize[1])
    plt.tight_layout()
    plt.axis("scaled")
    plt.axis("off")
    plt.savefig(fsave, format="png", bbox_inches="tight", dpi=300)


def fillcontours(imgshape, contours):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    for cnt in contours:
        cv2.fillPoly(mask, [cnt], (255))
    return mask


def mask_dilation(morph_shape, dilatation_size, mask_in):
    global dilation
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


def MAIN(img_raw):
    edges = cv2.Canny(
        img, cannythresh1, cannythresh2, apertureSize=SobelSize, L2gradient=L2gradient
    )

    # find contours coordinates in binary edge image. contours here is a list of np.arrays containing all coordinates of each individual edge/contour.
    contours_canny, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print("Total Number of Contours Found: ", str(len(contours_canny)))

    ##############################
    # Contours filtering

    # remove small contours
    area = np.array([cv2.contourArea(cnt) for cnt in contours_canny])
    selector = area > min_size
    contours_filtered_size = list(compress(contours_canny, selector))
    print(
        "Contours larger than minimal size: ",
        str(100 * len(contours_filtered_size) / len(contours_canny)),
        "%",
    )

    # filter by intensity
    # make mask for intensity calculations
    selector = []
    for cnt in contours_filtered_size:
        mask = cnt_fill(img.shape, cnt)
        selector.append(cv2.mean(img, mask=mask)[0] > min_intensity)

    contours_filtered_size_int = list(compress(contours_filtered_size, selector))
    print(
        "Contours larger than minimal intensity: ",
        str(100 * len(contours_filtered_size_int) / len(contours_filtered_size)),
        "%",
    )

    # filter by extent (area/boxsize)
    extent = calc_extent(contours_in=contours_filtered_size_int)
    selector = extent > min_extent
    contours_filtered_size_int_extent = list(
        compress(contours_filtered_size_int, selector)
    )
    print(
        "Contours with a round shape: ",
        str(
            100
            * len(contours_filtered_size_int_extent)
            / len(contours_filtered_size_int)
        ),
        "%",
    )

    ##############################
    # Merge overlapping contours, and dilation by 1 pixel
    mask = fillcontours(imgshape=img.shape, contours=contours_filtered_size_int)
    mask_dilated = mask_dilation(
        dilation,
        morph_shape=morph_shape,
        dilatation_size=dilatation_size,
        mask_in=mask,
    )
    contours_final, _ = cv2.findContours(
        mask_dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    return contours_final


##############################
# Main body

for fpath in track(lst_tifs):
    img_raw = imread(fpath)
    img_denoise = medfilt(img_raw, med_size)
    # obtain canny contours
    contours = MAIN(img_denoise)
    # saving
    fpath_pkl = fpath.strip(".tif") + "_contours.pkl"
    fpath_img = fpath.strip(".tif") + "_contours.png"
    pickle.dump([contours, img_average], open(fpath_pkl, "wb"))
    pltcontours(img_average, contours, fpath_img, rescale_contrast)
