import os
from os.path import dirname
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from rich.progress import track

####################################
# Parameters
med_size = 3  # pixels
threshold = 0.55  # threshold * (max - min) + min
min_intensity = 0  # filter on average intensity within a contour

dilation = False
morph_shape = cv2.MORPH_ELLIPSE
dilatation_size = 1

plow = 0.05  # imshow intensity percentile
phigh = 95

lst_tifs = [
    "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-PB.tif",
    "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-HOPS.tif",
]
os.chdir(dirname(lst_tifs[0]))

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
    # img_denoise = medfilt(img_raw, med_size)
    img_denoise = img_raw
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

    fpath_img = fpath[:-4] + "_Threshold.png"
    if switch_plot:
        pltcontours(img_raw, contours_final, fpath_img)
    else:
        continue
