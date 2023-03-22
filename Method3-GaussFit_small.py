from tkinter import filedialog as fd
import os
import shutil
from os.path import join, exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skimage import exposure
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log
from lmfit.models import Gaussian2dModel
from tifffile import imread
from rich.progress import track

####################################
# Parameters
# DoG detector
blob_LoG_threshold = 0.001
max_sig = 5
# Gauss Fit
crop_size = 3  # pixels, half size of crop for Gauss fit
chisqr_threshold = 1000  # Goodness of fit
Nsigma = 2.355  # boundary will be Nsigma * sigmax/y, use 2.355 for FWHM

plow = 0.05  # imshow intensity percentile
phigh = 99

folder = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Simulated-1024"
)
os.chdir(folder)
lst_tifs = [f for f in os.listdir(folder) if f.endswith(".tif")]

switch_plot = True  # a switch to turn off plotting


####################################
# Main
if exists("Method-3-GaussFit"):
    shutil.rmtree("Method-3-GaussFit")
os.mkdir("Method-3-GaussFit")

lst_index = []
lst_contours = []
centerx = []
centery = []
sigmax = []
sigmay = []
lst_chisqr = []
fitx = []
fity = []
for fpath in track(lst_tifs):
    index = int(fpath.split("FOVindex-")[-1][:-4])
    img = imread(fpath)

    blobs = blob_log(
        img, threshold=blob_LoG_threshold, exclude_border=3, max_sigma=max_sig
    )

    # make local crops around
    lst_GaussCrop = []
    for initial_x, initial_y, initial_sigma in blobs:
        GaussCrop = img[
            int(initial_x) - crop_size : int(initial_x) + crop_size + 1,
            int(initial_y) - crop_size : int(initial_y) + crop_size + 1,
        ]
        if GaussCrop.size > 0:
            # This mean a large blob is NOT near boundary and a full crop CAN be obtained
            lst_GaussCrop.append(GaussCrop)

    # Gauss Fit
    for GaussCrop, blob in zip(lst_GaussCrop, blobs):
        initial_x, initial_y, initial_sigma = blob
        GaussCrop = GaussCrop - GaussCrop.min()
        # call lmfit model
        model = Gaussian2dModel()
        # vectorize image
        xx, yy = np.meshgrid(
            np.arange(GaussCrop.shape[0]), np.arange(GaussCrop.shape[1])
        )
        vev_x = np.reshape(xx, -1)
        vev_y = np.reshape(yy, -1)
        vec_img = np.reshape(GaussCrop, -1)
        # fit with lmfit
        params = model.guess(vec_img, vev_x, vev_y)
        params["centerx"].set(min=0.5 * crop_size, max=1.5 * crop_size)
        params["centery"].set(min=0.5 * crop_size, max=1.5 * crop_size)
        params["sigmax"].set(min=0, max=1.5 * initial_sigma)
        params["sigmay"].set(min=0, max=1.5 * initial_sigma)
        weights = 1 / np.sqrt(vec_img + 1)
        result = model.fit(vec_img, x=vev_x, y=vev_y, params=params, weights=weights)

        lst_index.append(index)
        lst_chisqr.append(result.chisqr)
        fitx.append(result.best_values["centerx"] + initial_x - crop_size)
        fity.append(result.best_values["centery"] + initial_y - crop_size)
        if result.chisqr < chisqr_threshold:
            # if False:
            centerx.append(result.best_values["centerx"] + initial_x - crop_size)
            centery.append(result.best_values["centery"] + initial_y - crop_size)
            sigmax.append(result.best_values["sigmax"])
            sigmay.append(result.best_values["sigmay"])
        else:
            centerx.append(initial_x)
            centery.append(initial_y)
            sigmax.append(initial_sigma)
            sigmay.append(initial_sigma)


df_result = pd.DataFrame(
    {
        "index": lst_index,
        "centerx": centerx,
        "centery": centery,
        "sigmax": sigmax,
        "sigmay": sigmay,
        "chisqr": lst_chisqr,
        "fitx": fitx,
        "fity": fity,
    },
    dtype=float,
)
df_result = df_result.sort_values("index")
fpath_save = join("Method-3-GaussFit", "GaussFit.csv")
df_result.to_csv(fpath_save, index=False)


if switch_plot:
    for fpath in track(lst_tifs):
        index = int(fpath.split("FOVindex-")[-1][:-4])
        df_current = df_result[df_result["index"] == index]
        img = imread(fpath)
        fig, ax = plt.subplots()
        # Contrast stretching
        vmin, vmax = np.percentile(img, (plow, phigh))
        ax.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)

        for note, row in df_current.iterrows():
            x = row.centerx
            y = row.centery
            sigmax = row.sigmax
            sigmay = row.sigmay
            condensate = Ellipse(
                (y, x),
                Nsigma * sigmax,
                Nsigma * sigmay,
                color="black",
                fill=False,
                lw=2,
            )  # plot as FWHM
            ax.add_patch(condensate)

        plt.xlim(0, img.shape[0])
        plt.ylim(0, img.shape[1])
        plt.tight_layout()
        plt.axis("scaled")
        plt.axis("off")
        fpath_save = join("Method-3-GaussFit", fpath[:-4] + "_GaussFit.png")
        plt.savefig(fpath_save, format="png", bbox_inches="tight", dpi=300)
        plt.close()
