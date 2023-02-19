import os
from os.path import dirname
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.feature import blob_log
from lmfit.models import Gaussian2dModel
from tifffile import imread

####################################
# Parameters

# DoG detector
blob_LoG_threshold = 0
max_sig = 5
# Gauss Fit
crop_size = 5  # pixels, half size of crop for Gauss fit
chisqr_threshold = 100  # Goodness of fit
Nsigma = 2.355  # boundary will be Nsigma * sigmax/y, use 2.355 for FWHM

rescale_contrast = True
plow = 0.05  # imshow intensity percentile
phigh = 95

fpath = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-HOPS.tif"
folder = dirname(fpath)
os.chdir(folder)


img = imread(fpath)

blobs = blob_log(img, threshold=blob_LoG_threshold, exclude_border=2, max_sigma=max_sig)


fig, ax = plt.subplots()
# Contrast stretching
vmin, vmax = np.percentile(img, (plow, phigh))
ax.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
for x, y, sig in blobs:
    condensate = Circle(
        (y, x),
        Nsigma * sig,
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
fpath_save = fpath[:-4] + "_GaussFit.png"
plt.savefig(fpath_save, format="png", bbox_inches="tight", dpi=300)
