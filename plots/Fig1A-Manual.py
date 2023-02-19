import cv2
from os.path import dirname
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from tifffile import imread


rescale_contrast = True
plow = 0.05  # imshow intensity percentile
phigh = 95

fpath_img = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-HOPS.tif"
fpath_mask = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-HOPS-mannual.tif"
folder = dirname(fpath_img)

img = imread(fpath_img)
mask = cv2.imread(fpath_mask, cv2.IMREAD_GRAYSCALE)
# mask_binary = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)[1]

fig, ax = plt.subplots()
# Contrast stretching
vmin, vmax = np.percentile(img, (plow, phigh))
ax.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    x = cnt[:, 0][:, 0]
    y = cnt[:, 0][:, 1]
    plt.plot(x, y, "k-", lw=2)
    # still the last closing line will be missing, get it below
    xlast = [x[-1], x[0]]
    ylast = [y[-1], y[0]]
    plt.plot(xlast, ylast, "k-", lw=2)
plt.xlim(0, img.shape[0])
plt.ylim(0, img.shape[1])
plt.tight_layout()
plt.axis("scaled")
plt.axis("off")
fpath_save = fpath_mask[:-4] + ".png"
plt.savefig(fpath_save, format="png", bbox_inches="tight", dpi=300)
