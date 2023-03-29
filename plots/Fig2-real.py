import os
import numpy as np
from numpy.random import normal, poisson
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tifffile import imwrite, imread
import plotly.graph_objects as go


##################################
# Parameters
folder_save = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials"
os.chdir(folder_save)


def plt_blue(img, fsave):
    plt.figure(dpi=300)
    # Contrast stretching
    vmin, vmax = np.percentile(img, (0, 90))
    plt.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
    plt.xlim(0, img.shape[0])
    plt.ylim(0, img.shape[1])
    plt.tight_layout()
    plt.axis("scaled")
    plt.axis("off")
    plt.savefig(fsave, format="png", bbox_inches="tight", dpi=300)
    plt.close()


img_real = imread("RealData-PB.tif")
plt_blue(img_real, "Fig2-6-realdata.png")
