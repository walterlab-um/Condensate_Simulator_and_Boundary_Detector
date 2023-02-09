import os
from os.path import join, dirname
import numpy as np
import pandas as pd
import pims
import trackpy as tp
import matplotlib as mpl
import matplotlib.pyplot as plt

path_RNA = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Example-Slow Dwelling-bandpass-RNA.tif"
path_condensate = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Example-Slow Dwelling-bandpass-condensate.tif"
os.chdir(dirname(path_RNA))
os.mkdir("")

# testing code
# index = 6
# f = tp.locate(frames[index], diameter=5, separation=3, minmass=5e5, preprocess=False)
# tp.annotate(f, frames[index])

# detect tracks with trackpy
frames_RNA = pims.open(path_RNA)
spots_RNA = tp.batch(
    frames_RNA, diameter=5, separation=3, minmass=5e5, preprocess=False
)
tracks = tp.link(spots_RNA, search_range=3)
tracks_RNA = tp.filter_stubs(tracks, threshold=5)

idx = 1
# plot condensate channel
plt.figure(figsize=(5, 5), dpi=300)
img = frames_RNA[idx]
vmin, vmax = np.percentile(img, (0.05, 99))
plt.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
plt.axis("scaled")
plt.axis("off")
fname_save = "frame_" + str(idx) + "_"
