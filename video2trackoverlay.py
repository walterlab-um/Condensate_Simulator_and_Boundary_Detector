import os
from os.path import join, dirname
import shutil
import numpy as np
import pandas as pd
import pims
import trackpy as tp
import matplotlib as mpl
import matplotlib.pyplot as plt

tp.quiet()

path_RNA = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Example-Slow Dwelling-bandpass-RNA.tif"
path_condensate = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Example-Slow Dwelling-bandpass-condensate.tif"
os.chdir(dirname(path_RNA))
try:
    os.mkdir("montages")
except:
    shutil.rmtree("montages")
    os.mkdir("montages")
os.chdir("montages")


# load video
frames_condensate = pims.open(path_condensate)
frames_RNA = pims.open(path_RNA)

# detect tracks with trackpy
# testing code
# index = 6
# f = tp.locate(frames[index], diameter=5, separation=3, minmass=5e5, preprocess=False)
# tp.annotate(f, frames[index])
spots_RNA = tp.batch(
    frames_RNA, diameter=5, separation=3, minmass=4e5, preprocess=False, processes=1
)
tracks = tp.link(spots_RNA, search_range=3)
tracks_RNA = tp.filter_stubs(tracks, threshold=5)
tracks_RNA.to_csv("tracks.csv", index=False)


idx = 1
for idx in range(tracks_RNA.shape[0]):
    if idx % 5 != 0:
        continue

    # plot condensate channel
    img = frames_condensate[idx]
    fname_save = "frame_" + str(idx) + "_condensate.png"

    plt.figure(figsize=(5, 5), dpi=300)
    vmin, vmax = np.percentile(img, (0.05, 99))
    plt.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
    plt.axis("scaled")
    plt.axis("off")
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()

    # plot RNA channel
    img = frames_RNA[idx]
    fname_save = "frame_" + str(idx) + "_RNA.png"

    plt.figure(figsize=(5, 5), dpi=300)
    vmin, vmax = np.percentile(img, (0.05, 99))
    plt.imshow(img, cmap="Reds", vmin=vmin, vmax=vmax)
    plt.axis("scaled")
    plt.axis("off")
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()
