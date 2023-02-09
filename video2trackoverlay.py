import os
from os.path import join, dirname
import shutil
import numpy as np
import pandas as pd
import pims
import trackpy as tp
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from rich.progress import track

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
spots = tp.batch(
    frames_RNA, diameter=5, separation=3, minmass=4e5, preprocess=False, processes=1
)
tracks = tp.link(spots, search_range=3)
tracks_RNA = tp.filter_stubs(tracks, threshold=5)
tracks_RNA.to_csv("tracks_RNA.csv", index=False)

spots = tp.batch(
    frames_condensate,
    diameter=11,
    separation=3,
    minmass=4e5,
    preprocess=False,
    processes=1,
)
tracks = tp.link(spots, search_range=3)
tracks_condensate = tp.filter_stubs(tracks, threshold=5)
tracks_condensate.to_csv("tracks_condensate.csv", index=False)


for idx in track(range(frames_condensate.shape[0])):
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
    plt.gca().invert_yaxis()  # must invert AFTER xlim, otherwise it goes back uninverted
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()

    # plot condensate track
    # current_track = tracks_condensate[tracks_condensate["frame"] <= idx]
    # cmap = get_cmap("Blues")
    # color = cmap(0.9)
    # fname_save = "frame_" + str(idx) + "_condensate_track.png"

    # plt.figure(figsize=(5, 5), dpi=300)
    # x = current_track.x.to_numpy(dtype=float)
    # y = current_track.y.to_numpy(dtype=float)
    # size = current_track["size"].to_numpy(dtype=float)[-1]
    # plt.plot(x, y, ls="-", lw=1, color=color, alpha=0.3)
    # plt.scatter(x[-1], y[-1], s=size * 10000, color=color, alpha=0.3)
    # plt.axis("scaled")
    # plt.xlim(0, frames_condensate.shape[1])
    # plt.ylim(0, frames_condensate.shape[2])
    # plt.savefig(fname_save, format="png", bbox_inches="tight")
    # plt.close()

    # plot RNA channel
    img = frames_RNA[idx]
    fname_save = "frame_" + str(idx) + "_RNA.png"

    plt.figure(figsize=(5, 5), dpi=300)
    vmin, vmax = np.percentile(img, (0.05, 99))
    plt.imshow(img, cmap="Reds", vmin=vmin, vmax=vmax)
    plt.axis("scaled")
    plt.axis("off")
    plt.gca().invert_yaxis()  # must invert AFTER xlim, otherwise it goes back uninverted
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()

    # plot RNA track
    # current_track = tracks_RNA[tracks_RNA["frame"] <= idx]
    # color = "firebrick"
    # fname_save = "frame_" + str(idx) + "_RNA_track.png"

    # plt.figure(figsize=(5, 5), dpi=300)
    # x = current_track.x.to_numpy(dtype=float)
    # y = current_track.y.to_numpy(dtype=float)
    # plt.plot(x, y, ls="-", lw=1, color=color)
    # plt.scatter(x[-1], y[-1], s=200, color=color)
    # plt.axis("scaled")
    # plt.xlim(0, frames_condensate.shape[1])
    # plt.ylim(0, frames_condensate.shape[2])
    # plt.savefig(fname_save, format="png", bbox_inches="tight")
    # plt.close()

    # plot overlay tracks
    current_track = tracks_condensate[tracks_condensate["frame"] <= idx]
    cmap = get_cmap("Blues")
    color = cmap(0.9)
    fname_save = "frame_" + str(idx) + "_track.png"

    plt.figure(figsize=(5, 5), dpi=300)
    x = current_track.x.to_numpy(dtype=float)
    y = current_track.y.to_numpy(dtype=float)
    size = current_track["size"].to_numpy(dtype=float)[-1]
    plt.plot(x, y, ls="-", lw=1, color=color, alpha=0.3)
    plt.scatter(x[-1], y[-1], s=size * 10000, color=color, alpha=0.3)

    current_track = tracks_RNA[tracks_RNA["frame"] <= idx]
    color = "firebrick"

    x = current_track.x.to_numpy(dtype=float)
    y = current_track.y.to_numpy(dtype=float)
    plt.plot(x, y, ls="-", lw=1, color=color)
    plt.scatter(x[-1], y[-1], s=200, color=color)
    plt.axis("scaled")
    plt.xlim(0, frames_condensate.shape[1])
    plt.ylim(0, frames_condensate.shape[2])
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()
