import os
from os.path import join, dirname
import shutil
import numpy as np
import pims
import trackpy as tp
from matplotlib.cm import get_cmap
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from rich.progress import track

tp.quiet()

path_RNA = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-montage/Example-Slow Dwelling-bandpass-RNA.tif"
path_condensate = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-montage/Example-Slow Dwelling-bandpass-condensate.tif"
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

    img_condensate = frames_condensate[idx]
    img_RNA = frames_RNA[idx]
    current_track_condensate = tracks_condensate[tracks_condensate["frame"] <= idx]
    x_condensate = current_track_condensate.x.to_numpy(dtype=float)
    y_condensate = current_track_condensate.y.to_numpy(dtype=float)
    size_condensate = current_track_condensate["size"].to_numpy(dtype=float)[-1]
    current_track_RNA = tracks_RNA[tracks_RNA["frame"] <= idx]
    x_RNA = current_track_RNA.x.to_numpy(dtype=float)
    y_RNA = current_track_RNA.y.to_numpy(dtype=float)
    size_RNA = current_track_RNA["size"].to_numpy(dtype=float)[-1]

    color_con = get_cmap("Blues")(0.9)
    color_RNA = get_cmap("Reds")(0.9)

    # plot condensate channel
    fname_save = "frame_" + str(idx) + "_condensate.png"
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    vmin, vmax = np.percentile(img_condensate, (0.05, 99))
    ax.imshow(img_condensate, cmap="Blues", vmin=vmin, vmax=vmax)
    spot = Circle(
        (x_condensate[-1], y_condensate[-1]),
        size_condensate,
        color=color_con,
        ls="--",
        fill=False,
        lw=2,
    )
    ax.add_patch(spot)
    plt.axis("scaled")
    plt.axis("off")
    plt.gca().invert_yaxis()  # must invert AFTER xlim, otherwise it goes back uninverted
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()

    # plot RNA channel
    fname_save = "frame_" + str(idx) + "_RNA.png"
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    vmin, vmax = np.percentile(img_RNA, (0.05, 99))
    plt.imshow(img_RNA, cmap="Reds", vmin=vmin, vmax=vmax)
    spot = Circle(
        (x_RNA[-1], y_RNA[-1]),
        size_RNA,
        color=color_RNA,
        ls="--",
        fill=False,
        lw=2,
    )
    ax.add_patch(spot)
    plt.axis("scaled")
    plt.axis("off")
    plt.gca().invert_yaxis()  # must invert AFTER xlim, otherwise it goes back uninverted
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()

    # plot overlay tracks
    fname_save = "frame_" + str(idx) + "_track.png"
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.plot(x_condensate, y_condensate, ls="-", lw=1, color=color_con)
    spot = Circle(
        (x_condensate[-1], y_condensate[-1]),
        size_condensate,
        color=color_con,
        ls="--",
        fill=False,
        lw=2,
    )
    ax.add_patch(spot)
    spot = Circle(
        (x_condensate[-1], y_condensate[-1]),
        size_condensate,
        color=color_con,
        alpha=0.3,
        lw=0,
    )
    ax.add_patch(spot)

    ax.plot(x_RNA, y_RNA, ls="-", lw=1, color=color_RNA)
    plt.scatter(x_RNA[-1], y_RNA[-1], s=200, color=color_RNA)
    plt.axis("scaled")
    plt.axis("off")
    plt.xlim(-0.5, frames_condensate.shape[2] - 0.5)
    plt.ylim(-0.5, frames_condensate.shape[1] - 0.5)
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()
