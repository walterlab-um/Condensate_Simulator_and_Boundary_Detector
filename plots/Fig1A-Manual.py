import os
from os.path import dirname, basename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from tifffile import imread


rescale_contrast = True
plow = 0.05  # imshow intensity percentile
phigh = 95

fpath = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-HOPS.tif"
folder = dirname(fpath)
os.chdir(folder)

lst_rois = [
    f
    for f in os.listdir(folder)
    if (f.startswith(basename(fpath)[:-4]) & f.endswith(".txt"))
]

img = imread(fpath)


fig, ax = plt.subplots()
# Contrast stretching
vmin, vmax = np.percentile(img, (plow, phigh))
ax.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
for path_roi in lst_rois:
    df_roi = pd.read_csv(path_roi, sep="	", header=None)
    coords_roi_raw = []
    for _, row in df_roi.iterrows():
        current_coord = row.to_numpy(dtype=float)
        # round up to int pixels
        current_coord_round = np.around(current_coord)
        coords_roi_raw.append(current_coord_round)
    # Remove the duplicating roi coordinates after round, and return to original indexes
    coords_roi_array = np.stack(coords_roi_raw)
    _, indexes = np.unique(coords_roi_array, axis=0, return_index=True)
    coords_roi_final = [coords_roi_array[index] for index in sorted(indexes)]
    # Plot
    # condensate = Polygon([tuple(row) for row in np.flip(coords_roi_final, 1)])
    condensate = Polygon([tuple(row - 1) for row in coords_roi_final])
    x, y = condensate.exterior.xy
    ax.plot(x, y, "-k")

plt.xlim(0, img.shape[0])
plt.ylim(0, img.shape[1])
plt.tight_layout()
plt.axis("scaled")
plt.axis("off")
fpath_save = fpath[:-4] + "_Mannual.png"
plt.savefig(fpath_save, format="png", bbox_inches="tight", dpi=300)
