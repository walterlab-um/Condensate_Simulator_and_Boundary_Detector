import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

plow = 0.05  # imshow intensity percentile
phigh = 95

fpath_img = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-PB.tif"
fpath_mask = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-HOPS-mannual.tif"
os.chdir(os.path.dirname(fpath_img))

img = imread(fpath_img) / 10
mask = cv2.imread(fpath_mask, cv2.IMREAD_GRAYSCALE)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=600)
X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
surf = ax.plot_surface(
    X,
    Y,
    img,
    linewidth=2,
    antialiased=False,
    cmap="Blues",
    alpha=1,
    vmin=0,
    vmax=img.max() * 0.9,
)
ax.xaxis.set_pane_color((0.9, 0.9, 0.9))
ax.yaxis.set_pane_color((0.9, 0.9, 0.9))
ax.zaxis.set_pane_color((0.9, 0.9, 0.9))
plt.tick_params(labelsize=5, pad=0, direction="out")
ax.set_xticks(np.arange(0, img.shape[1], 2))
ax.set_yticks(np.arange(0, img.shape[0], 2))
for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)
plt.savefig("Fig1B-1.png", format="png", bbox_inches="tight", transparent=True)
