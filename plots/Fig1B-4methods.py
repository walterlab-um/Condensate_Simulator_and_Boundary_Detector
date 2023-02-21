import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from tifffile import imread

plow = 0.05  # imshow intensity percentile
phigh = 95

fpath_img = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-PB.tif"
fpath_mask = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-HOPS-mannual.tif"

img = imread(fpath_img)
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
    alpha=0.7,
)
# ax.plot_wireframe(
#     X,
#     Y,
#     img,
#     color="black",
#     alpha=0.5,
#     lw=1,
# )
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, img.shape[0])
plt.ylim(0, img.shape[1])
plt.tight_layout()
# plt.axis("scaled")
# plt.axis("off")
fpath_save = fpath_img[:-4] + "-3d.png"
plt.savefig(fpath_save, format="png", bbox_inches="tight", dpi=300)
