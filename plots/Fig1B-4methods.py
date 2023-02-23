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

# fig1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=600)
X, Y = np.meshgrid(np.arange(25), np.arange(25))
ax.plot_surface(
    X,
    Y,
    img[:, :],
    antialiased=True,
    cmap="Blues",
    edgecolor="darkblue",
    lw=0.1,
    alpha=0.3,
    vmin=0,
    vmax=img.max() * 0.9,
)
# ax.plot(
#     ys=np.repeat(13, 25),
#     xs=np.arange(25),
#     zs=img[13, :] + 0.5,
#     color="black",
#     lw=3,
#     visible=True,
# )
ax.plot(
    xs=np.arange(25),
    ys=img[13, :] + 0.5,
    zdir="y",
    zs=25,
    color="black",
    lw=3,
    visible=True,
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
ax.view_init(elev=20, azim=-55, roll=0)
plt.savefig("Fig1B-1.png", format="png", bbox_inches="tight", transparent=True)
plt.close()


# fig2
plt.figure(figsize=(12, 4), dpi=300)
intensity = img[13, :]
x = np.arange(25)
plt.plot(x, intensity, lw=10, color="black")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim(0, 24)
plt.xlabel("x, pixels", weight="bold")
plt.ylabel(
    (r"$I$"),
    fontsize=80,
    math_fontfamily="cm",
    weight="bold",
)
plt.savefig("Fig1B-2.png", format="png", bbox_inches="tight")
plt.close()

# fig3
plt.figure(figsize=(12, 4), dpi=300)
gradient = np.diff(intensity, 1)
x = np.arange(25)
plt.plot((x[:-1] + x[1:]) / 2, gradient, lw=10, color="black")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim(0, 24)
plt.xlabel("x, pixels", weight="bold")
# plt.ylabel(r"$abs(\nabla \mathtt{I})$", weight="bold", fontsize=30)
plt.ylabel(
    r"$| \frac{\partial I}{\partial x} |$",
    math_fontfamily="cm",
    weight="bold",
    fontsize=80,
)
plt.savefig("Fig1B-3.png", format="png", bbox_inches="tight")
plt.close()


# fig4
plt.figure(figsize=(12, 4), dpi=300)
gradient = np.diff(intensity, 2)
x = np.arange(25)
plt.plot((x[:-2] + x[2:]) / 2, gradient, lw=10, color="black")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim(0, 24)
plt.xlabel("x, pixels", weight="bold")
# plt.ylabel(r"$abs(\nabla \mathtt{I})$", weight="bold", fontsize=30)
plt.ylabel(
    r"$| \frac{\partial^2 I}{\partial x^2} |$",
    math_fontfamily="cm",
    weight="bold",
    fontsize=80,
)
plt.savefig("Fig1B-4.png", format="png", bbox_inches="tight")
plt.close()
