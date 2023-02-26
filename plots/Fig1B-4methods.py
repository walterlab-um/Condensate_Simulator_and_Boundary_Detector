import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
from lmfit.models import GaussianModel
from tifffile import imread
import pandas as pd

plow = 0.05  # imshow intensity percentile
phigh = 95

fpath_img_PB = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-PB.tif"
fpath_img_HOPS = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-HOPS.tif"
fpath_mask = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/RealData-PB-mannual.tif"
os.chdir(os.path.dirname(fpath_img_PB))

img_PB = imread(fpath_img_PB) / 10
img_HOPS = imread(fpath_img_HOPS)
mask = cv2.imread(fpath_mask, cv2.IMREAD_GRAYSCALE)


def plot_style():
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xlim(0, 24)
    plt.locator_params(axis="y", nbins=3)
    plt.locator_params(axis="x", nbins=4)
    plt.yticks(fontsize=25, rotation="vertical", weight="bold", va="center")
    plt.xticks(fontsize=25, weight="bold")
    plt.xlabel("")
    plt.ylabel("")


# fig1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=600)
X, Y = np.meshgrid(np.arange(25), np.arange(25))
ax.plot_surface(
    X,
    Y,
    img_PB,
    antialiased=True,
    cmap="Blues",
    edgecolor="darkblue",
    lw=0.1,
    alpha=0.3,
    vmin=0,
    vmax=img_PB.max() * 0.9,
)
ax.plot(
    xs=np.arange(25),
    ys=img_PB[13, :],
    zdir="y",
    zs=13,
    color="black",
    lw=3,
)
xx, zz = np.meshgrid(np.arange(25), np.linspace(img_PB.min(), img_PB.max() + 50, 25))
yy = np.ones((25, 25)) * 13
ax.plot_surface(
    xx,
    yy,
    zz,
    antialiased=True,
    lw=0,
    alpha=0.3,
    color="gray",
)
ax.xaxis.set_pane_color((0.9, 0.9, 0.9))
ax.yaxis.set_pane_color((0.9, 0.9, 0.9))
ax.zaxis.set_pane_color((0.9, 0.9, 0.9))
plt.tick_params(labelsize=5, pad=0, direction="out")
ax.set_xticks(np.arange(0, img_PB.shape[1], 2))
ax.set_yticks(np.arange(0, img_PB.shape[0], 2))
for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)
ax.view_init(elev=20, azim=-45, roll=0)
plt.savefig(
    "Fig1B-3dsurface-PB-final.png", format="png", bbox_inches="tight", transparent=True
)
plt.close()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=600)
X, Y = np.meshgrid(np.arange(img_HOPS.shape[1]), np.arange(img_HOPS.shape[0]))
ax.plot_surface(
    X,
    Y,
    img_HOPS,
    antialiased=True,
    cmap="Blues",
    edgecolor="darkblue",
    lw=0.1,
    alpha=0.3,
    vmin=0,
    vmax=img_HOPS.max() * 0.9,
)
ax.plot(
    xs=np.arange(13),
    ys=img_HOPS[9, :],
    zdir="y",
    zs=img_HOPS.shape[0],
    color="black",
    lw=3,
)
ax.xaxis.set_pane_color((0.9, 0.9, 0.9))
ax.yaxis.set_pane_color((0.9, 0.9, 0.9))
ax.zaxis.set_pane_color((0.9, 0.9, 0.9))
plt.tick_params(labelsize=5, pad=0, direction="out")
ax.set_xticks(np.arange(0, img_HOPS.shape[1], 2))
ax.set_yticks(np.arange(0, img_HOPS.shape[0], 2))
for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)
ax.view_init(elev=20, azim=-55, roll=0)
plt.savefig(
    "Fig1B-3dsurface-HOPS.png", format="png", bbox_inches="tight", transparent=True
)
plt.close()


# fig2
plt.figure(figsize=(12, 4), dpi=300)
intensity = img_PB[13, :]
smoothed = gaussian_filter1d(intensity, 2)
x = np.arange(25)
plt.plot(x, smoothed, lw=10, color="black")
plot_style()
plt.savefig("Fig1B-intensity.png", format="png", bbox_inches="tight")
plt.close()

fig = Figure(facecolor="none")
fig.text(0, 0, r"$G \otimes I$", math_fontfamily="cm", weight=100)
fig.savefig(
    "Fig1B-intensity-equation.png",
    format="png",
    bbox_inches="tight",
    dpi=2000,
)
fig.clear()


# fig3
plt.figure(figsize=(12, 4), dpi=300)
gradient = np.gradient(gaussian_filter1d(intensity, 2))
plt.plot(x, gradient, lw=10, color="black")
plot_style()
plt.savefig("Fig1B-gradient.png", format="png", bbox_inches="tight")
plt.close()

fig = Figure(facecolor="none")
fig.text(
    0,
    0,
    r"$\frac{d}{dx} G \otimes I$",
    math_fontfamily="cm",
    weight=100,
)
fig.savefig(
    "Fig1B-gradient-equation.png",
    format="png",
    bbox_inches="tight",
    dpi=2000,
)
fig.clear()

# fig4
plt.figure(figsize=(12, 4), dpi=300)
# intensity = img_HOPS[9, :]
LoG = np.gradient(np.gradient(gaussian_filter1d(intensity, 1)))
plt.plot(x, LoG, lw=10, color="black")
plot_style()
plt.savefig("Fig1B-laplacian.png", format="png", bbox_inches="tight")
plt.close()

fig = Figure(facecolor="none")
fig.text(
    0,
    0,
    r"$\frac{d^2}{dx^2} G \otimes I$",
    math_fontfamily="cm",
    weight=100,
)
fig.savefig(
    "Fig1B-laplacian-equation.png",
    format="png",
    bbox_inches="tight",
    dpi=2000,
)
fig.clear()

plt.figure(figsize=(12, 4), dpi=300)
plt.plot(x, intensity, lw=10, color="black")

mod = GaussianModel()
pars = mod.guess(intensity, x=x)
result = mod.fit(intensity, pars, x=x)

plt.plot(
    x,
    result.best_fit,
    color="firebrick",
    linewidth=5,
    ls="--",
)

plt.axvline(result.best_values["center"], color="gray", lw=0.1)
plt.axvline(
    result.best_values["center"] + result.best_values["sigma"], color="gray", lw=0.1
)
plt.axvline(
    result.best_values["center"] - result.best_values["sigma"], color="gray", lw=0.1
)
plot_style()
plt.savefig("Fig1B-laplacian-GaussFit.png", format="png", bbox_inches="tight")
plt.close()


# kernels
x = np.linspace(-3, 3, 120)

G = np.exp(-np.power(x - 0, 2) / (2 * np.power(1, 2)))
plt.figure(figsize=(5, 5), dpi=300, linewidth=20, edgecolor="black")
plt.plot(x, G, lw=5, color="black")
plt.xlim(x.min(), x.max())
plt.axis("off")
plt.savefig("Fig1B-kernel-1.png", bbox_inches="tight", format="png")
plt.close()

gofG = np.gradient(G)
plt.figure(figsize=(5, 5), dpi=300, linewidth=20, edgecolor="black")
plt.plot(x, gofG, lw=5, color="black")
plt.xlim(x.min(), x.max())
plt.axis("off")
plt.savefig("Fig1B-kernel-2.png", bbox_inches="tight", format="png")
plt.close()

LoG = np.gradient(np.gradient(G))
plt.figure(figsize=(5, 5), dpi=300, linewidth=20, edgecolor="black")
plt.plot(x, LoG, lw=5, color="black")
plt.xlim(x.min(), x.max())
plt.axis("off")
plt.savefig("Fig1B-kernel-3.png", bbox_inches="tight", format="png")
plt.close()

# last panel: manual labeling and ML
plt.figure(figsize=(5, 5), dpi=300)
vmin, vmax = np.percentile(img_PB, (plow, phigh))
plt.imshow(img_PB, cmap="Blues", vmin=vmin, vmax=vmax)
plt.axis("off")
plt.savefig("Fig1B-ML-orginal.png", bbox_inches="tight", format="png")
plt.close()

plt.figure(figsize=(5, 5), dpi=300)
plt.imshow(img_PB, cmap="Blues", vmin=vmin, vmax=vmax)
plt.imshow(mask, cmap="Oranges", alpha=0.3, vmax=2)
plt.axis("off")
plt.savefig("Fig1B-ML-manualmask.png", bbox_inches="tight", format="png")
plt.close()


df = pd.read_csv(
    "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig1-detailed4methods/Fig1B-ML-probability.csv"
)
data = df["Gray_Value"].to_numpy(dtype=float)
plt.figure(figsize=(12, 4), dpi=300)
plt.plot(np.arange(data.shape[0]), data, lw=10, color="black")
plot_style()
plt.savefig("Fig1B-ML-probability.png", format="png", bbox_inches="tight")
plt.close()
