from os.path import join, dirname
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from lmfit.models import ConstantModel, GaussianModel

sns.set(color_codes=True, style="white")

path = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/HOPS_Dcp1a_Camera_Noise.tif"
video_noise = imread(path)
vector_nosie = np.reshape(video_noise, -1)


plt.figure(figsize=(6, 4), dpi=200)
g = sns.histplot(
    data=vector_nosie,
    fill=True,
    stat="count",
    alpha=0.5,
    color="dimgray",
    bins=50,
    binrange=(350, 450),
)
plt.xlim(350, 450)
plt.title("Camera Noise Determined by Experiment", fontsize=13, fontweight="bold")
plt.xlabel("Intensity (a.u.)", weight="bold")


# Fit to Gauss
counts, bins = np.histogram(vector_nosie, bins=50, range=(350, 450))

mod = GaussianModel()
pars = mod.guess(counts, x=(bins[1:] + bins[:-1]) / 2)
result = mod.fit(counts, pars, x=(bins[1:] + bins[:-1]) / 2)

plt.plot(
    (bins[1:] + bins[:-1]) / 2,
    result.best_fit,
    color=sns.color_palette()[3],
    linewidth=2,
)

# label with text
plt.text(
    0.65,
    0.83,
    r"$\mu$ = "
    + str(round(result.best_values["center"], 1))
    + ", $\sigma$ = "
    + str(round(result.best_values["sigma"], 1)),
    weight="bold",
    fontsize=15,
    color=sns.color_palette()[3],
    transform=plt.gcf().transFigure,
)

plt.tight_layout()
fsave = join(dirname(path), "Camera Noise Determined by Experiment.png")
plt.savefig(fsave, format="png")
