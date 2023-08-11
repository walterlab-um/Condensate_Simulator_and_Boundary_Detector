import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

bins = 6
os.chdir(
    "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Fig_TOC/"
)
metric = "rmsd_edge"
title = "Edge Deviation RMSD, nm"
cmap = "magma"
vrange = (50, 200)


def assemble_heatmap(heatmap, metric=None, operation="rate"):
    global r, pc, df_result
    # assemble heatmap for different quantities
    for row in np.arange(len(r) - 1):
        for column in np.arange(len(pc) - 1):
            range_r = (r[row], r[row + 1])
            range_pc = (pc[column], pc[column + 1])

            within_r_range = df_result[
                (df_result["truth_r"] > range_r[0])
                & (df_result["truth_r"] <= range_r[1])
            ]
            within_r_and_pc_range = within_r_range[
                (within_r_range["truth_pc"] > range_pc[0])
                & (within_r_range["truth_pc"] <= range_pc[1])
            ]

            if operation == "rate":
                rate = (
                    within_r_and_pc_range[
                        within_r_and_pc_range["success"] == False
                    ].shape[0]
                    / within_r_and_pc_range.shape[0]
                )
                heatmap[row, column] = rate
                continue

            if (
                np.isnan(within_r_and_pc_range[metric]).sum()
                == within_r_and_pc_range.shape[0]
            ):
                heatmap[row, column] = np.nan
                continue

            if operation == "mean":
                heatmap[row, column] = np.nanmean(within_r_and_pc_range[metric])

            if operation == "var":
                heatmap[row, column] = np.nanvar(within_r_and_pc_range[metric])

    return heatmap


# initialize heatmaps
r = np.linspace(100, 600, bins)
pc = np.linspace(2, 10, bins)
heatmap_mean = np.zeros((bins - 1, bins - 1))

lst_fourmethods = [
    "Method1_results.csv",
    "Method2_results.csv",
    "Method3_results.csv",
    "Method4_results.csv",
]
for fname in track(lst_fourmethods):
    df_result = pd.read_csv(fname, dtype=float)
    # assemble heatmap for different quantities
    heatmap_mean = assemble_heatmap(heatmap_mean, metric, "mean")
    norm = LogNorm(vmin=vrange[0], vmax=vrange[1])

    plt.figure(figsize=(5, 5), dpi=300)
    ax = sns.heatmap(
        data=heatmap_mean,
        cmap=cmap,
        norm=norm,
        cbar=False,
    )
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(fname[:-4] + ".png", format="png", bbox_inches="tight")
    plt.close()
