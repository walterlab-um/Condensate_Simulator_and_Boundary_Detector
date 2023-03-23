import os
from os.path import join, isdir, exists
import shutil
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

###################################
# Parameters
bins = 10

folder = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Simulated-1024"
)
os.chdir(folder)
lst_metric = [
    "deviation_center",
    "rmsd_edge",
    "fold_deviation_area",
    "fold_deviation_PC",
]
dict_subtitle = {
    "deviation_center": "Center Deviation",
    "rmsd_edge": "Edge Deviation RMSD",
    "fold_deviation_area": "Area Deviation Fold Change",
    "fold_deviation_PC": "PC Deviation Fold Change",
}
dict_cmap = {
    "deviation_center": "magma",
    "rmsd_edge": "magma",
    "fold_deviation_area": "seismic",
    "fold_deviation_PC": "seismic",
}
cmap_default = "magma"

divnorm = colors.TwoSlopeNorm(0)


###################################
# Functions
def assemble_heatmap(heatmap, metric, operation):
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

            if operation == "rate":
                rate = (
                    within_r_and_pc_range[
                        within_r_and_pc_range["success" == False]
                    ].shape[0]
                    / within_r_and_pc_range.shape[0]
                )
                heatmap[row, column] = rate

    return heatmap


###################################
# Main
if exists("Results-heatmap"):
    shutil.rmtree("Results-heatmap")
os.mkdir("Results-heatmap")

# initialize heatmaps
r = np.linspace(100, 600, bins)
pc = np.linspace(2, 10, bins)
heatmap_mean = np.zeros((bins - 1, bins - 1))
heatmap_var = np.zeros((bins - 1, bins - 1))
heatmap_fail = np.zeros((bins - 1, bins - 1))

# ticks labels for all heatmaps
xticks = ((pc[:-1] + pc[1:]) / 2).astype(int)
yticks = ((r[:-1] + r[1:]) / 2).astype(int)

# loop through all subfolders
lst_subfolders = [
    f for f in os.listdir(folder) if isdir(f) & (not f.startswith("Results"))
]
for subfolder in track(lst_subfolders):
    fname = [f for f in os.listdir(subfolder) if f.endswith("results.csv")][0]
    df_result = pd.read_csv(join(subfolder, fname), dtype=float)

    # plot fail rate as heatmap
    plt.figure(dpi=300)
    data = df_result[df_result["success"] == False]
    sns.histplot(
        data,
        x="truth_pc",
        y="truth_r",
        cbar=True,
        bins=(pc, r),
        cmap=cmap_default,
        stat="density",
        fill=True,
    )
    plt.xlim(2, 10)
    plt.ylim(100, 600)
    plt.xlabel("Partition Coefficient", weight="bold")
    plt.ylabel("Condensate Radius, nm", weight="bold")
    title = (
        " ".join([s.capitalize() for s in subfolder.split("-")]) + "\n" + "- Fail Rate"
    )
    path_save = join(
        "Results-heatmap",
        ("_".join([s.capitalize() for s in subfolder.split("-")]) + "-Fail Rate.png"),
    )
    plt.title(title, weight="bold")
    plt.tight_layout()
    plt.savefig(path_save, format="png", bbox_inches="tight")
    plt.close()

    for metric in lst_metric:
        # assemble heatmap for different quantities
        heatmap_mean = assemble_heatmap(heatmap_mean, metric, "mean")
        heatmap_var = assemble_heatmap(heatmap_var, metric, "var")

        # plot heatmaps for different quantities, in both mean and varience
        plt.figure(dpi=300)
        ax = sns.heatmap(
            data=heatmap_mean,
            xticklabels=xticks,
            yticklabels=yticks,
            robust=True,
            cmap=dict_cmap[metric],
        )
        ax.invert_yaxis()
        plt.xlabel("Partition Coefficient", weight="bold")
        plt.ylabel("Condensate Radius, nm", weight="bold")
        title = (
            " ".join([s.capitalize() for s in subfolder.split("-")])
            + "\n"
            + dict_subtitle[metric]
            + " - Mean"
        )
        path_save = join(
            "Results-heatmap",
            (
                "_".join([s.capitalize() for s in subfolder.split("-")])
                + "-"
                + dict_subtitle[metric]
                + "-Mean.png"
            ),
        )
        plt.title(title, weight="bold")
        plt.tight_layout()
        plt.savefig(path_save, format="png", bbox_inches="tight")
        plt.close()

        plt.figure(dpi=300)
        ax = sns.heatmap(
            data=heatmap_var,
            xticklabels=xticks,
            yticklabels=yticks,
            robust=True,
            cmap=cmap_default,
        )
        ax.invert_yaxis()
        plt.xlabel("Partition Coefficient", weight="bold")
        plt.ylabel("Condensate Radius, nm", weight="bold")
        title = (
            " ".join([s.capitalize() for s in subfolder.split("-")])
            + "\n"
            + dict_subtitle[metric]
            + " - Variance"
        )
        path_save = join(
            "Results-heatmap",
            (
                "_".join([s.capitalize() for s in subfolder.split("-")])
                + "-"
                + dict_subtitle[metric]
                + "-Variance.png"
            ),
        )
        plt.title(title, weight="bold")
        plt.tight_layout()
        plt.savefig(path_save, format="png", bbox_inches="tight")
        plt.close()