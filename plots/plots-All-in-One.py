import os
from os.path import join, isdir, exists
import shutil
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

###################################
# Parameters
bins = 6

folder = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Simulated-4096"
)
os.chdir(folder)
lst_metric = [
    "deviation_center",
    "rmsd_edge",
    "fold_deviation_area",
    "fold_deviation_PC",
    "fold_deviation_PC_max",
]
dict_subtitle = {
    "deviation_center": "Center Deviation, nm",
    "rmsd_edge": "Edge Deviation RMSD, nm",
    "fold_deviation_area": "Area Deviation Fold Change",
    "fold_deviation_PC": "PC Deviation Fold Change",
    "fold_deviation_PC_max": "PC-max Deviation Fold Change",
}
dict_cmap = {
    "deviation_center": "magma",
    "rmsd_edge": "magma",
    "fold_deviation_area": "seismic",
    "fold_deviation_PC": "seismic",
    "fold_deviation_PC_max": "seismic",
}
cmap_default = "magma"
dict_vrange = {
    "deviation_center": (70, 150),
    "rmsd_edge": (50, 200),
    "fold_deviation_area": (0, 2),
    "fold_deviation_PC": (0, 2),
    "fold_deviation_PC_max": (0, 2),
}
dict_vrange_var = {
    "deviation_center": (10**2, 10**3),
    "rmsd_edge": (1, 10**3),
    "fold_deviation_area": (10 ** (-4), 10 ** (-1)),
    "fold_deviation_PC": (10 ** (-5), 10 ** (-3)),
    "fold_deviation_PC_max": (10 ** (-5), 10 ** (-3)),
}


###################################
# Functions
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


def plot_heatmap(heatmap, subfolder, subtitle, cmap, norm=None):
    global xticks, yticks
    # plot heatmaps for different quantities, in both mean and varience
    plt.figure(figsize=(6, 5), dpi=300)
    if norm == None:
        ax = sns.heatmap(
            data=heatmap,
            xticklabels=xticks,
            yticklabels=yticks,
            annot=True,
            cmap=cmap,
            vmax=1,
            vmin=0,
        )
    else:
        ax = sns.heatmap(
            data=heatmap,
            xticklabels=xticks,
            yticklabels=yticks,
            annot=True,
            cmap=cmap,
            norm=norm,
        )
    ax.invert_yaxis()
    plt.xlabel("Partition Coefficient", weight="bold")
    plt.ylabel("Condensate Radius, nm", weight="bold")
    title = " ".join([s.capitalize() for s in subfolder.split("-")]) + "\n" + subtitle
    path_save = join(
        "Results-heatmap",
        (
            "_".join([s.capitalize() for s in subfolder.split("-")])
            + "-"
            + subtitle
            + ".png"
        ),
    )
    plt.title(title, weight="bold")
    plt.tight_layout()
    plt.savefig(path_save, format="png", bbox_inches="tight")
    plt.close()


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
xticks = [round(x, 1) for x in (pc[:-1] + pc[1:]) / 2]
yticks = [round(x, 1) for x in (r[:-1] + r[1:]) / 2]

# loop through all subfolders
lst_subfolders = [
    f for f in os.listdir(folder) if isdir(f) & (not f.startswith("Results"))
]
for subfolder in track(lst_subfolders):
    fname = [f for f in os.listdir(subfolder) if f.endswith("results.csv")][0]
    df_result = pd.read_csv(join(subfolder, fname), dtype=float)

    # plot fail rate as heatmap
    heatmap_fail = assemble_heatmap(heatmap_fail)
    plot_heatmap(heatmap_fail, subfolder, "Fail Rate", cmap_default)

    for metric in lst_metric:
        # assemble heatmap for different quantities
        heatmap_mean = assemble_heatmap(heatmap_mean, metric, "mean")
        heatmap_var = assemble_heatmap(heatmap_var, metric, "var")

        # plot heatmaps for different quantities, in both mean and varience
        if metric in ["deviation_center", "rmsd_edge"]:
            norm = LogNorm(vmin=dict_vrange[metric][0], vmax=dict_vrange[metric][1])
            var_norm = LogNorm(
                vmin=dict_vrange_var[metric][0], vmax=dict_vrange_var[metric][1]
            )
        elif metric in [
            "fold_deviation_area",
            "fold_deviation_PC",
            "fold_deviation_PC_max",
        ]:
            norm = TwoSlopeNorm(
                1, vmin=dict_vrange[metric][0], vmax=dict_vrange[metric][1]
            )
            var_norm = LogNorm(
                vmin=dict_vrange_var[metric][0], vmax=dict_vrange_var[metric][1]
            )

        plot_heatmap(
            heatmap_mean,
            subfolder,
            dict_subtitle[metric],
            dict_cmap[metric],
            norm,
        )
        plot_heatmap(
            heatmap_var,
            subfolder,
            dict_subtitle[metric] + "-" + "Variance",
            cmap_default,
            var_norm,
        )
