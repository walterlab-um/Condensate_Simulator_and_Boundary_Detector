import os
from os.path import join, dirname
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
import seaborn as sns

sns.set(color_codes=True, style="white")
from statannot import add_stat_annotation


path_truth = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_Dcp1a_HOPS/groundtruth.csv"
path_threshold = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_Dcp1a_HOPS/Contours_Denoise_Threshold_validation_results.csv"
path_canny = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_Dcp1a_HOPS/Contours_Canny_validation_results.csv"
path_blob = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_Dcp1a_HOPS/GaussFit_validation_results.csv"


# path_truth = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_PB/groundtruth.csv"
# path_threshold = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_PB/Contours_Denoise_Threshold_validation_results.csv"
# path_canny = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_PB/Contours_Canny_validation_results.csv"
# path_blob = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_PB/GaussFit_validation_results.csv"


# path_truth = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/highC-small/groundtruth.csv"
# path_threshold = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/highC-small/Contours_Denoise_Threshold_validation_results.csv"
# path_canny = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/highC-small/Contours_Canny_validation_results.csv"
# path_blob = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/highC-small/GaussFit_validation_results.csv"
os.chdir(dirname(path_truth))


def rmsd_d2edge(df, tag):
    array_d2edge = np.array([], dtype=float)
    for row in df["distance2edge"]:
        if tag == "BlobDetector":
            array_d2edge = np.append(array_d2edge, row)
            continue

        if type(row) != str:
            # lst_rmsd_d2edge.append(np.nan)
            continue
        lst_current_d2edge = (
            row.replace("\n", "").replace("[", "").replace("]", "").split(" ")
        )
        current_d2edge = np.array([x for x in lst_current_d2edge if x], dtype=float)
        array_d2edge = np.concatenate((array_d2edge, current_d2edge))
    rmsd_d2edge = np.sqrt(np.nanmean(array_d2edge**2))

    return rmsd_d2edge


def compute_chunck(df, tag):
    falserate = []
    rmsd_area_nm2 = []
    rmsd_aspect_ratio = []
    rmsd_distance2center = []
    rmsd_distance2edge = []
    for i in range(30):
        df_current = df.iloc[i * 100 : (i + 1) * 100 - 1]
        falserate.append(np.sum(df_current["success"]))
        rmsd_distance2edge.append(rmsd_d2edge(df_current, tag))
        rmsd_area_nm2.append(np.sqrt(np.nanmean(df_current["rmsd_area_nm2"] ** 2)))
        rmsd_aspect_ratio.append(
            np.sqrt(np.nanmean(df_current["rmsd_aspect_ratio"] ** 2))
        )
        rmsd_distance2center.append(
            np.sqrt(np.nanmean(df_current["rmsd_distance2center"] ** 2))
        )
    df_new = pd.DataFrame(
        {
            "class": np.repeat(tag, len(rmsd_area_nm2)),
            "falserate": falserate,
            "rmsd_area_nm2": rmsd_area_nm2,
            "rmsd_aspect_ratio": rmsd_aspect_ratio,
            "rmsd_distance2center": rmsd_distance2center,
            "rmsd_distance2edge": rmsd_distance2edge,
        },
        dtype=object,
    )

    return df_new


# Calculate RMSD
df_truth = pd.read_csv(path_truth)
df_truth["area_nm2"] = df_truth["r_nm"].to_numpy(dtype=float) ** 2 * np.pi

# for method 1
df_threshold = pd.read_csv(path_threshold)
df_threshold["index"] = df_threshold["index"].to_numpy(dtype=int)
df_threshold["rmsd_area_nm2"] = df_truth["area_nm2"] - df_threshold["area_nm2"]
df_threshold["rmsd_aspect_ratio"] = 1 - df_threshold["aspect_ratio"]
df_threshold["rmsd_distance2center"] = df_threshold["distance2center"]
df1 = compute_chunck(df_threshold, "Thresholding")


# for method 2
df_canny = pd.read_csv(path_canny)
df_canny["index"] = df_canny["index"].to_numpy(dtype=int)
df_canny["rmsd_area_nm2"] = df_truth["area_nm2"] - df_canny["area_nm2"]
df_canny["rmsd_aspect_ratio"] = 1 - df_canny["aspect_ratio"]
df_canny["rmsd_distance2center"] = df_canny["distance2center"]
df2 = compute_chunck(df_canny, "CannyEdge")

# for method 3
df_blob = pd.read_csv(path_blob)
df_blob["index"] = df_blob["index"].to_numpy(dtype=int)
df_blob["rmsd_area_nm2"] = df_truth["area_nm2"] - df_blob["area_nm2"]
df_blob["rmsd_aspect_ratio"] = 1 - df_blob["aspect_ratio"]
df_blob["rmsd_distance2center"] = df_blob["distance2center"]
df3 = compute_chunck(df_blob, "BlobDetector")

data = pd.concat([df1, df2, df3])
# data.dropna(inplace=True)
data.to_csv("data.csv", index=False)

# area
x = "class"
y = "rmsd_area_nm2"
order = ["Thresholding", "CannyEdge", "BlobDetector"]
box_pairs = [
    ("Thresholding", "CannyEdge"),
    ("Thresholding", "BlobDetector"),
    ("CannyEdge", "BlobDetector"),
]
plt.figure(figsize=(3, 6), dpi=200)
ax = sns.boxplot(data=data, x=x, y=y, order=order)
ax = sns.swarmplot(data=data, x=x, y=y, order=order, color=".25")
test_results = add_stat_annotation(
    ax,
    data=data,
    x=x,
    y=y,
    order=order,
    box_pairs=box_pairs,
    test="t-test_ind",
    comparisons_correction=None,
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.xticks(rotation=30)
plt.ylabel("RMSD Area (nm${^2}$)", weight="bold")
ax.set_xlabel(None)
plt.tight_layout()
plt.savefig("RMSD-Area.png", format="png", bbox_inches="tight")
plt.close()

# rmsd_aspect_ratio
x = "class"
y = "rmsd_aspect_ratio"
order = ["Thresholding", "CannyEdge", "BlobDetector"]
box_pairs = [
    ("Thresholding", "CannyEdge"),
    ("Thresholding", "BlobDetector"),
    ("CannyEdge", "BlobDetector"),
]
plt.figure(figsize=(3, 6), dpi=200)
ax = sns.boxplot(data=data, x=x, y=y, order=order)
ax = sns.swarmplot(data=data, x=x, y=y, order=order, color=".25")
test_results = add_stat_annotation(
    ax,
    data=data,
    x=x,
    y=y,
    order=order,
    box_pairs=box_pairs,
    test="t-test_ind",
    comparisons_correction=None,
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.xticks(rotation=30)
plt.ylabel("RMSD Aspect Ratio", weight="bold")
ax.set_xlabel(None)
plt.tight_layout()
plt.savefig("RMSD-AspectRatio.png", format="png", bbox_inches="tight")
plt.close()

# rmsd_distance2center
x = "class"
y = "rmsd_distance2center"
order = ["Thresholding", "CannyEdge", "BlobDetector"]
box_pairs = [
    ("Thresholding", "CannyEdge"),
    ("Thresholding", "BlobDetector"),
    ("CannyEdge", "BlobDetector"),
]
plt.figure(figsize=(3, 6), dpi=200)
ax = sns.boxplot(data=data, x=x, y=y, order=order)
ax = sns.swarmplot(data=data, x=x, y=y, order=order, color=".25")
test_results = add_stat_annotation(
    ax,
    data=data,
    x=x,
    y=y,
    order=order,
    box_pairs=box_pairs,
    test="t-test_ind",
    comparisons_correction=None,
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.xticks(rotation=30)
plt.ylabel("RMSD, center (nm)", weight="bold")
ax.set_xlabel(None)
plt.tight_layout()
plt.savefig("RMSD-center.png", format="png", bbox_inches="tight")
plt.close()

# rmsd_distance2edge

x = "class"
y = "rmsd_distance2edge"
order = ["Thresholding", "CannyEdge", "BlobDetector"]
box_pairs = [
    ("Thresholding", "CannyEdge"),
    ("Thresholding", "BlobDetector"),
    ("CannyEdge", "BlobDetector"),
]
plt.figure(figsize=(3, 6), dpi=200)
ax = sns.boxplot(data=data, x=x, y=y, order=order)
ax = sns.swarmplot(data=data, x=x, y=y, order=order, color=".25")
test_results = add_stat_annotation(
    ax,
    data=data,
    x=x,
    y=y,
    order=order,
    box_pairs=box_pairs,
    test="t-test_ind",
    comparisons_correction=None,
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.xticks(rotation=30)
plt.ylabel("RMSD, edge (nm)", weight="bold")
ax.set_xlabel(None)
plt.tight_layout()
plt.savefig("RMSD-edge.png", format="png", bbox_inches="tight")
plt.close()

# False rate
x = "class"
y = "falserate"
order = ["Thresholding", "CannyEdge", "BlobDetector"]
box_pairs = [
    ("Thresholding", "CannyEdge"),
    ("Thresholding", "BlobDetector"),
    ("CannyEdge", "BlobDetector"),
]
plt.figure(figsize=(3, 6), dpi=200)
ax = sns.boxplot(data=data, x=x, y=y, order=order)
ax = sns.swarmplot(data=data, x=x, y=y, order=order, color=".25")
test_results = add_stat_annotation(
    ax,
    data=data,
    x=x,
    y=y,
    order=order,
    box_pairs=box_pairs,
    test="t-test_ind",
    comparisons_correction=None,
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.xticks(rotation=30)
plt.ylabel("Positive Rate %", weight="bold")
ax.set_xlabel(None)
plt.tight_layout()
plt.savefig("positiverate.png", format="png", bbox_inches="tight")
plt.close()
