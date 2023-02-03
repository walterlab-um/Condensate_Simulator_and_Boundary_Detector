import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
import seaborn as sns

sns.set(color_codes=True, style="white")
from statannot import add_stat_annotation


path_truth = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_Dcp1a_HOPS/groundtruth.csv"
path_threshold = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_Dcp1a_HOPS/groundtruth.csv"
path_canny = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_Dcp1a_HOPS/groundtruth.csv"
path_blob = "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/mimic_Dcp1a_HOPS/groundtruth.csv"


data = df_all[df_all.HOPScondition == "300 mM Na+"]
x = "RNA"
y = "Percentage of Colocalization"
order = [
    "miR-21 double strand",
    "miR-21 guide strand",
    "THOR lncRNA",
    "THOR-delta lncRNA",
    "L941 lncRNA",
    "beta-Actin mRNA",
    "SOX2 mRNA",
]
box_pairs = [
    ("miR-21 double strand", "miR-21 guide strand"),
    ("THOR lncRNA", "THOR-delta lncRNA"),
    ("THOR lncRNA", "L941 lncRNA"),
    ("beta-Actin mRNA", "SOX2 mRNA"),
]

plt.figure(figsize=(5, 6), dpi=200)
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
plt.title("RNA Colocalization with HOPS Condensates (per cell)")
plt.tight_layout()
os.chdir(folderpath_save)
plt.savefig("RNA Colocalization with HOPS Condensates-percell.png", format="png")
plt.close()


y = "Number of RNAs per cell"
plt.figure(figsize=(5, 6), dpi=200)
sns.boxplot(data=data, x=x, y=y, order=order, showfliers=False)
plt.xticks(rotation=30)
plt.title("Number of RNAs per cell")
plt.tight_layout()
os.chdir(folderpath_save)
plt.savefig("Number of RNAs-percell.png", format="png")
