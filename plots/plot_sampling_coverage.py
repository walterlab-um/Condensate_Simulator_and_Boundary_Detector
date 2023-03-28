import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(color_codes=True, style="white")

###################################
# Parameters
bins = 10

folder = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/JPCB-CondensateBoundaryDetection/Simulated-4096"
)
os.chdir(folder)
data = pd.read_csv("groundtruth.csv", dtype=float)


###################################
# Main
plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(data["r_nm"], data["C_condensed"], color="gray", alpha=0.5, s=1)
plt.xlim(100, 600)
plt.ylim(2, 10)
plt.xlabel("Condensate Radius, nm", weight="bold")
plt.ylabel("Partition Coefficient", weight="bold")
plt.title("Sampling Coverage", weight="bold")
plt.tight_layout()
plt.savefig("Sampling Coverage.png", format="png", bbox_inches="tight")
plt.close()
