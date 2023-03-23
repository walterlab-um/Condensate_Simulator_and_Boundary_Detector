import os
from os.path import join, exists
import shutil
from tkinter import filedialog as fd
from rich.progress import track

folder = fd.askdirectory()
os.chdir(folder)

lst_subfolder = [
    "PC-Deviation",
    "Fail-Rate",
    "Edge-RMSD",
    "Center-Deviation",
    "Area-Deviation",
]

for subfolder in track(lst_subfolder):
    if not exists(subfolder):
        os.mkdir(subfolder)

    marker = subfolder.split("-")[0]
    for f in os.listdir(folder):
        if f.endswith(".png") & (marker in f):
            shutil.move(f, join(subfolder, f))

    if not exists(join(subfolder, "Variance")):
        os.mkdir(join(subfolder, "Variance"))

    for f in os.listdir(subfolder):
        if f.endswith("Variance.png"):
            shutil.move(join(subfolder, f), join(subfolder, "Variance", f))
