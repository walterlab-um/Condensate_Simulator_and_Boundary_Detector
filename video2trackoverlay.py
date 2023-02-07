import numpy as np
import pandas as pd
import pims
import trackpy as tp
import matplotlib as mpl
import matplotlib.pyplot as plt

path = "/Users/GGM/Documents/Graduate_Work/Nils_Walter_Lab/Writing/MyPublications/ResearchArticle-JPCB/figure-materials/Example-Slow Dwelling-bandpass-RNA.tif"
frames = pims.open(path)


spots = tp.batch(frames, diameter=5, separation=3, minmass=1)
tracks = tp.link(spots, search_range=3)
tracks_filtered = tp.filter_stubs(tracks, threshold=5)
