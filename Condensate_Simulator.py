import numpy as np
from numpy.random import rand
import pandas as pd

fovsize = 5000  # unit: nm
real_img_pxlsize = 100  # unit: nm
truth_img_pxlsize = 10  # unit: nm
condensate_r_range = (100, 2000)  # unit: nm
N = 1  # number of condensates per field of view
