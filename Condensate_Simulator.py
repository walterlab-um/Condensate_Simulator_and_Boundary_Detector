import numpy as np
from numpy.random import rand
import pandas as pd
from tifffile import imwrite

##################################
# Parameters
fovsize = 5000  # unit: nm
real_img_pxlsize = 100  # unit: nm
truth_img_pxlsize = 10  # unit: nm
condensate_r_range = (100, 2000)  # unit: nm
pad_size = 200  # push condensates back from FOV edges. unit: nm
N_condensate = 1  # number of condensates per field of view
N_fov = 100  # number of total im


##################################
# Main
# generate condensate radius
condensate_r = (
    rand(N_fov * N_condensate) * (condensate_r_range[1] - condensate_r_range[0])
    + condensate_r_range[0]
)
# make sure condensates are padded from FOV edges
coor_min = condensate_r + pad_size
coor_max = fovsize - condensate_r - pad_size
# generate condensate center coordinates
center_x = []
center_y = []
for current_min, current_max in zip(coor_min, coor_max):
    center_x.append(rand() * (current_max - current_min) + current_min)
    center_y.append(rand() * (current_max - current_min) + current_min)
