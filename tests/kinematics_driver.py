#!/usr/bin/env python3


import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

from sim_engine_3d import *

simulation = SimEngine3D("../models/four_link.mdl")

# import numpy as np
# e1_crank = 0.71
# e2_crank = 0.0
# e3_crank = 0.0
# e0_crank = np.sqrt(1 - e1_crank**2 - e2_crank**2 - e3_crank**2)
# print(e0_crank)
# # 0.7042016756583301
#
# e1_rod = -0.21
# e2_rod = 0.40
# e3_rod = -0.1
# e0_rod = np.sqrt(1 - e1_rod**2 - e2_rod**2 - e3_rod**2)
# print(e0_rod)
# # 0.8865100112237876

# e1_slider = 0.0
# e2_slider = 0.0
# e3_slider = 0.0
# e0_slider = np.sqrt(1 - e1_slider**2 - e2_slider**2 - e3_slider**2)