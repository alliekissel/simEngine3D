#!/usr/bin/env python3


import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

from sim_engine_3d import *


def print_results(gcon, t):
    # sloppily prints the return values of each function
    print("Phi: ", gcon.phi(t), "\nnu: ", gcon.nu(t), "\ngamma: ", gcon.gamma(t),
          "\npartial_r: ", gcon.partial_r(), "\npartial_p: ", gcon.partial_p())

simulation = SimEngine3D("../models/revJoint.mdl")
#for i in range(0, len(simulation.constraint_list)):
#    print(print_results(simulation.constraint_list[i], 0))
