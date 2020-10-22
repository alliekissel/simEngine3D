#!/usr/bin/env python3

import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

from sim_engine_3d import *


def print_results(gcon, t):
    print("Phi: ", gcon.phi(t), "\nnu: ", gcon.nu(t), "\ngamma: ", gcon.gamma(t),
          "\npartial_r: ", gcon.partial_r(), "\npartial_p: ", gcon.partial_p())


simulation = SimEngine3D("../models/revJoint.mdl")
print(print_results(simulation.constraint_list[0], 0))