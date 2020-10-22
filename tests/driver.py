#!/usr/bin/env python3

import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

from sim_engine_3d import *


def print_results(gcon, t):
    print("Phi: ", gcon.phi(t), "\nnu: ", gcon.nu(t), "\ngamma: ", gcon.gamma(t),
          "\npartial_r: ", gcon.partial_r(), "\npartial_p: ", gcon.partial_p())


simulation = SimEngine3D("../models/constraints_test.mdl")
print(print_results(simulation.constraint_list[0], 0))
print(print_results(simulation.constraint_list[1], 0))

# these norm steps needed if checking against example
'''
self.p_i = self.p_i / np.linalg.norm(self.p_i)
self.p_dot_i[3] = -np.dot([self.p_dot_i[0][0], self.p_dot_i[1][0], self.p_dot_i[2][0]],
                          [self.p_i[0][0], self.p_i[1][0], self.p_i[2][0]]) / self.p_i[3]
self.p_dot_i = self.p_dot_i / np.linalg.norm(self.p_dot_i)
self.p_j = self.p_j / np.linalg.norm(self.p_j)
self.p_dot_j[3] = -np.dot([self.p_dot_j[0][0], self.p_dot_j[1][0], self.p_dot_j[2][0]],
                          [self.p_j[0][0], self.p_j[1][0], self.p_j[2][0]]) / self.p_j[3]
self.p_dot_j = self.p_dot_j / np.linalg.norm(self.p_dot_j)
'''