#!/usr/bin/env python3


import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from sim_engine_3d import SimEngine3D
import gcons

sys = SimEngine3D("../models/four_link.mdl")

L = 2  # [m] - length of the bar
w = 0.05  # [m] - side length of bar
rho = 7800  # [kg/m^3] - density of the bar
r = 2  # [m] - radius of the rotor
height = 0.1  # [m] - the height of the rotor

# body 1 properties
V = height*np.pi * r**2
sys.bodies_list[1].m = rho * V
J_xx = 1/2 * sys.bodies_list[1].m * r
J_yz = 1/12 * sys.bodies_list[1].m * (3*r**2 + height**2)
sys.bodies_list[1].J = np.diag([J_xx, J_yz, J_yz])
sys.n_bodies += 1

# body 2 properties
V = 2 * L * w ** 2
sys.bodies_list[2].m = rho * V
J_xx = 1 / 6 * sys.bodies_list[2].m * w ** 2
J_yz = 1 / 12 * sys.bodies_list[2].m * (w ** 2 + (2 * L) ** 2)
sys.bodies_list[2].J = np.diag([J_xx, J_yz, J_yz])
sys.n_bodies += 1

# body 3 properties
V = L * w ** 2
sys.bodies_list[3].m = rho * V
J_xx = 1 / 6 * sys.bodies_list[3].m * w ** 2
J_yz = 1 / 12 * sys.bodies_list[3].m * (w ** 2 + L ** 2)
sys.bodies_list[3].J = np.diag([J_xx, J_yz, J_yz])
sys.n_bodies += 1

# Alternative driving constraint for singularity encounter
sys.alternative_driver = copy(sys.constraint_list[-1])
sys.alternative_driver.a_bar_j = np.array([[0], [0], [1]])
sys.alternative_driver.prescribed_val = gcons.DrivingConstraint("cos(-pi * t - pi/2 + pi/2)",
                                                            "-pi*cos(pi*t + pi/2)",
                                                            "pi**2*sin(pi*t + pi/2)")

sys.t_start = 0
sys.t_end = 5
sys.timestep = 0.001

sys.max_iters = 20
sys.tol = 1e-2

sys.dynamics_solver()

# print positions of bodies
link1 = sys.r_sol[:, 0:3]
link2 = sys.r_sol[:, 3:6]
link3 = sys.r_sol[:, 6:9]

_, ax1 = plt.subplots()
ax1.plot(sys.t_grid, link1[:, 0])
ax1.set_title('Link 1, x')

_, ax2 = plt.subplots()
ax2.plot(sys.t_grid, link1[:, 1])
ax2.set_title('Link 1, y')

_, ax3 = plt.subplots()
ax3.plot(sys.t_grid, link1[:, 2])
ax3.set_title('Link 1, z')

_, ax4 = plt.subplots()
ax4.plot(sys.t_grid, link2[:, 0])
ax4.set_title('Link 2, x')

_, ax5 = plt.subplots()
ax5.plot(sys.t_grid, link2[:, 1])
ax5.set_title('Link 2, y')

_, ax6 = plt.subplots()
ax6.plot(sys.t_grid, link2[:, 2])
ax6.set_title('Link 2, z')

_, ax7 = plt.subplots()
ax7.plot(sys.t_grid, link3[:, 0])
ax7.set_title('Link 3, x')

_, ax8 = plt.subplots()
ax8.plot(sys.t_grid, link3[:, 1])
ax8.set_title('Link 3, y')

_, ax9 = plt.subplots()
ax9.plot(sys.t_grid, link3[:, 2])
ax9.set_title('Link 3, z')

plt.show()

