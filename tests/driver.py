#!/usr/bin/env python3

# This is a temporary file. @TODO create model parser in sim_engine_3d.py

import sys
sys.path.append("../")

import numpy as np
from src.rigidbody import RigidBody
from src.driving_constraint import DrivingConstraint
from src.gcons import GConDP1, GConCD


def print_results(gcon):
    print("Phi: ", gcon.phi(), "\nnu: ", gcon.nu(), "\ngamma: ", gcon.gamma(),
          "\npartial_r: ", gcon.partial_r(), "\npartial_p: ", gcon.partial_p())


# ------------------------------------- Test GConDP1 -----------------------------------------

# create body 1 object
body_1_id = 1
r_1 = np.array([[4],
                [0],
                [0]])
r_dot_1 = np.array([[0],
                    [0],
                    [0]])
p_1 = np.array([[0.5],
                [0.5],
                [0.5],
                [0.5]])
p_dot_1 = np.array([[0],
                    [0],
                    [0],
                    [0]])
body_1 = RigidBody(body_1_id, r_1, r_dot_1, p_1, p_dot_1)

# create body 2 object
body_2_id = 2
r_2 = np.array([[2],
                [0],
                [0]])
r_dot_2 = np.array([[0],
                    [0],
                    [0]])
p_2 = np.array([[0.5],
                [0.5],
                [0.5],
                [0.5]])
p_dot_2 = np.array([[1],
                    [1],
                    [1],
                    [1]])
body_2 = RigidBody(body_2_id, r_2, r_dot_2, p_2, p_dot_2)

# define DP1 attributes
a_1 = np.array([[1],
                [3],
                [1]])
a_2 = np.array([[1],
                [2],
                [3]])
prescribed_val_dp1 = DrivingConstraint(3.0, 1.5, 0.0)

dp1_constraint = GConDP1(body_1, a_1, body_2, a_2, prescribed_val_dp1)

# @TODO: add flag options for return values
print_results(dp1_constraint)

# ------------------------------------- Test GConCD -----------------------------------------

# reuse body 1 and body 2 from DP1 test

# define CD attributes
c = np.array([[1],
              [0],
              [0]])
s_bar_p_1 = np.array([[3],
                      [2],
                      [2]])
s_bar_q_2 = np.array([[2],
                      [4],
                      [1]])
prescribed_val_cd = DrivingConstraint(3.0, 1.5, 0.0)

cd_constraint = GConCD(c, body_1, s_bar_p_1, body_2, s_bar_q_2, prescribed_val_cd)

# @TODO: add flag options for return values
print_results(cd_constraint)
