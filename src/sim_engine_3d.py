#!/usr/bin/env python3

# this file will instantiate a system

# read in .mdl file - prefer to have .mdl file input at command line
#     also command line - flags for desired returned values
# parse model parameters

# maybe do a loop through the .mdl file to count number of constraints
# and instantiate the constraints based on the type name
# @TODO: write .mdl parser and create system in this file

import sys
sys.path.append("../")

import numpy as np
from src.rigidbody import RigidBody
from src.driving_constraint import DrivingConstraint
from src.gcons import GConDP1, GConCD

r_1 = np.array([[0],
                [0],
                [0]])
r_dot_1 = np.array([[0],
                    [0],
                    [0]])
p_1 = np.array([[0],
                [0],
                [0],
                [0]])
p_dot_1 = np.array([[0],
                    [0],
                    [0],
                    [0]])
body_1 = RigidBody(r_1, r_dot_1, p_1, p_dot_1)

r_2 = np.array([[0],
                [0],
                [0]])
r_dot_2 = np.array([[0],
                    [0],
                    [0]])
p_2 = np.array([[0],
                [0],
                [0],
                [0]])
p_dot_2 = np.array([[0],
                    [0],
                    [0],
                    [0]])
body_2 = RigidBody(r_2, r_dot_2, p_2, p_dot_2)

a_1 = np.array([[0],
                [0],
                [0]])
a_2 = np.array([[0],
                [0],
                [0]])

func = 0

constraint = GConDP1(body_1, a_1, body_2, a_2, func)

print(constraint.gamma())

