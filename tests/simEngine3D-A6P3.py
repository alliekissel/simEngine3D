#!/usr/bin/env python3

# This is a temporary file. @TODO create model parser in sim_engine_3d.py

import sys
sys.path.append("../")

import numpy as np
from src.rigidbody import RigidBody
from src.driving_constraint import DrivingConstraint
from src.gcons import GConDP1, GConCD
from src.newton_raphson import *


def print_results(gcon, t):
    # sloppily prints the return values of each function
    print("Phi: ", gcon.phi(t)[0][0], "\nnu: ", gcon.nu(t), "\ngamma: ", gcon.gamma(t)[0][0],
          "\npartial_r: ", gcon.partial_r()[0], "\npartial_p: ", gcon.partial_p()[0])


# create ground object
ground_id = 0
r_j = np.array([[0],
                [0],
                [0]])
r_dot_j = np.array([[0],
                    [0],
                    [0]])
p_j = np.array([[0],
                [0],
                [0],
                [0]])
p_dot_j = np.array([[0],
                    [0],
                    [0],
                    [0]])
ground = RigidBody(ground_id, r_j, r_dot_j, p_j, p_dot_j)

# create body 1 pendulum object
body_i_id = 1
r_i = np.array([[0],
                [np.sqrt(2)],
                [-np.sqrt(2)]])
r_dot_i = np.array([[0],
                    [0],
                    [0]])
# p_i converted from euler angles
p_i = np.array([[0.6533],
                [0.2706],
                [0.6533],
                [0.2706]])
p_dot_i = np.array([[0],
                    [0],
                    [0],
                    [0]])
body_i = RigidBody(body_i_id, r_i, r_dot_i, p_i, p_dot_i)

# define DP1 attributes
# make unit vectors
a_bar_j = np.array([[0],
                    [0],
                    [-1]])
a_bar_i = np.array([[1],
                    [0],
                    [0]])
f = lambda t: np.cos(np.pi/4*np.cos(2*t))
f_dot = lambda t: np.pi/2*np.sin(2*t)*np.sin(np.pi/4*np.cos(2*t))
f_ddot = lambda t: np.pi*np.cos(2*t)*np.sin(np.pi/4*np.cos(2*t)) - np.pi**2/4*np.sin(2*t)**2*np.cos(np.pi/4*np.cos(2*t))
prescribed_val_dp1 = DrivingConstraint(f, f_dot, f_ddot)

dp1_constraint = GConDP1(body_i, a_bar_i, ground, a_bar_j, prescribed_val_dp1)

# @TODO: add flag options for return values
print_results(dp1_constraint, t=0)

# Setup problem specs

dt = 1e-3
sim_time = 10
t_steps = sim_time/dt
t_grid = np.arange(0, sim_time, dt)

# store time history of point Q
r_q = np.zeros((3, t_steps))
r_q_dot = np.zeros((3, t_steps))
r_q_ddot = np.zeros((3, t_steps))

# store time history of point O'
r_o = np.zeros((3, t_steps))
r_o_dot = np.zeros((3, t_steps))
r_o_ddot = np.zeros((3, t_steps))

for i, t in enumerate(t_grid):
    q_o = newton_raphson(body_i, ground, dp1_constraint, t)
    r_o = q_o[:3, i]

    # Get Jacobian
    Phi_q = get_vals(body_i, ground, dp1_constraint, t)

    # Solve velocity
    nu = dp1_constraint.nu(t)
    q_o_dot = np.linalg.solve(Phi_q, nu)
    r_o_dot = q_o_dot[:3, i]

    # Solve acceleration
    gamma = dp1_constraint.gamma(t)
    q_o_ddot = np.linalg.solve(Phi_q, gamma)
    r_o_ddot = q_o_ddot[:3, i]
