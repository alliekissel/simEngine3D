#!/usr/bin/env python3

# class to initialize system and perform analysis

import sys
import pathlib as pl
src_folder = pl.Path('./src/')
sys.path.append(str(src_folder))

from gcons import *
from newton_raphson import *

import json as js
import numpy as np
import matplotlib.pyplot as plt


class SimEngine3D:
    def __init__(self, filename, analysis=0):
        self.bodies_list = []
        self.n_bodies = 0  # number of bodies that don't include the ground!
        self.constraint_list = []

        self.init_system(filename)

        self.timestep = 0.01
        self.tspan = 5
        if analysis == 0:
            self.kinematics_solver()
        else:
            self.inverse_dynamics_solver()

    def init_system(self, filename):
        # setup initial system based on model parameters
        with open(filename) as f:
            model = js.load(f)
            bodies = model['bodies']
            constraints = model['constraints']

        for body in bodies:
            self.bodies_list.append(RigidBody(body))

        for con in constraints:
            if con['type'] == 'DP1':
                self.constraint_list.append(GConDP1(con, self.bodies_list))
            elif con['type'] == 'DP2':
                self.constraint_list.append(GConDP2(con, self.bodies_list))
            elif con['type'] == 'D':
                self.constraint_list.append(GConD(con, self.bodies_list))
            elif con['type'] == 'CD':
                self.constraint_list.append(GConCD(con, self.bodies_list))
            else:
                print("Incorrect geometric constraint type given.")
        return

    def set_simulation_duration(self, tspan):
        self.tspan = tspan
        return self.tspan

    def set_system_timestep(self, dt):
        self.timestep = dt
        return self.timestep

    def set_q(self, q_new):
        i = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                # update generalized coordinates for bodies
                rdim = 3
                pdim = 4
                r_start = i * (rdim + pdim)
                p_start = (r_start + pdim) - 1
                body.r = q_new[r_start:r_start + rdim]
                body.p = q_new[p_start:p_start + pdim]
                i += 1

    def get_q(self):
        #q = np.zeros((len(self.constraint_list) + self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                q = np.concatenate((body.r, body.p), axis=0)
                # rdim = 3
                # pdim = 4
                # r_start = idx * (rdim + pdim)
                # p_start = (r_start + pdim)-1
                # q[r_start:r_start+rdim] = body.r[0:]
                # q[p_start:p_start+pdim] = body.p[0:]
                # idx += 1
        return q

    def get_phi(self, t):
        # includes all kinematic constraints
        Phi_K = np.concatenate([con.phi(t) for con in self.constraint_list], axis=0)
        # create euler parameter constraints to add to full Phi
        Phi_p = np.zeros((self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                Phi_p[idx] = body.p.T @ body.p - 1.0
                idx += 1
        return np.concatenate((Phi_K, Phi_p), axis=0)

    def get_phi_q(self):
        Phi_r = np.concatenate([con.partial_r() for con in self.constraint_list], axis=0)
        Phi_p = np.concatenate([con.partial_p() for con in self.constraint_list], axis=0)
        Phi_q = np.concatenate((Phi_r, Phi_p), axis=1)

        Phi_euler = np.zeros((self.n_bodies, 7))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                Phi_euler[idx, 3:] = 2 * body.p.T
                idx += 1
        return np.concatenate((Phi_q, Phi_euler), axis=0)

    def get_nu(self, t):
        nu_G = np.concatenate([con.nu(t) for con in self.constraint_list], axis=0)
        nu_euler = np.zeros((self.n_bodies, 1))
        return np.concatenate((nu_G, nu_euler), axis=0)

    def get_gamma(self, t):
        gamma_G = np.concatenate([con.gamma(t) for con in self.constraint_list], axis=0)
        gamma_euler = np.zeros((self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                gamma_euler[idx, :] = -2 * body.p_dot.T @ body.p_dot
                idx += 1
        return np.concatenate((gamma_G, gamma_euler), axis=0)

    def kinematics_solver(self):
        for body in self.bodies_list:
            if not body.is_ground:
                self.n_bodies += 1

        N = int(self.tspan/self.timestep)
        t_start = 0
        t_end = self.tspan
        t_grid = np.linspace(t_start, t_end, N)

        max_iters = 50
        tol = 1e-4

        q_0 = self.get_q()
        Phi_q_0_inv = np.linalg.inv(self.get_phi_q())

        # initialize position, velocity and acceleration storage arrays
        r = np.zeros((N, 3))
        r_dot = np.zeros((N, 3))
        r_ddot = np.zeros((N, 3))
        r[0, :] = q_0[0:3, 0].T
        r_dot[0, :] = (Phi_q_0_inv @ self.get_nu(t_start))[0:3, 0].T
        r_ddot[0, :] = (Phi_q_0_inv @ self.get_gamma(t_start))[0:3, 0].T

        # Set initial conditions and begin time integration
        q_k = q_0
        for i, t in enumerate(t_grid):
            # perform Newton iteration at each time step
            # initialize the norm to be greater than the tolerance so loop begins
            delta_q_norm = 2 * tol
            iteration = 0

            Phi_k = self.get_phi(t)
            Phi_q_k_inv = np.linalg.inv(self.get_phi_q())
            delta_q = Phi_q_k_inv @ Phi_k
            q_k1 = q_k - delta_q

            while delta_q_norm > tol:

                if iteration >= max_iters:
                    print("Newton-Raphson has not converged after", str(max_iters), "iterations. Stopping.")
                    break

                q_k = q_k1

                # update body's generalized coordinates
                self.set_q(q_k)

                # Update Phi, Phi_q for next iteration
                Phi_k = self.get_phi(t)
                Phi_q_k_inv = np.linalg.inv(self.get_phi_q())

                delta_q = Phi_q_k_inv @ Phi_k
                q_k1 = q_k - delta_q

                # Calculate norm(delta_q) to check convergence
                delta_q_norm = np.linalg.norm(delta_q)
                iteration += 1

            # update body's generalized coordinates to converged q
            self.set_q(q_k1)

            # store position for this timestep
            r[i, :] = q_k1[0:3, 0].T

            # velocity analysis
            q_dot = np.linalg.inv(self.get_phi_q()) @ self.get_nu(t)
            r_dot[i, :] = q_dot[0:3, 0].T

            # acceleration analysis
            q_ddot = np.linalg.inv(self.get_phi_q()) @ self.get_gamma(t)
            r_ddot[i, :] = q_ddot[0:3, 0].T

        # plot position, velocity and acceleration for full time duration
        # position
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(t_grid, r[:, 0])
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel('X Position [m]')

        ax2.plot(t_grid, r[:, 1])
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('Y Position [m]')

        ax3.plot(t_grid, r[:, 2])
        ax3.set_xlabel('t [s]')
        ax3.set_ylabel('Z Position [m]')

        # velocity
        f_v, (ax1_v, ax2_v, ax3_v) = plt.subplots(3, 1, sharex=True)
        ax1_v.plot(t_grid, r_dot[:, 0])
        ax1_v.set_xlabel('t [s]')
        ax1_v.set_ylabel('X Velocity [m/s]')

        ax2_v.plot(t_grid, r_dot[:, 1])
        ax2_v.set_xlabel('t [s]')
        ax2_v.set_ylabel('Y Velocity [m/s]')

        ax3_v.plot(t_grid, r_dot[:, 2])
        ax3_v.set_xlabel('t [s]')
        ax3_v.set_ylabel('Z Velocity [m/s]')

        # acceleration
        f_a, (ax1_a, ax2_a, ax3_a) = plt.subplots(3, 1, sharex=True)
        ax1_a.plot(t_grid, r_ddot[:, 0])
        ax1_a.set_xlabel('t [s]')
        ax1_a.set_ylabel('X Acceleration [m/s^2]')

        ax2_a.plot(t_grid, r_ddot[:, 1])
        ax2_a.set_xlabel('t [s]')
        ax2_a.set_ylabel('Y Acceleration [m/s^2]')

        ax3_a.plot(t_grid, r_ddot[:, 2])
        ax3_a.set_xlabel('t [s]')
        ax3_a.set_ylabel('Z Acceleration [m/s^2]')

        plt.show()
        return

    def inverse_dynamics_solver(self):
        # perform inverse dynamics analysis
        # q, q_dot, q_ddot = self.kinematics_solver()
        return


class RigidBody:
    def __init__(self, body_dict):
        if body_dict['name'] == 'ground':
            self.is_ground = True
            self.body_id = body_dict['id']
            self.r = np.array([[0],
                               [0],
                               [0]])
            self.r_dot = np.array([[0],
                                   [0],
                                   [0]])
            self.p = np.array([[1],
                               [0],
                               [0],
                               [0]])
            self.p_dot = np.array([[0],
                                   [0],
                                   [0],
                                   [0]])
        else:
            self.is_ground = False
            self.body_id = body_dict['id']
            self.r = np.array([body_dict['r']]).T
            self.r_dot = np.array([body_dict['r_dot']]).T
            self.p = np.array([body_dict['p']]).T
            self.p_dot = np.array([body_dict['p_dot']]).T
