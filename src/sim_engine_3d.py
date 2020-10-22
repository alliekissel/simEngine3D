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


class SimEngine3D:
    def __init__(self, filename, analysis=0):
        self.bodies_list = []
        self.constraint_list = []

        self.init_system(filename)

        self.history = {'r': [], 'p': [], 'r_dot': [], 'p_dot': [], 'r_ddot': [], 'p_ddot': []}
        self.timestep = 0
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

    def set_system_timestep(self, dt):
        self.timestep = dt

    def kinematics_solver(self):
        n_constraints = len(self.constraint_list)
        n_bodies = 0
        for body in self.bodies_list:
            if not body.is_ground:
                n_bodies += 1

        # build initial full phi and q with dim(nc+nb,1)
        phi_0 = np.zeros((n_constraints + n_bodies, 1))
        q_0 = np.zeros((n_constraints + n_bodies, 1))
        # build initial jacobian with dim (nc+nb, nc+nb)
        phi_q_0 = np.zeros((n_constraints + n_bodies, n_constraints + n_bodies))
        for i, con in enumerate(self.constraint_list):
            phi_0[i] = con.phi(t=0)
            phi_q_0[i, 0:3*n_bodies] = con.partial_r()
            phi_q_0[i, 3*n_bodies:] = con.partial_p()

        i = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                phi_0[i+n_constraints] = body.p.T @ body.p - 1.0

                rdim = 3
                pdim = 4
                r_start = i * (rdim + pdim)
                p_start = (r_start + pdim)-1
                q_0[r_start:r_start+rdim] = body.r[0:]
                q_0[p_start:p_start+pdim] = body.p[0:]
                phi_q_0[i + n_constraints, r_start:r_start + pdim] = 2*body.p.T

                i += 1

        # position analysis - call newton-raphson
        tol = 10e-3
        ################# Putting it here for now ##############################
        q_k = q_0
        Phi_k = phi_0
        Phi_q_k = phi_q_0
        # initialize the norm to be greater than the tolerance so loop begins
        delta_q_norm = 2 * tol

        iteration = 0

        while delta_q_norm > tol:

            delta_q = np.linalg.solve(Phi_q_k, Phi_k)
            q_new = q_k - delta_q

            i = 0
            n_bodies = 0
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
                    n_bodies += 1
                    i += 1

            # Get updated Phi, Phi_q
            # @TODO: should create a get function for these and use in kinematics_solver() too
            n_constraints = len(self.constraint_list)
            for i, con in enumerate(self.constraint_list):
                Phi_k[i] = con.phi(1)
                Phi_q_k[i, 0:3 * n_bodies] = con.partial_r()
                Phi_q_k[i, 3 * n_bodies:] = con.partial_p()

            i = 0
            for body in self.bodies_list:
                if body.is_ground:
                    pass
                else:
                    rdim = 3
                    pdim = 4
                    r_start = i * (rdim + pdim)
                    Phi_k[i + n_constraints] = body.p.T @ body.p - 1.0
                    Phi_q_k[i + n_constraints, r_start:r_start + pdim] = 2 * body.p.T
                    i += 1

            q_k = q_new
            delta_q_norm = np.linalg.norm(delta_q)
            iteration += 1

        #q, iteration = newton_raphson(self.constraint_list, self.bodies_list, q, phi, phi_q, 2, 10e-6)
        print(iteration)
        print(q_k)
        # velocity analysis
        # acceleration analysis
        return q_k

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
