#!/usr/bin/env python3

# class to initialize system and perform analysis

from .gcons import *

import json as js


class SimEngine3D:
    def __init__(self, filename, analysis=0):
        self.bodies_list = []
        self.constraint_list = []

        self.init_system(filename)

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

    def kinematics_solver(self):
        # position analysis - call newton-raphson
        # velocity analysis
        # acceleration analysis
        return

    def inverse_dynamics_solver(self):
        # perform inverse dynamics analysis
        return


class RigidBody:
    def __init__(self, body_dict):
        if body_dict['type'] == 'ground':
            self.ground = True
            self.body_id = body_dict['id']
            self.r = np.array([[0],
                               [0],
                               [0]])
            self.r_dot = np.array([[0],
                                   [0],
                                   [0]])
            self.p = np.array([[0],
                               [0],
                               [0],
                               [0]])
            self.p_dot = np.array([[0],
                                   [0],
                                   [0],
                                   [0]])
        else:
            self.ground = False
            self.body_id = body_dict['id']
            self.r = np.array([body_dict['r']]).T
            self.r_dot = np.array([body_dict['r_dot']]).T
            self.p = np.array([body_dict['p']]).T
            self.p_dot = np.array([body_dict['p_dot']]).T
