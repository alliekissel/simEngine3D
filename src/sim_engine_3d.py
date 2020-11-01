#!/usr/bin/env python3

# class to initialize system and perform analysis

import sys
import pathlib as pl

src_folder = pl.Path('./src/')
sys.path.append(str(src_folder))

from gcons import *

import json as js
import numpy as np
import matplotlib.pyplot as plt

import time


class SimEngine3D:
    def __init__(self, filename, analysis=0):
        self.bodies_list = []
        self.n_bodies = 0  # number of bodies that don't include the ground!
        self.constraint_list = []

        self.init_system(filename)

        self.g = 9.81

        self.lam = 0
        self.lambda_p = 0

        self.timestep = 0.01
        self.tspan = 10
        if analysis == 0:
            self.kinematics_solver()
        else:
            self.dynamics_solver()

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
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                # update generalized coordinates for bodies
                rdim = 3
                pdim = 4
                r_start = idx * (rdim + pdim)
                p_start = (r_start + pdim) - 1
                body.r = q_new[r_start:r_start + rdim]
                body.p = q_new[p_start:p_start + pdim]
                idx += 1

    def get_q(self):
        r = np.zeros((3 * self.n_bodies, 1))
        p = np.zeros((4 * self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                r[idx*3:idx*3 + 3] = body.r
                p[idx*4:idx*4 + 4] = body.p
                idx += 1
        return r, p

    def set_q_dot(self, r_dot, p_dot):
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                # update generalized coordinates for bodies
                body.r_dot = r_dot[idx*3:idx*3 + 3]
                body.p_dot = p_dot[idx*4:idx*4 + 4]
                idx += 1

    def get_q_dot(self):
        r_dot = np.zeros((3 * self.n_bodies, 1))
        p_dot = np.zeros((4 * self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                r_dot[idx*3:idx*3 + 3] = body.r_dot
                p_dot[idx*4:idx*4 + 4] = body.p_dot
                idx += 1
        return r_dot, p_dot

    def set_q_ddot(self, r_ddot, p_ddot):
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                # update generalized coordinates for bodies
                body.r_ddot = r_ddot[idx*3:idx*3 + 3]
                body.p_ddot = p_ddot[idx*4:idx*4 + 4]
                idx += 1

    def get_q_ddot(self):
        r_ddot = np.zeros((3 * self.n_bodies, 1))
        p_ddot = np.zeros((4 * self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                r_ddot[idx*3:idx*3 + 3] = body.r_ddot
                p_ddot[idx*4:idx*4 + 4] = body.p_ddot
                idx += 1
        return r_ddot, p_ddot

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

    def get_M(self):
        m_mat = np.zeros((3*self.n_bodies, 3*self.n_bodies))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                idx = idx*3
                m_mat[idx:idx+3, idx:idx+3] = body.m*np.eye(3)
                idx += 1
        return m_mat

    def get_J_P(self):
        # J_P reference Lecture 13 slide 15
        j_p_mat = np.zeros((4*self.n_bodies, 4*self.n_bodies))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                idx = idx * 4
                G = g_mat(body.p)
                j_p_mat[idx:idx + 4, idx:idx + 4] = 4 * G.T @ body.J @ G
                idx += 1
        return j_p_mat

    def get_P(self):
        p_mat = np.zeros((self.n_bodies, 4*self.n_bodies))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                p_mat[idx, idx:idx + 4] = body.p.T
                idx += 1
        return p_mat

    def get_F_g(self):
        # return F when gravity is the only force
        f_g_mat = np.zeros((3*self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                idx = idx * 3
                f_g_mat[idx:idx+3] = np.array([[0], [0], [body.m*self.g]])
                idx += 1
        return f_g_mat

    def get_tau(self):
        tau = np.zeros((4 * self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                idx = idx * 4
                G_dot = g_dot_mat(body.p_dot)
                tau[idx:idx + 4] = 8 * G_dot.T @ body.J @ G_dot @ body.p
                idx += 1
        return tau

    def set_lambda(self, lam):
        self.lam = lam

    def set_lambda_p(self, lambda_p):
        self.lambda_p = lambda_p

    def residual(self, order, t):
        if order == 1:
            beta_0 = 1
        elif order == 2:
            beta_0 = 2/3
        else:
            print("BDF of order greater than 2 not implemented yet.")

        h = self.timestep

        nb = self.n_bodies
        nc = len(self.constraint_list)

        r_ddot, p_ddot = self.get_q_ddot()

        g = np.zeros((nc + 8*nb, 1))
        g[0:3*nb] = self.get_M() @ r_ddot + self.get_phi_q()[0:nc, 0:3].T @ self.lam - self.get_F_g()
        g[3*nb:7*nb] = self.get_J_P() @ p_ddot + self.get_phi_q()[0:nc, 3:].T @ self.lam \
                       + self.get_P().T @ self.lambda_p - self.get_tau()
        g[7*nb:] = 1/(beta_0**2 * h**2) * self.get_phi(t)

        return g

    def psi(self):
        nc = len(self.constraint_list)
        nb = self.n_bodies

        M = self.get_M()
        J_P = self.get_J_P()
        P = self.get_P()
        Phi_r = self.get_phi_q()[0:nc, 0:3*nb]
        Phi_p = self.get_phi_q()[0:nc, 3*nb:]

        # build Psi(nu), our quasi-newton iteration matrix
        psi = np.zeros((nc + 8*nb, nc + 8*nb))
        psi[0:3*nb, 0:3*nb] = M
        psi[0:3*nb, 8*nb:] = Phi_r.T
        psi[3*nb:7*nb, 3*nb:7*nb] = J_P
        psi[3*nb:7*nb, 7*nb:8*nb] = P.T
        psi[3*nb:7*nb, 8*nb:] = Phi_p.T
        psi[7*nb:8*nb, 3*nb:7*nb] = P
        psi[8*nb:, 0:3*nb] = Phi_r
        psi[8*nb:, 3*nb:7*nb] = Phi_p

        return psi

    def reaction_torque(self):
        nc = len(self.constraint_list)
        nb = self.n_bodies
        Phi_p = self.get_phi_q()[0:nc, 3:]  # this isn't going to be right with more than one body
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                pi = 1 / 2 * Phi_p @ e_mat(body.p).T
                torque = -pi.T @ self.lam
                idx += 1

        return torque

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

        r_0, p_0 = self.get_q()
        q_0 = np.vstack((r_0, p_0))
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

            # this is the initial guess for Newton iteration
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

                # Newton step
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

    def dynamics_solver(self, order=1):
        # putting model parameters here for now. @TODO: move to .mdl file or a setter function
        L = 2
        w = 0.05
        rho = 7800
        self.g = 9.81
        m = rho*2*L*w**2
        Jxx = 1/12*m*2*w**2
        Jyy = 1/12*m*((2*L)**2 + w**2)
        Jzz = Jyy
        J = np.diag(np.array([Jxx, Jyy, Jzz]))

        for body in self.bodies_list:
            if not body.is_ground:
                body.m = m
                body.J = J
                self.n_bodies += 1

        # simulation parameters
        N = int(self.tspan/self.timestep)
        t_start = 0
        t_end = self.tspan
        t_grid = np.linspace(t_start, t_end, N)
        max_iters = 50
        tol = 1e-2
        h = self.timestep

        start = time.time()

        # ------------------------ INITIAL CONDITIONS -------------------------------------------------
        # solve for initial conditions, reference Lecture 17 slide 7
        F = self.get_F_g()
        tau = self.get_tau()
        gamma_p = self.get_gamma(t_start)[(len(self.constraint_list) + self.n_bodies)-1:, :]
        gamma = self.get_gamma(t_start)[0:len(self.constraint_list), :]

        nc = len(self.constraint_list)
        nb = self.n_bodies

        # get initial psi
        psi = self.psi()

        # build full RHS matrix
        eom_RHS = np.zeros((nc + 8*nb, 1))
        eom_RHS[0:3*nb] = F
        eom_RHS[3*nb:7*nb] = tau
        eom_RHS[7*nb:8*nb] = gamma_p
        eom_RHS[8*nb:] = gamma

        # solve to find initial accelerations and lagrange multipliers, vector z
        z_0 = np.linalg.solve(psi, eom_RHS)
        r_ddot_0 = z_0[0:3*nb]
        p_ddot_0 = z_0[3*nb:7*nb]
        self.set_q_ddot(r_ddot_0, p_ddot_0)
        lambda_p_0 = z_0[7*nb:8*nb]
        self.set_lambda_p(lambda_p_0)
        lambda_0 = z_0[8*nb:]
        self.set_lambda(lambda_0)

        # ------------------------ BEGIN TIME EVOLUTION -------------------------------------------------
        # solution storage arrays
        r_0, p_0 = self.get_q()
        r_dot_0, p_dot_0 = self.get_q_dot()

        r_sol = np.zeros((N, 3))
        r_dot_sol = np.zeros((N, 3))
        r_ddot_sol = np.zeros((N, 3))
        r_sol[0, :] = r_0.T
        r_dot_sol[0, :] = r_dot_0.T
        r_ddot_sol[0, :] = r_ddot_0.T

        p_sol = np.zeros((N, 4))
        p_dot_sol = np.zeros((N, 4))
        p_ddot_sol = np.zeros((N, 4))
        p_sol[0, :] = p_0.T
        p_dot_sol[0, :] = p_dot_0.T
        p_ddot_sol[0, :] = p_ddot_0.T

        omega_sol = np.zeros((N, 3))
        omega_sol[0, :] = omega(p_0, p_ddot_0).T

        torque_sol = np.zeros((N, 3))
        torque_sol[0, :] = self.reaction_torque().T

        for i, t in enumerate(t_grid):
            # we already have our initial conditions. want to start at i = 1
            if i == 0:
                continue

            # STAGE 0 - Prime new time step
            r_ddot, p_ddot = self.get_q_ddot()
            lam = self.lam  #@TODO: generalize these to more bodies
            lambda_p = self.lambda_p

            # nu = 0
            z = np.zeros((nc + 8*nb, 1))
            z[0:3*nb] = r_ddot
            z[3*nb:7*nb] = p_ddot
            z[7*nb:8*nb] = lambda_p
            z[8*nb:] = lam

            if order == 2 and i == 1:
                # seed BDF 1
                beta_0 = 1
                c_r_dot = np.array([r_dot_sol[i - 1, :]]).T
                c_p_dot = np.array([p_dot_sol[i - 1, :]]).T
                c_r = np.array([r_sol[i - 1, :]]).T + beta_0 * h * c_r_dot
                c_p = np.array([p_sol[i - 1, :]]).T + beta_0 * h * c_p_dot
            else:
                # STAGE 1 - Compute position and velocity using BDF and most recent accelerations ---------------
                if order == 1:
                    beta_0 = 1
                    c_r_dot = np.array([r_dot_sol[i - 1, :]]).T
                    c_p_dot = np.array([p_dot_sol[i - 1, :]]).T
                    c_r = np.array([r_sol[i - 1, :]]).T + beta_0 * h * c_r_dot
                    c_p = np.array([p_sol[i - 1, :]]).T + beta_0 * h * c_p_dot

                elif order == 2:
                    beta_0 = 2 / 3
                    c_r_dot = 4 / 3 * r_dot_sol[i - 1, :].T - 1 / 3 * r_dot_sol[i - 2, :].T
                    c_p_dot = 4 / 3 * p_dot_sol[i - 1, :].T - 1 / 3 * p_dot_sol[i - 2, :].T
                    c_r = 4 / 3 * r_sol[i - 1, :].T - 1 / 3 * r_sol[i - 2, :].T + beta_0 * h * c_r_dot
                    c_p = 4 / 3 * p_sol[i - 1, :].T - 1 / 3 * p_sol[i - 2, :].T + beta_0 * h * c_p_dot
                else:
                    print("BDF of order greater than 2 not implemented yet.")

            iteration = 0
            delta_norm = 2*tol  # initialize larger than tolerance so loop begins
            while delta_norm > tol:
                if iteration >= max_iters:
                    print("Solution has not converged after", str(max_iters), "iterations. Stopping.")
                    break

                # get updated level 0 and level 1 values
                r, p = self.get_q()
                r_dot, p_dot = self.get_q_dot()

                # STAGE 1 - Compute position and velocity using BDF and most recent accelerations ---------------
                r_n = c_r + beta_0 ** 2 * h ** 2 * r_ddot
                p_n = c_p + beta_0 ** 2 * h ** 2 * p_ddot
                self.set_q(np.concatenate((r_n, p_n), axis=0))
                r_dot_n = c_r_dot + beta_0 * h * r_ddot
                p_dot_n = c_p_dot + beta_0 * h * p_ddot
                self.set_q_dot(r_dot_n, p_dot_n)

                # STAGE 2 - Compute the residual, g_n -----------------------------------------------------------
                gn = self.residual(order, t)

                # STAGE 3 - Solve linear system Psi*delta z = -gn for correction delta z ------------------------
                delta = np.linalg.solve(self.psi(), -gn)

                # STAGE 4 Improve quality of the approximate solution, z^(nu+1)= z^(nu) + delta z^(nu) ----------
                z = z + delta

                r_ddot = z[0:3*nb]
                p_ddot = z[3*nb:7*nb]
                self.set_q_ddot(r_ddot, p_ddot)
                lambda_p = z[7*nb:8*nb]
                self.set_lambda_p(lambda_p)
                lam = z[8*nb:]
                self.set_lambda(lam)


                # STAGE 5 - Set nu = nu + 1, exit loop if norm of correction is small enough --------------------
                delta_norm = np.linalg.norm(delta)
                iteration += 1

            # STAGE 6 - Store solution and move onto next time step ---------------------------------------------
            # set r, p, lam, lambda_p so that n_0 guess is correct
            r_n = c_r + beta_0 ** 2 * h ** 2 * r_ddot
            p_n = c_p + beta_0 ** 2 * h ** 2 * p_ddot
            self.set_q(np.concatenate((r_n, p_n), axis=0))
            r_dot_n = c_r_dot + beta_0 * h * r_ddot
            p_dot_n = c_p_dot + beta_0 * h * p_ddot
            self.set_q_dot(r_dot_n, p_dot_n)

            # calculate omega at this time step
            omega_sol[i, :] = omega(p_n, p_ddot).T

            # calculate reaction torque at this time step
            torque_sol[i, :] = self.reaction_torque().T

            # store solutions for plotting
            r_sol[i, :] = r_n.T
            r_dot_sol[i, :] = r_dot_n.T
            r_ddot_sol[i, :] = r_ddot.T
            p_sol[i, :] = p_n.T
            p_dot_sol[i, :] = p_dot_n.T
            p_ddot_sol[i, :] = p_ddot.T

        end = time.time()
        print("Simulation time:", (end-start))

        # plot x, y, z position for full time duration
        # position
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(t_grid, r_sol[:, 0])
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel('X Position [m]')

        ax2.plot(t_grid, r_sol[:, 1])
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('Y Position [m]')

        ax3.plot(t_grid, r_sol[:, 2])
        ax3.set_xlabel('t [s]')
        ax3.set_ylabel('Z Position [m]')

        # plot omega for full time duration
        # position
        f_omega, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(t_grid, omega_sol[:, 0])
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel('X Omega')

        ax2.plot(t_grid, omega_sol[:, 1])
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('Y Omega')

        ax3.plot(t_grid, omega_sol[:, 2])
        ax3.set_xlabel('t [s]')
        ax3.set_ylabel('Z Omega')

        # plot torque for full time duration
        f_torque, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(t_grid, torque_sol[:, 0])
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel('X Reaction torque')
        ax2.plot(t_grid, torque_sol[:, 1])
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('Y Reaction torque')
        ax3.plot(t_grid, torque_sol[:, 2])
        ax3.set_xlabel('t [s]')
        ax3.set_ylabel('Z Reaction torque')

        plt.show()

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
            self.r_ddot = np.array([[0],
                                    [0],
                                    [0]])
            self.p = np.array([body_dict['p']]).T
            self.p_dot = np.array([body_dict['p_dot']]).T
            self.p_ddot = np.array([[0],
                                    [0],
                                    [0],
                                    [0]])
            self.m = 0
            self.J = 0
