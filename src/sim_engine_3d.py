#!/usr/bin/env python3

# class to initialize system and perform analysis

import sys
import pathlib as pl

src_folder = pl.Path('./src/')
sys.path.append(str(src_folder))

import gcons

import json as js
import numpy as np

import time


class SimEngine3D:
    def __init__(self, filename):
        self.bodies_list = []
        self.n_bodies = 0  # number of bodies that don't include the ground!
        self.constraint_list = []

        self.init_system(filename)

        self.g = -9.81

        self.lam = 0
        self.lambda_p = 0

        self.timestep = 0.001
        self.t_start = 0
        self.t_end = 10
        self.tspan = self.t_end - self.t_start
        self.N = int(self.tspan / self.timestep)
        self.t_grid = np.linspace(self.t_start, self.t_end, self.N)

        self.tol = 1e-3
        self.max_iters = 20

        self.alternative_driver = None

        self.r_sol = None
        self.r_dot_sol = None
        self.r_ddot_sol = None

    def init_system(self, filename):
        # setup initial system based on model parameters
        with open(filename) as f:
            model = js.load(f)
            bodies = model['bodies']
            constraints = model['constraints']

        for body in bodies:
            self.bodies_list.append(RigidBody(body))

        for con in constraints:
            for body in self.bodies_list:
                if body.body_id == con['body_i']:
                    body_i = body
                    print("body_i found")
                if body.body_id == con['body_j']:
                    body_j = body
                    print("body_j found")
            if con['type'] == 'DP1':
                self.constraint_list.append(gcons.GConDP1(con, body_i, body_j))
            elif con['type'] == 'DP2':
                self.constraint_list.append(gcons.GConDP2(con, body_i, body_j))
            elif con['type'] == 'D':
                self.constraint_list.append(gcons.GConD(con, body_i, body_j))
            elif con['type'] == 'CD':
                self.constraint_list.append(gcons.GConCD(con, body_i, body_j))
            else:
                print("Incorrect geometric constraint type given.")
        return

    def kinematics_solver(self):
        for body in self.bodies_list:
            if not body.is_ground:
                self.n_bodies += 1

        nb = self.n_bodies
        print("Number of bodies counted:", nb)
        N = self.N

        r_0, p_0 = self.get_q()
        q_0 = np.vstack((r_0, p_0))

        # initialize position, velocity and acceleration storage arrays
        self.r_sol = np.zeros((N, 3 * nb))
        self.r_dot_sol = np.zeros((N, 3 * nb))
        self.r_ddot_sol = np.zeros((N, 3 * nb))

        iterations = np.zeros((N, 1))

        self.r_sol[0, :] = q_0[0:3 * nb, 0].T
        Phi_q_0_inv = np.linalg.inv(self.get_phi_q())
        self.r_dot_sol[0, :] = (Phi_q_0_inv @ self.get_nu(self.t_start))[0:3 * nb, 0].T
        self.r_ddot_sol[0, :] = (Phi_q_0_inv @ self.get_gamma(self.t_start))[0:3 * nb, 0].T

        # Set initial conditions and begin time integration
        q_k = q_0

        tic = time.perf_counter()
        for i, t in enumerate(self.t_grid):
            if i == 0:
                continue

            # check for driving constraint singularity
            if np.abs(np.abs(self.constraint_list[-1].prescribed_val.f(t)) - 1) < 0.1:
                if self.alternative_driver is None:
                    print("Alternative driving constraint not defined.")
                    break
                self.constraint_list[-1], self.alternative_driver = self.alternative_driver, self.constraint_list[-1]

            Phi_q_k_inv = np.linalg.inv(self.get_phi_q())

            # initialize the norm to be greater than the tolerance so loop begins
            delta_q_norm = 2 * self.tol
            iteration = 0
            while delta_q_norm > self.tol:

                Phi_k = self.get_phi(t)

                # Newton step
                delta_q = -Phi_q_k_inv @ Phi_k
                q_k = q_k + delta_q

                # update body's generalized coordinates
                self.set_q(q_k)

                # Calculate norm(delta_q) to check convergence
                delta_q_norm = np.linalg.norm(delta_q)

                iteration += 1
                if iteration >= self.max_iters:
                    print("Newton-Raphson has not converged after", str(self.max_iters), "iterations. Stopping.")
                    break

            print("Newton-Raphson took", str(iteration), "iterations to converge.")
            iterations[i] = iteration

            # store position for this timestep
            self.r_sol[i, :] = q_k[0:3 * nb, 0].T

            # calculate velocity
            q_dot = np.linalg.inv(self.get_phi_q()) @ self.get_nu(t)
            self.r_dot_sol[i, :] = q_dot[0:3 * nb, 0].T

            self.set_q_dot(q_dot[0:3 * nb], q_dot[3 * nb:])

            # calculate acceleration
            q_ddot = np.linalg.inv(self.get_phi_q()) @ self.get_gamma(t)
            self.r_ddot_sol[i, :] = q_ddot[0:3 * nb, 0].T

        toc = time.perf_counter()
        duration = toc - tic
        avg_iteration = np.average(iterations)
        print("Method took", avg_iteration, "iterations on average and ", duration,
              "seconds to complete.")

        return

    def dynamics_solver(self, order=1):
        h = self.timestep
        N = self.N

        start = time.time()
        # ------------------------ INITIAL CONDITIONS -------------------------------------------------
        # solve for initial conditions, reference Lecture 17 slide 7
        nc = len(self.constraint_list)
        nb = self.n_bodies

        # build full RHS matrix
        eom_RHS = np.zeros((nc + 8 * nb, 1))
        eom_RHS[0:3 * nb] = self.get_F_g()
        eom_RHS[3 * nb:7 * nb] = self.get_tau()
        eom_RHS[7 * nb:8 * nb] = self.get_gamma(self.t_start)[len(self.constraint_list):, :]
        eom_RHS[8 * nb:] = self.get_gamma(self.t_start)[0:len(self.constraint_list), :]

        # solve to find initial accelerations and lagrange multipliers, vector z
        z_0 = np.linalg.solve(self.psi(), eom_RHS)
        r_ddot_0 = z_0[0:3 * nb]
        p_ddot_0 = z_0[3 * nb:7 * nb]
        self.set_q_ddot(r_ddot_0, p_ddot_0)
        lambda_p_0 = z_0[7 * nb:8 * nb]
        self.set_lambda_p(lambda_p_0)
        lambda_0 = z_0[8 * nb:]
        self.set_lambda(lambda_0)

        # ------------------------ BEGIN TIME EVOLUTION -------------------------------------------------
        # solution storage arrays
        r_0, p_0 = self.get_q()
        r_dot_0, p_dot_0 = self.get_q_dot()
        self.r_sol = np.zeros((N, 3 * nb))
        self.r_dot_sol = np.zeros((N, 3 * nb))
        self.r_ddot_sol = np.zeros((N, 3 * nb))
        self.r_sol[0, :] = r_0.T
        self.r_dot_sol[0, :] = r_dot_0.T
        self.r_ddot_sol[0, :] = r_ddot_0.T

        p_sol = np.zeros((N, 4*self.n_bodies))
        p_dot_sol = np.zeros((N, 4*self.n_bodies))
        p_ddot_sol = np.zeros((N, 4*self.n_bodies))
        p_sol[0, :] = p_0.T
        p_dot_sol[0, :] = p_dot_0.T
        p_ddot_sol[0, :] = p_ddot_0.T

        for i, t in enumerate(self.t_grid):
            # we already have our initial conditions. want to start at i = 1
            if i == 0:
                continue

            # check for driving constraint singularity
            if np.abs(np.abs(self.constraint_list[-1].prescribed_val.f(t)) - 1) < 0.1:
                if self.alternative_driver is None:
                    print("Alternative driving constraint not defined.")
                    break
                self.constraint_list[-1], self.alternative_driver = self.alternative_driver, self.constraint_list[-1]

            # print status updates
            print('t: {:.3f}, cond: {:f}, rank: {:f}'.format(t, np.linalg.cond(self.psi()),
                                                             np.linalg.matrix_rank(self.psi())))

            # STAGE 0 - Prime new time step
            r_ddot, p_ddot = self.get_q_ddot()
            lam = self.lam
            lambda_p = self.lambda_p

            # nu = 0
            z = np.zeros((nc + 8 * nb, 1))
            z[0:3 * nb] = r_ddot
            z[3 * nb:7 * nb] = p_ddot
            z[7 * nb:8 * nb] = lambda_p
            z[8 * nb:] = lam

            if order == 2 and i == 1:
                # seed BDF 1
                beta_0 = 1
                c_r_dot = np.array([self.r_dot_sol[i - 1, :]]).T
                c_p_dot = np.array([p_dot_sol[i - 1, :]]).T
                c_r = np.array([self.r_sol[i - 1, :]]).T + beta_0 * h * c_r_dot
                c_p = np.array([p_sol[i - 1, :]]).T + beta_0 * h * c_p_dot
            else:
                # STAGE 1 - Compute position and velocity using BDF and most recent accelerations ---------------
                if order == 1:
                    beta_0 = 1
                    c_r_dot = np.array([self.r_dot_sol[i - 1, :]]).T
                    c_p_dot = np.array([p_dot_sol[i - 1, :]]).T
                    c_r = np.array([self.r_sol[i - 1, :]]).T + beta_0 * h * c_r_dot
                    c_p = np.array([p_sol[i - 1, :]]).T + beta_0 * h * c_p_dot

                elif order == 2:
                    beta_0 = 2 / 3
                    c_r_dot = 4 / 3 * self.r_dot_sol[i - 1, :].T - 1 / 3 * self.r_dot_sol[i - 2, :].T
                    c_p_dot = 4 / 3 * p_dot_sol[i - 1, :].T - 1 / 3 * p_dot_sol[i - 2, :].T
                    c_r = 4 / 3 * self.r_sol[i - 1, :].T - 1 / 3 * self.r_sol[i - 2, :].T + beta_0 * h * c_r_dot
                    c_p = 4 / 3 * p_sol[i - 1, :].T - 1 / 3 * p_sol[i - 2, :].T + beta_0 * h * c_p_dot
                else:
                    print("BDF of order greater than 2 not implemented yet.")

            iteration = 0
            delta_norm = 2 * self.tol  # initialize larger than tolerance so loop begins
            while delta_norm > self.tol:
                if iteration >= self.max_iters:
                    print("Solution has not converged after", str(self.max_iters), "iterations. Stopping.")
                    break

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

                r_ddot = z[0:3 * nb]
                p_ddot = z[3 * nb:7 * nb]
                self.set_q_ddot(r_ddot, p_ddot)
                lambda_p = z[7 * nb:8 * nb]
                self.set_lambda_p(lambda_p)
                lam = z[8 * nb:]
                self.set_lambda(lam)

                # STAGE 5 - Set nu = nu + 1, exit loop if norm of correction is small enough --------------------
                delta_norm = np.linalg.norm(delta)
                iteration += 1
                # print("Iteration: ", iteration)

            # STAGE 6 - Store solution and move onto next time step ---------------------------------------------
            # set r, p, lam, lambda_p so that n_0 guess is correct
            r_n = c_r + beta_0 ** 2 * h ** 2 * r_ddot
            p_n = c_p + beta_0 ** 2 * h ** 2 * p_ddot
            self.set_q(np.concatenate((r_n, p_n), axis=0))
            r_dot_n = c_r_dot + beta_0 * h * r_ddot
            p_dot_n = c_p_dot + beta_0 * h * p_ddot
            self.set_q_dot(r_dot_n, p_dot_n)

            # store solutions for plotting
            self.r_sol[i, :] = r_n.T
            self.r_dot_sol[i, :] = r_dot_n.T
            self.r_ddot_sol[i, :] = r_ddot.T
            p_sol[i, :] = p_n.T
            p_dot_sol[i, :] = p_dot_n.T
            p_ddot_sol[i, :] = p_ddot.T

        end = time.time()
        print("Simulation time:", (end - start))

        return

    def set_q(self, q_new):
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                # update generalized coordinates for bodies
                rdim = 3
                pdim = 4
                r_start = idx * rdim
                p_start = 3 * self.n_bodies + idx * pdim
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
                rdim = 3
                pdim = 4
                r_start = idx * rdim
                p_start = idx * pdim
                r[r_start:r_start + rdim] = body.r
                p[p_start:p_start + pdim] = body.p
                idx += 1
        return r, p

    def set_q_dot(self, r_dot, p_dot):
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                # update generalized coordinates for bodies
                rdim = 3
                pdim = 4
                r_start = idx * rdim
                p_start = idx * pdim
                body.r_dot = r_dot[r_start:r_start + rdim]
                body.p_dot = p_dot[p_start:p_start + pdim]
                idx += 1

    def get_q_dot(self):
        r_dot = np.zeros((3 * self.n_bodies, 1))
        p_dot = np.zeros((4 * self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                rdim = 3
                pdim = 4
                r_start = idx * rdim
                p_start = idx * pdim
                r_dot[r_start:r_start + rdim] = body.r_dot
                p_dot[p_start:p_start + pdim] = body.p_dot
                idx += 1
        return r_dot, p_dot

    def set_q_ddot(self, r_ddot, p_ddot):
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                # update generalized coordinates for bodies
                rdim = 3
                pdim = 4
                r_start = idx * rdim
                p_start = idx * pdim
                body.r_ddot = r_ddot[r_start:r_start + rdim]
                body.p_ddot = p_ddot[p_start:p_start + pdim]
                idx += 1

    def get_q_ddot(self):
        r_ddot = np.zeros((3 * self.n_bodies, 1))
        p_ddot = np.zeros((4 * self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                rdim = 3
                pdim = 4
                r_start = idx * rdim
                p_start = idx * pdim
                r_ddot[r_start:r_start + rdim] = body.r_ddot
                p_ddot[p_start:p_start + pdim] = body.p_ddot
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
        nc = len(self.constraint_list)
        nb = self.n_bodies
        jacobian = np.zeros((nc + nb, 7 * nb))
        offset = 3 * nb

        for row, con in enumerate(self.constraint_list):
            # subtract 1 since ground body does not show up in jacobian
            idi = con.body_i.body_id - 1
            idj = con.body_j.body_id - 1
            if con.body_i.is_ground:
                # fill row of jacobian with only body j
                jacobian[row, 3 * idj:3 * idj + 3] = con.partial_r()
                jacobian[row, offset + 4 * idj:offset + 4 * idj + 4] = con.partial_p()
            elif con.body_j.is_ground:
                # fill row of jacobian with only body i
                jacobian[row, 3 * idi:3 * idi + 3] = con.partial_r()
                jacobian[row, offset + 4 * idi:offset + 4 * idi + 4] = con.partial_p()
            else:
                # fill row of jacobian with both body i and body j
                jacobian[row, 3 * idi:3 * idi + 3] = con.partial_r()[0]
                jacobian[row, offset + 4 * idi:offset + 4 * idi + 4] = con.partial_p()[0]
                jacobian[row, 3 * idj:3 * idj + 3] = con.partial_r()[1]
                jacobian[row, offset + 4 * idj:offset + 4 * idj + 4] = con.partial_p()[1]

        # Euler parameter rows for each body
        row_euler = nc
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                jacobian[row_euler, offset + 4 * idx:offset + 4 * idx + 4] = 2 * body.p.T
                row_euler += 1
                idx += 1

        return jacobian

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
        m_mat = np.zeros((3 * self.n_bodies, 3 * self.n_bodies))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                m_mat[idx * 3:idx * 3 + 3, idx * 3:idx * 3 + 3] = body.m * np.eye(3)
                idx += 1
        return m_mat

    def get_J_P(self):
        # J_P reference Lecture 13 slide 15
        j_p_mat = np.zeros((4 * self.n_bodies, 4 * self.n_bodies))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                G = gcons.g_mat(body.p)
                j_p_mat[idx * 4:idx * 4 + 4, idx * 4:idx * 4 + 4] = 4 * G.T @ body.J @ G
                idx += 1
        return j_p_mat

    def get_P(self):
        p_mat = np.zeros((self.n_bodies, 4 * self.n_bodies))
        # p_mat = np.zeros((4 * self.n_bodies, self.n_bodies))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                p_mat[idx, 4 * idx:4 * idx + 4] = 2 * body.p.T
                # p_mat[4*idx:4*idx + 4, idx:idx + 1] = 2 * body.p
                idx += 1
        return p_mat

    def get_F_g(self):
        # return F when gravity is the only force
        f_g_mat = np.zeros((3 * self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                f_g_mat[idx * 3:idx * 3 + 3] = np.array([[0], [0], [body.m * self.g]])
                idx += 1
        return f_g_mat

    def get_tau(self):
        tau = np.zeros((4 * self.n_bodies, 1))
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                G_dot = gcons.g_dot_mat(body.p_dot)
                tau[idx * 4:idx * 4 + 4] = 8 * G_dot.T @ body.J @ G_dot @ body.p
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
            beta_0 = 2 / 3
        else:
            print("BDF of order greater than 2 not implemented yet.")

        h = self.timestep

        nb = self.n_bodies
        nc = len(self.constraint_list)

        r_ddot, p_ddot = self.get_q_ddot()
        Phi = self.get_phi(t)[:nc, :]
        Phi_euler = self.get_phi(t)[nc:nc + nb, :]
        Phi_r = self.get_phi_q()[0:nc, 0:3 * nb]
        Phi_p = self.get_phi_q()[0:nc, 3 * nb:]

        g_row1 = self.get_M() @ r_ddot + Phi_r.T @ self.lam - self.get_F_g()
        g_row2 = self.get_J_P() @ p_ddot + Phi_p.T @ self.lam \
                 + self.get_P().T @ self.lambda_p - self.get_tau()
        g_row3 = 1 / (beta_0 ** 2 * h ** 2) * Phi_euler
        g_row4 = 1 / (beta_0 ** 2 * h ** 2) * Phi
        g = np.block([[g_row1],
                      [g_row2],
                      [g_row3],
                      [g_row4]])
        return g

    def psi(self):
        nc = len(self.constraint_list)
        nb = self.n_bodies

        M = self.get_M()
        J_P = self.get_J_P()
        P = self.get_P()
        Phi_r = self.get_phi_q()[0:nc, 0:3 * nb]
        Phi_p = self.get_phi_q()[0:nc, 3 * nb:]

        # build Psi, our quasi-newton iteration matrix
        zero_block_12 = np.zeros((3 * nb, 4 * nb))
        zero_block_13 = np.zeros((3 * nb, nb))
        zero_block_21 = np.zeros((4 * nb, 3 * nb))
        zero_block_31 = np.zeros((nb, 3 * nb))
        zero_block_33 = np.zeros((nb, nb))
        zero_block_34 = np.zeros((nb, nc))
        zero_block_43 = np.zeros((nc, nb))
        zero_block_44 = np.zeros((nc, nc))
        psi = np.block([[M, zero_block_12, zero_block_13, Phi_r.T],
                        [zero_block_21, J_P, P.T, Phi_p.T],
                        [zero_block_31, P, zero_block_33, zero_block_34],
                        [Phi_r, Phi_p, zero_block_43, zero_block_44]])
        # psi = np.block([[M, zero_block_12, zero_block_13, Phi_r.T],
        #                 [zero_block_21, J_P, P, Phi_p.T],
        #                 [zero_block_31, P.T, zero_block_33, zero_block_34],
        #                 [Phi_r, Phi_p, zero_block_43, zero_block_44]])

        return psi

    def reaction_torque(self):
        nc = len(self.constraint_list)
        Phi_p = self.get_phi_q()[0:nc, 3:]  # this isn't going to be right with more than one body
        idx = 0
        for body in self.bodies_list:
            if body.is_ground:
                pass
            else:
                pi = 1 / 2 * Phi_p @ gcons.e_mat(body.p).T
                torque = -pi.T @ self.lam
                idx += 1

        return torque


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
