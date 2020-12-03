#!/usr/bin/env python3

# this file provides functions to calculate quantities needed for
# flexible multi-body dynamics

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def get_S(L, W, H, xi, eta, zeta):
    S_1 = 1/4 * (xi**3 - 3*xi + 2)
    S_2 = L/8 * (xi**3 - xi**2 - xi + 1)
    S_3 = W*eta/4 * (1 - xi)
    S_4 = H*zeta/4 * (1 - xi)
    S_5 = 1/4 * (-xi**3 + 3*xi + 2)
    S_6 = L/8 * (xi**3 + xi**2 - xi - 1)
    S_7 = W*eta/4 * (1 + xi)
    S_8 = H*zeta/4 * (1 + xi)

    S_1_block = S_1 * np.eye(3)
    S_2_block = S_2 * np.eye(3)
    S_3_block = S_3 * np.eye(3)
    S_4_block = S_4 * np.eye(3)
    S_5_block = S_5 * np.eye(3)
    S_6_block = S_6 * np.eye(3)
    S_7_block = S_7 * np.eye(3)
    S_8_block = S_8 * np.eye(3)

    S = np.vstack((S_1_block, S_2_block, S_3_block, S_4_block,
                   S_5_block, S_6_block, S_7_block, S_8_block)).T
    return S


def get_S_xi(L, W, H, xi, eta, zeta):
    S_1 = 3/4 * (xi**2 - 1)
    S_2 = L/8 * (3*xi**2 - 2*xi - 1)
    S_3 = -W*eta/4
    S_4 = -H*zeta/4
    S_5 = 3/4 * (-xi**2 + 1)
    S_6 = L/8 * (3*xi**2 + 2*xi - 1)
    S_7 = W*eta/4
    S_8 = H*zeta/4

    S_1_block = S_1 * np.eye(3)
    S_2_block = S_2 * np.eye(3)
    S_3_block = S_3 * np.eye(3)
    S_4_block = S_4 * np.eye(3)
    S_5_block = S_5 * np.eye(3)
    S_6_block = S_6 * np.eye(3)
    S_7_block = S_7 * np.eye(3)
    S_8_block = S_8 * np.eye(3)

    S_xi = np.vstack((S_1_block, S_2_block, S_3_block, S_4_block,
                      S_5_block, S_6_block, S_7_block, S_8_block)).T
    return S_xi


def get_S_eta(L, W, H, xi, eta, zeta):
    S_1 = 0
    S_2 = 0
    S_3 = W/4 * (-xi + 1)
    S_4 = 0
    S_5 = 0
    S_6 = 0
    S_7 = W/4 * (xi + 1)
    S_8 = 0

    S_1_block = S_1 * np.eye(3)
    S_2_block = S_2 * np.eye(3)
    S_3_block = S_3 * np.eye(3)
    S_4_block = S_4 * np.eye(3)
    S_5_block = S_5 * np.eye(3)
    S_6_block = S_6 * np.eye(3)
    S_7_block = S_7 * np.eye(3)
    S_8_block = S_8 * np.eye(3)

    S_eta = np.vstack((S_1_block, S_2_block, S_3_block, S_4_block,
                       S_5_block, S_6_block, S_7_block, S_8_block)).T
    return S_eta


def get_S_zeta(L, W, H, xi, eta, zeta):
    S_1 = 0
    S_2 = 0
    S_3 = 0
    S_4 = H/4 * (-xi + 1)
    S_5 = 0
    S_6 = 0
    S_7 = 0
    S_8 = H/4 * (xi + 1)

    S_1_block = S_1 * np.eye(3)
    S_2_block = S_2 * np.eye(3)
    S_3_block = S_3 * np.eye(3)
    S_4_block = S_4 * np.eye(3)
    S_5_block = S_5 * np.eye(3)
    S_6_block = S_6 * np.eye(3)
    S_7_block = S_7 * np.eye(3)
    S_8_block = S_8 * np.eye(3)

    S_zeta = np.vstack((S_1_block, S_2_block, S_3_block, S_4_block,
                        S_5_block, S_6_block, S_7_block, S_8_block)).T
    return S_zeta


def get_M(rho=7700, W=0.003, H=0.003, L=0.5):
    # 2 GQ points for i, j (eta, zeta)
    W_i = [1, 1]
    W_j = W_i
    # 6 GQ points for k (xi)
    W_k = [0.1713, 0.3608, 0.4679, 0.4679, 0.3608, 0.1713]

    X_i = [-np.sqrt(3)/3, np.sqrt(3)/3]
    X_j = X_i
    X_k = [-0.9325, -0.6612, -0.2386, 0.2386, 0.6612, 0.9325]

    e_0 = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, L, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]).T

    M = np.zeros((24, 24))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                M += W_i[i] * W_j[j] * W_k[k] * get_S(L, W, H, X_k[k], X_j[j], X_i[i]).T \
                     @ get_S(L, W, H, X_k[k], X_j[j], X_i[i]) \
                     * np.linalg.det(np.concatenate((get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                     get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                     get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0), axis=1))
    return rho*M


def get_Q_gravity(rho=7700, W=0.003, H=0.003, L=0.5):
    # 2 GQ points for i, j (eta, zeta)
    W_i = [1, 1]
    W_j = W_i
    # 6 GQ points for k (xi)
    W_k = [0.1713, 0.3608, 0.4679, 0.4679, 0.3608, 0.1713]

    X_i = [-np.sqrt(3) / 3, np.sqrt(3) / 3]
    X_j = X_i
    X_k = [-0.9325, -0.6612, -0.2386, 0.2386, 0.6612, 0.9325]

    e_0 = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, L, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]).T

    F_gravity = np.array([[0, 0, -9.81]]).T
    Q = np.zeros((24, 1))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                Q += W_i[i] * W_j[j] * W_k[k] * get_S(L, W, H, X_k[k], X_j[j], X_i[i]).T \
                     @ F_gravity \
                     * np.linalg.det(np.concatenate((get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                     get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                     get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0), axis=1))

    return rho * Q


def get_Q_internal_e(e, W=0.003, H=0.003, L=0.5, E=2.0e11, nu=0.3):
    # element of the jacobian for the internal force vector

    # 2 GQ points for i, j (eta, zeta)
    W_i = [1, 1]
    W_j = W_i
    # 4 GQ points for k (xi)
    W_k = [0.3479, 0.6521, 0.6521, 0.3479]

    X_i = [-np.sqrt(3)/3, np.sqrt(3)/3]
    X_j = X_i
    X_k = [-0.8611, -0.3399, 0.3399, 0.8611]

    k1 = 10 * ((1 + nu) / (12 + 11 * nu))
    k2 = k1
    D = E * nu / ((1 + nu) * (1 - 2 * nu)) * np.array([[(1 - nu) / nu, 1, 1, 0, 0, 0],
                                                       [1, (1 - nu) / nu, 1, 0, 0, 0],
                                                       [1, 1, (1 - nu) / nu, 0, 0, 0],
                                                       [0, 0, 0, (1 - 2 * nu) / (2 * nu), 0, 0],
                                                       [0, 0, 0, 0, (1 - 2 * nu) / (2 * nu) * k1, 0],
                                                       [0, 0, 0, 0, 0, (1 - 2 * nu) / (2 * nu) * k2]])

    e_0 = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, L, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]).T
    Q_e = np.zeros((24, 24))
    Q_inner_sum = np.zeros((24, 24))
    for i in range(2):
        for j in range(2):
            for k in range(4):
                J_inv = np.linalg.inv(np.concatenate((get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                      get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                      get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0), axis=1))
                S_1_F = J_inv[0, 0] * get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[1, 0] * get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[2, 0] * get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i])
                S_2_F = J_inv[0, 1] * get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[1, 1] * get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[2, 1] * get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i])
                S_3_F = J_inv[0, 2] * get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[1, 2] * get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[2, 2] * get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i])
                S_F = np.concatenate((S_1_F, S_2_F, S_3_F), axis=1)
                for l in range(3):
                    for m in range(3):
                        Q_inner_sum += D[l, m] * (1/2 * (S_F[:, 24 * l:24 * l + 24].T @ S_F[:, 24 * l:24 * l + 24]) * (
                                      e.T @ S_F[:, 24 * m:24 * m + 24].T @ S_F[:, 24 * m:24 * m + 24] @ e - 1)
                                                  + (S_F[:, 24 * l:24 * l + 24].T @ S_F[:, 24 * l:24 * l + 24] @ e
                                                     @ e.T @ S_F[:, 24 * m:24 * m + 24].T @ S_F[:, 24 * m:24 * m + 24]))
                Q_e += W_i[i] * W_j[j] * W_k[k] \
                     * (Q_inner_sum
                        + (D[3, 3] * (S_2_F.T @ S_3_F + S_3_F.T @ S_2_F) * (e.T @ S_2_F.T @ S_3_F @ e))
                        + (D[3, 3] * (S_2_F.T @ S_3_F + S_3_F.T @ S_2_F) @ e @ e.T @ (S_2_F.T @ S_3_F + S_3_F.T @ S_2_F))
                        + (D[4, 4] * (S_1_F.T @ S_3_F + S_3_F.T @ S_1_F) * (e.T @ S_1_F.T @ S_3_F @ e))
                        + (D[4, 4] * (S_1_F.T @ S_3_F + S_3_F.T @ S_1_F) @ e @ e.T @ (S_1_F.T @ S_3_F + S_3_F.T @ S_1_F))
                        + (D[5, 5] * (S_1_F.T @ S_2_F + S_2_F.T @ S_1_F) * (e.T @ S_1_F.T @ S_2_F @ e))
                        + (D[5, 5] * (S_1_F.T @ S_2_F + S_2_F.T @ S_1_F) @ e @ e.T @ (S_1_F.T @ S_2_F + S_2_F.T @ S_1_F)) ) \
                     * np.linalg.det(np.concatenate((get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                     get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                     get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0), axis=1))

    return -Q_e


def get_Q_internal(e, W=0.003, H=0.003, L=0.5, E=2.0e11, nu=0.3):
    # e is a 24x1 time dependent vector
    # 3 GQ points for i, j (eta, zeta)
    W_i = [0.5556, 0.8889, 0.5556]
    W_j = W_i
    # 5 GQ points for k (xi)
    W_k = [0.2369, 0.4786, 0.5689, 0.4786, 0.2369]

    X_i = [-0.7746, 0, 0.7746]
    X_j = X_i
    X_k = [-0.9062, -0.5385, 0.0000, 0.5385, 0.9062]

    k1 = 10*((1 + nu)/(12 + 11*nu))
    k2 = k1
    D = E*nu/((1 + nu) * (1 - 2*nu)) * np.array([[(1-nu)/nu, 1, 1, 0, 0, 0],
                                                 [1, (1-nu)/nu, 1, 0, 0, 0],
                                                 [1, 1, (1-nu)/nu, 0, 0, 0],
                                                 [0, 0, 0, (1-2*nu)/(2*nu), 0, 0],
                                                 [0, 0, 0, 0, (1-2*nu)/(2*nu)*k1, 0],
                                                 [0, 0, 0, 0, 0, (1-2*nu)/(2*nu)*k2]])

    e_0 = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, L, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]).T
    Q = np.zeros((24, 1))
    Q_inner_sum = np.zeros((24, 1))
    for i in range(3):
        for j in range(3):
            for k in range(5):
                J_inv = np.linalg.inv(np.concatenate((get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                      get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                      get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0), axis=1))
                S_1_F = J_inv[0, 0]*get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[1, 0]*get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[2, 0]*get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i])
                S_2_F = J_inv[0, 1] * get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[1, 1] * get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[2, 1] * get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i])
                S_3_F = J_inv[0, 2] * get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[1, 2] * get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) \
                        + J_inv[2, 2] * get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i])
                S_F = np.concatenate((S_1_F, S_2_F, S_3_F), axis=1)
                for l in range(3):
                    for m in range(3):
                        Q_inner_sum += ((S_F[:, 24*l:24*l+24].T @ S_F[:, 24*l:24*l+24]) @ e) \
                         @ (D[l, m]/2 * (e.T @ (S_F[:, 24*m:24*m+24].T @ S_F[:, 24*m:24*m+24]) @ e - 1))
                Q += W_i[i] * W_j[j] * W_k[k] \
                     * (Q_inner_sum
                     + ((S_2_F.T @ S_3_F + S_3_F.T @ S_2_F) @ e) * (D[3, 3] * (e.T @ S_2_F.T @ S_3_F @ e))
                     + ((S_1_F.T @ S_3_F + S_3_F.T @ S_1_F) @ e) * (D[4, 4] * (e.T @ S_1_F.T @ S_3_F @ e))
                     + ((S_1_F.T @ S_2_F + S_2_F.T @ S_1_F) @ e) * (D[5, 5] * (e.T @ S_1_F.T @ S_2_F @ e))) \
                     * np.linalg.det(np.concatenate((get_S_xi(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                     get_S_eta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0,
                                                     get_S_zeta(L, W, H, X_k[k], X_j[j], X_i[i]) @ e_0), axis=1))

    return -Q


def get_global_position(e, xi, eta, zeta, W=0.003, H=0.003, L=0.5):
    # solve r(U,t) = S(U)e(t) - 3x1 vector
    r = get_S(L, W, H, xi, eta, zeta) @ e

    return r


def get_Q_external(f_n_applied, n=1, W=0.003, H=0.003, L=0.5, xi=1, eta=0, zeta=0):
    # assume f_n_applied is given as an array of 3x1 arrays
    Q = np.zeros((24, 1))
    for i in range(n):
        Q += get_S(L, W, H, xi, eta, zeta).T @ f_n_applied[i*3:i*3 + 3]

    return Q

def get_jacobian(e):
    zero_block = np.zeros((24, 24))
    identity_block = np.eye(24, 24)
    nonzero_block = get_M() - get_Q_internal_e(e)

    return np.block([
        [nonzero_block, identity_block],
        [identity_block, zero_block]
    ])

# testing
def main():
    # ========================= HOMEWORK 9 TESTING ========================
    #print(get_S(1,1,1,1,1,1))
    #print(get_S_zeta(1,1,1,1,1,1))
    #print(get_M().diagonal())
    #print(get_Q_gravity())
    #print(get_Q_internal((np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5, 0, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0]]).T)))

    # # plot entire beam axis
    # ax = plt.axes(projection='3d')
    # e = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5, 0, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0]]).T
    # N = 50
    # grid = np.linspace(-1, 1, N)
    # eta = zeta = 0
    # r = np.zeros((N, 3))
    # for iter, xi in enumerate(grid):
    #     r[iter, :] = get_global_position(e, xi, eta, zeta).T
    # xdata = r[:, 0]
    # ydata = r[:, 1]
    # zdata = r[:, 2]
    # ax.plot3D(xdata, ydata, zdata)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.savefig("3D")
    #
    # ax_2d = plt.axes(aspect='equal')
    # ax_2d.plot(xdata, zdata)
    # ax_2d.set_xlabel('x')
    # ax_2d.set_ylabel('z')
    # plt.savefig("2D")

    #print(get_Q_external(np.array([[10 * np.cos(45 * np.pi/180), 0, 10 * np.sin(45 * np.pi/180)]]).T))

    # ========================= HOMEWORK 10 TESTING ========================

    e = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5, 0, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0]]).T
    #print(get_Q_internal((np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5, 0, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0]]).T)))
    #print(np.linalg.inv(get_jacobian(e)))

if __name__ == '__main__':
    main()