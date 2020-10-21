#!/usr/bin/env python3

"""Implementations of the primitive GCons

@TODO: provide file description here
@TODO: additional function descriptions
@TODO: refactor code to better handle reused quantities, i.e. e_0, e, e_tilde, etc. Perhaps group as
       reference frame attributes
"""

import numpy as np


# ------------------------------------- Utility functions -----------------------------------------
def skew(vector):
    """ Function to transform 3x1 vector into a skew symmetric cross product matrix

    This function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)
    function credit: https://stackoverflow.com/questions/36915774/form-numpy-array-from-possible-numpy-array

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    vector = np.array(vector)

    return np.array([[0, -vector.item(2), vector.item(1)],
                     [vector.item(2), 0, -vector.item(0)],
                     [-vector.item(1), vector.item(0), 0]])


def rotation(p):
    # calculate rotation matrix A
    e_0 = p[0][0]
    e = np.array([[p[1][0]],
                  [p[2][0]],
                  [p[3][0]]])
    e_tilde = skew(e)
    return (e_0 ** 2 - e.T @ e) * np.eye(3) + 2 * (e @ e.T + e_0 * e_tilde)


def g_mat(p):
    e_0 = p[0][0]
    e = np.array([[p[1][0]],
                  [p[2][0]],
                  [p[3][0]]])
    e_tilde = skew(e)
    return np.concatenate((-e, e_tilde), axis=1)


def b_mat(p, a_bar):
    e_0 = p[0][0]
    e = np.array([[p[1][0]],
                  [p[2][0]],
                  [p[3][0]]])
    e_tilde = skew(e)
    a_bar_tilde = skew(a_bar)
    column_1 = (e_0 * np.eye(3) + e_tilde) @ a_bar
    column_2 = (e @ a_bar.T - (e_0 * np.eye(3) + e_tilde) @ a_bar_tilde)
    return np.concatenate((2 * column_1, 2 * column_2), axis=1)


# Don't need these yet, but may be useful later
def omega_bar(p, p_dot):
    G = g_mat(p)
    return 2 * (G @ p_dot)


def omega_bar_tilde(p, p_dot):
    G = g_mat(p)
    G_p_dot_tilde = skew(G @ p_dot)
    return 2 * G_p_dot_tilde


# ------------------------------------- Driving Constraint -----------------------------------------
class DrivingConstraint:
    """This class defines a driving function and its first and second derivatives

    This class stores the driving constraint function f(t) and its first and
    second derivatives.
    @TODO: .mdl input -> lambda function
    """
    def __init__(self, f, f_dot, f_ddot):
        self.f = f
        self.f_dot = f_dot
        self.f_ddot = f_ddot


# ------------------------------------- DP1 Constraint --------------------------------------------
class GConDP1:
    """This class implements the Dot Product 1 (DP1) geometric constraint.

    The DP1 constraint reflects the fact that motion is such that the dot
    product between a vector attached to body i and a second vector attached
    to body j assumes a specified value.
    Description credit: Dan Negrut, ME 751, Lecture 8, Slide 34
    """

    def __init__(self, body_i, a_bar_i, body_j, a_bar_j, prescribed_val):
        # initialize constraint attributes
        # body i and the associated L_RF_i attributes
        self.body_i = body_i
        self.p_i = body_i.p
        self.p_dot_i = body_i.p_dot
        # algebraic vector a_bar_i
        self.a_bar_i = a_bar_i
        # body j and the associated L_RF_j attributes
        self.body_j = body_j
        self.p_j = body_j.p
        self.p_dot_j = body_j.p_dot
        #  algebraic vector a_bar_j
        self.a_bar_j = a_bar_j
        # prescribed value the dot product should assume, specified through f(t)
        #     * most often, f(t)=0, indicating vectors are orthogonal
        #     * f(t) nonzero leads to a driving (rheonomic) constraint
        # this object has f, f_dot and f_ddot attributes
        self.prescribed_val = prescribed_val

        # calculated quantities
        self.A_i = rotation(self.p_i)
        self.A_j = rotation(self.p_j)
        self.omega_bar_i = omega_bar(self.p_i, self.p_dot_i)
        self.omega_bar_j = omega_bar(self.p_j, self.p_dot_j)
        self.omega_bar_tilde_i = omega_bar_tilde(self.p_i, self.p_dot_i)
        self.omega_bar_tilde_j = omega_bar_tilde(self.p_j, self.p_dot_j)

    def phi(self, t):
        return self.a_bar_i.T @ self.A_i.T @ self.A_j @ self.a_bar_j - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return self.prescribed_val.f_dot(t)

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        a_i = self.A_i @ self.a_bar_i
        a_j = self.A_j @ self.a_bar_j
        a_dot_i = b_mat(self.p_i, self.a_bar_i) @ self.p_dot_i
        a_dot_j = b_mat(self.p_j, self.a_bar_j) @ self.p_dot_j
        return - a_i.T @ b_mat(self.p_dot_j, self.a_bar_j) @ self.p_dot_j \
               - a_j.T @ b_mat(self.p_dot_i, self.a_bar_i) @ self.p_dot_i \
               - 2 * (a_dot_i.T @ a_dot_j) + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        # no r dependence, so the partial derivatives are zero
        # check for ground body
        if self.body_j.body_id == 0:
            return np.zeros((1, 3))
        else:
            return np.zeros((1, 6))

    def partial_p(self):
        # calculate partial_phi/partial_p
        # check for ground body
        col_1 = self.a_bar_j.T @ b_mat(self.p_i, self.a_bar_i)
        col_2 = self.a_bar_i.T @ b_mat(self.p_j, self.a_bar_j)
        if self.body_j.body_id == 0:
            return col_1
        else:
            return np.concatenate((col_1, col_2), axis=1)


# ------------------------------------- DP2 Constraint --------------------------------------------
class GConDP2:
    """This class implements the Dot Product 2 (DP2) geometric constraint.

    The DP2 constraint reflects the fact that motion is such that the dot product between a vector
    a_bar_i on body i and a second vector P_iQ_j from body i to body j assumes a specified value
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 9
    """

    def __init__(self, body_i, a_bar_i, s_bar_p_i, body_j, s_bar_q_j, prescribed_val):
        # initialize constraint attributes
        # body i and the associated L_RF_i attributes
        self.body_i = body_i
        self.p_i = body_i.p
        self.p_dot_i = body_i.p_dot
        # algebraic vector a_bar_i
        self.a_bar_i = a_bar_i
        # location of point P
        self.s_bar_p_i = s_bar_p_i
        # body j and the associated L_RF_j attributes
        self.body_j = body_j
        self.p_j = body_j.p
        self.p_dot_j = body_j.p_dot
        #  location of point Q
        self.s_bar_q_j = s_bar_q_j
        # prescribed value the dot product should assume, specified through f(t)
        #     * most often, f(t)=0, indicating vectors are orthogonal
        #     * f(t) nonzero leads to a driving (rheonomic) constraint
        # this object has f, f_dot and f_ddot attributes
        self.prescribed_val = prescribed_val

        # calculated quantities
        self.A_i = rotation(self.p_i)
        self.A_j = rotation(self.p_j)
        self.omega_bar_i = omega_bar(self.p_i, self.p_dot_i)
        self.omega_bar_j = omega_bar(self.p_j, self.p_dot_j)
        self.omega_bar_tilde_i = omega_bar_tilde(self.p_i, self.p_dot_i)
        self.omega_bar_tilde_j = omega_bar_tilde(self.p_j, self.p_dot_j)

    def d_ij(self):
        # calculate d_ij, the distance between point P and point Q
        r_p = self.body_i.r + self.A_i @ self.s_bar_p_i
        r_q = self.body_j.r + self.A_j @ self.s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        return self.a_bar_i.T @ self.A_i.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return self.prescribed_val.f_dot(t)

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        a_i = self.A_i @ self.a_bar_i
        a_dot_i = b_mat(self.p_i, self.a_bar_i) @ self.p_dot_i
        d_dot_ij = self.r_dot_j + b_mat(self.p_j, self.s_bar_q_j) @ self.p_dot_j \
                   - self.r_dot_i - b_mat(self.p_i, self.s_bar_p_i) @ self.p_dot_j
        return - a_i.T @ b_mat(self.p_dot_j, self.s_bar_q_j) @ self.p_dot_j \
               + a_i.T @ b_mat(self.p_dot_i, self.s_bar_p_i) @ self.p_dot_i \
               - self.d_ij().T @ b_mat(self.p_dot_i, self.a_bar_i) @ self.p_dot_i \
               - 2 * a_dot_i.T @ d_dot_ij - self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        # check for ground body
        col_1 = -self.a_bar_i.T
        col_2 = self.a_bar_i.T
        if self.body_j.body_id == 0:
            return col_1
        else:
            return np.concatenate((col_1, col_2), axis=1)

    def partial_p(self):
        # calculate partial_phi/partial_p
        # check for ground body
        col_1 = self.d_ij().T @ b_mat(self.p_i, self.a_bar_i) \
                - self.a_bar_i.T @ b_mat(self.p_i, self.s_bar_p_i)
        col_2 = self.a_bar_i.T @ b_mat(self.p_j, self.s_bar_q_j)
        if self.body_j.body_id == 0:
            return col_1
        else:
            return np.concatenate((col_1, col_2), axis=1)


# ------------------------------------- D Constraint --------------------------------------------
class GConD:
    """This class implements the Distance (D) geometric constraint.

    The D constraint reflects the fact that motion is such that the distance between point P on
    body i and point Q on body j assumes a specified value greater than zero.
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 12
    """

    def __init__(self, body_i, s_bar_p_i, body_j, s_bar_q_j, prescribed_val):
        # initialize constraint attributes
        # body i and the associated L_RF_i attributes
        self.body_i = body_i
        self.p_i = body_i.p
        self.p_dot_i = body_i.p_dot
        # location of point P
        self.s_bar_p_i = s_bar_p_i
        # body j and the associated L_RF_j attributes
        self.body_j = body_j
        self.p_j = body_j.p
        self.p_dot_j = body_j.p_dot
        #  location of point Q
        self.s_bar_q_j = s_bar_q_j
        # prescribed value the dot product should assume, specified through f(t)
        #     * most often, f(t)=0, indicating vectors are orthogonal
        #     * f(t) nonzero leads to a driving (rheonomic) constraint
        # this object has f, f_dot and f_ddot attributes
        self.prescribed_val = prescribed_val

        # calculated quantities
        self.A_i = rotation(self.p_i)
        self.A_j = rotation(self.p_j)
        self.omega_bar_i = omega_bar(self.p_i, self.p_dot_i)
        self.omega_bar_j = omega_bar(self.p_j, self.p_dot_j)
        self.omega_bar_tilde_i = omega_bar_tilde(self.p_i, self.p_dot_i)
        self.omega_bar_tilde_j = omega_bar_tilde(self.p_j, self.p_dot_j)

    def d_ij(self):
        # calculate d_ij, the distance between point P and point Q
        r_p = self.body_i.r + self.A_i @ self.s_bar_p_i
        r_q = self.body_j.r + self.A_j @ self.s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        return self.d_ij().T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return self.prescribed_val.f_dot(t)

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        d_dot_ij = self.r_dot_j + b_mat(self.p_j, self.s_bar_q_j) @ self.p_dot_j \
                   - self.r_dot_i - b_mat(self.p_i, self.s_bar_p_i) @ self.p_dot_j
        return - 2 * self.d_ij().T @ b_mat(self.p_dot_j, self.s_bar_q_j) @ self.p_dot_j \
               + 2 * self.d_ij().T @ b_mat(self.p_dot_i, self.s_bar_p_i) @ self.p_dot_i \
               - 2 * d_dot_ij.T @ d_dot_ij + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        # check for ground body
        col_1 = -2 * self.d_ij().T
        col_2 = 2 * self.d_ij().T
        if self.body_j.body_id == 0:
            return col_1
        else:
            return np.concatenate((col_1, col_2), axis=1)

    def partial_p(self):
        # calculate partial_phi/partial_p
        # check for ground body
        col_1 = -2 * self.d_ij().T @ b_mat(self.p_i, self.s_bar_p_i)
        col_2 = 2 * self.d_ij().T @ b_mat(self.p_j, self.s_bar_q_j)
        if self.body_j.body_id == 0:
            return col_1
        else:
            return np.concatenate((col_1, col_2), axis=1)


# ------------------------------------- CD Constraint ---------------------------------------------
class GConCD:
    """This class implements the Coordinate Difference (CD) geometric constraint.

    The CD geometric constraint reflects the fact that motion is such that the difference
    between the x (or y or z) coordinate of point P on body i and the x (or y or z) coordinate
    of point Q on body j assumes a specified value.
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 15
    """

    def __init__(self, c, body_i, s_bar_p_i, body_j, s_bar_q_j, prescribed_val):

        # initialize constraint attributes
        # coordinate c of interest
        self.c = c
        # body i and the associated L_RF_i attributes
        self.body_i = body_i
        self.p_i = body_i.p
        self.p_dot_i = body_i.p_dot
        # location of point P
        self.s_bar_p_i = s_bar_p_i
        # body j and the associated L_RF_j attributes
        self.body_j = body_j
        self.p_j = body_j.p
        self.p_dot_j = body_j.p_dot
        # location of point Q
        self.s_bar_q_j = s_bar_q_j
        # prescribed value the dot product should assume, specified through f(t)
        #     * most often, f(t)=0, indicating vectors are orthogonal
        #     * f(t) nonzero leads to a driving (rheonomic) constraint
        # this object has f, f_dot and f_ddot attributes
        self.prescribed_val = prescribed_val

        # calculated quantities
        self.A_i = rotation(self.p_i)
        self.A_j = rotation(self.p_j)

    def d_ij(self):
        # calculate d_ij, the distance between point P and point Q
        r_p = self.body_i.r + self.A_i @ self.s_bar_p_i
        r_q = self.body_j.r + self.A_j @ self.s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        return self.c.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return self.prescribed_val.f_dot(t)

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        return self.c.T @ b_mat(self.p_dot_i, self.s_bar_p_i) @ self.p_dot_i \
               - self.c.T @ b_mat(self.p_dot_j, self.s_bar_q_j) @ self.p_dot_j \
               + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        # no r dependence, so the partial derivatives are zero
        # check for ground body
        col_1 = -self.c.T
        col_2 = self.c.T
        if self.body_j.body_id == 0:
            return col_1
        else:
            return np.concatenate((col_1, col_2), axis=1)

    def partial_p(self):
        # calculate partial_phi/partial_p
        # check for ground body
        col_1 = -self.c.T @ b_mat(self.p_i, self.s_bar_p_i)
        col_2 = self.c.T @ b_mat(self.p_j, self.s_bar_q_j)
        if self.body_j.body_id == 0:
            return col_1
        else:
            return np.concatenate((col_1, col_2), axis=1)
