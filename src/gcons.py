#!/usr/bin/env python3

"""Implementations of the primitive GCons

@TODO: provide file description here
@TODO: additional function descriptions
"""

import numpy as np

#@TODO: think of better way to handle numpy objects in eval()
from numpy import cos, sin, pi


# ------------------------------------- Utility functions -----------------------------------------
def skew(vector):
    """ Function to transform 3x1 vector into a skew symmetric cross product matrix """
    return np.array([[0, -vector.item(2), vector.item(1)],
                     [vector.item(2), 0, -vector.item(0)],
                     [-vector.item(1), vector.item(0), 0]])


def rotation(p):
    """ Function to create a rotation matrix from the Euler parameters """
    e_0 = p[0][0]
    e = np.array([[p[1][0]],
                  [p[2][0]],
                  [p[3][0]]])
    e_tilde = skew(e)
    return (2*e_0 ** 2 - 1) * np.eye(3) + 2 * (e @ e.T + e_0 * e_tilde)


def g_mat(p):
    """ Function to create a G matrix from the Euler parameters """
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
    column_2 = e @ a_bar.T - (e_0 * np.eye(3) + e_tilde) @ a_bar_tilde
    return np.concatenate((2 * column_1, 2 * column_2), axis=1)


# ------------------------------------- Driving Constraint -----------------------------------------
class DrivingConstraint:
    """This class defines a driving function and its first and second derivatives

    This class stores the driving constraint function f(t) and its first and
    second derivatives as lambda functions.
    """

    def __init__(self, f_string, f_dot_string, f_ddot_string):
        self.f = lambda t: eval(f_string)
        self.f_dot = lambda t: eval(f_dot_string)
        self.f_ddot = lambda t: eval(f_ddot_string)


# ------------------------------------- DP1 Constraint --------------------------------------------
class GConDP1:
    """This class implements the Dot Product 1 (DP1) geometric constraint.

    The DP1 constraint reflects the fact that motion is such that the dot
    product between a vector attached to body i and a second vector attached
    to body j assumes a specified value.
    Description credit: Dan Negrut, ME 751, Lecture 8, Slide 34
    """

    def __init__(self, constraint_dict, body_list):
        self.body_i = body_list[1]
        self.p_i = self.body_i.p
        self.p_dot_i = self.body_i.p_dot

        self.body_j = body_list[0]
        self.p_j = self.body_j.p
        self.p_dot_j = self.body_j.p_dot

        self.a_bar_i = np.array([constraint_dict['a_bar_i']]).T
        self.a_bar_j = np.array([constraint_dict['a_bar_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def phi(self, t):
        A_i = rotation(self.p_i)
        A_j = rotation(self.p_j)
        return self.a_bar_i.T @ A_i.T @ A_j @ self.a_bar_j - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        A_i = rotation(self.p_i)
        A_j = rotation(self.p_j)
        a_i = A_i @ self.a_bar_i
        a_j = A_j @ self.a_bar_j
        a_dot_i = b_mat(self.p_i, self.a_bar_i) @ self.p_dot_i
        a_dot_j = b_mat(self.p_j, self.a_bar_j) @ self.p_dot_j
        return - a_i.T @ b_mat(self.p_dot_j, self.a_bar_j) @ self.p_dot_j \
               - a_j.T @ b_mat(self.p_dot_i, self.a_bar_i) @ self.p_dot_i \
               - 2 * (a_dot_i.T @ a_dot_j) + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        # no r dependence, so the partial derivatives are zero
        # check for ground body
        if self.body_i.is_ground or self.body_j.is_ground:
            return np.zeros((1, 3))
        return np.zeros((1, 6))

    def partial_p(self):
        # calculate partial_phi/partial_p
        # check for ground body
        A_i = rotation(self.p_i)
        A_j = rotation(self.p_j)
        a_i = A_i @ self.a_bar_i
        a_j = A_j @ self.a_bar_j
        phi_p_i = a_j.T @ b_mat(self.p_i, self.a_bar_i)
        phi_p_j = a_i.T @ b_mat(self.p_j, self.a_bar_j)
        if self.body_i.is_ground:
            return phi_p_j
        if self.body_j.is_ground:
            return phi_p_i
        return np.concatenate((phi_p_i, phi_p_j), axis=1)


# ------------------------------------- DP2 Constraint --------------------------------------------
class GConDP2:
    """This class implements the Dot Product 2 (DP2) geometric constraint.

    The DP2 constraint reflects the fact that motion is such that the dot product between a vector
    a_bar_i on body i and a second vector P_iQ_j from body i to body j assumes a specified value
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 9
    """

    def __init__(self, constraint_dict, body_list):
        self.body_i = body_list[1]
        self.r_i = self.body_i.r
        self.r_dot_i = self.body_i.r_dot
        self.p_i = self.body_i.p
        self.p_dot_i = self.body_i.p_dot

        self.body_j = body_list[0]
        self.p_j = self.body_j.p
        self.p_dot_j = self.body_j.p_dot
        self.r_j = self.body_j.r
        self.r_dot_j = self.body_j.r_dot

        self.a_bar_i = np.array([constraint_dict['a_bar_i']]).T
        self.s_bar_p_i = np.array([constraint_dict['s_bar_p_i']]).T
        self.s_bar_q_j = np.array([constraint_dict['s_bar_q_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def d_ij(self):
        # calculate d_ij, the distance between point P and point Q
        A_i = rotation(self.p_i)
        A_j = rotation(self.p_j)
        r_p = self.body_i.r + A_i @ self.s_bar_p_i
        r_q = self.body_j.r + A_j @ self.s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        A_i = rotation(self.p_i)
        return self.a_bar_i.T @ A_i.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        A_i = rotation(self.p_i)
        a_i = A_i @ self.a_bar_i
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
        phi_r_i = -self.a_bar_i.T
        phi_r_j = self.a_bar_i.T

        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        return np.concatenate((phi_r_i, phi_r_j), axis=1)

    def partial_p(self):
        # calculate partial_phi/partial_p
        # check for ground body
        phi_p_i = self.d_ij().T @ b_mat(self.p_i, self.a_bar_i) \
                - self.a_bar_i.T @ b_mat(self.p_i, self.s_bar_p_i)
        phi_p_j = self.a_bar_i.T @ b_mat(self.p_j, self.s_bar_q_j)

        if self.body_i.is_ground:
            return phi_p_j
        if self.body_j.is_grounf:
            return phi_p_i
        return np.concatenate((phi_p_i, phi_p_j), axis=1)


# ------------------------------------- D Constraint --------------------------------------------
class GConD:
    """This class implements the Distance (D) geometric constraint.

    The D constraint reflects the fact that motion is such that the distance between point P on
    body i and point Q on body j assumes a specified value greater than zero.
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 12
    """

    def __init__(self, constraint_dict, body_list):
        self.body_i = body_list[1]
        self.r_i = self.body_i.r
        self.r_dot_i = self.body_i.r_dot
        self.p_i = self.body_i.p
        self.p_dot_i = self.body_i.p_dot

        self.body_j = body_list[0]
        self.p_j = self.body_j.p
        self.p_dot_j = self.body_j.p_dot
        self.r_j = self.body_j.r
        self.r_dot_j = self.body_j.r_dot

        self.s_bar_p_i = np.array([constraint_dict['s_bar_p_i']]).T
        self.s_bar_q_j = np.array([constraint_dict['s_bar_q_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def d_ij(self):
        # calculate d_ij, the distance between point P and point Q
        A_i = rotation(self.p_i)
        A_j = rotation(self.p_j)
        r_p = self.body_i.r + A_i @ self.s_bar_p_i
        r_q = self.body_j.r + A_j @ self.s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        return self.d_ij().T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

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
        phi_r_i = -2 * self.d_ij().T
        phi_r_j = 2 * self.d_ij().T
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        else:
            return np.concatenate((phi_r_i, phi_r_j), axis=1)

    def partial_p(self):
        # calculate partial_phi/partial_p
        # check for ground body
        phi_p_i = -2 * self.d_ij().T @ b_mat(self.p_i, self.s_bar_p_i)
        phi_p_j = 2 * self.d_ij().T @ b_mat(self.p_j, self.s_bar_q_j)
        if self.body_i.is_ground:
            return phi_p_j
        if self.body_j.is_ground:
            return phi_p_i
        else:
            return np.concatenate((phi_p_i, phi_p_j), axis=1)


# ------------------------------------- CD Constraint ---------------------------------------------
class GConCD:
    """This class implements the Coordinate Difference (CD) geometric constraint.

    The CD geometric constraint reflects the fact that motion is such that the difference
    between the x (or y or z) coordinate of point P on body i and the x (or y or z) coordinate
    of point Q on body j assumes a specified value.
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 15
    """

    def __init__(self, constraint_dict, body_list):
        self.body_i = body_list[1]
        self.r_i = self.body_i.r
        self.r_dot_i = self.body_i.r_dot
        self.p_i = self.body_i.p
        self.p_dot_i = self.body_i.p_dot

        self.body_j = body_list[0]
        self.p_j = self.body_j.p
        self.p_dot_j = self.body_j.p_dot
        self.r_j = self.body_j.r
        self.r_dot_j = self.body_j.r_dot

        self.c = np.array([constraint_dict['c']]).T
        self.s_bar_p_i = np.array([constraint_dict['s_bar_p_i']]).T
        self.s_bar_q_j = np.array([constraint_dict['s_bar_q_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def d_ij(self):
        # calculate d_ij, the distance between point P and point Q
        A_i = rotation(self.p_i)
        A_j = rotation(self.p_j)
        r_p = self.body_i.r + A_i @ self.s_bar_p_i
        r_q = self.body_j.r + A_j @ self.s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        return self.c.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        return self.c.T @ b_mat(self.p_dot_i, self.s_bar_p_i) @ self.p_dot_i \
               - self.c.T @ b_mat(self.p_dot_j, self.s_bar_q_j) @ self.p_dot_j \
               + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        # no r dependence, so the partial derivatives are zero
        # check for ground body
        phi_r_i = -self.c.T
        phi_r_j = self.c.T
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        else:
            return np.concatenate((phi_r_i, phi_r_j), axis=1)

    def partial_p(self):
        # calculate partial_phi/partial_p
        # check for ground body
        phi_p_i = -self.c.T @ b_mat(self.p_i, self.s_bar_p_i)
        phi_p_j = self.c.T @ b_mat(self.p_j, self.s_bar_q_j)
        if self.body_i.is_ground:
            return phi_p_j
        if self.body_j.is_ground:
            return phi_p_i
        else:
            return np.concatenate((phi_p_i, phi_p_j), axis=1)
