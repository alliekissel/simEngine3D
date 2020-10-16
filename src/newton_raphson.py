#!/usr/bin/env python3

import numpy as np


def get_vals(body_i, body_j, gcon, t):

    # Get Phi, Phi_q, and initialize q
    # check for ground body
    if body_j.body_id == 0:
        # no r, p for ground body
        Phi = np.concatenate((gcon.phi(t), 0.5*body_i.p.T @ body_i.p - 0.5), axis=None)
        Phi_q_g = np.concatenate((gcon.partial_r(), gcon.partial_p()), axis=None)
        print(Phi_q_g)
        print(body_i.p.T)
        Phi_q = np.concatenate((Phi_q_g, body_i.p.T), axis=1)
        q = np.concatenate((body_i.r, body_i.p), axis=None)
    else:
        Phi = np.concatenate((gcon.phi(t), 0.5*body_i.p.T @ body_i.p - 0.5, 0.5*body_j.p.T @ body_j.p - 0.5),
                             axis=None)
        Phi_q_g = np.concatenate((gcon.partial_r(), gcon.partial_p()), axis=None)
        Phi_q = np.concatenate((Phi_q_g, body_i.p.T, body_j.p.T), axis=1)
        q = np.concatenate((body_i.r, body_i.p, body_j.r, body_j.p), axis=None)

    return q, Phi, Phi_q


def newton_raphson(body_i, body_j, gcon, t, tol):
    # Return the approximate solution to phi(q,t)=0 via Newton-Raphson Method
    # @TODO: generalize to handle more than one gcon and more than two bodies

    q0, Phi, Phi_q = get_vals(body_i, body_j, gcon, t)

    qk = q0
    delta_q_norm = 1
    while delta_q_norm > tol:

        delta_q = np.linalg.solve(Phi_q, Phi)
        q_new = qk - delta_q

        delta_q_norm = np.linalg.norm(delta_q)

        # update body attributes
        if body_j.body_id == 0:
            body_i.r = q_new[0:3]
            body_i.p = q_new[3:]
        else:
            body_i.r = q_new[0:3]
            body_i.p = q_new[3:7]
            body_j.r = q_new[7:10]
            body_j.p = q_new[10:]

        # Get updated Phi, Phi_q
        q, Phi, Phi_q = get_vals(body_i, body_j, gcon, t)

        qk = q_new

    return qk
