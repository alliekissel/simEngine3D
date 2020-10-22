#!/usr/bin/env python3

import numpy as np

def newton_raphson(cons_list, bodies_list, q_0, Phi_0, Phi_q_0, t, tol):
    # Return the approximate solution to phi(q,t)=0 via Newton-Raphson Method

    q_k = q_0
    Phi_k = Phi_0
    Phi_q_k = Phi_q_0
    # initialize the norm to be greater than the tolerance so loop begins
    delta_q_norm = 2*tol

    iteration = 0

    while delta_q_norm > tol:

        delta_q = np.linalg.solve(Phi_q_k, Phi_k)
        q_new = q_k - delta_q

        i = 0
        n_bodies = 0
        for body in bodies_list:
            if body.is_ground:
                pass
            else:
                # update generalized coordinates for bodies
                rdim = 3
                pdim = 4
                r_start = i * (rdim + pdim)
                p_start = (r_start + pdim) - 1
                body.r = q_new[r_start:r_start+rdim]
                body.p = q_new[p_start:p_start+pdim]
                n_bodies += 1
                i += 1

        # Get updated Phi, Phi_q
        #@TODO: should create a get function for these and use in kinematics_solver() too
        n_constraints = len(cons_list)
        for i, con in enumerate(cons_list):
            Phi_k[i] = con.phi(t)
            Phi_q_k[i, 0:3 * n_bodies] = con.partial_r()
            Phi_q_k[i, 3 * n_bodies:] = con.partial_p()

        i = 0
        for body in bodies_list:
            if body.is_ground:
                pass
            else:
                rdim = 3
                pdim = 4
                r_start = i * (rdim + pdim)
                Phi_k[i+n_constraints] = body.p.T @ body.p - 1.0
                Phi_q_k[i + n_constraints, r_start:r_start + pdim] = 2*body.p.T
                i += 1

        q_k = q_new
        delta_q_norm = np.linalg.norm(delta_q)
        iteration += 1

    return q_k, iteration
