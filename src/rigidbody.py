#!/usr/bin/env python3

"""Class to define a rigid body object and assign reference frame attributes

@TODO: provide file description here
@TODO: decide if this class should be moved to sim file or elsewhere
"""


class RigidBody:
    def __init__(self, body_id, r, r_dot, p, p_dot):
        self.body_id = body_id
        self.r = r
        self.r_dot = r_dot
        self.p = p
        self.p_dot = p_dot
