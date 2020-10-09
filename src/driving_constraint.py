#!/usr/bin/env python3

"""Class to define a driving function and its first and second derivatives

@TODO: provide file description here
@TODO: decide if this class should be moved to sim file or elsewhere
"""


class DrivingConstraint:
    def __init__(self, f, f_dot, f_ddot):
        self.f = f
        self.f_dot = f_dot
        self.f_ddot = f_ddot
