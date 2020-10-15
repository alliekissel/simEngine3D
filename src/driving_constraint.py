#!/usr/bin/env python3

"""Class to define a driving function and its first and second derivatives

This class stores the driving constraint function f(t) and its first and
second derivatives. It take lambda functions as it's parameters and assumes
f, f_dot, and f_ddot will be given in .mdl file.

@TODO: decide if this class should be moved to sim file or elsewhere
"""


class DrivingConstraint:
    def __init__(self, f, f_dot, f_ddot):
        # These are lambda functions with t dependence
        self.f = f
        self.f_dot = f_dot
        self.f_ddot = f_ddot
