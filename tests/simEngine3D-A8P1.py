#!/usr/bin/env python3


import sys
import pathlib as pl
src_folder = pl.Path('../src/')
sys.path.append(str(src_folder))

from sim_engine_3d import *

simulation = SimEngine3D("../models/revJoint.mdl", analysis=1)