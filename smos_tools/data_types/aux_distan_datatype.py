#!/usr/bin/env python3
"""
The data types for an AUX_DISTAN file
"""

import numpy as np

# Grid_Point_Type, repeated 2621441+1 times

datatype = [
    ("Grid_Point_ID", np.uint32),
    ("Flag", np.uint8),
    ("Dist", np.float32),
    ("Tg_resol_max_ocean", np.float32),
    ("Sea_Ice_Mask", np.uint16),
]

flag_datatype = [
    ("Fg_Land_Sea_Coast1_tot", bool),
    ("Fg_Land_Sea_Coast2_tot", bool),
    ("Blank1", bool),
    ("Blank2", bool),
    ("Blank3", bool),
    ("Blank4", bool),
    ("Blank5", bool),
    ("Blank6", bool),
]

# Boolean. Ice Mask. Twelve bits one per month. January is 2^0 and December 2^11
flag_sea_ice_datatype = [
    ("Month1", bool),
    ("Month2", bool),
    ("Month3", bool),
    ("Month4", bool),
    ("Month5", bool),
    ("Month6", bool),
    ("Month7", bool),
    ("Month8", bool),
    ("Month9", bool),
    ("Month10", bool),
    ("Month11", bool),
    ("Month12", bool),
    ("Blank13", bool),
    ("Blank14", bool),
    ("Blank15", bool),
    ("Blank16", bool),
    ("Blank17", bool),
    ("Blank18", bool),
]
