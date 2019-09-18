#!/usr/bin/env python3
"""
The data types for an AUX_DISTAN file
"""

import numpy as np

# Grid_Point_Type, repeated 2621441+1 times

datatype = [('Grid_Point_ID', np.uint32),
            ('Flag', np.uint8),
            ('Dist', np.float32),
            ('Tg_resol_max_ocean', np.float32),
            ('Sea_Ice_Mask', np.uint16)
            ]

flag_datatype = [('Fg_Land_Sea_Coast1_tot', np.bool),
                 ('Fg_Land_Sea_Coast2_tot', np.bool),
                 ('Blank1', np.bool),
                 ('Blank2', np.bool),
                 ('Blank3', np.bool),
                 ('Blank4', np.bool),
                 ('Blank5', np.bool),
                 ('Blank6', np.bool)
                ]

# Boolean. Ice Mask. Twelve bits one per month. January is 2^0 and December 2^11
flag_sea_ice_datatype = [('Month1', np.bool),
                 ('Month2', np.bool),
                 ('Month3', np.bool),
                 ('Month4', np.bool),
                 ('Month5', np.bool),
                 ('Month6', np.bool),
                 ('Month7', np.bool),
                 ('Month8', np.bool),
                 ('Month9', np.bool),
                 ('Month10', np.bool),
                 ('Month11', np.bool),
                 ('Month12', np.bool),
                 ('Blank13', np.bool),
                 ('Blank14', np.bool),
                 ('Blank15', np.bool),
                 ('Blank16', np.bool),
                 ('Blank17', np.bool),
                 ('Blank18', np.bool)
                ]
