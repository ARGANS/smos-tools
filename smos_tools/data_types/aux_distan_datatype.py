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
