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
