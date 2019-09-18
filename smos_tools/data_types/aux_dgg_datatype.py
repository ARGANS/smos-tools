#!/usr/bin/env python3
"""
The data types for an AUX_DGG___ file
"""

import numpy as np

# Grid_Point_Data_Type, repeated Grid_Point_Counter times, (nested and itself repeated 10 times)

datatype = [('Grid_Point_ID', np.uint32),
            ('Latitude', np.float32),
            ('Longitude', np.float32),
            ('Altitude', np.float32)
            ]
