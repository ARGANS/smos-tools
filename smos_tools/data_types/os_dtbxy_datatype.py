#!/usr/bin/env python3
"""
The data types for an Ocean Salinity DTBXY
"""

import numpy as np

# data type list of lists. The inner lists are there for each file sub-block.
datatype = [
            [('MaxValid', 'float32')],
            [('MinValid', 'float32')],

            # REGION
            [('region_count', 'uint32')],
            # Region ID, start and stop snap time, start and stop snap id
            [('Region_ID', 'uint32'), ('Days', 'int32'), ('Seconds', 'uint32'), ('Microseconds', 'uint32'),
             ('stop_Days', 'int32'), ('stop_Seconds', 'uint32'), ('stop_Microseconds', 'uint32'),
             ('Start_Snapshot_ID', 'uint32'), ('Stop_Snapshot_ID', 'uint32')],
            # stats (just one number) repeated for the 3 models, 8 pols, 12 fov zones
            [('mean', 'float32'), ('median', 'float32'), ('min', 'float32'), ('max', 'float32'), ('std', 'float32')],
            # counts, dTb, std_dTb, flags repeated 129 x 129
            [('count_deltaTB', 'uint32'), ('deltaTB', 'float32'), ('std_deltaTB', 'float32'), ('flags', 'ushort')],

            # SNAPSHOTS
            [('snap_count', 'uint32')],
            # snapshot general info
            [('Snapshot_ID', 'uint32'), ('Snapshot_OBET', 'uint64'), ('Snapshot_Latitude', 'float32'),
             ('Snapshot_Longitude', 'float32'), ('Snapshot_Altitude', 'float32'), ('Snapshot_Flags', 'ushort'),
             ('L1c_TEC', 'int16')],
            [('measurement_count', 'ushort')],
            # measured Tb mean and std
            [('L1cTB', 'ushort'), ('std_L1cTB', 'ushort')],
            # BOA fwd model components
            [('atmosTB', 'int16'), ('std_atmosTB', 'ushort'), ('flatSeaTB', 'int16'), ('std_flatSeaTB', 'ushort'),
             ('roughTB', 'int16'), ('std_roughTB', 'ushort'), ('galTB', 'int16'), ('std_galTB', 'ushort'),
             ('sunTB', 'int16'), ('std_sunTB', 'ushort'), ('sumTB', 'int16'), ('std_sumTB', 'ushort')],
            # TOA fwd model components with L1c TEC
            [('atmosTB', 'int16'), ('std_atmosTB', 'ushort'), ('flatSeaTB', 'int16'), ('std_flatSeaTB', 'ushort'),
             ('roughTB', 'int16'), ('std_roughTB', 'ushort'), ('galTB', 'int16'), ('std_galTB', 'ushort'),
             ('sunTB', 'int16'), ('std_sunTB', 'ushort'), ('sumTB', 'int16'), ('std_sumTB', 'ushort')],
            # TOA fwd model components with A3 TEC
            [('atmosTB', 'int16'), ('std_atmosTB', 'ushort'), ('flatSeaTB', 'int16'), ('std_flatSeaTB', 'ushort'),
             ('roughTB', 'int16'), ('std_roughTB', 'ushort'), ('galTB', 'int16'), ('std_galTB', 'ushort'),
             ('sunTB', 'int16'), ('std_sunTB', 'ushort'), ('sumTB', 'int16'), ('std_sumTB', 'ushort')],
            # geophysics
            [('SSS', 'int16'), ('std_SSS', 'ushort'), ('SST', 'int16'), ('std_SST', 'ushort'), ('WS', 'int16'),
             ('std_WS', 'ushort'), ('A3TEC', 'int16'), ('std_A3TEC', 'ushort'), ('Tair', 'int16'),
             ('std_Tair', 'ushort'), ('SP', 'int16'), ('std_SP', 'ushort'), ('TCWV', 'int16'), ('std_TCWV', 'ushort'),
             ('HS', 'int16'), ('std_HS', 'ushort')],
            # flags
            [('coast', 'ushort'), ('sun_point', 'ushort'), ('sun_tails', 'ushort'), ('rfi', 'ushort'),
             ('rain', 'ushort'), ('ice', 'ushort')],

            [('gp_count', 'uint32')],
            # grid points
            [('Grid_Point_ID', 'uint32'), ('Grid_Point_Latitude', 'float32'), ('Grid_Point_Longitude', 'float32')],
            [('measurement_count', 'ushort')],
            [('Snapshot_Index', 'ushort'), ('Zone_Bits', 'ushort')]
            ]