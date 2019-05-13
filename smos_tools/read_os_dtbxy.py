#!/usr/bin/env python3
"""
Read the AUX_DTBXY file (L2OS v67x).

Read the delta Tb product data block. The file contains a set of delta Tbs on a xi-eta grid for each region, together
with associated statistics. See product specs for the contents of the file.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import zipfile


# data type list of lists. The inner lists are there for each file sub-block.
data_types = [
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


log_level = logging.DEBUG
logger = logging.getLogger()
logger.setLevel(log_level)
stream = logging.StreamHandler()
stream.setLevel(log_level)
formatter = logging.Formatter(fmt='%(asctime)s [%(filename)s - %(levelname)s]: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S',)
stream.setFormatter(formatter)
logger.addHandler(stream)


def read_dtbxy(datatypes, dtbxy_file):
    """
    Read the data block of an AUX_DTBXY file (L2OS v67x).

    :param datatypes: list of lists; in each sublist, there is one or more tuples
    :param dtbxy_file: string filename
    :return: a list of numpy structures arrays, sometimes nested with other lists and arrays.
    """

    try:
        data_file = open(dtbxy_file, 'r')
    except:
        logging.error('Unable to open file')

    # L2OS processor version
    version_number = 670

    # max valid xi, eta
    max_valid = np.fromfile(data_file, data_types[0], 2)
    # print(max_valid)

    # min valid xi, eta
    min_valid = np.fromfile(data_file, data_types[1], 2)
    # print(min_valid)

    # get number of regions
    count = np.fromfile(data_file, data_types[2], 1)
    # print(count)

    # initialise the lists for region block
    region_stats_dtbs = [0] * 12 * 129 * 129
    pol_region_stats_dtbs = [0] * 8
    mod_pol_region_stats_dtbs = [0] * 3
    region_block = []

    # REGION
    # check whether there are regions
    if count[0][0] > 0:
        big_block = [0] * count[0][0]
        # extract region information
        for region in range(0, count[0][0]):
            region_info = np.fromfile(data_file, data_types[3], 1)
            for model in range(0, 3):
                for pol in range(0, 8):
                    stats_info = np.fromfile(data_file, data_types[4], 12)
                    delta_tbs = np.fromfile(data_file, data_types[5], 129 * 129)
                    region_stats_dtbs = [stats_info, delta_tbs]
                    pol_region_stats_dtbs[pol] = region_stats_dtbs

                mod_pol_region_stats_dtbs[model] = pol_region_stats_dtbs
                pol_region_stats_dtbs = [0] * 8
            # append into region block
            region_block.append([region_info, mod_pol_region_stats_dtbs])
            mod_pol_region_stats_dtbs = [0] * 3
            big_block[region] = region_block

            region_block = []
    else:

        logging.warning('Number of regions is less than 1 ...')
        region_stats = [np.zeros(1, dtype=data_types[4]), np.zeros(1, dtype=data_types[5])]
        pol_region_stats_dtbs[:] = [region_stats] * 8
        mod_pol_region_stats_dtbs[:] = pol_region_stats_dtbs * 3

        region_block.append([np.zeros(1, dtype=data_types[3]), mod_pol_region_stats_dtbs])
        big_block = [0]
        big_block[0] = region_block

    # SNAPSHOTS
    snap_count = np.fromfile(data_file, data_types[6], 1)

    # set up snap block and zone stats
    # snap_info = []
    snap_block = []
    zone_stats = [0] * 32

    # check if there are snaps
    if snap_count[0][0] > 0:
        # if snap count greater than 0 extract snap stuff
        for snap in range(0, snap_count[0][0]):
            snap_info = np.fromfile(data_file, data_types[7], 1)
            for measurement_zone in range(0, 32):
                # extract all zones
                measurement_count = np.fromfile(data_file, data_types[8], 1)
                l1c_stokes_stats = np.fromfile(data_file, data_types[9], 4)
                b_o_a_model_stats = np.fromfile(data_file, data_types[10], 4)
                t_o_a__l1c_t_e_c_model_stats = np.fromfile(data_file, data_types[11], 4)
                t_o_a__a3_t_e_c_model_stats = np.fromfile(data_file, data_types[12], 4)
                geophysical_stats = np.fromfile(data_file, data_types[13], 1)
                flags = np.fromfile(data_file, data_types[14], 1)

                zone_stats[measurement_zone] = [measurement_count, l1c_stokes_stats, b_o_a_model_stats,
                                                t_o_a__l1c_t_e_c_model_stats, t_o_a__a3_t_e_c_model_stats,
                                                geophysical_stats, flags]

            snap_block.append([snap_info, zone_stats[:]])
            zone_stats = [0] * 32
    if snap_count[0][0] == 0:
        logging.warning('Number of snapshots in region is 0.')
        # if no snaps. Make empty list to preserve shape.
        empty_snap_info = np.zeros(1, dtype=data_types[7])

        empty_zone_stats = [np.zeros(1, dtype=data_types[8]), np.zeros(1, dtype=data_types[9]),
                            np.zeros(1, dtype=data_types[10]), np.zeros(1, dtype=data_types[11]),
                            np.zeros(1, dtype=data_types[12]), np.zeros(1, dtype=data_types[13]),
                            np.zeros(1, dtype=data_types[13])]

        snap_block.append([empty_snap_info, empty_zone_stats])

    # extract gridpoint count
    gridpoint_count = np.fromfile(data_file, data_types[15], 1)
    meas_block = []
    gridpoint_block = []
    # check based on gridpoint count
    if gridpoint_count[0][0] > 0:
        # if grid points extract
        for gp in range(0, gridpoint_count[0][0]):

            gridpoint_info = np.fromfile(data_file, data_types[16], 1)  # grid point block
            gp_meas_count = np.fromfile(data_file, data_types[17], 1)  # gp_meas_count

            if gp_meas_count[0][0] > 0:
                meas_block = np.fromfile(data_file, data_types[18], gp_meas_count[0][0])  # meas block
            else:  # make blank to preserve shape
                meas_block = np.zeros(1, dtype=data_types[18])

            gridpoint_block.append([gridpoint_info, gp_meas_count, meas_block])

    else:
        gridpoint_block.append([np.zeros(1, dtype=data_types[16]), np.zeros(1, dtype=data_types[17]),
                                np.zeros(1, dtype=data_types[18])])

    data_file.close()

    return [version_number, max_valid, min_valid, count, big_block, snap_count, snap_block,
            gridpoint_count, gridpoint_block]


if __name__ == '__main__':

    dtbxy_file = '/home/famico/work/TESTS/SM-186_CNFOSF_for_v671/TC36/Outputs/' \
                 'SM_TEST_AUX_DTBXY__20120101T174428_20120101T183740_671_001_0/' \
                 'SM_TEST_AUX_DTBXY__20120101T174428_20120101T183740_671_001_0.DBL'
    logging.debug(dtbxy_file)

    import time
    t_start = time.time()
    read_dtbxy(data_types, dtbxy_file)
    t_end = time.time()
    t = t_end - t_start
    print('function read_dtbxy runs in ', t, ' seconds.')

