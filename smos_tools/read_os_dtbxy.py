#!/usr/bin/env python3
"""
Read the AUX_DTBXY file (L2OS v67x).

Read the delta Tb product data block. The file contains a set of delta Tbs on a xi-eta grid for each region, together
with associated statistics. See product specs for the contents of the file.
"""


import numpy as np

import logging
import logging.config

from smos_tools.data_types.os_dtbxy_datatype import datatype
from smos_tools.logger.logging_config import logging_config


def read_dtbxy(datatype, dtbxy_file):
    """
    Read the data block of an AUX_DTBXY file (L2OS v67x).

    :param datatypes: list of lists; in each sublist, there is one or more tuples
    :param dtbxy_file: string filename
    :return: a list of numpy structures arrays, sometimes nested with other lists and arrays.
    """

    try:
        data_file = open(dtbxy_file, 'r')
    except IOError:
        logging.exception('file {} does not exist'.format(dtbxy_file))
        raise

    # L2OS processor version
    version_number = 670

    # max valid xi, eta
    max_valid = np.fromfile(data_file, datatype[0], 2)
    # print(max_valid)

    # min valid xi, eta
    min_valid = np.fromfile(data_file, datatype[1], 2)
    # print(min_valid)

    # get number of regions
    count = np.fromfile(data_file, datatype[2], 1)
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
            region_info = np.fromfile(data_file, datatype[3], 1)
            for model in range(0, 3):
                for pol in range(0, 8):
                    stats_info = np.fromfile(data_file, datatype[4], 12)
                    delta_tbs = np.fromfile(data_file, datatype[5], 129 * 129)
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
        region_stats = [np.zeros(1, dtype=datatype[4]), np.zeros(1, dtype=datatype[5])]
        pol_region_stats_dtbs[:] = [region_stats] * 8
        mod_pol_region_stats_dtbs[:] = pol_region_stats_dtbs * 3

        region_block.append([np.zeros(1, dtype=datatype[3]), mod_pol_region_stats_dtbs])
        big_block = [0]
        big_block[0] = region_block

    # SNAPSHOTS
    snap_count = np.fromfile(data_file, datatype[6], 1)

    # set up snap block and zone stats
    # snap_info = []
    snap_block = []
    zone_stats = [0] * 32

    # check if there are snaps
    if snap_count[0][0] > 0:
        # if snap count greater than 0 extract snap stuff
        for snap in range(0, snap_count[0][0]):
            snap_info = np.fromfile(data_file, datatype[7], 1)
            for measurement_zone in range(0, 32):
                # extract all zones
                measurement_count = np.fromfile(data_file, datatype[8], 1)
                l1c_stokes_stats = np.fromfile(data_file, datatype[9], 4)
                b_o_a_model_stats = np.fromfile(data_file, datatype[10], 4)
                t_o_a__l1c_t_e_c_model_stats = np.fromfile(data_file, datatype[11], 4)
                t_o_a__a3_t_e_c_model_stats = np.fromfile(data_file, datatype[12], 4)
                geophysical_stats = np.fromfile(data_file, datatype[13], 1)
                flags = np.fromfile(data_file, datatype[14], 1)

                zone_stats[measurement_zone] = [measurement_count, l1c_stokes_stats, b_o_a_model_stats,
                                                t_o_a__l1c_t_e_c_model_stats, t_o_a__a3_t_e_c_model_stats,
                                                geophysical_stats, flags]

            snap_block.append([snap_info, zone_stats[:]])
            zone_stats = [0] * 32
    if snap_count[0][0] == 0:
        logging.warning('Number of snapshots in region is 0.')
        # if no snaps. Make empty list to preserve shape.
        empty_snap_info = np.zeros(1, dtype=datatype[7])

        empty_zone_stats = [np.zeros(1, dtype=datatype[8]), np.zeros(1, dtype=datatype[9]),
                            np.zeros(1, dtype=datatype[10]), np.zeros(1, dtype=datatype[11]),
                            np.zeros(1, dtype=datatype[12]), np.zeros(1, dtype=datatype[13]),
                            np.zeros(1, dtype=datatype[13])]

        snap_block.append([empty_snap_info, empty_zone_stats])

    # extract gridpoint count
    gridpoint_count = np.fromfile(data_file, datatype[15], 1)
    meas_block = []
    gridpoint_block = []
    # check based on gridpoint count
    if gridpoint_count[0][0] > 0:
        # if grid points extract
        for gp in range(0, gridpoint_count[0][0]):

            gridpoint_info = np.fromfile(data_file, datatype[16], 1)  # grid point block
            gp_meas_count = np.fromfile(data_file, datatype[17], 1)  # gp_meas_count

            if gp_meas_count[0][0] > 0:
                meas_block = np.fromfile(data_file, datatype[18], gp_meas_count[0][0])  # meas block
            else:  # make blank to preserve shape
                meas_block = np.zeros(1, dtype=datatype[18])

            gridpoint_block.append([gridpoint_info, gp_meas_count, meas_block])

    else:
        gridpoint_block.append([np.zeros(1, dtype=datatype[16]), np.zeros(1, dtype=datatype[17]),
                                np.zeros(1, dtype=datatype[18])])

    data_file.close()

    return [version_number, max_valid, min_valid, count, big_block, snap_count, snap_block,
            gridpoint_count, gridpoint_block]


if __name__ == '__main__':

    logging.config.dictConfig(logging_config)

    logging.getLogger(__name__)

    dtbxy_file = '/home/rdavies/workspace/v670/test_data_v670/v670/' \
                 'SM_TEST_AUX_DTBXY__20140402T010641_20140402T015956_670_001_8/' \
                 'SM_TEST_AUX_DTBXY__20140402T010641_20140402T015956_670_001_8.DBL'

    logging.info(dtbxy_file)

    import time
    t_start = time.time()
    read_dtbxy(datatype, dtbxy_file)
    t_end = time.time()
    t = t_end - t_start
    logging.info('function read_dtbxy runs in {}  seconds.'.format(t))

