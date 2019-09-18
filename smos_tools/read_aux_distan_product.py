#!/usr/bin/env python

import numpy as np
import pandas as pd
import logging
import logging.config
import os
from smos_tools.data_types.aux_distan_datatype import datatype, flag_datatype, flag_sea_ice_datatype
from smos_tools.logger.logging_config import logging_config


def read_aux_distan(filename):
    """
    Read the AUX_DISTAN Product file.

    :param filename: path to .DBL file
    :return: numpy structured array
    """
    # check the files is an AUX_DISTAN file
    if os.path.basename(filename)[8:18] != 'AUX_DISTAN':
        logging.exception('{} is not an AUX_DISTAN file'.format(filename))

    try:
        file = open(filename, 'rb')
    except IOError:
        logging.exception('file {} does not exist'.format(filename))
        raise

    logging.debug('Reading AUX_DISTAN file...')
    data = np.fromfile(file, dtype=np.dtype(datatype), count=2621442)

    file.close()
    logging.debug('Done.')

    return data


def unpack_flag(distan_flags):
    """
    Unpacks the distan flags into a numpy structured array, including blanks

    :param distan_flags: Flag component of the numpy structured data array for AUX_DISTAN
    :return: a numpy structured array
    """

    # Make the empty array with the right dtype for the data

    unpacked_flags = np.empty((len(distan_flags)), dtype=flag_datatype)

    # unpack from Least Significant Bit

    logging.debug('Unpacking flags...')
    for position in range(0, len(flag_datatype)):
        unpacked_flags[flag_datatype[position][0]] = (distan_flags >> position) & 1

    return unpacked_flags


def unpack_sea_ice_flag(sea_ice_flags):
    """
    Unpacks the distan sea_ice_mask flags into a numpy structured array, including blanks

    :param sea_ice_flags: Sea-ice mask flag component of the numpy structured data array for AUX_DISTAN
    :return: a numpy structured array
    """

    # Make the empty array with the right dtype for the data

    unpacked_flags = np.empty((len(sea_ice_flags)), dtype=flag_sea_ice_datatype)

    # unpack from Least Significant Bit

    logging.debug('Unpacking sea-ice mask flags...')
    for position in range(0, len(flag_sea_ice_datatype)):
        unpacked_flags[flag_sea_ice_datatype[position][0]] = (sea_ice_flags >> position) & 1

    return unpacked_flags


def read_gridpoint_to_is_sea_and_ice(filename):
    """
    Read the AUX_DISTAN Product file, unpack flags and manipulate to get pandas dataframe, with Grid_Point_ID, Is_Sea, Is_Sometimes_Sea_Ice, Is_Always_Sea_Ice

    :param filename: path to AUX_DISTAN .DBL file
    :return: Pandas dataframe, Grid_Point_ID, Is_Sea
    """

    # Read all data
    data = read_aux_distan(filename)

    # Unpack the flags into a matching length array
    flags = unpack_flag(data['Flag'])

    # Unpack the sea-ice mask flags
    sea_ice_flags = unpack_sea_ice_flag(data['Sea_Ice_Mask'])

    # Make a dataframe to manipulate these
    dataframe = pd.DataFrame({'Grid_Point_ID': data['Grid_Point_ID'],
        'Fg_Land_Sea_Coast1_tot': flags['Fg_Land_Sea_Coast1_tot'],
        'Fg_Land_Sea_Coast2_tot': flags['Fg_Land_Sea_Coast2_tot'],
        'Sea_Ice_Month_1': sea_ice_flags['Month1'],
        'Sea_Ice_Month_2': sea_ice_flags['Month2'],
        'Sea_Ice_Month_3': sea_ice_flags['Month3'],
        'Sea_Ice_Month_4': sea_ice_flags['Month4'],
        'Sea_Ice_Month_5': sea_ice_flags['Month5'],
        'Sea_Ice_Month_6': sea_ice_flags['Month6'],
        'Sea_Ice_Month_7': sea_ice_flags['Month7'],
        'Sea_Ice_Month_8': sea_ice_flags['Month8'],
        'Sea_Ice_Month_9': sea_ice_flags['Month9'],
        'Sea_Ice_Month_10': sea_ice_flags['Month10'],
        'Sea_Ice_Month_11': sea_ice_flags['Month11'],
        'Sea_Ice_Month_12': sea_ice_flags['Month12']
        })
    # TODO then same apply as below

    logging.debug('Converting flag values to Is_Sea boolean...')
    # Replace flag columns with a single column to signify sea. Is land when both flags are false, sea when either flag is true
    dataframe['Is_Sea'] = dataframe.apply(lambda row: (row['Fg_Land_Sea_Coast1_tot'] or row['Fg_Land_Sea_Coast2_tot']), axis=1)
    dataframe.drop(columns=['Fg_Land_Sea_Coast1_tot', 'Fg_Land_Sea_Coast2_tot'], inplace=True)

    logging.debug('Converting sea-ice mask flag values to Is_Sometimes_Sea_Ice, Is_Always_Sea_Ice booleans...')
    dataframe['Is_Sometimes_Sea_Ice'] = dataframe.apply(lambda row: (row['Sea_Ice_Month_1'] or
        row['Sea_Ice_Month_2'] or
        row['Sea_Ice_Month_3'] or
        row['Sea_Ice_Month_4'] or
        row['Sea_Ice_Month_5'] or
        row['Sea_Ice_Month_6'] or
        row['Sea_Ice_Month_7'] or
        row['Sea_Ice_Month_8'] or
        row['Sea_Ice_Month_9'] or
        row['Sea_Ice_Month_10'] or
        row['Sea_Ice_Month_11'] or
        row['Sea_Ice_Month_12']), axis=1)
    dataframe['Is_Always_Sea_Ice'] = dataframe.apply(lambda row: (row['Sea_Ice_Month_1'] and
        row['Sea_Ice_Month_2'] and
        row['Sea_Ice_Month_3'] and
        row['Sea_Ice_Month_4'] and
        row['Sea_Ice_Month_5'] and
        row['Sea_Ice_Month_6'] and
        row['Sea_Ice_Month_7'] and
        row['Sea_Ice_Month_8'] and
        row['Sea_Ice_Month_9'] and
        row['Sea_Ice_Month_10'] and
        row['Sea_Ice_Month_11'] and
        row['Sea_Ice_Month_12']), axis=1)
    dataframe.drop(columns=[
        'Sea_Ice_Month_1',
        'Sea_Ice_Month_2',
        'Sea_Ice_Month_3',
        'Sea_Ice_Month_4',
        'Sea_Ice_Month_5',
        'Sea_Ice_Month_6',
        'Sea_Ice_Month_7',
        'Sea_Ice_Month_8',
        'Sea_Ice_Month_9',
        'Sea_Ice_Month_10',
        'Sea_Ice_Month_11',
        'Sea_Ice_Month_12'], inplace=True)

    return dataframe


if __name__ == '__main__':

    data = read_gridpoint_to_is_sea_and_ice("/home/smos/builds/v6.71/smos/products/AUX_/DISTAN/SM_OPER_AUX_DISTAN_20050101T000000_20500101T000000_001_011_3/SM_OPER_AUX_DISTAN_20050101T000000_20500101T000000_001_011_3.DBL")
