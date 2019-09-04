#!/usr/bin/env python

import numpy as np
import pandas as pd
import logging
import logging.config
import os
from smos_tools.data_types.aux_distan_datatype import datatype, flag_datatype
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

    logging.info('Reading file...')
    data = np.fromfile(file, dtype=np.dtype(datatype), count=2621442)

    file.close()
    logging.info('Done.')

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

    for position in range(0, len(flag_datatype)):
        unpacked_flags[flag_datatype[position][0]] = (distan_flags >> position) & 1

    return unpacked_flags


def read_gridpoint_to_is_sea(filename):

    # Read all data
    data = read_aux_distan(filename)

    # Unpack the flags into a matching length array
    flags = unpack_flag(data['Flag'])

    # Make a dataframe to manipulate these
    dataframe = pd.DataFrame({'Grid_Point_ID': data['Grid_Point_ID'],
        'Fg_Land_Sea_Coast1_tot': flags['Fg_Land_Sea_Coast1_tot'],
        'Fg_Land_Sea_Coast2_tot': flags['Fg_Land_Sea_Coast2_tot']})

    # Replace flag columns with a single column to signify sea. Is land when both flags are false, sea when either flag is true
    dataframe['Is_Sea'] = dataframe.apply(lambda row: (row['Fg_Land_Sea_Coast1_tot'] or row['Fg_Land_Sea_Coast2_tot']), axis=1)
    dataframe.drop(columns=['Fg_Land_Sea_Coast1_tot', 'Fg_Land_Sea_Coast2_tot'], inplace=True)

    return dataframe


if __name__ == '__main__':

    data = read_gridpoint_to_is_sea("/home/smos/builds/v6.71/smos/products/AUX_/DISTAN/SM_OPER_AUX_DISTAN_20050101T000000_20500101T000000_001_011_3/SM_OPER_AUX_DISTAN_20050101T000000_20500101T000000_001_011_3.DBL")
