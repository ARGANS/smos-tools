#!/usr/bin/env python

import numpy as np
import pandas as pd
import logging
import logging.config
import os
from smos_tools.data_types.aux_dgg_datatype import datatype
from smos_tools.logger.logging_config import logging_config


def read_aux_dgg(filename):
    """
    Read the AUX_DGG___ Product file.

    :param filename: path to .DBL file
    :return: numpy structured array
    """
    # check the file is an AUX_DGG file
    if os.path.basename(filename)[8:18] != 'AUX_DGG___':
        logging.exception('{} is not an AUX_DGG___ file'.format(filename))

    try:
        file = open(filename, 'rb')
    except IOError:
        logging.exception('file {} does not exist'.format(filename))
        raise

    data = np.empty(0, dtype=np.dtype(datatype))

    logging.debug('Reading AUX_DGG___ file...')
    for i in range(0,10):
        np.fromfile(file, dtype=np.uint64, count=1)[0] # zoneid
        grid_pt_counter = np.fromfile(file, dtype=np.uint32, count=1)[0]
        this_zone_data = np.fromfile(file, dtype=np.dtype(datatype), count=grid_pt_counter)
        data = np.concatenate((data, this_zone_data), axis=0)

    file.close()
    logging.debug('Done.')

    return data


if __name__ == '__main__':

    dataframes = read_aux_dgg("/home/smos/builds/v6.71/smos/products/AUX_/DGG___/SM_OPER_AUX_DGG____20050101T000000_20500101T000000_300_003_3/SM_OPER_AUX_DGG____20050101T000000_20500101T000000_300_003_3.DBL")
