#!/usr/bin/env python

import numpy as np
import pandas as pd
import logging
import logging.config
import os
from smos_tools.data_types.aux_distan_datatype import datatype
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


if __name__ == '__main__':

    dataframes = read_aux_distan("/home/smos/builds/v6.71/smos/products/AUX_/DISTAN/SM_OPER_AUX_DISTAN_20050101T000000_20500101T000000_001_011_3/SM_OPER_AUX_DISTAN_20050101T000000_20500101T000000_001_011_3.DBL")
