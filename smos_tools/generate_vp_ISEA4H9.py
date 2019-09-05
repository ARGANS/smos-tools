#!/usr/bin/env python

import numpy as np
import pandas as pd
import logging
import logging.config
import os
from smos_tools.read_aux_dgg_product import read_aux_dgg
from smos_tools.read_aux_distan_product import read_gridpoint_to_is_sea
from smos_tools.logger.logging_config import logging_config


def generate_ISEA4H9(aux_dgg_filepath, aux_distan_filepath, output_directory):
    """
    Generate ISEA4H9 grid specification CSV, containing columns Grid_Point_ID, Latitude, Longitude, Is_Sea

    :param aux_dgg_filepath: Path to an AUX_DGG___ .DBL file
    :param aux_distan_filepath: Path to an AUX_DISTAN .DBL file
    :param output_directory: Path to a directory to save output csv
    :return: Success/fail
    """

    output_filepath = os.path.join(output_directory, 'ISEA4H9_from_AUX_files.csv')

    # Do some basic IO checks
    if not os.path.exists(aux_dgg_filepath):
        raise FileNotFoundError(aux_dgg_filepath)
    if not os.path.exists(aux_distan_filepath):
        raise FileNotFoundError(aux_distan_filepath)

    if not os.path.exists(output_directory):
        raise IOError('No directory: {}'.format(output_directory))
    if os.path.exists(output_filepath):
        raise FileExistsError(output_filepath)

    logging.info('Output location will be: {}'.format(output_filepath))

    # First read the two input files
    logging.info('Reading input files...')
    dgg = pd.DataFrame(read_aux_dgg(aux_dgg_filepath))
    distan = read_gridpoint_to_is_sea(aux_distan_filepath)
    # Dataframe contents
    # Grid_Point_ID   Latitude   Longitude   Altitude
    # Grid_Point_ID  Is_Sea

    logging.info('Joining AUX file contents...')
    # Join on the Grid_Point_ID column
    out = dgg.merge(distan, on='Grid_Point_ID', how='inner')
    # Don't want the Altitude column in the CSV
    out = out.drop(labels='Altitude', axis=1)

    logging.info('Converting and writing output CSV...')
    # Save the file as a CSV with column headers
    out.to_csv(output_filepath, columns=['Grid_Point_ID', 'Latitude', 'Longitude', 'Is_Sea'], index=False)

    logging.info('ISEA4H9 creation complete. File saved.')


if __name__ == '__main__':

    logging.config.dictConfig(logging_config)
    logging.getLogger(__name__)

    dgg_fp = '/home/smos/builds/v6.71/smos/products/AUX_/DGG___/SM_OPER_AUX_DGG____20050101T000000_20500101T000000_300_003_3/SM_OPER_AUX_DGG____20050101T000000_20500101T000000_300_003_3.DBL'
    distan_fp = '/home/smos/builds/v6.71/smos/products/AUX_/DISTAN/SM_OPER_AUX_DISTAN_20050101T000000_20500101T000000_001_011_3/SM_OPER_AUX_DISTAN_20050101T000000_20500101T000000_001_011_3.DBL'
    output_fp = 'insert_output_directory_here'
    generate_ISEA4H9(dgg_fp, distan_fp, output_fp)
