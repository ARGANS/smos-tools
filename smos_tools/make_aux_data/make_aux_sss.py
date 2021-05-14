#!/usr/bin/env python

"""
make_aux_sss: functions to make an AUX_SSS binary file for L2OS

"""
import struct
import numpy as np
from netCDF4 import Dataset
import os
from datetime import datetime

def test_read_aux_sss(aux_sss_file):
    """

    :param aux_sss_file: Path to the binary AUX_SSS .DBL file
    :return:
    """

    num_gps = 5  # in test

    dtype = [('Grid_Point_ID', np.uint32),
             ('1', np.ushort),
             ('2', np.ushort),
             ('3', np.ushort),
             ('4', np.ushort),
             ('5', np.ushort),
             ('6', np.ushort),
             ('7', np.ushort),
             ('8', np.ushort),
             ('9', np.ushort),
             ('10', np.ushort),
             ('11', np.ushort),
             ('12', np.ushort),
             ('13', np.ushort),
             ('14', np.ushort),
             ('15', np.ushort),
             ('16', np.ushort),
             ('17', np.ushort),
             ('18', np.ushort),
             ('19', np.ushort),
             ('20', np.ushort),
             ('21', np.ushort),
             ('22', np.ushort),
             ('23', np.ushort),
             ('24', np.ushort),
             ('25', np.ushort),
             ('26', np.ushort),
             ('27', np.ushort),
             ('28', np.ushort),
             ('29', np.ushort),
             ('30', np.ushort),
             ('31', np.ushort),
             ('32', np.ushort),
             ('33', np.ushort),
             ('34', np.ushort),
             ]

    aux_sss_file = open(aux_sss_file, 'rb')

    aux_sss_asc = np.fromfile(aux_sss_file, dtype=dtype, count=num_gps)

    print(aux_sss_asc)

    aux_sss_desc = np.fromfile(aux_sss_file, dtype=dtype, count=num_gps)

    print(aux_sss_desc)


def format_aux_sss(input_file):
    """

    :param input_file: The .nc file containing the climatology data
    :return: numpy arrays for gpids and scaled sss_clim
    """
    dataset = Dataset(input_file, "r")
    gpids = np.asarray(dataset.variables["GPID"])
    sss_clim = np.asarray(dataset.variables["SSSmean"]).transpose()

    # need to remove GPS with all NaN clim
    no_nans = sss_clim[np.where(~np.isnan(sss_clim).all(axis=1))]
    gpids = gpids[np.where(~np.isnan(sss_clim).all(axis=1))]

    # now we need to scale the data to be ushort.
    # * 1000 and truncate
    # should we round here?
    scaled = (no_nans * 1000).astype(np.ushort)
    print('Input file: {}'.format(input_file))
    print('Num.valid gpts: {}'.format(len(gpids)))
    print('Num.valid clim values: {}'.format(len(scaled)))
    
    valid_gpt_file = input_file + '_valid.txt'
    print('Writing info on valid grid points to {}'.format(valid_gpt_file))
    f = open(valid_gpt_file, 'w')
    f.write('Input file: {}\n'.format(input_file))
    f.write('Num.valid gpts: {}\n'.format(len(gpids)))
    f.write('Num.valid clim values: {}\n'.format(len(scaled)))
    f.close()

    return gpids, scaled


def write_to_binary(gridpoints, clim, fout):
    """

    :param gridpoints: The numpy array of gridpoints,
    :param clim: The numpy array of climatology, shape (number_gridpoints, 34)
    :param fout: The open file to write to
    :return:
    """
    # L = uint32, H = ushort
    fmt_gpid = '< L'
    fmt_sss = '< 34H'

    for i in range(0, len(gridpoints)):
        fout.write(struct.pack(fmt_gpid, gridpoints[i]))
        fout.write(struct.pack(fmt_sss, *clim[i]))

    return fout


def make_aux_sss(asc_file, desc_file, output_loc):
    """

    :param asc_file: The path to the .nc file for ascending orbits
    :param desc_file: The path to the .nc file for descending orbits
    :return:
    """

    fout = open(output_loc, 'wb')

    # write the ascending
    gpids, asc_sss_clim = format_aux_sss(asc_file)
    fout = write_to_binary(gpids, asc_sss_clim, fout)

    gpids, desc_sss_clim = format_aux_sss(desc_file)
    fout = write_to_binary(gpids, desc_sss_clim, fout)

    fout.close()


def read_acri_clim_manuel(clim_file):
    clim_dataset = Dataset(clim_file)
    gp_id = np.array(clim_dataset.variables['GPID'][:], dtype=np.int32)
    clim = np.array(clim_dataset.variables['SSSmean'][:, :], dtype=np.float32)
    x_swath = np.array(clim_dataset.variables['xswath'][:], dtype=np.float32)
    clim_dataset.close()

    return gp_id, clim

def make_test_aux_sss(output_loc):
    """

    :param output_loc: The ouput file to write to
    :return:
    """

    gpids = np.asarray([1, 2, 3, 4, 5], dtype=np.uint32)
    asc_sss = np.zeros((5, 34), dtype=np.ushort)
    vals = np.arange(1, 35, dtype=np.ushort)
    for i in range(0, 5):
        asc_sss[i, :] = vals
    desc_sss = asc_sss * 2

    fout = open(output_loc, 'wb')

    # write the ascending
    fout = write_to_binary(gpids, asc_sss, fout)

    fout = write_to_binary(gpids, desc_sss, fout)

    fout.close()


if __name__ == "__main__":
    asc_file = '/mnt/smos_int/smos/ANOMALY_ACRI/SMOSmean_nocorrTB_A_002.nc'
    desc_file = '/mnt/smos_int/smos/ANOMALY_ACRI/SMOSmean_nocorrTB_D_002.nc'
    #asc_data, desc_data = read_acri_clim_manuel(asc_file, desc_file)
    
    output_file = 'SM_TEST_AUX_SSSCLI_20050101T000000_20500101T000000_001_002_8.DBL'
    output_file_info = os.path.basename(output_file).replace('.DBL', '.info')
    f = open(output_file_info, 'w')
    f.write('Input netCDF file used for ascending: {}\n'.format(asc_file))
    f.write('Input netCDF file used for descending: {}\n'.format(desc_file))
    f.write('Script used for generation: {}, in {}\n'.format(__file__, os.getcwd()))
    f.write('Creation date: {}'.format(datetime.now()))
    f.close()
    
    #my_asc = format_aux_sss(asc_file)
    #my_desc = format_aux_sss(desc_file)
    make_aux_sss(asc_file, desc_file, output_file)

    #make_test_aux_sss('SM_TEST_AUX_SSSCLI_20050101T000000_20500101T000000_001_014_3.DBL')
    #test_read_aux_sss('/home/rdavies/workspace/SMOS_processor/smos/products/AUX_/'
    #                  'SSSCLI/SM_TEST_AUX_SSSCLI_20050101T000000_20500101T000000_001_014_3/'
    #                  'SM_TEST_AUX_SSSCLI_20050101T000000_20500101T000000_001_014_3.DBL')
