#!/usr/bin/env python

"""
make_aux_sss: functions to make an AUX_SSS binary file for L2OS

"""
import struct
import numpy as np
from netCDF4 import Dataset


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
    dataset = Dataset(asc_file, "r")
    gpids = np.asarray(dataset.variables["GPID"])
    sss_clim = np.asarray(dataset.variables["SSSmean"]).transpose()

    # need to remove GPS with all NaN clim
    no_nans = sss_clim[np.where(~np.isnan(sss_clim).all(axis=1))]
    gpids = gpids[np.where(~np.isnan(sss_clim).all(axis=1))]

    # now we need to scale the data to be ushort.
    # * 1000 and truncate

    scaled = (no_nans * 1000).astype(np.ushort)

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


def make_test_aux_sss(output_loc):
    """

    :param output_loc: The ouput file to write to
    :return:
    """

    gpids = np.asarray([1, 2, 3, 4, 5], dtype=np.uint32)
    asc_sss = np.ones((5, 34), dtype=np.ushort)
    desc_sss = asc_sss * 2

    fout = open(output_loc, 'wb')

    # write the ascending
    fout = write_to_binary(gpids, asc_sss, fout)

    fout = write_to_binary(gpids, desc_sss, fout)

    fout.close()


if __name__ == "__main__":
    asc_file = '/mnt/smos_int/smos/Manuel/SSS_anomaly/01-climatologies/ACRI-ST/SMOSmean_nocorrTB_A_001.nc'
    desc_file = '/mnt/smos_int/smos/Manuel/SSS_anomaly/01-climatologies/ACRI-ST/SMOSmean_nocorrTB_D_001.nc'

    # make_aux_sss(asc_file, desc_file, 'SM_OPER_AUX_SSSCLI_20050101T000000_20500101T000000_001_014_3.DBL')
    # make_test_aux_sss('SM_TEST_AUX_SSSCLI_20050101T000000_20500101T000000_001_014_3.DBL')
    test_read_aux_sss('/home/rdavies/workspace/SMOS_processor/smos/products/AUX_/'
                      'SSSCLI/SM_TEST_AUX_SSSCLI_20050101T000000_20500101T000000_001_014_3/'
                      'SM_TEST_AUX_SSSCLI_20050101T000000_20500101T000000_001_014_3.DBL')
