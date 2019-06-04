#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import logging
import logging.config
import os

from smos_tools.data_types.os_udp_datatype import datatype, science_flags_dtype, control_flags_dtype
from smos_tools.logger.logging_config import logging_config


def read_os_udp(filename):
    """
    Read the Ocean Salinity User Data Product file.
    
    :param filename: path to .DBL file
    :return: numpy structured array
    """
    # check the files are udp files
    if os.path.basename(filename)[14:17] != 'UDP':
        logging.exception('{} is not a UDP file'.format(filename))
        raise

    try:
        file = open(filename, 'rb')
    except IOError:
        logging.exception('file {} does not exist'.format(filename))
        raise

    logging.info('Reading file...')
    # Read first unsigned int32, containing number of grid points to iterate over
    n_grid_points = np.fromfile(file, dtype=np.uint32, count=1)[0]
    data = np.fromfile(file, dtype=np.dtype(datatype), count=n_grid_points)
    file.close()
    logging.info('Done.')
    
    return data


def unpack_control_flags(control_flag_data):
    """
    Unpacks the control flags into a numpy structured array so they can be made into a dataframe later

    :param control_flag_data: A control flag part of the data from read_os_udp
    :return: a numpy structured array
    """

    # Make the empty array with the right dtype for the data

    unpacked_flags = np.empty((len(control_flag_data)), dtype=control_flags_dtype)

    # unpack from Least Significant Bit

    for position in range(0, len(control_flags_dtype)):
        unpacked_flags[control_flags_dtype[position][0]] = (control_flag_data >> position) & 1

    return unpacked_flags


def unpack_science_flags(science_flag_data):
    """
    Unpacks the control flags into a numpy structured array so they can be made into a dataframe later

    :param science_flag_data: A science flag part of the data from read_os_udp
    :return: a numpy structured array
    """

    # Make the empty array with the right dtype for the data

    unpacked_flags = np.empty((len(science_flag_data)), dtype=science_flags_dtype)

    # unpack from Least Significant Bit

    for position in range(0, len(science_flags_dtype)):
        unpacked_flags[science_flags_dtype[position][0]] = (science_flag_data >> position) & 1

    return unpacked_flags


def extract_field(data, fieldname='SSS1'):
    """
    Converts the structured array into a pandas small dataframe.

    :param data: numpy structured array (record array). 
    :param fieldname: string (a field name from variable dtypes). 
    :return: pandas dataframe (columns are Mean_acq_time, Latitude, Longitude and fieldname) 
    Mean_acq_time is expressed in UTC decimal days (MJD2000 reference).
    """

    # NOTE there is a difference between how OS and SM handle mean acquisition time.
    # For OS this is a float expressed in UTC decimal days (in MJD2000) reference,
    # while for SM this has already been split into Days, seconds and microseconds
    time_frame = pd.DataFrame(data['Geophysical_Parameters_Data']['Mean_acq_time'], columns=['Mean_acq_time'])
    gridpoint_id_frame = pd.DataFrame(data['Grid_Point_Data']['Grid_Point_ID'], columns=['Grid_Point_ID'])
    lat_frame = pd.DataFrame(data['Grid_Point_Data']['Latitude'], columns=['Latitude'])
    lon_frame = pd.DataFrame(data['Grid_Point_Data']['Longitude'], columns=['Longitude'])

    # Hande which sub dictionary the field might be in
    geophys = [elem[0] for elem in datatype[1][1]]
    confidence = [elem[0] for elem in datatype[6][1]]
    if fieldname in geophys:
        dict_part = 'Geophysical_Parameters_Data'
    elif fieldname in confidence:
        dict_part = 'Product_Confidence_Descriptors'
    else:
        logging.error("ERROR: Couldn't find fieldname '{}' in "
                      "'Geophysical_Parameters_Data' or 'Product_Confidence_Descriptors'".format(fieldname))
        raise KeyError("{} not in Geophysical_Parameters_Data or Product_Confidence_Descriptors".format(fieldname))

    field_frame = pd.DataFrame(data[dict_part][fieldname], columns=[fieldname])

    dataframe = pd.concat([time_frame,
                           gridpoint_id_frame, lat_frame, lon_frame, field_frame], axis=1)

    dataframe = dataframe.replace(-999, np.NaN)
    dataframe.dropna(axis=0, inplace=True)
    
    return dataframe


def setup_os_plot(lat, long):
    """
    Sets up the orbit plot for ocean salinity
    :param lat: a list of latitudes
    :param long: a list of longitudes
    :return: figure object, basemap object, dot_size
    """
    fig1 = plt.figure()
    centre_lon = long.mean()
    centre_lat = lat.mean()
    # find a min and max lat and long
    # +-4 took from soil moisture plotting funct
    min_lon = max(long.min() - 4, -180.)
    max_lon = min(long.max() + 4, +180.)
    min_lat = max(lat.min() - 4, -90.)
    max_lat = min(lat.max() + 4, +90.)
    delta_lon = np.abs(max_lon - min_lon)
    delta_lat = np.abs(max_lat - min_lat)

    if delta_lat > 45:  # for  full orbit
        # lat_0 = 10. for soil moisture is 10
        lat_0 = 5.
        lon_0 = centre_lon
        width = 110574 * 70  # ~100km * 70 deg
        # height = 140 * 10**5 # 100km * 140 deg
        height = 10 ** 5 * 170  # 100km * 140 deg
        dot_size = 1
    else:
        lat_0 = centre_lat
        lon_0 = centre_lon
        width = delta_lon * 110574
        height = delta_lat * 10 ** 5
        dot_size = 5

    m = Basemap(
        projection='poly',
        lat_0=lat_0,
        lon_0=lon_0,
        width=width,
        height=height,
        resolution='l')

    m.drawcoastlines(linewidth=0.5)
    m.fillcontinents()
    # labels [left, right, top, bottom]
    m.drawparallels(np.arange(-80., 80., 20.), labels=[True, False, False, False], fontsize=8)
    m.drawmeridians(np.arange(-180, 180, 20.), labels=[False, False, False, True], fontsize=8, rotation=45)
    m.drawmapboundary()

    return fig1, m, dot_size


def plot_os_orbit(os_df, fieldname='SSS1', vmin=None, vmax=None):
    """
    Plot the ocean salinity UDP field fieldname.
    
    :param os_df: pandas dataframe containing Soil Moisture with index Days, Seconds, Microseconds, Grid_Point_ID
    :param fieldname: string fieldname of the data field to compare
    :return:
    """
    
    logging.info('Plotting {} orbit...'.format(fieldname))

    figure, m, dot_size = setup_os_plot(os_df['Latitude'].values, os_df['Longitude'].values)
    
    if fieldname in ['SSS1', 'SSS2']:
        plt.title(fieldname)
        cmap = 'viridis'
        c = os_df[fieldname]  # geophysical variable to plot
        if vmin == None:
            vmin = 32.
        if vmax == None:
            vmax = 38.
        m.scatter(os_df['Longitude'].values,
                  os_df['Latitude'].values,
                  latlon=True,
                  c=c,
                  s=dot_size,
                  zorder=10,
                  cmap=cmap,
                  vmin=vmin,
                  vmax=vmax,
                  )
        cbar = m.colorbar()
        cbar.set_label('[pss]')
        
    elif fieldname == 'SSS3':  # SSS anomaly
        plt.title('SSS anomaly')
        cmap = 'bwr'
        c = os_df[fieldname]  # geophysical variable to plot
        if vmin == None:
            vmin = -0.5
        if  vmax== None:
            vmax = +0.5
        m.scatter(os_df['Longitude'].values,
                  os_df['Latitude'].values,
                  latlon=True,
                  c=c,
                  s=dot_size,
                  zorder=10,
                  cmap=cmap,
                  )
        cbar = m.colorbar()
    
    else:
        plt.title(fieldname)
        cmap = 'viridis'
        c = os_df[fieldname] # geophysical variable to plot
        m.scatter(os_df['Longitude'].values,
                  os_df['Latitude'].values,
                  latlon=True,
                  c=c,
                  s=dot_size,
                  zorder=10,
                  cmap=cmap,
                  )
        cbar = m.colorbar()

    plt.show()


def plot_os_difference(os_df, fieldname='SSS1', vmin=-1, vmax=+1):
    """
        Plot the ocean salinity UDP difference for fieldname.

        :param os_df: pandas dataframe containing Soil Moisture with index Days, Seconds, Microseconds, Grid_Point_ID
        :param fieldname: string fieldname of the data field to compare
        :return:
        """

    logging.info('Plotting {} ...'.format(fieldname))

    figure, m, dot_size = setup_os_plot(os_df['Latitude'].values, os_df['Longitude'].values)

    plt.title(fieldname)
    cmap = 'bwr'
    c = os_df[fieldname]  # geophysical variable to plot

    m.scatter(os_df['Longitude'].values,
              os_df['Latitude'].values,
              latlon=True,
              c=c,
              s=dot_size,
              zorder=10,
              cmap=cmap,
              vmin=vmin,
              vmax=vmax)
    cbar = m.colorbar()

    plt.show()


def evaluate_field_diff(frame1, frame2, fieldname='SSS1', vmin=-1, vmax=+1, xaxis='Latitude'):
    """
    Plot the difference between two dataframes for a given field. Gives map plots and scatter.
    :param frame1: pandas dataframe containing the requested data field and index (Days, Seconds, Microseconds, Grid_Point_ID)
    :param frame2: pandas dataframe containing the requested data field and index (Days, Seconds, Microseconds, Grid_Point_ID)
    :param fieldname: String fieldname of the data field to compare
    :param vmin: Minimum value visible on plot. Lower values saturate.
    :param vmax: Maximum value visible on plot. Higher values saturate.
    :param xaxis: Varible againt which the fieldname is plotted. One of: {'Latitude', 'Grid_Point_ID'}
    :return:
    """
    logging.info('Evaluating difference between 2 dataframes for field {}...'.format(fieldname))

    # Print record counts
    logging.info('Dataset 1 contains {} valid datarows'.format(len(frame1.index)))
    logging.info('Dataset 2 contains {} valid datarows'.format(len(frame2.index)))

    # Get records in common
    common = pd.merge(frame1, frame2, how='inner', on=['Mean_acq_time', 'Grid_Point_ID'])
    common.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude'}, inplace=True)
    common.drop('Latitude_y', axis=1, inplace=True)
    common.drop('Longitude_y', axis=1, inplace=True)
    common[fieldname + '_Diff'] = common[fieldname + '_y'] - common[fieldname + '_x']
    common.reset_index(inplace=True)

    # Outer merge ready for getting new records
    outer = pd.merge(frame1, frame2, how='outer', on=['Mean_acq_time', 'Grid_Point_ID'],
                     indicator=True)
    # Get records in 1 but not 2
    leftonly = outer[outer['_merge'] == 'left_only'].copy()
    leftonly.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude', fieldname + '_x': fieldname},
                    inplace=True)
    leftonly.drop('Latitude_y', axis=1, inplace=True)
    leftonly.drop('Longitude_y', axis=1, inplace=True)
    leftonly.drop(fieldname + '_y', axis=1, inplace=True)
    leftonly.drop('_merge', axis=1, inplace=True)

    # Get records in 2 but not 1
    rightonly = outer[outer['_merge'] == 'right_only'].copy()
    rightonly.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude', fieldname + '_y': fieldname},
                     inplace=True)
    rightonly.drop('Latitude_y', axis=1, inplace=True)
    rightonly.drop('Longitude_y', axis=1, inplace=True)
    rightonly.drop(fieldname + '_x', axis=1, inplace=True)
    rightonly.drop('_merge', axis=1, inplace=True)

    logging.info('Dataset analysis:')
    logging.info('{} rows common to both datasets.'.format(len(common.index)))
    logging.info('{} rows in dataset 1 only.'.format(len(leftonly.index)))
    logging.info('{} rows in dataset 2 only.'.format(len(rightonly.index)))

    # Get records in common that are same/diff

    plot_os_difference(common, fieldname=fieldname + '_Diff', vmin=vmin, vmax=vmax)

    fig2, ax2 = plt.subplots(1)
    # plot each difference against the index grid point id
    common.plot(x=xaxis, y=fieldname + '_Diff', ax=ax2, legend=False, rot=90,
                fontsize=8, clip_on=False, style='o')
    ax2.set_ylabel(fieldname + ' Diff')
    ax2.axhline(y=0, linestyle=':', linewidth='0.5', color='k')
    fig2.tight_layout()

    # plot only the ones with a non-zero difference?
    non_zero_diff = common[common[fieldname + '_Diff'] != 0]
    if non_zero_diff.empty:
        logging.info('No differences to plot')
    else:
        fig3, ax3 = plt.subplots(1)
        non_zero_diff.plot(x=xaxis, y=fieldname + '_Diff', ax=ax3, legend=False,
                           rot=90, fontsize=8, clip_on=False, style='o')
        ax3.axhline(y=0, linestyle=':', linewidth='0.5', color='k')
        ax3.set_ylabel(fieldname + ' Diff')
        fig3.tight_layout()

    plt.show()


if __name__ == '__main__':

    logging.config.dictConfig(logging_config)

    logging.getLogger(__name__)

    udp1 = '/home/famico/repos/SMOS-L2OS-Processor/Outputs_ref/' \
        'SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_671_001_0/' \
        'SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_671_001_0.DBL'

    udp2 = '/home/famico/repos/SMOS-L2OS-Processor/Outputs_v673/' \
        'SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_673_001_0/' \
        'SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_673_001_0.DBL'
    print('==========')
    print(os.path.basename(udp1)[14:17])
    print(os.path.basename(udp1))
    print('==========')

    data1 = read_os_udp(udp1)
    df1 = extract_field(data1)
    #print(df1)

    data2 = read_os_udp(udp2)
    df2 = extract_field(data2)
    #print(df2)

    #evaluate_field_diff(df1, df2, fieldname='SSS1', vmin=-0.001, vmax=0.001, xaxis='Latitude')
    plot_os_orbit(df2, fieldname='SSS1', vmin=33, vmax=37)
    import sys
    sys.exit(0)


