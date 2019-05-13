#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import argparse
import os
import sys
import logging
import logging.config

from smos_tools.data_types.sm_udp_datatype import datatype
from smos_tools.logger.logging_config import logging_config




def read_sm_product(filepath):
    """
    Read the Soil Moisture UDP file
    :param filepath: path to .DBL file
    :return: numpy structured array
    """
    # Open the data file for reading

    try:
        file = open(filepath, 'rb')
    except IOError:
        logging.exception('file {} does not exist'.format(filepath))
        raise

    # Read first unsigned int32, containing number of datapoints to iterate over
    n_grid_points = np.fromfile(file, dtype=np.uint32, count=1)[0]
    logging.info('Data file contains {} data points'.format(n_grid_points))
    logging.info('Reading file... ')
    data = np.fromfile(file, dtype=datatype, count=n_grid_points)
    file.close()
    logging.info('Done')

    return data


def extract_field(data, fieldname):
    """
    Take numpy structured array and extract with requested field into pandas dataframe.
    :param data: numpy structured array
    :param fieldname: string of fieldname inside 'Retrieval_Results_Data' to extract
    :return: pandas dataframe containing requested field along with index (Days, Seconds, Microseconds, Grid_Point_ID)
    """
    # Make every nested structured numpy array into a dataframe of its own
    base_frame = pd.DataFrame(data)
    time_frame = pd.DataFrame(data['Mean_Acq_Time'])
    retrieval_frame = pd.DataFrame(data['Retrieval_Results_Data'])

    # Look to see if user requested fieldname exists.
    if fieldname not in retrieval_frame.columns.values:
        logging.error("ERROR: Couldn't find fieldname '{}' in 'Retrieval_Results_Data'".format(fieldname))
        raise KeyError("{} not one of {}".format(fieldname, retrieval_frame.columns.values))

    # Make a dataframe with the columns we care about

    extracted_data = pd.concat([time_frame['Days'], time_frame['Seconds'],
                                time_frame['Microseconds'], base_frame['Grid_Point_ID'],
                                base_frame['Latitude'], base_frame['Longitude'],
                                retrieval_frame[fieldname]], axis=1)

    # The time fields, and the gridpoint ID combine to make a unique index we can join over
    extracted_data = extracted_data.set_index(['Days', 'Seconds', 'Microseconds', 'Grid_Point_ID'])

    # convert all -999 to NaN and drop
    extracted_data.replace(to_replace=-999.0, value=np.nan, inplace=True)
    extracted_data.dropna(axis=0, inplace=True)

    return extracted_data


# Plot difference between 2 dataframes containing soil moisture
def evaluate_field_diff(smdf1, smdf2, fieldname):
    """
    Plot the difference between two dataframes for a given field. Gives map plots and scatter.
    :param smdf1: pandas dataframe containing the requested data field and index (Days, Seconds, Microseconds, Grid_Point_ID)
    :param smdf2: pandas dataframe containing the requested data field and index (Days, Seconds, Microseconds, Grid_Point_ID)
    :param fieldname: String fieldname of the data field to compare
    :return:
    """
    logging.info('Evaluating difference between 2 dataframes for field {}...'.format(fieldname))

    # Exclude NaN records (reported as fieldname = -999.0)
    frame1 = smdf1[smdf1[fieldname] != -999.0]
    frame2 = smdf2[smdf2[fieldname] != -999.0]

    # Print record counts
    logging.info('Dataset 1 contains {}/{} valid datarows'.format(len(frame1.index), len(smdf1)))
    logging.info('Dataset 2 contains {}/{} valid datarows'.format(len(frame2.index), len(smdf2)))

    # Get records in common
    common = pd.merge(frame1, frame2, how='inner', on=['Days', 'Seconds', 'Microseconds', 'Grid_Point_ID'])
    common.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude'}, inplace=True)
    common.drop('Latitude_y', axis=1, inplace=True)
    common.drop('Longitude_y', axis=1, inplace=True)
    common[fieldname+'_Diff'] = common[fieldname+'_y'] - common[fieldname+'_x']
    common.reset_index(inplace=True)

    # Outer merge ready for getting new records
    outer = pd.merge(frame1, frame2, how='outer', on=['Days', 'Seconds', 'Microseconds', 'Grid_Point_ID'],
                     indicator=True)
    # Get records in 1 but not 2
    leftonly = outer[outer['_merge'] == 'left_only'].copy()
    leftonly.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude', fieldname+'_x': fieldname},
                    inplace=True)
    leftonly.drop('Latitude_y', axis=1, inplace=True)
    leftonly.drop('Longitude_y', axis=1, inplace=True)
    leftonly.drop(fieldname+'_y', axis=1, inplace=True)
    leftonly.drop('_merge', axis=1, inplace=True)

    # Get records in 2 but not 1
    rightonly = outer[outer['_merge'] == 'right_only'].copy()
    rightonly.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude', fieldname+'_y': fieldname},
                     inplace=True)
    rightonly.drop('Latitude_y', axis=1, inplace=True)
    rightonly.drop('Longitude_y', axis=1, inplace=True)
    rightonly.drop(fieldname+'_x', axis=1, inplace=True)
    rightonly.drop('_merge', axis=1, inplace=True)

    logging.info('Dataset analysis:')
    logging.info('{} rows common to both datasets.'.format(len(common.index)))
    logging.info('{} rows in dataset 1 only.'.format(len(leftonly.index)))
    logging.info('{} rows in dataset 2 only.'.format(len(rightonly.index)))

    # Get records in common that are same/diff

    plot_sm_difference(common, fieldname=fieldname+'_Diff')

    fig2, ax2 = plt.subplots(1)
    # plot each difference against the index grid point id
    common.plot(x='Grid_Point_ID', y=fieldname+'_Diff', ax=ax2, legend=False, rot=90,
                fontsize=8, clip_on=False, style='o')
    ax2.set_ylabel(fieldname + ' Diff')
    ax2.axhline(y=0, linestyle=':', linewidth='0.5', color='k')
    fig2.tight_layout()

    # plot only the ones with a non-zero difference?
    non_zero_diff = common[common[fieldname+'_Diff'] != 0]
    if non_zero_diff.empty:
        logging.info('No differences to plot')
    else:
        fig3, ax3 = plt.subplots(1)
        non_zero_diff.plot(x='Grid_Point_ID', y=fieldname+'_Diff', ax=ax3, legend=False,
                           rot=90, fontsize=8, clip_on=False, style='o')
        ax3.axhline(y=0, linestyle=':', linewidth='0.5', color='k')
        ax3.set_ylabel(fieldname + ' Diff')
        fig3.tight_layout()

    plt.show()


def setup_sm_plot(lat, long):
    fig1 = plt.figure()
    # Set up plot
    # find a central lon and lat
    centre_lon = long.mean()
    centre_lat = lat.mean()
    # find a min and max lat and long
    min_lon = max(long.min() - 4, -180.)
    max_lon = min(long.max() + 4, +180.)
    delta_lon = np.abs(max_lon - min_lon)

    min_lat = max(lat.min() - 4, -90.)
    max_lat = min(lat.max() + 4, +90.)
    delta_lat = np.abs(max_lat - min_lat)

    # for a full orbit?
    # width=110574 * 90,
    # height=16 * 10**6

    if delta_lat > 45:
        lat_0 = 10.
        lon_0 = centre_lon
        width = 110574 * 70
        height = 14 * 10 ** 6
        dot_size = 1
    else:
        lat_0 = centre_lat
        lon_0 = centre_lon
        width = delta_lon * 110574
        height = delta_lat * 10 ** 5
        dot_size = 5

    m = Basemap(
        # projection='cyl',
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


# Plot a SM orbit from a pandas dataframe
def plot_sm_orbit(smdf, fieldname='Soil_Moisture'):
    """
     Plot the soil moisture orbit. Gives map plots and scatter.
    
    :param smdf: pandas dataframe containing Soil Moisture with index Days, Seconds, Microseconds, Grid_Point_ID
    :param fieldname: string fieldname of the data field to compare
    :return:
    """

    logging.info('Plotting {} orbit...'.format(fieldname))
    
    fig, m, dot_size = setup_sm_plot(smdf['Latitude'].values, smdf['Longitude'].values)
    
    if fieldname == 'Soil_Moisture':
        plt.title(fieldname)
        cmap = 'viridis_r'
        c = smdf[fieldname]  # geophysical variable to plot
        vmin = 0.
        vmax = 1.
        m.scatter(smdf['Longitude'].values,
          smdf['Latitude'].values,
          latlon=True,
          c=c,
          s=dot_size,
          zorder=10,
          cmap=cmap,
          vmin=vmin,
          vmax=vmax,
          )
        cbar = m.colorbar()
        cbar.set_label(r'[m$^3$/m$^3$]')

    else:
        plt.title(fieldname)
        cmap = 'viridis'
        c = smdf[fieldname] # geophysical variable to plot 
        m.scatter(smdf['Longitude'].values,
          smdf['Latitude'].values,
          latlon=True,
          c=c,
          s=dot_size,
          zorder=10,
          cmap=cmap,
          )    
        cbar = m.colorbar()

    plt.show()


def plot_sm_difference(smdf, fieldname='Soil_Moisture'):
    """
        Plot the difference between two dataframes.

        :param smdf: pandas dataframe containing Soil Moisture with index Days, Seconds, Microseconds, Grid_Point_ID
        :param fieldname: string fieldname of the data field to compare
        :return:
        """
    logging.info('Plotting {} difference ...'.format(fieldname))

    fig, m, dot_size = setup_sm_plot(smdf['Latitude'].values, smdf['Longitude'].values)

    plt.title(fieldname)
    cmap = 'bwr'
    c = smdf[fieldname]  # geophysical variable to plot
    vmin = -1.
    vmax = +1.
    m.scatter(smdf['Longitude'].values,
              smdf['Latitude'].values,
              latlon=True,
              c=c,
              s=dot_size,
              zorder=10,
              cmap=cmap,
              vmin=vmin,
              vmax=vmax)
    cbar = m.colorbar()

    plt.show()


if __name__ == '__main__':

    logging.config.dictConfig(logging_config)

    logging.getLogger(__name__)
