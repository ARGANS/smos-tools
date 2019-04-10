#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import argparse
import os
import sys


# Primary array datatype, repeated n_grid_points times, Grid_Point_Data_Type
datatype = [('Grid_Point_ID', np.uint32),
            ('Latitude', np.float32),
            ('Longitude', np.float32),
            ('Altitude', np.float32),
            # UTC_Type
            ('Mean_Acq_Time', [
                ('Days', np.int32),
                ('Seconds', np.uint32),
                ('Microseconds', np.uint32)
            ]),
            # Retrieval_Results_Data_Type
            ('Retrieval_Results_Data', [
                ('Soil_Moisture', np.float32),
                ('Soil_Moisture_DQX', np.float32),
                ('Optical_Thickness_Nad', np.float32),
                ('Optical_Thickness_Nad_DQX', np.float32),
                ('Surface_Temperature', np.float32),
                ('Surface_Temperature_DQX', np.float32),
                ('TTH', np.float32),
                ('TTH_DQX', np.float32),
                ('RTT', np.float32),
                ('RTT_DQX', np.float32),
                ('Scattering_Albedo_H', np.float32),
                ('Scattering_Albedo_H_DQX', np.float32),
                ('DIFF_Albedos', np.float32),
                ('DIFF_Albedos_DQX', np.float32),
                ('Roughness_Param', np.float32),
                ('Roughness_Param_DQX', np.float32),
                ('Dielect_Const_MD_RE', np.float32),
                ('Dielect_Const_MD_RE_DQX', np.float32),
                ('Dielect_Const_MD_IM', np.float32),
                ('Dielect_Const_MD_IM_DQX', np.float32),
                ('Dielect_Const_Non_MD_RE', np.float32),
                ('Dielect_Const_Non_MD_RE_DQX', np.float32),
                ('Dielect_Const_Non_MD_IM', np.float32),
                ('Dielect_Const_Non_MD_IM_DQX', np.float32),
                ('TB_ASL_Theta_B_H', np.float32),
                ('TB_ASL_Theta_B_H_DQX', np.float32),
                ('TB_ASL_Theta_B_V', np.float32),
                ('TB_ASL_Theta_B_V_DQX', np.float32),
                ('TB_TOA_Theta_B_H', np.float32),
                ('TB_TOA_Theta_B_H_DQX', np.float32),
                ('TB_TOA_Theta_B_V', np.float32),
                ('TB_TOA_Theta_B_V_DQX', np.float32)
            ]),
            # Confidence_Descriptors_Data_Type
            ('Confidence_Descriptors_Data', [
                ('Confidence_Flags', np.uint16),
                ('GQX', np.uint8),
                ('Chi_2', np.uint8),
                ('Chi_2_P', np.uint8),
                ('N_Wild', np.uint16),
                ('M_AVA0', np.uint16),
                ('M_AVA', np.uint16),
                ('AFP', np.float32),
                ('N_AF_FOV', np.uint16),
                ('N_Sun_Tails', np.uint16),
                ('N_Sun_Glint_Area', np.uint16),
                ('N_Sun_FOV', np.uint16),
                ('N_RFI_Mitigations', np.uint16),
                ('N_Strong_RFI', np.uint16),
                ('N_Point_Source_RFI', np.uint16),
                ('N_Tails_Point_Source_RFI', np.uint16),
                ('N_Software_Error', np.uint16),
                ('N_Instrument_Error', np.uint16),
                ('N_ADF_Error', np.uint16),
                ('N_Calibration_Error', np.uint16),
                ('N_X_Band', np.uint16)
            ]),
            # Science_Descriptors_Data_Type
            ('Science_Descriptors_Data', [
                ('Science_Flags', np.uint32),
                ('N_Sky', np.uint16)
            ]),
            # Processing_Descriptors_Data_Type
            ('Processing_Descriptors_Data', [
                ('Processing_Flags', np.uint16),
                ('S_Tree_1', np.uint8),
                ('S_Tree_2', np.uint8)
            ]),
            # DGG_Current_Data_Type
            ('DGG_Current_Data', [
                ('DGG_Current_Flags', np.uint8),
                ('Tau_Cur_DQX', np.float32),
                ('HR_Cur_DQX', np.float32),
                ('N_RFI_X', np.uint16),
                ('N_RFI_Y', np.uint16),
                ('RFI_Prob', np.uint8)
            ]),
            ('X_Swath', np.uint16)
            ]


# Read in a SMOS SM data product, and return a numpy structured array.
def read_sm_product(filepath):
    # Open the data file for reading
    with open(filepath) as file:
        # Read first unsigned int32, containing number of datapoints to iterate over
        n_grid_points = np.fromfile(file, dtype=np.uint32, count=1)[0]
        print('Data file contains {} data points'.format(n_grid_points))

        print('Reading file... ', end='')
        data = np.fromfile(file, dtype=datatype, count=n_grid_points)
        print('Done')

    return data


# Take a numpy structured SM format array and extract just the Soil Moisture information
# into a pandas dataframe
def extract_sm(data):
    # Make every nested structured numpy array into a dataframe of its own
    base_frame = pd.DataFrame(data)
    time_frame = pd.DataFrame(data['Mean_Acq_Time'])
    retrieval_frame = pd.DataFrame(data['Retrieval_Results_Data'])

    # Make a dataframe with the columns we care about
    soil_moisture = pd.concat([time_frame['Days'], time_frame['Seconds'],
            time_frame['Microseconds'], base_frame['Grid_Point_ID'],
            base_frame['Latitude'], base_frame['Longitude'],
            retrieval_frame['Soil_Moisture']],
            axis=1)

    # The time fields, and the gridpoint ID combine to make a unique index we can join over
    soil_moisture = soil_moisture.set_index(['Days', 'Seconds', 'Microseconds', 'Grid_Point_ID'])

    return soil_moisture


# Plot any SM values from a dataframe that aren't NaN, with their lat/lon position
def plot_sm(data_frame):
    # Assume a roughly continuous data region for now, just plot all datapoints that aren't -999.

    # Take out -999. float values
    data_frame = data_frame[data_frame['Soil_Moisture'] != -999.0]

    axes = data_frame.plot.scatter('Grid_Point_ID', 'Soil_Moisture')
    plt.show()


# Plot difference between 2 dataframes containing soil moisture
def evaluate_sm_diff(smdf1, smdf2):
    print('Evaluating difference between 2 dataframes...')

    # Exclude NaN records (reported as Soil_Moisture = -999.0)
    frame1 = smdf1[smdf1["Soil_Moisture"] != -999.0]
    frame2 = smdf2[smdf2["Soil_Moisture"] != -999.0]

    # Print record counts
    print('Dataset 1 contains {}/{} valid datarows'.format(len(frame1.index), len(smdf1)))
    print('Dataset 2 contains {}/{} valid datarows'.format(len(frame2.index), len(smdf2)))

    # Get records in common
    common = pd.merge(frame1, frame2, how='inner', on=['Days', 'Seconds', 'Microseconds', 'Grid_Point_ID'])
    common.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude'}, inplace=True)
    common.drop('Latitude_y', axis=1, inplace=True)
    common.drop('Longitude_y', axis=1, inplace=True)
    common['Soil_Moisture_Diff'] = common['Soil_Moisture_y'] - common['Soil_Moisture_x']
    common.reset_index(inplace=True)

    # Outer merge ready for getting new records
    outer = pd.merge(frame1, frame2, how='outer', on=['Days', 'Seconds', 'Microseconds', 'Grid_Point_ID'],
                     indicator=True)
    # Get records in 1 but not 2
    leftonly = outer[outer['_merge'] == 'left_only'].copy()
    leftonly.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude', 'Soil_Moisture_x': 'Soil_Moisture'},
                    inplace=True)
    leftonly.drop('Latitude_y', axis=1, inplace=True)
    leftonly.drop('Longitude_y', axis=1, inplace=True)
    leftonly.drop('Soil_Moisture_y', axis=1, inplace=True)
    leftonly.drop('_merge', axis=1, inplace=True)

    # Get records in 2 but not 1
    rightonly = outer[outer['_merge'] == 'right_only'].copy()
    rightonly.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude', 'Soil_Moisture_y': 'Soil_Moisture'},
                     inplace=True)
    rightonly.drop('Latitude_y', axis=1, inplace=True)
    rightonly.drop('Longitude_y', axis=1, inplace=True)
    rightonly.drop('Soil_Moisture_x', axis=1, inplace=True)
    rightonly.drop('_merge', axis=1, inplace=True)

    print('Dataset analysis:')
    print('{} rows common to both datasets.'.format(len(common.index)))
    print('{} rows in dataset 1 only.'.format(len(leftonly.index)))
    print('{} rows in dataset 2 only.'.format(len(rightonly.index)))

    # Get records in common that are same/diff

    fig1 = plt.figure()
    # Set up plot
    # find a central lon and lat
    center_lon = common['Longitude'].mean()
    centre_lat = common['Latitude'].mean()

    # find a min and max lat and long
    min_lon = common['Longitude'].min() - 4
    max_lon = common['Longitude'].max() + 4

    min_lat = common['Latitude'].min() - 4
    max_lat = common['Latitude'].max() + 4

    # for a full orbit?
    # width=110574 * 90,
    # height=16 * 10**6

    m = Basemap(projection='poly',
                llcrnrlon=min_lon,
                llcrnrlat=min_lat,
                urcrnrlat=max_lat,
                urcrnrlon=max_lon,
                lat_0=centre_lat, lon_0=center_lon,
                resolution='l')

    m.drawcoastlines()
    m.fillcontinents()
    # labels [left, right, top, bottom]
    m.drawparallels(np.arange(-80., 80., 20.), labels=[True, False, False, False], fontsize=8)
    m.drawmeridians(np.arange(-180, 180, 60.), labels=[False, False, False, True], fontsize=8)
    m.drawmapboundary()

    m.scatter(common['Longitude'].values,
              common['Latitude'].values,
              latlon=True,
              c=common['Soil_Moisture_Diff'],
              s=5,
              zorder=10)

    # add colorbar
    m.colorbar()

    plt.title("Difference in SM")

    fig2, ax2 = plt.subplots(1)
    # plot each difference against the index
    common.plot(x='Grid_Point_ID', y='Soil_Moisture_Diff', ax=ax2, legend=False, rot=90,
                fontsize=8, clip_on=False, style='o')
    ax2.set_ylabel('Soil Moisture Diff')
    ax2.axhline(y=0, linestyle=':', linewidth='0.5', color='k')
    fig2.tight_layout()

    # plot only the ones with a non-zero difference?
    fig3, ax3 = plt.subplots(1)
    common[common['Soil_Moisture_Diff'] != 0].plot(x='Grid_Point_ID', y='Soil_Moisture_Diff', ax=ax3, legend=False,
                                                   rot=90, fontsize=8, clip_on=False, style='o')
    ax3.axhline(y=0, linestyle=':', linewidth='0.5', color='k')
    ax3.set_ylabel('Soil Moisture Diff')
    fig3.tight_layout()

    plt.show()


if __name__ == '__main__':

    # TODO: Reorganise the argparse stuff?
    parser = argparse.ArgumentParser(description='Read L2SM Processor UDP files')
    parser.add_argument('--plot-diff', '-d', nargs=2,
                        help='Evaluate and plot the difference between two UDP files.')

    args = parser.parse_args()

    if args.plot_diff:
        # Requested to plot the difference between two UDP files
        file1 = os.path.abspath(args.plot_diff[0])
        file2 = os.path.abspath(args.plot_diff[1])

        print('UDP file 1: {}'.format(file1))
        fail = False
        if not os.path.isfile(file1):
            print('ERROR: UDP file not found.')
            fail = True
        print('UDP file 2: {}'.format(file2))
        if not os.path.isfile(file2):
            print('ERROR: UDP file not found.')
            fail = True
        if fail:
            sys.exit(1)

        dataframe1 = extract_sm(read_sm_product(file1))
        dataframe2 = extract_sm(read_sm_product(file2))
        evaluate_sm_diff(dataframe1, dataframe2)
    else:
        # For now this is the only possible command
        print('No arguments given.')
        print('Try -h for help')

