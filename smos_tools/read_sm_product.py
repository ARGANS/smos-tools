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


def read_sm_product(filepath):
    """
    Read the Soil Moisture UDP file
    :param filepath: path to .DBL file
    :return: numpy structured array
    """
    # Open the data file for reading
    with open(filepath) as file:
        # Read first unsigned int32, containing number of datapoints to iterate over
        n_grid_points = np.fromfile(file, dtype=np.uint32, count=1)[0]
        print('Data file contains {} data points'.format(n_grid_points))
        print('Reading file... ', end='')
        data = np.fromfile(file, dtype=datatype, count=n_grid_points)
        print('Done')

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
    if not fieldname in retrieval_frame.columns.values:
        print("ERROR: Couldn't find fieldname '{}' in 'Retrieval_Results_Data'".format(fieldname))
        sys.exit(1)

    # Make a dataframe with the columns we care about
    extracted_data = pd.concat([time_frame['Days'], time_frame['Seconds'],
            time_frame['Microseconds'], base_frame['Grid_Point_ID'],
            base_frame['Latitude'], base_frame['Longitude'],
            retrieval_frame[fieldname]],
            axis=1)

    # The time fields, and the gridpoint ID combine to make a unique index we can join over
    extracted_data = extracted_data.set_index(['Days', 'Seconds', 'Microseconds', 'Grid_Point_ID'])

    return extracted_data


def plot_field(data_frame, fieldname):
    """
    Plot data on a scatter plot

    Plots only values which are not NaN (-999.0), against their gridpoint ID.
    :param data_frame: pandas dataframe containing a field value and index Grid_Point_ID
    :param fieldname: string of fieldname to plot
    :return:
    """
    # Assume a roughly continuous data region for now, just plot all datapoints that aren't -999.

    # Take out -999. float values
    data_frame = data_frame[data_frame[fieldname] != -999.0]

    axes = data_frame.plot.scatter('Grid_Point_ID', fieldname)
    plt.show()


def evaluate_field_diff(smdf1, smdf2, fieldname):
    """
    Plot the difference between two dataframes for a given field. Gives map plots and scatter.
    :param smdf1: pandas dataframe containing the requested data field and index (Days, Seconds, Microseconds, Grid_Point_ID)
    :param smdf2: pandas dataframe containing the requested data field and index (Days, Seconds, Microseconds, Grid_Point_ID)
    :param fieldname: String fieldname of the data field to compare
    :return:
    """
    print('Evaluating difference between 2 dataframes for field {}...'.format(fieldname))

    # Exclude NaN records (reported as fieldname = -999.0)
    frame1 = smdf1[smdf1[fieldname] != -999.0]
    frame2 = smdf2[smdf2[fieldname] != -999.0]

    # Print record counts
    print('Dataset 1 contains {}/{} valid datarows'.format(len(frame1.index), len(smdf1)))
    print('Dataset 2 contains {}/{} valid datarows'.format(len(frame2.index), len(smdf2)))

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
              c=common[fieldname+'_Diff'],
              s=5,
              zorder=10)

    # add colorbar
    m.colorbar()

    plt.title("Difference in " + fieldname)

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
        print('No differences to plot')
    else:
        fig3, ax3 = plt.subplots(1)
        non_zero_diff.plot(x='Grid_Point_ID', y=fieldname+'_Diff', ax=ax3, legend=False,
                           rot=90, fontsize=8, clip_on=False, style='o')
        ax3.axhline(y=0, linestyle=':', linewidth='0.5', color='k')
        ax3.set_ylabel(fieldname + ' Diff')
        fig3.tight_layout()

    plt.show()


if __name__ == '__main__':

    # TODO: Reorganise the argparse stuff?
    parser = argparse.ArgumentParser(description='Read L2SM Processor UDP files')
    parser.add_argument('--plot-diff', '-d', nargs=2, metavar='FILE',
                        help='Evaluate and plot the difference between two UDP files.')
    parser.add_argument('--field-name', '-f', default='Soil_Moisture',
                        help="Field name to extract and diff. Default 'Soil_Moisture'.")

    args = parser.parse_args()

    if args.plot_diff:
        # Requested to plot the difference between two UDP files
        file1 = os.path.abspath(args.plot_diff[0])
        file2 = os.path.abspath(args.plot_diff[1])
        field = args.field_name

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

        print('Extracting field: {}.'.format(field))

        dataframe1 = extract_field(read_sm_product(file1), field)
        dataframe2 = extract_field(read_sm_product(file2), field)
        evaluate_field_diff(dataframe1, dataframe2, field)
    else:
        # For now this is the only possible command
        print('ERROR: Invalid or no flags given.')
        print('       Try -h for help.')

