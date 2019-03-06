#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hardcode paths
schema_path = 'schema/DBL_SM_XXXX_MIR_SMUDP2_0400.binXschema.xml'
data_path = 'data/SM_TEST_MIR_SMUDP2_20150721T102717_20150721T112036_650_001_9.DBL'

# Primary array datatype, repeated n_grid_points times, Grid_Point_Data_Type
datatype = [('Grid_Point_ID',np.uint32),
            ('Latitude',np.float32),
            ('Longitude',np.float32),
            ('Altitude',np.float32),
            # UTC_Type
            ('Mean_Acq_Time', [
                ('Days',np.int32),
                ('Seconds',np.uint32),
                ('Microseconds',np.uint32)
            ]),
            # Retrieval_Results_Data_Type
            ('Retrieval_Results_Data', [
                ('Soil_Moisture',np.float32),
                ('Soil_Moisture_DQX',np.float32),
                ('Optical_Thickness_Nad',np.float32),
                ('Optical_Thickness_Nad_DQX',np.float32),
                ('Surface_Temperature',np.float32),
                ('Surface_Temperature_DQX',np.float32),
                ('TTH',np.float32),
                ('TTH_DQX',np.float32),
                ('RTT',np.float32),
                ('RTT_DQX',np.float32),
                ('Scattering_Albedo_H',np.float32),
                ('Scattering_Albedo_H_DQX',np.float32),
                ('DIFF_Albedos',np.float32),
                ('DIFF_Albedos_DQX',np.float32),
                ('Roughness_Param',np.float32),
                ('Roughness_Param_DQX',np.float32),
                ('Dielect_Const_MD_RE',np.float32),
                ('Dielect_Const_MD_RE_DQX',np.float32),
                ('Dielect_Const_MD_IM',np.float32),
                ('Dielect_Const_MD_IM_DQX',np.float32),
                ('Dielect_Const_Non_MD_RE',np.float32),
                ('Dielect_Const_Non_MD_RE_DQX',np.float32),
                ('Dielect_Const_Non_MD_IM',np.float32),
                ('Dielect_Const_Non_MD_IM_DQX',np.float32),
                ('TB_ASL_Theta_B_H',np.float32),
                ('TB_ASL_Theta_B_H_DQX',np.float32),
                ('TB_ASL_Theta_B_V',np.float32),
                ('TB_ASL_Theta_B_V_DQX',np.float32),
                ('TB_TOA_Theta_B_H',np.float32),
                ('TB_TOA_Theta_B_H_DQX',np.float32),
                ('TB_TOA_Theta_B_V',np.float32),
                ('TB_TOA_Theta_B_V_DQX',np.float32)
            ]),
            # Confidence_Descriptors_Data_Type
            ('Confidence_Descriptors_Data', [
                ('Confidence_Flags',np.uint16),
                ('GQX',np.uint8),
                ('Chi_2',np.uint8),
                ('Chi_2_P',np.uint8),
                ('N_Wild',np.uint16),
                ('M_AVA0',np.uint16),
                ('M_AVA',np.uint16),
                ('AFP',np.float32),
                ('N_AF_FOV',np.uint16),
                ('N_Sun_Tails',np.uint16),
                ('N_Sun_Glint_Area',np.uint16),
                ('N_Sun_FOV',np.uint16),
                ('N_RFI_Mitigations',np.uint16),
                ('N_Strong_RFI',np.uint16),
                ('N_Point_Source_RFI',np.uint16),
                ('N_Tails_Point_Source_RFI',np.uint16),
                ('N_Software_Error',np.uint16),
                ('N_Instrument_Error',np.uint16),
                ('N_ADF_Error',np.uint16),
                ('N_Calibration_Error',np.uint16),
                ('N_X_Band',np.uint16)
            ]),
            # Science_Descriptors_Data_Type
            ('Science_Descriptors_Data', [
                ('Science_Flags',np.uint32),
                ('N_Sky',np.uint16)
            ]),
            # Processing_Descriptors_Data_Type
            ('Processing_Descriptors_Data', [
                ('Processing_Flags',np.uint16),
                ('S_Tree_1',np.uint8),
                ('S_Tree_2',np.uint8)
            ]),
            # DGG_Current_Data_Type
            ('DGG_Current_Data', [
                ('DGG_Current_Flags',np.uint8),
                ('Tau_Cur_DQX',np.float32),
                ('HR_Cur_DQX',np.float32),
                ('N_RFI_X',np.uint16),
                ('N_RFI_Y',np.uint16),
                ('RFI_Prob',np.uint8)
            ]),
            ('X_Swath',np.uint16)
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
    retrieval_frame = pd.DataFrame(data['Retrieval_Results_Data'])

    # Make a dataframe with the columns we care about
    soil_moisture = pd.concat([base_frame['Grid_Point_ID'], base_frame['Latitude'],
            base_frame['Longitude'], retrieval_frame['Soil_Moisture']],
            axis=1)
    #soil_moisture = soil_moisture.set_index('Grid_Point_ID')

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
    print('Dataset 1 contains {} valid datarows from {}'.format(len(frame1.index), len(smdf1)))
    print('Dataset 2 contains {} valid datarows from {}'.format(len(frame2.index), len(smdf2)))

    # Get records in 1 but not 2
    #extra1 = 
    # TODO: Need a unique ID for joining, Gridpoint ID not unique

    # Get records in 2 but not 1

    # Get records in common
    # Get records in common that are same/diff
    # Plot them

numpy_data = read_sm_product(data_path)

sm_df = extract_sm(numpy_data)

#plot_sm(sm_df)

# Artificially create a second dataframe for testing, and change a couple of rows
sm_df_mod = sm_df.copy(deep=True)
sm_df_mod.at[23902, 'Soil_Moisture'] = -999.0
sm_df_mod.at[23905, 'Soil_Moisture'] = -999.0
sm_df_mod.at[23907, 'Soil_Moisture'] = -999.0
sm_df_mod.at[23946, 'Soil_Moisture'] = sm_df_mod.at[23946, 'Soil_Moisture'] + 0.2

# Call function to evaluate the difference between the two
evaluate_sm_diff(sm_df, sm_df_mod)

