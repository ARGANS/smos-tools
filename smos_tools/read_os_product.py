#!/usr/bin/env python

# This file was moved and maintained in smos-tools git repo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import logging
import argparse
import os
import sys

#=================================================
# UDP data types
dtype = [('Grid_Point_ID', np.uint32),
         ('Latitude', np.float32),
         ('Longitude', np.float32),
         ('Equiv_ftprt_diam', np.float32),
         ('Mean_acq_time', np.float32),
         ('SSS1', np.float32),
         ('Sigma_SSS1', np.float32),
         ('SSS2', np.float32),
         ('Sigma_SSS2', np.float32),
         ('SSS3', np.float32),
         ('Sigma_SSS3', np.float32),
         ('A_card', np.float32),
         ('Sigma_A_card', np.float32),
         ('WS', np.float32),
         ('SST', np.float32),
         ('Tb_42_5H', np.float32),
         ('Sigma_Tb_42_5H', np.float32),
         ('Tb_42_5V', np.float32),
         ('Sigma_Tb_42_5V', np.float32),
         ('Tb_42_5X', np.float32),
         ('Sigma_Tb_42_5X', np.float32),
         ('Tb_42_5Y', np.float32),
         ('Sigma_Tb_42_5Y', np.float32),
         ('Control_Flags_1', np.uint32),
         ('Control_Flags_2', np.uint32),
         ('Control_Flags_3', np.uint32),
         ('Control_Flags_4', np.uint32),
         ('Dg_chi2_1', np.uint16),
         ('Dg_chi2_2', np.uint16),
         ('Dg_chi2_3', np.uint16),
         ('Dg_chi2_Acard', np.uint16),
         ('Dg_chi2_P_1', np.uint16),
         ('Dg_chi2_P_2', np.uint16), 
         ('Dg_chi2_P_3', np.uint16),
         ('Dg_chi2_P_Acard', np.uint16),
         ('Dg_quality_SSS1', np.uint16),
         ('Dg_quality_SSS2', np.uint16),
         ('Dg_quality_SSS3', np.uint16),
         ('Dg_quality_Acard', np.uint16),
         ('Dg_num_iter_1', np.uint8),
         ('Dg_num_iter_2', np.uint8),
         ('Dg_num_iter_3', np.uint8),
         ('Dg_num_iter_4', np.uint8),
         ('Dg_num_meas_l1c', np.uint16),
         ('Dg_num_meas_valid', np.uint16),
         ('Dg_border_fov', np.uint16),
         ('Dg_af_fov', np.uint16),
         ('Dg_sun_tails', np.uint16),
         ('Dg_sun_glint_area', np.uint16),
         ('Dg_sun_glint_fov', np.uint16),
         ('Dg_sun_fov', np.uint16),
         ('Dg_sun_glint_L2', np.uint16),
         ('Dg_Suspect_ice', np.uint16),
         ('Dg_Galactic_Noise_Error', np.uint16),
         ('Dg_sky', np.uint16),
         ('Dg_moon_glint', np.uint16),
         ('Dg_RFI_L1', np.uint16),
         ('Dg_RFI_X', np.uint16),
         ('Dg_RFI_Y', np.uint16),
         ('Dg_RFI_probability', np.uint16),
         ('X_swath', np.float32),
         ('Science_Flags_1', np.uint32),
         ('Science_Flags_2', np.uint32),
         ('Science_Flags_3', np.uint32),
         ('Science_Flags_4', np.uint32)
        ] 



#=================================================
def read_os_udp(filename):
    """
    Read the Ocean Salinity User Data Product file.
    
    :param filename: path to .DBL file
    :return: numpy structured array
    """
    try:
        file = open(filename, 'rb')
    except IOError:
        logging.error('file {} does not exist'.format(filename))
    
    logging.info('Reading file...')
    # Read first unsigned int32, containing number of grid points to iterate over
    n_grid_points = np.fromfile(file, dtype=np.uint32, count=1)[0]
    data = np.fromfile(file, dtype=np.dtype(dtype), count=n_grid_points)
    file.close()
    logging.info('Done.')
    
    return data


def extract_fields(data):
    """
    Converts the udp structured array into a pandas dataframe.

    :param data: numpy structured array (record array).  
    :return: pandas dataframe (one column per field name). 
    """

    dataframe = pd.DataFrame(data)
    dataframe = dataframe.replace(-999, np.NaN)

    return dataframe
    
    
def extract_field(data, fieldname):
    """
    Converts the structured array into a pandas small dataframe.

    :param data: numpy structured array (record array). 
    :param fieldname: string (a field name from variable dtypes). 
    :return: pandas dataframe (columns are Mean_acq_time, Latitude, Longitude and fieldname) 
    Mean_acq_time is expressed in UTC decimal days (MJD2000 reference).
    """

    if not(fieldname in data.dtype.fields):
        logging.error('Argument {} not valid. Select a field name from variable dtypes.'.format(fieldname)) 

    time_frame = pd.DataFrame(data['Mean_acq_time'], columns=['Mean_acq_time'])
    lat_frame = pd.DataFrame(data['Latitude'], columns=['Latitude'])
    lon_frame = pd.DataFrame(data['Longitude'], columns=['Longitude'])
    field_frame = pd.DataFrame(data[fieldname], columns=[fieldname])

    dataframe = pd.concat([time_frame, lat_frame, lon_frame, field_frame], axis=1)
    dataframe = dataframe.replace(-999, np.NaN)
    
    return dataframe    


### Plot an OS orbit from a pandas dataframe
def plot_os_orbit(df, fieldname='SSS1', mode='default'):
    """
    Plot the difference between two dataframes. Gives map plots and scatter.
    
    :param df: pandas dataframe containing Soil Moisture with index Days, Seconds, Microseconds, Grid_Point_ID
    :param fieldname: string fieldname of the data field to compare
    :param mode: string 'default' or 'diff' (for plotting differences)
    :return:
    """
 
    # TODO check if fieldname is correct
    if not(fieldname in df.columns):
        logging.error('Incorrect field name.')
        #sys.exit(1)
    
    print('Plotting {} for the full orbit...'.format(fieldname))
    
    # Exclude NaN records 
    df = df[df[fieldname] != np.NaN]
    
    fig1 = plt.figure()
    centre_lon = df['Longitude'].mean()
    centre_lat = df['Latitude'].mean()
    # find a min and max lat and long 
    # +-4 took from soil moisture plotting funct
    min_lon = max(df['Longitude'].min() - 4, -180.)
    max_lon = min(df['Longitude'].max() + 4, +180.)
    min_lat = max(df['Latitude'].min() - 4, -90.)
    max_lat = min(df['Latitude'].max() + 4, +90.)
    delta_lon = np.abs(max_lon - min_lon)
    delta_lat = np.abs(max_lat - min_lat)
    
            
    if delta_lat > 45: # for  full orbit
        #lat_0 = 10. for soil moisture is 10
        lat_0 = 5.
        lon_0 = centre_lon
        width = 110574 * 70 # ~100km * 70 deg
        #height = 140 * 10**5 # 100km * 140 deg        
        height = 10**5 * 170 # 100km * 140 deg
        dot_size = 1
    else:
        lat_0=centre_lat
        lon_0=centre_lon
        width=delta_lon * 110574
        height=delta_lat * 10**5        
        dot_size = 5
    
    m = Basemap(
            #projection='cyl',
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
    m.drawmeridians(np.arange(-180, 180, 20.), labels=[False, False, False, True], fontsize=8)
    m.drawmapboundary()
    
    if (mode == 'default') & (fieldname in ['SSS1', 'SSS2']):
        plt.title(fieldname)
        cmap = 'viridis'
        c = df[fieldname] # geophysical variable to plot 
        vmin = 32.
        vmax = 38.
        m.scatter(df['Longitude'].values,
          df['Latitude'].values,
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
        
    elif (mode == 'default') & (fieldname == 'SSS3'): # SSS anomaly
        plt.title('SSS anomaly')
        cmap = 'bwr'
        c = df[fieldname] # geophysical variable to plot 
        vmin = -0.5
        vmax = +0.5
        m.scatter(df['Longitude'].values,
          df['Latitude'].values,
          latlon=True,
          c=c,
          s=dot_size,
          zorder=10,
          cmap=cmap,
          )    
        cbar = m.colorbar()
    
    elif (mode == 'default') & (fieldname not in ['SSS1', 'SSS2', 'SSS3']):
        plt.title(fieldname)
        cmap = 'viridis'
        c = df[fieldname] # geophysical variable to plot 
        m.scatter(df['Longitude'].values,
          df['Latitude'].values,
          latlon=True,
          c=c,
          s=dot_size,
          zorder=10,
          cmap=cmap,
          )    
        cbar = m.colorbar()

    
    elif mode == 'diff':
        plt.title(fieldname)
        cmap = 'bwr'        
        c = df[fieldname] # geophysical variable to plot
        vmin = -1.
        vmax = +1.
        m.scatter(df['Longitude'].values,
          df['Latitude'].values,
          latlon=True,
          c=c,
          s=dot_size,
          zorder=10,
          cmap=cmap,
          vmin=vmin,
          vmax=vmax)
        cbar = m.colorbar()
    
    else:
        logging.error('Incorrect mode argument in function plot_os_orbit()')
        
    plt.show()   
    
    
def set_logger(log_level=logging.INFO):
    """ Set logging"""

    #log_level = logging.DEBUG
    logger = logging.getLogger()
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    formatter = logging.Formatter(
                            fmt='%(asctime)s [%(filename)s - %(levelname)s]: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    return
    
#=========================================================
if __name__ == '__main__':
    
    dir_udp = '/home/smos/builds/v6.71/Outputs_ref/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_671_001_0'
    file_udp = 'SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_671_001_0.DBL'
    filename = os.path.join(dir_udp, file_udp)
    
    data = read_os_udp(filename)
    df = extract_fields(data)
    df1 = extract_field(data, 'SSS1')
    plot_os_orbit(df, 'SSS1')
