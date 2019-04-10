#!/usr/bin/env python
import read_sm_product
import numpy as np
import pandas as pd
import argparse

if __name__ == '__main__':
    # Included test data
    schema_path = 'schema/DBL_SM_XXXX_MIR_SMUDP2_0400.binXschema.xml'
    data_path = 'data/SM_TEST_MIR_SMUDP2_20150721T102717_20150721T112036_650_001_9.DBL'

    numpy_data = read_sm_product.read_sm_product(data_path)

    sm_df = read_sm_product.extract_sm(numpy_data)
    # plot_sm(sm_df)

    # Artificially create a second dataframe for testing, and change a couple of rows
    sm_df_mod = sm_df.copy(deep=True)
    sm_df_mod.at[(5680, 39052, 366745, 1205371), 'Soil_Moisture'] = \
        sm_df_mod.at[(5680, 39052, 366745, 1205371), 'Soil_Moisture'] + 0.2  # 0.103672
    sm_df_mod.at[(5680, 39059, 827423, 1202803), 'Soil_Moisture'] = -999.0  # 0.068059
    sm_df_mod.at[(5680, 39055, 725343, 1203317), 'Soil_Moisture'] = -999.0  # 0.112978
    sm_df_mod.at[(5680, 39051, 937985, 1203830), 'Soil_Moisture'] = -999.0  # 0.116852
    sm_df_mod.at[(5680, 39050, 378980, 1204344), 'Soil_Moisture'] = -999.0  # 0.086155

    # Call function to evaluate the difference between the two
    read_sm_product.evaluate_sm_diff(sm_df, sm_df_mod)

