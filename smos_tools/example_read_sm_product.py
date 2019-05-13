#!/usr/bin/env python
import logging
import logging.config
import read_sm_product

from smos_tools.logger.logging_config import logging_config

if __name__ == '__main__':

    logging.config.dictConfig(logging_config)

    logging.getLogger(__name__)

    # Included test data
    schema_path = 'schema/DBL_SM_XXXX_MIR_SMUDP2_0400.binXschema.xml'
    data_path = 'data/SM_TEST_MIR_SMUDP2_20150721T102717_20150721T112036_650_001_9.DBL'

    field = 'Soil_Moisture'

    numpy_data = read_sm_product.read_sm_product(data_path)

    sm_df = read_sm_product.extract_field(numpy_data, field)
    # plot_field(sm_df, field)

    # Artificially create a second dataframe for testing, and change a couple of rows
    sm_df_mod = sm_df.copy(deep=True)
    sm_df_mod.at[(5680, 39052, 366745, 1205371), 'Soil_Moisture'] = \
        sm_df_mod.at[(5680, 39052, 366745, 1205371), 'Soil_Moisture'] + 0.2  # 0.103672
    sm_df_mod.at[(5680, 39059, 827423, 1202803), 'Soil_Moisture'] = -999.0  # 0.068059
    sm_df_mod.at[(5680, 39055, 725343, 1203317), 'Soil_Moisture'] = -999.0  # 0.112978
    sm_df_mod.at[(5680, 39051, 937985, 1203830), 'Soil_Moisture'] = -999.0  # 0.116852
    sm_df_mod.at[(5680, 39050, 378980, 1204344), 'Soil_Moisture'] = -999.0  # 0.086155

    # Call function to evaluate the difference between the two
    read_sm_product.evaluate_field_diff(sm_df, sm_df_mod, field)

