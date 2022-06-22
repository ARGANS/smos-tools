#!/usr/bin/env python3
"""
The data types for a Soil Moisture UDP
"""

import numpy as np

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


confidence_flags = [
                    "FL_VIEWS_T",
                    "FL_RFI_PRONE_H",
                    "FL_RFI_PRONE_V",
                    "FL_RETR_T",
                    "FL_NO_PROD",
                    "FL_RANGE",
                    "FL_DQX",
                    "FL_CHI2_P",
                    "FL_FARADAY_ROTATION_ANGLE",
                    "FL_TAU_BIOMASS",
                    ]
science_flags = [
                "FL_NON_NOM",
                "FL_SCENE_T",
                "FL_BARREN",
                "FL_TOPO_S",
                "FL_TOPO_M",
                "FL_OW",
                "FL_SNOW_MIX",
                "FL_SNOW_WET",
                "FL_SNOW_DRY",
                "FL_FOREST",
                "FL_NOMINAL",
                "FL_FROST",
                "FL_ICE",
                "FL_WETLANDS",
                "FL_FLOOD_PROB",
                "FL_URBAN_LOW",
                "FL_URBAN_HIGH",
                "FL_SAND",
                "FL_SEA_ICE",
                "FL_COAST",
                "FL_OCCUR_T",
                "FL_LITTER",
                "FL_PR",
                "FL_INTERCEP",
                "FL_EXTERNAL",
                "FL_RAIN",
                "FL_TEC",
                "FL_TAU_FO",
                "FL_WINTER_FOREST",
                "FL_DUAL_RETR_FNO_FFO",
                "FL_SFC_LOW_TEMP",
                "FL_VEG_LOW_TEMP",
        ]

