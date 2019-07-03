# SMOS Tools

Collection of helpful tools relating to SMOS activities (Soil Moisture and Ocean Salinity).

## Requirements

This is a python 3 library.

Required packages:

- Basemap
- numpy
- matplotlib
- pandas
- setuptools

## Installing

Uninstall any previous version of SMOS Tools you may have installed with

`pip uninstall smos_tools`

To build, `cd` to the root directory containing the `setup.py` and then run

`python setup.py bdist_wheel`

Then (preferably not in the conda environment you are using to develop this)

`cd dist && pip install smos_tools-1.0.1-py3-none-any.whl`

## Usage

The following command line utilities are provided in `bin/`, and should be
automatically added to your path during install.

### read_os_product

Utility to read and plot L2OS `UDP` files.

The options are:

- `--plot-diff FILE FILE`, `-d FILE FILE` : Evaluate the and plot the difference between two `UDP` `DBL` files.
- `--field-name NAME`, `-f NAME` : Field name to plot. Default is `SSS1`.
- `--plot-orbit FILE`, `-o FILE` : Plot the Ocean Salinity orbit from `UDP` file.

Example usage:

`$ read_os_product -o /data/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_670_001_0/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_670_001_0.DBL`

### read_sm_product

Utility to read and plot L2SM `UDP` files.

The options are:

- `--plot-diff FILE FILE`, `-d FILE FILE` : Evaluate the and plot the difference between two `UDP` `DBL` files.
- `--field-name NAME`, `-f NAME` : Field name to plot. Default is `Soil_Moisture`.
- `--plot-orbit FILE`, `-o FILE` : Plot the Soil Moisture orbit from `UDP` file.

Example usage:

`$ read_sm_product -d SM_TEST_MIR_SMUDP2_20180530T044823_20180530T054143_650_001_1/SM_TEST_MIR_SMUDP2_20180530T044823_20180530T054143_650_001_1.DBL SM_TEST_MIR_SMUDP2_20180530T044823_20180530T054143_650_001_1/SM_TEST_MIR_SMUDP2_20180530T044823_20180530T054143_650_001_1.DBL`

## Contents

| Feature | Description |
|---------|-------------|
| `read_sm_product.py` | Contains functions for reading, formatting and plotting and evaluating L2SM `UDP` files |
| `read_os_product.py` | Contains functions for reading, formatting and plotting and evaluating L2OS `UDP` files |
| `read_os_dtbxy.py` | Contains the function to read an L2OS `DTBXY` file |
