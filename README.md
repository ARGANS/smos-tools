# SMOS Tools

Collection of helpful tools relating to SMOS activities (Soil Moisture and Ocean Salinity).

## Version History

| Version | Notes |
| ------- | ----- |
| v1.1.0  | Update logging levels and exception raising, for inclusion in Validation Protocol |

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

`pip uninstall smos-tools`

To build, `cd` to the root directory containing the `setup.py` and then run

`python setup.py bdist_wheel`

Then (preferably not in the conda environment you are using to develop this)

`cd dist && pip install smos_tools-1.1.0-py3-none-any.whl`

One-liner:

`pip uninstall smos-tools && python setup.py bdist_wheel && pip install dist/smos_tools-1.1.0-py3-none-any.whl`

## Usage

The following command line utilities are provided in `bin/`, and should be
automatically added to your path during install.

### read_os_product

Utility to read and plot L2OS `UDP` files.

The options are:

- `--plot-diff FILE FILE`, `-d FILE FILE` : Evaluate the and plot the difference between two `UDP` `DBL` files.
- `--field-name NAME`, `-f NAME` : Field name to plot. Default is `SSS1`.
- `--plot-orbit FILE`, `-o FILE` : Plot the Ocean Salinity orbit from `UDP` file.
- `--vmin VALUE`, `-m VALUE` : Plots all values greater than VALUE. If smaller, it saturates.
- `--vmax VALUE`, `-M VALUE` : Plots all values smaller than VALUE. If higher, it saturates.
- `--x-axis NAME`, `-x NAME` : Varible againt which the fieldname is plotted. NAME is of: {'Latitude', 'Grid_Point_ID'}.

Example usage:

`$ read_os_product -o /data/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_670_001_0/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_670_001_0.DBL`

`$read_os_product --plot-orbit /home/famico/repos/SMOS-L2OS-Processor/Outputs_v673/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_673_001_0/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_673_001_0.DBL --vmin 20 --vmax 50`

`$read_os_product --plot-diff /home/famico/repos/SMOS-L2OS-Processor/Outputs_671/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_671_001_0/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_671_001_0.DBL
/home/famico/repos/SMOS-L2OS-Processor/Outputs_v673/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_673_001_0/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_673_001_0.DBL --vmin 20 --vmax 50 xaxis='Latitude'`



### read_sm_product

Utility to read and plot L2SM `UDP` files.

The commands are:

- `plot-diff`, `diff` : Evaluate and plot the difference between two `UDP` `.DBL` files.
- `plot-orbit`, `plot` : Plot the orbit from a Soil Moisture `UDP` `.DBL` file.

The options are:

- `--orbit-file FILE1 FILE2`, `-o FILE1 FILE2` : Direct path to one or more SM `UDP` `.DBL` files to evaluate
- `--orbit-name NAME1 NAME2`, `-n NAME1 NAME2` : Name to associate to each orbit, for plot output etc.
- `--field-name NAME`, `-f NAME` : Field name to plot. Default is `Soil_Moisture`.
- `--plot-orbit FILE`, `-o FILE` : Plot the Soil Moisture orbit from `UDP` file.
- `--vmin VALUE`, `-m VALUE` : Plots all values greater than VALUE. If smaller, saturate.
- `--vmax VALUE`, `-M VALUE` : Plots all values smaller than VALUE. If higher, saturate.
- `--x-axis NAME`, `-x NAME` : Varible againt which the fieldname is plotted. NAME is of: {'Latitude', 'Grid_Point_ID'}.

Example usage:

`$ read_sm_product diff -o SM_TEST_MIR_SMUDP2_20180530T044823_20180530T054143_650_001_1/SM_TEST_MIR_SMUDP2_20180530T044823_20180530T054143_650_001_1.DBL SM_TEST_MIR_SMUDP2_20180530T044823_20180530T054143_650_001_1/SM_TEST_MIR_SMUDP2_20180530T044823_20180530T054143_650_001_1.DBL -n v671 v680`

## Contents

| Feature | Description |
|---------|-------------|
| `read_sm_product.py` | Contains functions for reading, formatting and plotting and evaluating L2SM `UDP` files |
| `read_os_product.py` | Contains functions for reading, formatting and plotting and evaluating L2OS `UDP` files |
| `read_os_dtbxy.py` | Contains the function to read an L2OS `DTBXY` file |
| `read_aux_dgg_product.py` | Function to read an `AUX_DGG___` file into a numpy structured array |
| `read_aux_distan_product.py` | Functions to read an `AUX_DISTAN` file into a numpy structured array, and parse the flags -> create a pandas dataframe with an `Is_Sea` boolean from flags |
| `generate_vp_ISEA4H9.py` | Function to read `DGG` and `DISTAN` files, combining them into a single CSV in the format used by the Validation protocol (for definition of `ISEA4H9` grid) |
