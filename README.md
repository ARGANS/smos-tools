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
To build, `cd` to the same directory as the `setup.py` and then run

`python setup.py bdist_wheel`

Then (preferably not in the conda environment you are using to develop this)

`cd dist`

`pip install smos_tools-0.1-py3-none-any.whl` 


## Command line utilities

| Name | Description |
|------|-------------|
|`read_sm_product`| Handles reading and plotting L2SM `UDP` data files and the difference between two|

## Contents

| Feature | Description |
|---------|-------------|
| `read_sm_product` | A reader for the `UDP` data product from the SMOS L2SM Processor. Can get the difference between two similar UDP files and plot them |
| `read_os_product.py` | A reader for the `UDP` data product from the SMOS L2OS Processor.Can read the UDP, extract a pandas dataframe, plot one orbit.|
|`read_os_dtbxy.py`| A reader for the `DTBXY` data product from SMOS L2OS Procesor