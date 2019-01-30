#!/usr/bin/env python
import numpy as np

# Hardcode paths
schema_path = 'schema/DBL_SM_XXXX_MIR_SMUDP2_0400.binXschema.xml'
data_path = 'data/SM_TEST_MIR_SMUDP2_20150721T102717_20150721T112036_650_001_9.DBL'

# Open the data file for reading
# TODO: How to read binary data of differing sizes in same file? Need some
# kind of file pointer that knows how far the file we've gone?
# np.fromfile()
# Could use raw python file read to read the right number of bytes each time,
# then pass to numpy


