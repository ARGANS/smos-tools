import logging
import os
from smos_tools.logger.logging_config import logging_config
from smos_tools import read_os_product
logging.config.dictConfig(logging_config)

logging.getLogger(__name__)

if __name__ == '__main__':

    dir_udp = '/home/smos/builds/v6.71/Outputs_ref/SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_671_001_0'
    file_udp = 'SM_TEST_MIR_OSUDP2_20110501T141050_20110501T150408_671_001_0.DBL'
    filename = os.path.join(dir_udp, file_udp)

    data = read_os_product.read_os_udp(filename)

    df1 = read_os_product.extract_field(data, 'SSS1')

    # print(df1)
    read_os_product.plot_os_orbit(df1, 'SSS1')
