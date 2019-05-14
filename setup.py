from setuptools import setup

setup(name='smos_tools',
      version='1.0.0',
      description='Some open source tools for SMOS',
      url='https://www.argans.co.uk',
      author='ARGANS',
      author_email='smos@argans.co.uk',
      license='MIT',
      packages=['smos_tools', 'smos_tools.data', 'smos_tools.data_types',
                'smos_tools.logger', 'smos_tools.schema'],
      package_dir={'smos_tools.data': 'smos_tools/data'},
      package_data={'smos_tools.data':
                    ['smos_tools/data/SM_TEST_MIR_SMUDP2_20150721T102717_20150721T112036_650_001_9.DBL',
                     'smos_tools/data/SM_TEST_MIR_SMUDP2_20150721T102717_20150721T112036_650_001_9.HDR']
                    },
      include_package_data=True,
      scripts=['bin/read_sm_product', 'bin/read_os_product'],
      zip_safe=False,
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',

      ]
      )
