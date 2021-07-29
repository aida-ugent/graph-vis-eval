from setuptools import setup

setup(name='arope',
      version='1.0',
      install_requires=['numpy',
                        'scipy',
                        'networkx',
                        'pandas',
                        ],
      py_modules = ['utils'],
      package_dir = {'': 'arope/python'})