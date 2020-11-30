from setuptools import setup, find_packages

import os.path
# Change to the directory of this file before the build
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Build setup
setup(
    name='iccv19_attribute',
    version='1.0.0',
    license='',
    author='ang, Chufeng and Sheng, Lu and Zhang, Zhaoxiang and Hu, Xiaolin',
    url='https://github.com/chufengt/iccv19_attribute',
    packages=find_packages(),
    description='Improving Pedestrian Attribute Recognition With Weakly-Supervised Multi-Scale '
                'Attribute-Specific Localization',
    install_requires=['torch==1.3.1', 'numpy==1.19.4'],
    include_package_data=True,
    data_files=[('.', ['README.md', 'requirements.txt'])],
)
