from setuptools import setup, find_packages
setup(
name='pydentate',
version='1.1.0',
author='Jacob Toney',
author_email='jwt@mit.edu',
description='Graph Neural Networks for Predicting Metal-Ligand Coordination in Transition Metal Complexes',
long_description='Graph Neural Networks for Predicting Metal-Ligand Coordination in Transition Metal Complexes',
packages=find_packages(),
include_package_data=True,
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
python_requires='>=3.10, <3.13',
install_requires = [
  "pandas>=1.5.3",
  "numpy>=1.24.1",
  "torch>=2.1.0",
  "rdkit>=2023.3.3",
  "tqdm>=4.66.1",
  "jupyter>=1.1.1",
  "pip>=22.3.1"
]
)
