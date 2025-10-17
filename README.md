# pydentate
pydentate is an open source Python-based toolkit for predicting metal-ligand coordination in transition metal complexes (TMCs). Using only SMILES string representations as inputs, pydentate leverages graph neural networks to predict ligand denticity and coordinating atoms, enabling downstream generation of TMCs with novel metal-ligand combinations in physically realistic coordinations.

For more information and to cite our work, please see the corresponding publications:
1. [Graph Neural Networks for Predicting Metal–Ligand Coordination of Transition Metal Complexes](https://doi.org/10.1073/pnas.2415658122)
2. [Identifying Dynamic Metal–Ligand Coordination Modes with Ensemble Learning](https://chemrxiv.org/engage/chemrxiv/article-details/689f4370a94eede154e7a9de)

### Installation
Install via conda with the following commands:
1. `git clone https://github.com/hjkgrp/pydentate`
2. `cd pydentate`
3. `conda env create --name pydentate --file=pydentate.yml`
4. `conda activate pydentate`
5. `pip install -e .`

Alternatively, users may install via pip as follows:
`pip install pydentate`
