# pydentate
pydentate is an open source Python-based toolkit for predicting metal-ligand coordination in transition metal complexes (TMCs). Using only SMILES string representations as inputs, pydentate leverages graph neural networks to predict ligand denticity and coordinating atoms, enabling downstream generation of TMCs with novel metal-ligand combinations in physically realistic coordinations.

### Installation
Install using conda with the following commands:
1. `git clone https://github.com/hjkgrp/pydentate`
2. `cd pydentate`
3. `conda env create -f pydentate.yml`
4. `conda activate pydentate`
5. `pip install -e .`
