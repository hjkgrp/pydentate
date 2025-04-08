# pydentate
pydentate is an open source Python-based toolkit for predicting metal-ligand coordination in transition metal complexes (TMCs). Using only SMILES string representations as inputs, pydentate leverages graph neural networks to predict ligand denticity and coordinating atoms, enabling downstream generation of TMCs with novel metal-ligand combinations in physically realistic coordinations.

### Installation
Install via conda with the following commands:
1. `git clone https://github.com/hjkgrp/pydentate`
2. `cd pydentate`
3. `conda env create --name pydentate --file=pydentate.yml`
4. `conda activate pydentate`

Alternatively, users may install via pip as follows:
1. `git clone https://github.com/hjkgrp/pydentate`
2. `pip install rdkit==2023.3.3`
3. `pip install torch==2.1.0`
4. `pip install numpy==1.24.1`
5. `pip install pandas==1.5.3`
6. `pip install tqdm==4.66.1`
