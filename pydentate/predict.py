import numpy as np
import pandas as pd
from rdkit import Chem
import re
import torch
import torch.nn as nn
from tqdm import tqdm
import os

# featurization
class Featurization_parameters:
    def __init__(self):
        self.ATOM_FEATURES = {'atomic_num': list(range(100)),
                              'degree': [0, 1, 2, 3, 4, 5],
                              'formal_charge': [-1, -2, 1, 2, 0],
                              'chiral_tag': [0, 1, 2, 3],
                              'num_Hs': [0, 1, 2, 3, 4],
                              'hybridization': [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                                                Chem.rdchem.HybridizationType.SP3D2]}

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    encoding[choices.index(value) if value in choices else -1] = 1
    return encoding

def atom_features(atom):
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
        onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
        onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
        onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
        onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
        onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
        [1 if atom.GetIsAromatic() else 0] + [atom.GetMass() * 0.01]
    return features

def bond_features(bond):
    bt = bond.GetBondType()
    fbond = [0, bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
             bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
             (bond.GetIsConjugated() if bt is not None else 0), (bond.IsInRing() if bt is not None else 0)]
    fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond

# create molecular graph
class MolGraph:
    def __init__(self, mol):
        self.n_atoms = len(mol.GetAtoms())
        f_atoms_list = [atom_features(atom) for atom in mol.GetAtoms()]
        self.n_bonds, self.f_bonds, self.a2b, self.b2a, self.b2revb = 0, [], [[] for _ in range(self.n_atoms)], [], []
        self.b2br = np.zeros([len(mol.GetBonds()), 2])
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                f_bond = bond_features(bond)
                self.f_bonds.append(f_atoms_list[a1] + f_bond)
                self.f_bonds.append(f_atoms_list[a2] + f_bond)
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)
                self.b2a.append(a1)
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.b2br[bond.GetIdx(), :] = [self.n_bonds, self.n_bonds + 1]
                self.n_bonds += 2
        self.atom_fdim = 133
        self.bond_fdim = 14 + 133
        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        f_atoms.extend(f_atoms_list)
        f_bonds.extend(self.f_bonds)
        a2b.extend([[b + 1 for b in self.a2b[a]] for a in range(self.n_atoms)])
        b2a.extend([1 + self.b2a[b] for b in range(self.n_bonds)])
        b2revb.extend([1 + self.b2revb[b] for b in range(self.n_bonds)])
        self.n_atoms = 1 + self.n_atoms
        self.n_bonds = 1 + self.n_bonds
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
        self.f_atoms = torch.tensor(f_atoms, dtype=torch.float)
        self.f_bonds = torch.tensor(f_bonds, dtype=torch.float)
        self.a2b = torch.tensor([a + [0] * (self.max_num_bonds - len(a)) for a in a2b], dtype=torch.long)
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long)

def make_mol(s):
    params = Chem.SmilesParserParams()
    params.removeHs = True
    mol = Chem.MolFromSmiles(s, params)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

# model architectures
class MoleculeModel(nn.Module):
    def __init__(self, task, architecture):
        super().__init__()
        self.task = task
        self.encoder = MPN(task, architecture)
        if task=='coordination_number':
            self.multiclass_softmax = nn.Softmax(dim=1)
            self.readout = build_ffn(task, architecture)
        elif task=='coordinating_atoms':
            self.readout = MultiReadout(task, architecture)
        elif task=='hemilability':
            self.readout = build_ffn(task, architecture)

    def forward(self, mol_graph, features=None):
        encodings = self.encoder(mol_graph, features)
        output = self.readout(encodings)
        if self.task=='coordination_number':
            output = self.multiclass_softmax(output)
        elif self.task=='coordinating_atoms':
            output = [nn.Sigmoid()(x) for x in output]
        elif self.task=='hemilability':
            output = nn.Sigmoid()(output)
        return output

class MPNEncoder(nn.Module):
    def __init__(self, task, architecture):
        super().__init__()
        self.task = task
        self.architecture = architecture
        self.dropout = nn.Dropout(architecture['dropout'])
        self.act_func = nn.ReLU()
        self.W_i = nn.Linear(in_features=architecture['atom_fdim']+architecture['bond_fdim'],
                             out_features=architecture['hidden_size'], bias=False)
        self.W_h = nn.Linear(in_features=architecture['hidden_size'],
                             out_features=architecture['hidden_size'], bias=False)
        self.W_o = nn.Linear(in_features=architecture['atom_fdim']+architecture['hidden_size'],
                             out_features=architecture['hidden_size'])
    def forward(self, mol_graph):
        f_atoms, f_bonds, a2b, b2a, b2revb = mol_graph.f_atoms, mol_graph.f_bonds, mol_graph.a2b, mol_graph.b2a, mol_graph.b2revb
        input = self.W_i(f_bonds)
        message = self.act_func(input)
        for depth in range(self.architecture['depth'] - 1):
            nei_a_message = message.index_select(dim=0, index=a2b.view(-1)).view(a2b.size() + message.size()[1:])
            message = self.W_h(nei_a_message.sum(dim=1)[b2a] - message[b2revb])
            message = self.dropout(self.act_func(input + message))
        nei_a_message = message.index_select(dim=0, index=a2b.view(-1)).view(a2b.size() + message.size()[1:])
        a_message = nei_a_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.dropout(self.act_func(self.W_o(a_input)))
        if self.task in ['coordination_number', 'hemilability']:
            mol_vecs = torch.stack([sum(atom_hiddens) / (mol_graph.n_atoms-1)], dim=0)
            return mol_vecs
        elif self.task=='coordinating_atoms':
            return atom_hiddens

class MPN(nn.Module):
    def __init__(self, task, architecture):
        super(MPN, self).__init__()
        self.encoder = nn.ModuleList([MPNEncoder(task, architecture)])
        self.task = task
    def forward(self, mol_graph, features=None):
        output = self.encoder[0](mol_graph)
        if self.task in ['coordination_number', 'coordinating_atoms']:
            return output
        elif self.task=='hemilability':
            return torch.cat([output, features], dim=1)

def build_ffn(task, architecture):
    if task in ['coordination_number', 'coordinating_atoms']:
        layers = [nn.Dropout(architecture['dropout']), nn.Linear(in_features=architecture['hidden_size'], out_features=architecture['hidden_size'])]
    elif task=='hemilability':
        layers = [nn.Dropout(architecture['dropout']), nn.Linear(in_features=architecture['hidden_size']+2, out_features=architecture['hidden_size'])]
    layers.extend([nn.ReLU(), nn.Dropout(architecture['dropout']), nn.Linear(in_features=architecture['hidden_size'], out_features=architecture['hidden_size'])])
    if task in ['coordination_number', 'hemilability']:
        layers.extend([nn.ReLU(), nn.Dropout(architecture['dropout']), nn.Linear(architecture['hidden_size'], out_features=architecture['ffn_out_size'])])
    return nn.Sequential(*layers)

class MultiReadout(nn.Module):
    def __init__(self, task, architecture):
        super().__init__()
        self.ffn_list = nn.ModuleList([FFN(task, architecture)])
    def forward(self, input):
        return [self.ffn_list[0](input)]

class FFN(nn.Module):
    def __init__(self, task, architecture):
        super().__init__()
        self.ffn = nn.Sequential(build_ffn(task, architecture), nn.ReLU())
        self.ffn_readout = nn.Sequential(nn.Dropout(architecture['dropout']),
                                         nn.Linear(in_features=architecture['hidden_size'], out_features=architecture['ffn_out_size']))
    def forward(self, input):
        input = self.ffn(input)
        output = self.ffn_readout(input)[1:]
        return output

# generate predictions
def make_predictions(input_path, task, smiles_column='SMILES', output_path=False, features_path=False):
    '''
    Main function to load trained models and generate predictions of coordination number and coordinating atoms.
    INPUTS:
        input_path: str
            Path to csv where SMILES are stored.
        task: str
            Task for model prediction. Must be either 'coordination_number', 'coordinating_atoms', or 'hemilability'.
        smiles_column: str
            Column in input_path where SMILES are stored.
            default='SMILES'
        output_path: str
            Path to csv where results will be stored.
            default=task+'_preds.csv'
        features_path: str
            Path to csv containing coordination feature inputs. Only for use with hemilability prediction model.
            default=False
    '''
    
    # load model
    print('Loading training args')
    task = task.lower()
    if task not in ['coordination_number', 'coordinating_atoms', 'hemilability']:
        raise ValueError("task must be one of 'coordination_number', 'coordinating_atoms', or 'hemilability'")
    if task=='hemilability' and features_path==False:
        raise ValueError("coordination features must be specified in 'features_path' for hemilability prediction")
    output_path = task+'_preds.csv' if not output_path else output_path
    model_path = os.path.join(os.path.dirname(__file__), 'models', task+'.pt')
    state = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    loaded_state_dict = state['state_dict']
    if task=='coordination_number':
        architecture = {'atom_fdim': 133, 'bond_fdim': 14, 'depth': 6, 'dropout': 0.3, 'ffn_out_size': 6, 'hidden_size': 500}
    elif task=='coordinating_atoms':
        architecture = {'atom_fdim': 133, 'bond_fdim': 14, 'depth': 6, 'dropout': 0.35, 'ffn_out_size': 1, 'hidden_size': 600}
    elif task=='hemilability':
        architecture = {'atom_fdim': 133, 'bond_fdim': 14, 'depth': 6, 'dropout': 0.3, 'ffn_out_size': 1, 'hidden_size': 500}
    model = MoleculeModel(task, architecture)
    model_state_dict = model.state_dict()
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        if loaded_param_name in model_state_dict.keys():
            if re.match(r'(encoder\.encoder\.)([Wc])', loaded_param_name):
                param_name = loaded_param_name.replace('encoder.encoder', 'encoder.encoder.0')
            elif re.match(r'(^ffn)', loaded_param_name):
                param_name = loaded_param_name.replace('ffn', 'readout')
            else:
                param_name = loaded_param_name
            print(f'Loading pretrained parameter: {loaded_param_name}')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)
    # set features
    global PARAMS
    PARAMS = Featurization_parameters()
    # load data, generate predictions
    smiles_list = pd.read_csv(input_path)[smiles_column].tolist()
    mol_list = [make_mol(smiles) for smiles in smiles_list]
    if task=='hemilability':
        features_data = pd.read_csv(features_path)
        features_data = features_data[['coordination_number_uncertainties', 'coordinating_atoms_uncertainties']]
        features_data = [np.asarray(features_data.iloc[idx].tolist()) for idx in range(len(features_data))]
    elif task in ['coordination_number', 'coordinating_atoms']:
        features_data = [None]*len(mol_list)
    model.eval()
    preds = []
    for mol, features in tqdm(zip(mol_list, features_data)):
        mol_graph = MolGraph(mol)
        features = torch.from_numpy(features).float().unsqueeze(0) if task=='hemilability' else None
        with torch.no_grad():
            pred = model(mol_graph, features)
        preds.extend(pred)
    preds = [pred.flatten().tolist() for pred in preds]
    results = {smiles_column: smiles_list, task+'_probabilities': preds}
    print(f'Saving predictions to {output_path}')
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f'Done predicting {task}!')
    return
