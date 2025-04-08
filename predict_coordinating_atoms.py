import numpy as np
import pandas as pd
from rdkit import Chem
import re
import torch
import torch.nn as nn
from tqdm import tqdm

# featurization
class Featurization_parameters:
    def __init__(self):
        self.ATOM_FEATURES = {'atomic_num': list(range(100)),
                              'degree': [0, 1, 2, 3, 4, 5],
                              'formal_charge': [-1, -2, 1, 2, 0],
                              'chiral_tag': [0, 1, 2, 3],
                              'num_Hs': [0, 1, 2, 3, 4],
                              'hybridization': [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                                                Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]}

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
    def __init__(self):
        super().__init__()
        self.encoder = MPN()
        self.readout = MultiReadout()
    def forward(self, mol_graph):
        encodings = self.encoder(mol_graph)
        output = self.readout(encodings)
        output = [nn.Sigmoid()(x) for x in output]
        return output

class MPNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.35)
        self.act_func = nn.ReLU()
        self.W_i = nn.Linear(in_features=14+133, out_features=600, bias=False)
        self.W_h = nn.Linear(in_features=600, out_features=600, bias=False)
        self.W_o = nn.Linear(in_features=133+600, out_features=600)
        self.W_o_b = nn.Linear(in_features=14+133+600, out_features=600)
    def forward(self, mol_graph):
        f_atoms, f_bonds, a2b, b2a, b2revb = mol_graph.f_atoms, mol_graph.f_bonds, mol_graph.a2b, mol_graph.b2a, mol_graph.b2revb
        input = self.W_i(f_bonds)
        message = self.act_func(input)
        for depth in range(6 - 1):
            nei_a_message = message.index_select(dim=0, index=a2b.view(-1)).view(a2b.size() + message.size()[1:])
            message = self.W_h(nei_a_message.sum(dim=1)[b2a] - message[b2revb])
            message = self.dropout(self.act_func(input + message))
        nei_a_message = message.index_select(dim=0, index=a2b.view(-1)).view(a2b.size() + message.size()[1:])
        a_message = nei_a_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.dropout(self.act_func(self.W_o(a_input)))
        return atom_hiddens

class MPN(nn.Module):
    def __init__(self):
        super(MPN, self).__init__()
        self.encoder = nn.ModuleList([MPNEncoder()])
    def forward(self, mol_graph):
        return self.encoder[0](mol_graph)

def build_ffn():
    layers = [nn.Dropout(0.35), nn.Linear(in_features=600, out_features=600)]
    layers.extend([nn.ReLU(), nn.Dropout(0.35), nn.Linear(in_features=600, out_features=600)])
    return nn.Sequential(*layers)

class MultiReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn_list = nn.ModuleList([FFN()])
    def forward(self, input):
        return [self.ffn_list[0](input)]

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(build_ffn(), nn.ReLU())
        self.ffn_readout = nn.Sequential(nn.Dropout(0.35), nn.Linear(in_features=600, out_features=1))
    def forward(self, input):
        input = self.ffn(input)
        output = self.ffn_readout(input)[1:]
        return output

# generate predictions
def make_predictions():
    # load model
    print('Loading training args')
    state = torch.load('models/coordinating_atoms.pt', map_location=lambda storage, loc: storage)
    loaded_state_dict = state['state_dict']
    model = MoleculeModel()
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
    smiles_list = pd.read_csv('holdout_subset.csv')['smiles'].tolist()
    mol_list = [make_mol(smiles) for smiles in smiles_list]
    model.eval()
    preds = []
    for mol in tqdm(mol_list):
        mol_graph = MolGraph(mol)
        with torch.no_grad():
            pred = model(mol_graph)
        preds.extend(pred)
    preds = [pred.flatten().tolist() for pred in preds]
    results = {'smiles': smiles_list, 'coordinating_atom_probabilities': preds}
    print('Saving predictions to coordinating_atom_preds.csv')
    pd.DataFrame(results).to_csv('coordinating_atom_preds.csv', index=False)
    return preds

coord_atom_preds = make_predictions()
print('Done predicting!')
