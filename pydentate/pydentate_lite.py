import predict
import numpy as np
import pandas as pd
from rdkit import Chem
import re
import torch
import torch.nn as nn
from itertools import combinations

def pydentate_lite(smiles):
    all_tasks = ['coordination_number', 'coordinating_atoms', 'hemilability']
    all_preds = []
    for task in all_tasks:
        # load models
        model_path = 'models/' + task + '.pt'
        state = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        loaded_state_dict = state['state_dict']
        if task=='coordination_number':
            architecture = {'atom_fdim': 133, 'bond_fdim': 14, 'depth': 6, 'dropout': 0.3, 'ffn_out_size': 6, 'hidden_size': 500}
        elif task=='coordinating_atoms':
            architecture = {'atom_fdim': 133, 'bond_fdim': 14, 'depth': 6, 'dropout': 0.35, 'ffn_out_size': 1, 'hidden_size': 600}
        elif task=='hemilability':
            architecture = {'atom_fdim': 133, 'bond_fdim': 14, 'depth': 6, 'dropout': 0.3, 'ffn_out_size': 1, 'hidden_size': 500}
        model = predict.MoleculeModel(task, architecture)
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
                pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)
        # set features
        global PARAMS
        PARAMS = predict.Featurization_parameters()
        # load data, generate predictions
        mol = predict.make_mol(smiles)
        if task=='hemilability':
            coord_num_probs = all_preds[0]
            coord_atom_probs = all_preds[1]
            coord_num_uncertainty = np.max([1 - pred if pred >= 0.5 else pred for pred in coord_num_probs])
            coord_atom_uncertainty = np.max([1 - pred if pred >= 0.5 else pred for pred in coord_atom_probs])
            features = torch.tensor([coord_num_uncertainty, coord_atom_uncertainty], dtype=torch.float).unsqueeze(0)
        elif task in ['coordination_number', 'coordinating_atoms']:
            features = None
        model.eval()
        mol_graph = predict.MolGraph(mol, PARAMS)
        with torch.no_grad():
            pred = model(mol_graph, features)
        pred = pred[0] if type(pred) == list else pred
        pred = pred.flatten().tolist()
        all_preds.append(pred)

    # parse predictions
    coord_num_probs, coord_atom_probs, hemi_prob = all_preds
    hemi_prob = hemi_prob[0]
    coord_num = int(np.argmax(coord_num_probs)+1)
    coord_atoms = [idx for idx, atom in enumerate(np.round(coord_atom_probs)) if atom != 0]
    coord_syms = [mol.GetAtoms()[int(atom_idx)].GetSymbol() for atom_idx in coord_atoms]
    coord_prob = float(np.mean([max(coord_num_probs)] + [coord_atom_probs[atom] for atom in coord_atoms]))
    hemi = int(np.round(hemi_prob))

    # update predictions for internal consistency
    if coord_num != len(coord_atoms):
        if (coord_num_uncertainty <= coord_atom_uncertainty) or (len(coord_atoms) > 6):
            N = coord_num
            coord_atoms = [int(i) for i in np.argsort(coord_atom_probs)[-N:][::-1]]
            mol = Chem.MolFromSmiles(smiles)
            coord_syms = [mol.GetAtoms()[int(atom_idx)].GetSymbol() for atom_idx in coord_atoms]
        elif coord_atom_uncertainty < coord_num_uncertainty:
            coord_num = len(coord_atoms)

    # visualize coordination mode
    coord_img = Chem.Draw.MolToImage(mol, size=(500, 500), highlightAtoms=coord_atoms, dpi=500)

    # ensemble algorithm
    prob_cutoff = 0.1
    len_cutoff = 3
    if hemi_prob < 0.5:
        alternative_coordination_modes = None
        alt_img = None
    # ligand predicted to hemilabile; call ensemble algorithm
    elif hemi_prob >= 0.5:
        alternative_coordination_modes = {'alternative_coordination_numbers': [], 'alternative_coordinating_atoms': [], 'alternative_coordinating_atoms_symbols': [], 'alternative_probabilities': []}
        potential_catom_indices = [atom_idx for atom_idx, prob in enumerate(coord_atom_probs) if prob >= prob_cutoff]
        all_combos = []
        probabilities = []
        for num_idx, alt_coord_num in enumerate(coord_num_probs):
            combos = list(combinations(potential_catom_indices, num_idx+1))
            all_combos.extend(combos)
            probabilities.extend([np.mean([alt_coord_num] + [coord_atom_probs[catom_idx] for catom_idx in combo]) for combo in combos])
        all_combos = [set(combo) for combo in all_combos]
        if set(coord_atoms) in all_combos:
            remove_idx = all_combos.index(set(coord_atoms))
            all_combos.pop(remove_idx)
            probabilities.pop(remove_idx)

        sorted_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
        top_probabilities = [float(probabilities[i]) for i in sorted_indices][0:len_cutoff]
        top_combos = [list(all_combos[i]) for i in sorted_indices][0:len_cutoff]
        
        # store alternative coordination modes
        alternative_coordination_modes['alternative_coordination_numbers'] = [len(top_combo) for top_combo in top_combos]
        alternative_coordination_modes['alternative_coordinating_atoms'] = top_combos
        alternative_coordination_modes['alternative_coordinating_atoms_symbols'] = [[mol.GetAtoms()[atom_idx].GetSymbol() for atom_idx in top_combo] for top_combo in top_combos]
        alternative_coordination_modes['alternative_probabilities'] = top_probabilities
        
        # visualize coordination mode
        alt_img = Chem.Draw.MolToImage(mol, size=(500, 500), highlightAtoms=top_combos[0], dpi=500)
    
    # save all results to DataFrame
    results_summary = pd.DataFrame({'smiles': [smiles], 'coordination_number_probabilities': [coord_num_probs], 'predicted_coordination_number': [coord_num],
                                    'coordination_number_uncertainty': [float(coord_num_uncertainty)], 'coordinating_atoms_probabilities': [coord_atom_probs],
                                    'predicted_coordinating_atoms': [coord_atoms], 'predicted_coordinating_atoms_symbols': [coord_syms],
                                    'coordinating_atoms_uncertainty': [float(coord_atom_uncertainty)], 'coordination_probability': [coord_prob],
                                    'hemilability_probability': [hemi_prob], 'predicted_hemilability': [hemi], 'alternative_coordination_modes': [alternative_coordination_modes]})

    return coord_num, coord_atoms, coord_syms, coord_prob, hemi, alternative_coordination_modes, coord_img, alt_img, results_summary
