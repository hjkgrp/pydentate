import pandas as pd
import ast
import numpy as np
from rdkit import Chem

def parse_predictions(input_path, task, task_column=False, smiles_column='SMILES', output_path=False):
    '''
    Read predictions and parse into usable format.
    INPUTS:
        input_path: str
            Path to csv where predictions are stored.
        task: str
            prediction type, either 'coordination_number' or 'coordinating_atoms'
        task_column: str
            Column in input_path where predictions are stored.
            default=task+'_probabilities'
        smiles_column: str
            Column in input_path where SMILES are stored.
            default='SMILES'
        output_path: str
            Path to csv where results will be stored.
            default=task+'_preds.csv'
    '''
    task = task.lower()
    if task not in ['coordination_number', 'coordinating_atoms']:
        raise ValueError("task must be one of 'coordination_number' or 'coordinating_atoms'")

    preds = pd.read_csv(input_path)
    task_column = task+'_probabilities' if not task_column else task_column
    output_path = task+'_preds.csv' if not output_path else output_path
    
    # read predictions, parse into usable format
    preds[task_column] = preds[task_column].apply(ast.literal_eval)

    if task=='coordination_number':
        # take maximum prediction
        preds['predicted_coordination_number'] = preds[task_column].apply(lambda probs: np.argmax(probs)+1)
    elif task=='coordinating_atoms':
        # round predictions 
        preds['predicted_coordinating_atoms'] = [[idx for idx, atom in enumerate(np.round(preds[task_column][row_idx])) if atom != 0] for row_idx in range(len(preds))]
        # get coordinating atom symbols
        atom_symbols = []
        for idx, smiles in enumerate(preds[smiles_column]):
            mol = Chem.MolFromSmiles(smiles)
            atom_symbols.append([mol.GetAtoms()[atom_idx].GetSymbol() for atom_idx in preds['predicted_coordinating_atoms'][idx]])
        preds['predicted_coordinating_atoms_symbols'] = atom_symbols

    # save processed predictions
    preds.to_csv(output_path, index=False)
    print('Predictions processed!')
    return

def visualize(smiles, coord_atoms=False, save_image_path=None):
    '''
    Visualize ligand and (if applicable) predicted coordinating atoms.
    INPUTS:
        smiles: str
            SMILES string of ligand.
        coord_atoms: list
            Coordinating atom indices to highlight.
            default=False
        save_image_path: str
            Path to save image to.
            default=None        
    '''
    mol = Chem.MolFromSmiles(smiles)
    img = Chem.Draw.MolToImage(mol, size=(500, 500), highlightAtoms=coord_atoms, dpi=500)
    display(img)
    if save_image_path:
        img.save(save_image_path)
    return

def enforce_consistent_predictions(coordination_number_path, coordinating_atoms_path, output_path='combined_preds.csv', bias=False, smiles_column='SMILES'):
    '''
    Update predicted coordination number and coordinating atom indices for internal consistency (i.e., require predicted coordination number = len(coordinating atom indices)).
    INPUTS:
        coordination_number_path: str
            Path to csv where coordination number predictions are stored.
        coordinating_atoms_path: str
            Path to csv where coordinating atom predictions are stored.
        output_path: str
            Path to csv where results will be stored.
            default='combined_preds.csv'
        bias: str
            Keyword corresponding to which model to apply positive bias towards. If provided, must be either 'coordination_number', 'coordinating_atoms'.
            default=False
        smiles_column: str
            Column where SMILES are stored.
            default='SMILES'
    '''
    # read data
    coord_num_preds = pd.read_csv(coordination_number_path)
    coord_atom_preds = pd.read_csv(coordinating_atoms_path)

    # parse strings to lists
    coord_num_preds['coordination_number_probabilities'] = coord_num_preds['coordination_number_probabilities'].apply(ast.literal_eval)
    coord_atom_preds['coordinating_atoms_probabilities'] = coord_atom_preds['coordinating_atoms_probabilities'].apply(ast.literal_eval)
    coord_atom_preds['predicted_coordinating_atoms'] = coord_atom_preds['predicted_coordinating_atoms'].apply(ast.literal_eval)
    coord_atom_preds['predicted_coordinating_atoms_symbols'] = coord_atom_preds['predicted_coordinating_atoms_symbols'].apply(ast.literal_eval)
    
    predicted_coordination_number = list(coord_num_preds['predicted_coordination_number'])
    coordination_number_probabilities = coord_num_preds['coordination_number_probabilities']
    coordinating_atoms_probabilities = coord_atom_preds['coordinating_atoms_probabilities']
    predicted_coordinating_atoms = list(coord_atom_preds['predicted_coordinating_atoms'])
    predicted_coordinating_atoms_symbols = list(coord_atom_preds['predicted_coordinating_atoms_symbols'])
    
    # get model uncertainties
    coord_num_uncertainty = coord_num_preds['coordination_number_probabilities'].apply(lambda row: np.max([1 - pred if pred >= 0.5 else pred for pred in row]))
    coord_atom_uncertainty = coord_atom_preds['coordinating_atoms_probabilities'].apply(lambda row: np.max([1 - pred if pred >= 0.5 else pred for pred in row]))

    # update predictions to enforce consistency
    update_count = 0
    for idx in range(len(coord_num_preds)):
        if predicted_coordination_number[idx] != len(predicted_coordinating_atoms[idx]):
            # coordination number has lower uncertainty: overwrite coordinating atoms
            if (coord_num_uncertainty[idx] <= coord_atom_uncertainty[idx]) or (len(predicted_coordinating_atoms[idx]) > 6) or bias=='coordination_number':
                N = predicted_coordination_number[idx]
                predicted_coordinating_atoms[idx] = np.argsort(coordinating_atoms_probabilities[idx])[-N:][::-1]
                smiles = coord_atom_preds[smiles_column][idx]
                mol = Chem.MolFromSmiles(smiles)
                predicted_coordinating_atoms_symbols[idx] = [mol.GetAtoms()[int(atom_idx)].GetSymbol() for atom_idx in predicted_coordinating_atoms[idx]]
                update_count+=1
            # coordinating atoms have lower uncertainty: overwrite coordination number
            elif coord_atom_uncertainty[idx] < coord_num_uncertainty[idx] or bias=='coordinating_atoms':
                predicted_coordination_number[idx] = len(predicted_coordinating_atoms[idx])
                update_count+=1

    # save results
    coord_num_preds['predicted_coordination_number'] = predicted_coordination_number
    coord_num_preds['coordination_number_uncertainties'] = coord_num_uncertainty
    coord_num_preds['coordinating_atoms_probabilities'] = coordinating_atoms_probabilities
    coord_num_preds['predicted_coordinating_atoms'] = predicted_coordinating_atoms
    coord_num_preds['predicted_coordinating_atoms_symbols'] = predicted_coordinating_atoms_symbols
    coord_num_preds['coordinating_atoms_uncertainties'] = coord_atom_uncertainty
    coord_num_preds.to_csv(output_path, index=False)

    print(f'Internal consistency enforced. {update_count} predictions updated.')
    return