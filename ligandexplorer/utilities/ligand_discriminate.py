import os
import copy
import numpy as np
import networkx as nx
from collections import Counter
from Bio.PDB import *
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Superimposer import Superimposer
from ligandexplorer.utilities.calculated_docking_grid import calculated_docking_grid
from ligandexplorer.utilities.lgbm_workflow import load_model_and_pred, load_model_and_pred_ligand
from ligandexplorer.utilities.formating import get_parser

import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore',BiopythonWarning)

def generated_graph(structure):
    distance_thresholds = {
        ( "C", "C"): 1.84,  # 1.54
        ( "C", "N"): 1.76,  # 1.48
        ( "N", "C"): 1.76,
        ( "C", "O"): 1.72,  # 1.43
        ( "O", "C"): 1.72,  
        ( "C", "S"): 2.10,  # 1.82
        ( "S", "C"): 2.10,
        ( "C", "P"): 2.24,  # 1.87
        ( "P", "C"): 2.24,
        ( "O", "P"): 1.95,  #
        ( "P", "O"): 1.95,
        ( "C", "CL"): 1.90, # 1.77
        ( "CL", "C"): 1.90,
        ( "C", "BR"): 2.10, # 1.94
        ( "BR", "C"): 2.10,
        ( "C", "I"): 2.30,  # 2.14
        ( "I", "C"): 2.30,
        ( "NA", "C"): 2.30,
        ( "C", "NA"): 2.30
    }

    graph = nx.Graph()
    atoms = [ atom for atom in structure.get_atoms() if atom.element != 'H' ]
    for i, atom in enumerate(atoms):
        graph.add_node(i, 
                       element= atom.element,
                       coords= atom.get_coord())
        
    for i, atom1 in enumerate(atoms):
        coord1 = atom1.get_coord()
        for j, atom2 in enumerate(atoms[i+1:], i+1):
            coord2 = atom2.get_coord()
            distance = np.linalg.norm(coord1 - coord2)
            if (atom1.element, atom2.element) in distance_thresholds:
                if distance < distance_thresholds[(atom1.element, atom2.element)]:
                    graph.add_edge(i,j)
            elif distance <= 1.5:
                graph.add_edge(i, j)
            else:
                pass
    return graph

def is_subgraph_similar(G1, G2_list):
    for G2 in G2_list:
        for sub_graph_nodes in nx.connected_components(G2):
            sub_graph = G2.subgraph(sub_graph_nodes)
            if are_graphs_similar(G1, sub_graph):
                return True
    return False

def node_match(node1, node2):
    return node1['element'] == node2['element']

def are_graphs_similar(G1, G2):
    return nx.is_isomorphic(G1, G2, node_match=node_match)

def get_heavy_atoms(structure):
    element_count = Counter()
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element != 'H':
                        element_count[atom.element] += 1
    return element_count

def get_rmsd(str1, str2):
    atoms1 = [atom for atom in str1.get_atoms() if atom.element != 'H']
    atoms2 = [atom for atom in str2.get_atoms() if atom.element != 'H']    
    
    if len(atoms1) != len(atoms2):
        return 999
    
    if len(atoms1) == 0:
        return 0

    elements = ['C', 'N', 'O', 'S', 'P', 'CL', 'BR']
    str1_atoms = {element: sum(1 for atom in atoms1 if atom.element == element) for element in elements}
    str2_atoms = {element: sum(1 for atom in atoms2 if atom.element == element) for element in elements}
    if all(str1_atoms[elem] == str2_atoms[elem] for elem in elements):
        coords1 = [atom.get_coord() for atom in atoms1]
        coords2 = [atom.get_coord() for atom in atoms2]
        
        super_imposer = Superimposer()
        super_imposer.set_atoms(atoms1, atoms2) 
        super_imposer.apply(str2.get_atoms())
        return super_imposer.rms
    else:
        return 999

def compare_element_distributions(dist1, dist2):  
    return all(dist1[element] == dist2[element] for element in set(dist1) | set(dist2))  

def clean_alt_structure(structure):
    cleaned_structure = copy.deepcopy(structure)
    for model in cleaned_structure:
        for chain in model:
            for residue in chain:
                atoms_to_remove = []
                for atom in residue:
                    alt_loc = atom.get_altloc()
                    if alt_loc != ' ' and alt_loc != 'A':
                        atoms_to_remove.append(atom.get_id())
                
                for atom_id in atoms_to_remove:
                    residue.detach_child(atom_id)
                
                for atom in residue:
                    atom.set_altloc(' ')
    return cleaned_structure

def ligand_identify(work_path=None, input_pdb= None, search_mode= None, LGBM_Model_package= None, add_size= None):
    '''
    0: dna
    1: gly
    2: ions
    3: mem
    4: organic
    5: peptide
    6: rna
    '''
    ext = '.cif' if input_pdb.endswith('.cif') or input_pdb.endswith('.mmcif') else '.pdb'
    other_pdb_file = [ input_pdb, 'protein' + ext, 'water' + ext, 'fix' + ext]
    ligands_file = [ f for f in os.listdir(work_path) if f.endswith(ext) ]
    ligands = [ lig for lig in ligands_file if lig not in other_pdb_file ]
    for ligand in ligands:
        ligand_file = os.path.join(work_path, ligand)
        protein_file = os.path.join(work_path, 'protein' + ext)
        mol_classfication = load_model_and_pred(ligand_file, LGBM_Model_package )
        ligand_classfication = load_model_and_pred_ligand(protein_pdb= protein_file, ligand_pdb= ligand_file, LGBM_Model_package= LGBM_Model_package)
        if mol_classfication == 'dna':
            if ligand_classfication == 0:
                new_f = os.path.join(work_path, 'Other_DNA_' + ligand)
                print(f">>> {ligand} is DNA, it is not a ligand")
            else:
                new_f = os.path.join(work_path, 'Ligand_DNA_' + ligand)
                print(f">>> {ligand} is DNA, it is a ligand")
            os.rename(ligand_file, new_f)

        elif mol_classfication == 'gly':
            if ligand_classfication == 0:
                new_f = os.path.join(work_path, 'Other_carbohydrate_' + ligand)
                print(f">>> {ligand} is Carbohydrate, it is not a ligand")
            else:
                new_f = os.path.join(work_path, 'Ligand_carbohydrate_' + ligand)
                print(f">>> {ligand} is Carbohydrate, it is a ligand")
            os.rename(ligand_file, new_f)

        elif mol_classfication == 'ions':
            if ligand_classfication == 0:
                new_f = os.path.join(work_path, 'Other_ions_' + ligand)
                print(f">>> {ligand} is Ions, it is not a ligand")
            else:
                new_f = os.path.join(work_path, 'Ligand_ions_' + ligand)
                print(f">>> {ligand} is Ions, it is a ligand")
            os.rename(ligand_file, new_f)

        elif mol_classfication == 'mem':
            if ligand_classfication == 0:
                new_f = os.path.join(work_path, 'Other_phospholipids_' + ligand)
                print(f">>> {ligand} is phospholipids, it is not a ligand")
            else:
                new_f = os.path.join(work_path, 'Ligand_phospholipids_' + ligand)
                print(f">>> {ligand} is phospholipids, it is a ligand")
            os.rename(ligand_file, new_f)

        elif mol_classfication == 'organic':
            if ligand_classfication == 0:
                new_f = os.path.join(work_path, 'Other_organic_' + ligand)
                print(f">>> {ligand} is organic, it is not a ligand")
            else:
                new_f = os.path.join(work_path, 'Ligand_organic_' + ligand)
                print(f">>> {ligand} is organic, it is a ligand")
            os.rename(ligand_file, new_f)

        elif mol_classfication == 'peptide':
            if ligand_classfication == 0:
                new_f = os.path.join(work_path, 'Other_peptide_' + ligand)
                print(f">>> {ligand} is peptide, it is not a ligand")
            else:
                new_f = os.path.join(work_path, 'Ligand_peptide_' + ligand)
                print(f">>> {ligand} is peptide, it is a ligand")
            os.rename(ligand_file, new_f)

        elif mol_classfication == 'rna':
            if ligand_classfication == 0:
                new_f = os.path.join(work_path, 'Other_RNA_' + ligand)
                print(f">>> {ligand} is RNA, it is not a ligand")
            else:
                new_f = os.path.join(work_path, 'Ligand_RNA_' + ligand)
                print(f">>> {ligand} is RNA, it is a ligand")
            os.rename(ligand_file, new_f)
    # remove the same ligands
    parser = get_parser(input_pdb, QUIET= True)   
    if search_mode == 'strict':
        exclude_title = ('Other_')
        
        identify_ligand_files = [ f for f in os.listdir(work_path) 
                                 if f.endswith(ext) 
                                 and f not in other_pdb_file 
                                 and not f.startswith(exclude_title) ]
        rmsd_cut_off = 0.8
        if len(identify_ligand_files) >= 2:
            identify_str = {}
            atom_count_set = {}
            graphs_set = {}
            seen_files = set()
            for pdb_file in identify_ligand_files:
                pdb_f = os.path.join(work_path, pdb_file)
                no_alt_struct = clean_alt_structure(parser.get_structure(None, pdb_f))
                identify_str[pdb_file] = no_alt_struct
                atom_count_set[pdb_file] = get_heavy_atoms(no_alt_struct)
                graphs_set[pdb_file] = generated_graph(no_alt_struct)
            unique_files = []
            for i, ref_file in enumerate(identify_ligand_files):
                if ref_file in seen_files:
                    continue
                ref_str = identify_str[ref_file]
                unique_files.append(ref_file)
                seen_files.add(ref_file)

                for j in range(i + 1, len(identify_ligand_files)):
                    comp_f = identify_ligand_files[j]
                    if comp_f in seen_files:
                        continue

                    if compare_element_distributions(atom_count_set[ref_file], atom_count_set[comp_f]):
                        if are_graphs_similar(graphs_set[ref_file], graphs_set[comp_f]):
                            comp_str = identify_str[comp_f]
                            try:
                                rmsd = get_rmsd(ref_str, comp_str)
                                if rmsd < rmsd_cut_off:
                                    seen_files.add(comp_f)
                                    comp_f_path = os.path.join(work_path, comp_f)
                                    print(f">>> {comp_f} is similarity {ref_file}, remove {comp_f}")
                                    if os.path.exists(comp_f_path):
                                        os.remove(os.path.join(work_path, comp_f))
                            except ValueError as e:
                                print(f"Error calculating RMSD: {str(e)}")
                                continue

    print(f'>>> ligands identify: {work_path} Job Done')
    print('>>> Calculate docking grid')
    # calculated docking grid
    docking_grid_ligand = [ f for f in os.listdir(work_path) if f.endswith(ext) and f not in other_pdb_file and not f.startswith('Other_') ]
    for final_f in docking_grid_ligand:
        calculated_docking_grid(work_path, final_f, add_size= add_size)
    