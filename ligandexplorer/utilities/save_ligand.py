import os
from Bio.PDB import PDBParser, PDBIO, Select
from ligandexplorer.utilities.formating import get_parser, save_structure

class ResidueSelect(Select):
    def __init__(self, selected_residues):
        self.selected_residues = selected_residues
    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        resseq = residue.id[1]
        icode = residue.id[2]
        if icode != " " and icode is not None:
             res_id_str = f"{resseq}{icode}"
        else:
             res_id_str = str(resseq)
        
        res_name = residue.get_resname()
        return (chain_id, res_id_str, res_name) in self.selected_residues
    
def parser_residue_info(residue_info_list):
    selected_residues = set()
    for info in residue_info_list:
        parts = info.split('_')
        if len(parts) != 3:
            continue
        res_name, chain_id, res_id_str = parts
        
        # Keep res_id_str as string to handle insertion codes
        
        if chain_id == "!":
            selected_residues.add((None, res_id_str, res_name))
        else:
            selected_residues.add((chain_id, res_id_str, res_name))
    return selected_residues

def save_ligand(input_pdb, subgraph, work_path):
    parser = get_parser(input_pdb, QUIET= True)
    structure = parser.get_structure('structure', input_pdb)
    residue_info = list(subgraph.nodes())
    selected_residues = parser_residue_info(residue_info)
    first_res = residue_info[0]
    # res_name = first_res.split("_")[0]
    # print(f'+++ {first_res}')
    
    ext = '.cif' if input_pdb.endswith('.cif') or input_pdb.endswith('.mmcif') else '.pdb'
    output_filename = os.path.join(work_path, f"{first_res}{ext}")

    save_structure(structure, output_filename, select=ResidueSelect(selected_residues))
    # io = PDBIO()
    # io.set_structure(structure)
    # select = ResidueSelect(selected_residues)
    # io.save(output_filename, select)

def save_ligand_covalent(input_pdb, subgraph, work_path):
    parser = get_parser(input_pdb, QUIET= True)
    structure = parser.get_structure('structure', input_pdb)
    residue_info = list(subgraph.nodes())
    selected_residues = parser_residue_info(residue_info)
    first_res = residue_info[0]
    # res_name = first_res.split("_")[0]
    
    ext = '.cif' if input_pdb.endswith('.cif') or input_pdb.endswith('.mmcif') else '.pdb'
    output_filename = os.path.join(work_path, f"covalent_{first_res}{ext}")

    save_structure(structure, output_filename, select=ResidueSelect(selected_residues))
    # io = PDBIO()
    # io.set_structure(structure)
    # select = ResidueSelect(selected_residues)
    # io.save(output_filename, select)
