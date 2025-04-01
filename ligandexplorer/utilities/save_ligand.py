import os
from Bio.PDB import PDBParser, PDBIO, Select

class ResidueSelect(Select):
    def __init__(self, selected_residues):
        self.selected_residues = selected_residues
    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        res_id = residue.id[1]
        res_name = residue.get_resname()
        return (chain_id, res_id, res_name) in self.selected_residues
    
def parser_residue_info(residue_info_list):
    selected_residues = set()
    for info in residue_info_list:
        parts = info.split('_')
        if len(parts) != 3:
            continue
        res_name, chain_id, res_id_str = parts
        try:
            res_id = int(res_id_str)
        except:
            res_id = res_id_str
        if chain_id == "!":
            selected_residues.add((None, res_id, res_name))
        else:
            selected_residues.add((chain_id, res_id, res_name))
    return selected_residues

def save_ligand(input_pdb, subgraph, work_path):
    parser = PDBParser(QUIET= True)
    structure = parser.get_structure('structure', input_pdb)
    residue_info = list(subgraph.nodes())
    selected_residues = parser_residue_info(residue_info)
    first_res = residue_info[0]
    # res_name = first_res.split("_")[0]
    # print(f'+++ {first_res}')
    output_filename = os.path.join(work_path, f"{first_res}.pdb")

    io = PDBIO()
    io.set_structure(structure)
    select = ResidueSelect(selected_residues)
    io.save(output_filename, select)

def save_ligand_covalent(input_pdb, subgraph, work_path):
    parser = PDBParser(QUIET= True)
    structure = parser.get_structure('structure', input_pdb)
    residue_info = list(subgraph.nodes())
    selected_residues = parser_residue_info(residue_info)
    first_res = residue_info[0]
    # res_name = first_res.split("_")[0]
    output_filename = os.path.join(work_path, f"covalent_{first_res}.pdb")

    io = PDBIO()
    io.set_structure(structure)
    select = ResidueSelect(selected_residues)
    io.save(output_filename, select)
