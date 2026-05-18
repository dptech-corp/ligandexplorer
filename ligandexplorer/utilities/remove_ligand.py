import os
import shutil
from Bio.PDB.PDBIO import Select
from ligandexplorer.utilities.formating import get_parser, save_structure

def save_protein_without_ligand(work_path=None, protein_pdb=None, init_pdb=None):
    ext = '.cif' if protein_pdb.endswith('.cif') or protein_pdb.endswith('.mmcif') else '.pdb'
    input_protein = os.path.join(work_path, protein_pdb)
    output_protein = os.path.join(work_path, 'protein' + ext)
    other_pdb_files = [protein_pdb, 'protein' + ext, 'wat' + ext, 'fix' + ext, 'format' + ext, 'water' + ext]

    ligands_file = [f for f in os.listdir(work_path) 
                   if f.endswith(ext) and f not in other_pdb_files]
    
    if not ligands_file:
        shutil.copy(input_protein, output_protein)
        return
    
    parser = get_parser(input_protein, QUIET=True)
    structure = parser.get_structure('protein', input_protein)

    ligand_residue_ids = set()
    for ligand_file in ligands_file:
        ligand_path = os.path.join(work_path, ligand_file)
        ligand_structure = parser.get_structure('ligand', ligand_path)
        for model in ligand_structure:
            for chain in model:
                for residue in chain:
                    ligand_residue_ids.add((chain.get_id(), residue.id))

    class RemoveResiduesSelect(Select):
        def accept_residue(self, residue):
            chain_id = residue.get_parent().get_id()
            if (chain_id, residue.id) in ligand_residue_ids:
                return False
            return True

    save_structure(structure, output_protein, select=RemoveResiduesSelect())
    # io = PDBIO()
    # io.set_structure(structure)
    # io.save(output_protein, select=RemoveResiduesSelect())
