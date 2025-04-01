import os
import shutil
from Bio.PDB import PDBParser, PDBIO, NeighborSearch, Selection
from Bio.PDB.PDBIO import Select

def save_protein_without_ligand(work_path=None, protein_pdb=None, init_pdb=None):
    input_protein = os.path.join(work_path, protein_pdb)
    output_protein = os.path.join(work_path, 'protein.pdb')
    other_pdb_files = [protein_pdb, 'protein.pdb', 'wat.pdb', 'fix.pdb']

    ligands_file = [f for f in os.listdir(work_path) 
                   if f.endswith('.pdb') and f not in other_pdb_files]
    
    if not ligands_file:
        shutil.copy(input_protein, output_protein)
        return
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_protein)
    protein_atoms = Selection.unfold_entities(structure, 'A')  
    ns = NeighborSearch(protein_atoms)

    residues_to_remove = set()
    for ligand_file in ligands_file:
        ligand_path = os.path.join(work_path, ligand_file)
        ligand_structure = parser.get_structure('ligand', ligand_path)
        ligand_atoms = Selection.unfold_entities(ligand_structure, 'A')

        for atom in ligand_atoms:
            close_atoms = ns.search(atom.coord, 0.2, 'A')  # 0.2 埃范围内
            for close_atom in close_atoms:
                residue = close_atom.get_parent()
                residues_to_remove.add(residue)

    class RemoveResiduesSelect(Select):
        def accept_residue(self, residue):
            if residue in residues_to_remove:
                return False
            return True

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_protein, select=RemoveResiduesSelect())
