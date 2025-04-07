import os
import shutil
from Bio.PDB import PDBParser, PDBIO, Select
from ligandexplorer.utilities.pdbfixer_wf import fix_pdb_structure

def format_pdb_file(work_path=None, input_file=None, fix_pdb= False, debug= False):
    class AtomSelecter(Select):
        def __init__(self, keep_water= False):
            self.keep_water = keep_water
            self.processed_atoms = set()

        def accept_residue(self, residue):
            if self.keep_water:
                return residue.get_resname() in ['HOH', 'WAT', 'SOL', 'OPC', 'SPC', 'T3P', 'TIP']
            return residue.get_resname() not in ['HOH', 'WAT', 'SOL', 'OPC', 'SPC', 'T3P', 'TIP']
        
        def accept_atom(self, atom):
            residue = atom.get_parent()
            chain = residue.get_parent()
            # atom_id = (atom.get_parent().get_id(), atom.get_id())
            atom_id = (
                chain.id,
                residue.id, 
                atom.serial_number, 
                atom.get_name().strip()
            )
            if atom_id in self.processed_atoms:
                return False
            self.processed_atoms.add(atom_id)
            return True
                      
    parser = PDBParser(QUIET=True)
    io = PDBIO()

    input_pdb = os.path.join(work_path, input_file)
    format_pdb = os.path.join(work_path, 'format.pdb')
    water_pdb = os.path.join(work_path, 'water.pdb')
    fix_pdb_f = os.path.join(work_path, 'fix.pdb')
    
    if fix_pdb:
        if fix_pdb_structure(input_pdb, fix_pdb_f):
            structure = parser.get_structure('structure', fix_pdb_f)
            models = list(structure)
            num_models = len(models)
            if num_models > 1:
                first_model = models[0]
            else:
                first_model = structure[0]
            io.set_structure(first_model)
            io.save(format_pdb, select= AtomSelecter(keep_water= False))
            io.save(water_pdb, select= AtomSelecter(keep_water= True))
            shutil.move(format_pdb, input_pdb)
    else:
        structure = parser.get_structure('structure', input_pdb)
        models = list(structure)
        num_models = len(models)
        if num_models > 1:
            first_model = models[0]
        else:
            first_model = structure[0]
        io.set_structure(first_model)
        io.save(format_pdb, select= AtomSelecter(keep_water= False))
        io.save(water_pdb, select= AtomSelecter(keep_water= True))
        shutil.move(format_pdb, input_pdb)

        
    
    # try:
    #     structure = parser.get_structure('structure', input_pdb)
    # except Exception as e:
    #     print(f"Error when parser PDB file: {e}")
    #     return
    
    # models = list(structure)
    # num_models = len(models)
    # if num_models > 1:
    #     first_model = models[0]
    # else:
    #     first_model = structure[0]
    
    # io = PDBIO()
    # io.set_structure(first_model)
    # io.save(format_pdb, select= NonWaterSelect())
    # io.save(water_pdb, select= WaterSelect())
    # if not fix_pdb_structure(format_pdb, input_pdb):
    #     shutil.move(format_pdb, input_pdb)