import os
import shutil
from Bio.PDB import PDBParser, PDBIO, Select, MMCIFParser, MMCIFIO
from ligandexplorer.utilities.pdbfixer_wf import fix_pdb_structure

def get_parser(filename, QUIET=True):
    if filename.endswith('.cif') or filename.endswith('.mmcif'):
        return MMCIFParser(QUIET=QUIET)
    return PDBParser(QUIET=QUIET)

def save_structure(structure, filename, select=None):
    if filename.endswith('.cif') or filename.endswith('.mmcif'):
        io = MMCIFIO()
    else:
        io = PDBIO()
    io.set_structure(structure)
    if select:
        io.save(filename, select=select)
    else:
        io.save(filename)


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
                      
    input_full_path = os.path.join(work_path, input_file)
    ext = '.cif' if input_file.endswith('.cif') or input_file.endswith('.mmcif') else '.pdb'
    
    format_pdb = os.path.join(work_path, 'format' + ext)
    water_pdb = os.path.join(work_path, 'water' + ext)
    fix_pdb_f = os.path.join(work_path, 'fix' + ext)

    if fix_pdb:
        if fix_pdb_structure(input_full_path, fix_pdb_f):
            source_file = fix_pdb_f
        else:
            source_file = input_full_path
    else:
        source_file = input_full_path

    parser = get_parser(source_file, QUIET=True)
    structure = parser.get_structure('structure', source_file)

    models = list(structure)
    if len(models) > 1:
        first_model = models[0]
    else:
        first_model = structure[0]
    
    print(f"DEBUG: Saving format file to {format_pdb}")
    try:
        save_structure(first_model, format_pdb, select=AtomSelecter(keep_water= False))
    except Exception as e:
        print(f"DEBUG: Failed to save format file {format_pdb}: {e}")

    # save_structure(first_model, format_pdb, select=AtomSelecter(keep_water= False))
    # if os.path.exists(format_pdb):
    #    print(f"DEBUG: Saved {format_pdb}")
    # else:
    #    print(f"DEBUG: Failed to save {format_pdb}")

    try:
        save_structure(first_model, water_pdb, select=AtomSelecter(keep_water= True))
    except Exception as e:
         print(f"DEBUG: Failed to save water file {water_pdb}: {e}")
         
    if os.path.exists(format_pdb):
        shutil.move(format_pdb, input_full_path)
    else:
        print(f"DEBUG: format_pdb {format_pdb} not created, cannot move to {input_full_path}")        
    
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