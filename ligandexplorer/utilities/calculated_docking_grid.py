import os
import json
from Bio.PDB import *
from Bio.PDB import PDBParser

import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore',BiopythonWarning)


def calculated_docking_grid(work_path, ligand_file, add_size= 10):
    # input file was PDB
    input_path = os.path.join(work_path, ligand_file)
    output_grid = os.path.join(work_path, ligand_file.split('.')[0] + '.json')
    
    parser = PDBParser(QUIET= True)
    structure = parser.get_structure('mol', input_path)
    atoms = list(structure.get_atoms())
    if not atoms:
        print(f">>> Warning: No atoms found in {ligand_file}")
        with open(os.path.join(work_path, 'error.txt'), 'a') as error_out:
            output_str = f"{ligand_file}"
            error_out.write(output_str + '\n')
        return 999
    center = sum(atom.coord for atom in structure.get_atoms()) / len(atoms)

    coords = [atom.coord for atom in structure.get_atoms()]
    min_xyz = [min(coord[i] for coord in coords) for i in range(3)]
    max_xyz = [max(coord[i] for coord in coords) for i in range(3)]
    
    size = [abs(max_xyz[i] - min_xyz[i]) for i in range(3)]
    
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    
    size_x = size_x + add_size
    size_y = size_y + add_size
    size_z = size_z + add_size
    
    grid_info = {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_z": float(center_z),
        "size_x": float(size_x),
        "size_y": float(size_y),
        "size_z": float(size_z)
    }
    with open(output_grid, 'w') as f:
        json.dump(grid_info, f, indent=4)
    print(f">>> {ligand_file}")
    print('Center: ({:.6f}, {:.6f}, {:.6f})'.format(center_x, center_y, center_z))
    print('Size: ({:.6f}, {:.6f}, {:.6f})'.format(size_x, size_y, size_z))
 