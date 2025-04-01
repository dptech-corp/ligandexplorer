try:
    from pdbfixer import PDBFixer 
    from openmm.app import PDBFile
    import argparse
    pdbfixer_stat = True
except:
    pdbfixer_stat = False

def fix_pdb_structure(input_file, output_file):
    if pdbfixer_stat:
        try: 
            fixer = PDBFixer(filename= input_file)
            fixer.findMissingResidues()

            chains = list(fixer.topology.chains())
            keys = fixer.missingResidues.keys()
            for key in list(keys):
                chain = chains[key[0]]
                if key[1] == 0 or key[1] == len(list(chain.residues())):
                    del fixer.missingResidues[key]
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.findMissingAtoms()
            main_chain_atoms = ['N', 'CA', 'C', 'O']
            missing_atoms_copy = fixer.missingAtoms.copy()
            for key, atoms in missing_atoms_copy.items():
                new_atoms = [ atom for atom in atoms if atom.name in main_chain_atoms ]
                fixer.missingAtoms[key] = new_atoms
            fixer.addMissingAtoms()
            PDBFile.writeFile(
                fixer.topology,
                fixer.positions,
                open(output_file, 'w'), keepIds= True)
            print(f"Structure {input_file} has been fixed")
            return True
        except Exception as e:
            print(f"Error when fix PDB file {input_file}: {e}")
            return False
    else:
        return False