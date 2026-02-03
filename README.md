# What is LigandExplorer

<p>LigandExplorer is a workflow that combines cheminformatics tools and machine learning methods to automatically extract and classify ligands from PDB structures. It applies graph theory to identify covalent and non-covalent ligands based on molecular connectivity and uses machine learning models to filter out irrelevant molecules, ensuring that only biologically significant ligands are retained.</p>

# Core Feature

- **Automated Ligand Extraction:** Automatically identifies and extracts ligands from PDB files.
- **Intelligent Classification:** Uses machine learning models to accurately classify molecules into ions, solvents, nucleic acids, peptides, or biologically active ligands.
- **Graph Theory Application:** Distinguishes between covalent and non-covalent ligands based on molecular connectivity using graph theory algorithms.
- **High-Throughput Processing:** Supports multi-core parallel processing for rapid handling of large PDB datasets.
- **Flexible Customization:** Offers various command-line options for users to tailor the workflow to their specific needs.

# Installation Guide
1. Prerequisites

	Before installing, please ensure you have the necessary dependencies. You can install them using pip:
    
		`pip install numpy biopython networkx lightGBM scikit-learn`
    
2. Clone the Repository

	Clone the latest version of the code to your local machine using Git:

		git clone https://github.com/dptech-corp/ligandexplorer.git
   		cd ligandexplorer
    
3. Run the install script

	Use the Python setup script to install LigandExplorer and its dependencies:

		python setup.py install

# Quick Start

Once installed, you can run the program using the `ligandexplorer` command.

Use the `-h` or `--help` flag to view all available options:

`ligandexplorer -h`


# Example

Here is a basic usage example:

`ligandexplorer -i /path/to/your/input.zip -o /path/to/your/output_directory`

This command will process all PDB structures within **input.zip** and save the extracted and classified ligand information to the **output_directory**.

# Command-Line Options


```
usage: ligandexplorer [-h] -o OUTPUT_DIR -i INPUT_ZIP [-f FIX_PDB_FILE] [-b BOX_SIZE] [-c CORE] [-s STRICT_MODE]
                      [-l LIG_IDENTIFY] [-e SILENCE_MODE] [-t DEBUG]

Ligand explorer, version: v0.0.1

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        defined the output path
  -i INPUT_ZIP, --input_zip INPUT_ZIP
                        Path of the input zip file (rar, zip, tar, tar.gz, tar.bz2, bz2)
  -f FIX_PDB_FILE, --fix_pdb_file FIX_PDB_FILE
                        Fix missing main chain atom by PDBFixer (if PDBFixer is install)
  -b BOX_SIZE, --box_size BOX_SIZE
                        defined the docking grid box size (angstrom), default is 10
  -c CORE, --core CORE  defined the number of mult-process core, default is None <will use all core>
  -s STRICT_MODE, --strict_mode STRICT_MODE
                        If strict mode is on, workflow will remove identical ligands. default is True
  -l LIG_IDENTIFY, --lig_identify LIG_IDENTIFY
                        The workflow will determine the type of all ligand (ions, solvent, nucleic acid, ligand,
                        peptide. default is True)
  -e SILENCE_MODE, --silence_mode SILENCE_MODE
                        Disable all output information. The workflow will run in silence. default is False)
  -t DEBUG, --debug DEBUG
                        debug mode. output some debug information. default is False
```

# How to Cite

If you use LigandExplorer in your research, please cite this repository.
