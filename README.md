# What is LigandExplorer

<p>LigandExplorer is a workflow that combines cheminformatics tools and machine learning methods to automatically extract and classify ligands from PDB structures. It applies graph theory to identify covalent and non-covalent ligands based on molecular connectivity and uses machine learning models to filter out irrelevant molecules, ensuring that only biologically significant ligands are retained.</p>

---

## V2.0 Update

### Graph Neural Network (GNN) Backend

V2.0 introduces a new GNN-based classification backend, replacing the previous LightGBM models as the default. The GNN models learn molecular features end-to-end directly from raw atomic properties (atomic number, 3D coordinates, residue boundaries), eliminating the need for hand-crafted features.

**Two GNN models are included:**

- **Molecule Classifier :** 8-class classification — peptide, glycan, RNA, DNA, lipid, ion, organic, and **cyclic peptide** (new). Built on a SchNet. Test accuracy: **99.66%**.
- **Ligand Relevance Classifier :** Binary classification — determines whether a molecule is a biologically relevant ligand. Uses pocket-aware graph construction and dual-channel (ligand + protein) pooling. Test accuracy: **97.61%**, AUC: **99.41%**.

### Cyclic Peptide Support

V2.0 adds **cyclic peptide** as a new molecular category. Cyclic peptides are now correctly identified and classified rather than being grouped with linear peptides.

### Backend Selection

Users can choose between the GNN and legacy LightGBM backends via the `--backend` flag:

```bash
# Use GNN backend (default)
ligandexplorer -i input.zip -o output/

# Use legacy LightGBM backend (7-class, no cyclic peptide support)
ligandexplorer -i input.zip -o output/ --backend lgbm
```

### GPU Acceleration

When using the GNN backend, GPU acceleration is available via the `--device` flag:

```bash
# Run on CPU (default)
ligandexplorer -i input.zip -o output/

# Run on GPU
ligandexplorer -i input.zip -o output/ --device cuda
```

### Additional Dependencies

The GNN backend requires PyTorch and PyTorch Geometric:

```bash
pip install torch torch_geometric
```

---

# Core Feature

- **Automated Ligand Extraction:** Automatically identifies and extracts ligands from PDB files.
- **Intelligent Classification:** Uses GNN models to accurately classify molecules into ions, solvents, nucleic acids, peptides, cyclic peptides, or biologically active ligands.
- **Graph Theory Application:** Distinguishes between covalent and non-covalent ligands based on molecular connectivity using graph theory algorithms.
- **High-Throughput Processing:** Supports multi-core parallel processing for rapid handling of large PDB datasets.
- **Flexible Customization:** Offers various command-line options for users to tailor the workflow to their specific needs.

# Installation Guide
1. Prerequisites

	Before installing, please ensure you have the necessary dependencies. You can install them using pip:

		pip install numpy biopython networkx scikit-learn torch torch_geometric

	If you need the legacy LightGBM backend:

		pip install lightGBM

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

```bash
# Basic usage (GNN backend, CPU)
ligandexplorer -i /path/to/your/input.zip -o /path/to/your/output_directory

# With GPU acceleration
ligandexplorer -i /path/to/your/input.zip -o /path/to/your/output_directory --device cuda

# Using legacy LightGBM backend
ligandexplorer -i /path/to/your/input.zip -o /path/to/your/output_directory --backend lgbm
```

This command will process all PDB structures within **input.zip** and save the extracted and classified ligand information to the **output_directory**.

# Command-Line Options


```
usage: ligandexplorer [-h] -o OUTPUT_DIR -i INPUT_ZIP [-f FIX_PDB_FILE] [-b BOX_SIZE] [-c CORE]
                      [-s STRICT_MODE] [-l LIG_IDENTIFY] [-e SILENCE_MODE] [-t DEBUG]
                      [-m {gnn,lgbm}] [-d {cpu,cuda}]

Ligand explorer, version: v2.0

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
                        The workflow will determine the type of all ligand (ions, solvent, nucleic acid,
                        ligand, peptide, cyclic peptide). default is True
  -e SILENCE_MODE, --silence_mode SILENCE_MODE
                        Disable all output information. The workflow will run in silence. default is False
  -t DEBUG, --debug DEBUG
                        debug mode. output some debug information. default is False
  -m {gnn,lgbm}, --backend {gnn,lgbm}
                        Model backend for ligand identification: gnn (default) or lgbm
  -d {cpu,cuda}, --device {cpu,cuda}
                        Device for GNN inference: cpu (default) or cuda
```

# How to Cite

If you use LigandExplorer in your research, please cite:

> Li, Y.; Zou, R.; Yang, M.; Wang, Y.; Liu, Z.; Zheng, H. LigandExplorer: An Automated Tool for Ligand Extraction from PDB Structures. *J. Chem. Inf. Model.* **2026**, *66* (6), 3026–3035. DOI: [10.1021/acs.jcim.5c02921](https://doi.org/10.1021/acs.jcim.5c02921)

<details>
<summary>BibTeX</summary>

```bibtex
@article{Li2026LigandExplorer,
  author    = {Li, Yaqi and Zou, Rongfeng and Yang, Maohua and Wang, Ying and Liu, Zhonghua and Zheng, Hang},
  title     = {LigandExplorer: An Automated Tool for Ligand Extraction from PDB Structures},
  journal   = {Journal of Chemical Information and Modeling},
  year      = {2026},
  volume    = {66},
  number    = {6},
  pages     = {3026--3035},
  doi       = {10.1021/acs.jcim.5c02921},
  pmid      = {41762111},
}
```

</details>

<details>
<summary>RIS (EndNote, Zotero, Mendeley)</summary>

```ris
TY  - JOUR
AU  - Li, Yaqi
AU  - Zou, Rongfeng
AU  - Yang, Maohua
AU  - Wang, Ying
AU  - Liu, Zhonghua
AU  - Zheng, Hang
TI  - LigandExplorer: An Automated Tool for Ligand Extraction from PDB Structures
JO  - Journal of Chemical Information and Modeling
PY  - 2026
VL  - 66
IS  - 6
SP  - 3026
EP  - 3035
DO  - 10.1021/acs.jcim.5c02921
PM  - 41762111
ER  -
```

</details>
