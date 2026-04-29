"""
GNN-based inference replacing lgbm_workflow.

Provides the same function signatures:
  load_model_and_pred(input_pdb, LGBM_Model_package) -> str
  load_model_and_pred_ligand(protein_pdb, ligand_pdb, LGBM_Model_package) -> int
"""
import torch
from ligandexplorer.utilities.gnn_data import structure_to_graph, complex_to_graph
from ligandexplorer.utilities.gnn_models import MoleculeClassifier, LigandBinaryClassifier

CATEGORIES = ["peptide", "gly", "rna", "dna", "mem", "ions", "organic", "cyclic_peptide"]


def load_model_and_pred(input_pdb, LGBM_Model_package=None):
    """Classify molecule type (8-class).

    Returns one of: 'peptide', 'gly', 'rna', 'dna', 'mem', 'ions', 'organic', 'cyclic_peptide'
    """
    try:
        from ligandexplorer.workflow import ModelContainer
        model = ModelContainer.mol_classifier
        device = ModelContainer.device
        if model is None:
            raise RuntimeError("mol_classifier not loaded")

        graph = structure_to_graph(input_pdb, cutoff=5.0, max_neighbors=32)
        if graph is None or graph.z.size(0) == 0:
            return "organic"

        graph = graph.to(device)
        graph.batch = torch.zeros(graph.z.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(graph)
            pred = out.argmax(dim=-1).item()

        return CATEGORIES[pred]
    except Exception as e:
        print(f"GNN mol prediction error: {e}")
        return "organic"


def load_model_and_pred_ligand(protein_pdb, ligand_pdb, LGBM_Model_package=None):
    """Binary classification: is this a real ligand?

    Returns 0 (not ligand) or 1 (ligand).
    """
    try:
        from ligandexplorer.workflow import ModelContainer
        model = ModelContainer.ligand_classifier
        device = ModelContainer.device
        if model is None:
            raise RuntimeError("ligand_classifier not loaded")

        graph = complex_to_graph(protein_pdb, ligand_pdb,
                                 pocket_cutoff=6.0, edge_cutoff=5.0,
                                 max_neighbors=32)
        if graph is None or graph.z.size(0) < 3:
            return 0

        graph = graph.to(device)
        graph.batch = torch.zeros(graph.z.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(graph)
            pred = out.argmax(dim=-1).item()

        return pred
    except Exception as e:
        print(f"GNN ligand prediction error: {e}")
        return 0
