"""
Two-stage GNN-based molecule classification.

Stage 1: Main GNN classifies into 8 categories with chemical rule guardrails.
Stage 2: For peptide-group predictions, a specialized MLP sub-model refines
          into peptide / peptide_like / cyclic_peptide.

Provides the same function signatures:
  load_model_and_pred(input_pdb, LGBM_Model_package) -> str
  load_model_and_pred_ligand(protein_pdb, ligand_pdb, LGBM_Model_package) -> int

Supports two execution modes (transparent to callers):
  - CPU direct: model lives in worker process, inference runs directly.
  - CUDA proxy: graph is built on CPU, serialised to the GPU daemon
                 process via multiprocessing.Queue.
"""
import uuid
import numpy as np
import torch
from ligandexplorer.utilities.gnn_data import structure_to_graph, complex_to_graph
from ligandexplorer.utilities.ion_registry import get_registry

CATEGORIES = ["peptide", "peptide_like", "gly", "rna", "dna", "mem",
              "organic", "cyclic_peptide"]
SUB_CATEGORIES = ["peptide", "peptide_like", "cyclic_peptide"]

PEPTIDE_INDICES = {0, 1, 7}
DNA_RNA_INDICES = {3, 4}

IONS_MAX_ATOMS = 10


def match_ion_template(graph):
    """Check if a graph matches a known ion topology template."""
    z_list = graph.z.cpu().tolist()
    edge_index = graph.edge_index
    if edge_index is not None and edge_index.numel() > 0:
        edge_type = getattr(graph, 'edge_type', None)
        if edge_type is not None:
            cov_mask = edge_type == 1
            ei_np = edge_index[:, cov_mask].cpu().numpy()
        else:
            ei_np = edge_index.cpu().numpy()
    else:
        ei_np = np.zeros((2, 0), dtype=np.int64)
    return get_registry().match(z_list, ei_np)


def _apply_chemical_rules(probs, amide_count, elem_ratios, func_groups):
    """Apply hard chemical constraints to mask impossible classes.

    Rules:
      - amide_count == 0 -> cannot be peptide/peptide_like/cyclic_peptide
      - no phosphorus and no phosphoester -> cannot be DNA/RNA
    """
    mask = np.ones(len(CATEGORIES), dtype=bool)

    if amide_count == 0:
        for idx in PEPTIDE_INDICES:
            mask[idx] = False

    p_ratio = elem_ratios[4]
    phosphoester = func_groups[4]
    if p_ratio == 0 and phosphoester == 0:
        for idx in DNA_RNA_INDICES:
            mask[idx] = False

    masked = probs * mask
    total = masked.sum()
    if total > 0:
        masked /= total
    else:
        masked = probs
    return masked


def _run_sub_model(sub_model, norm_mean, norm_std, func_groups, elem_ratios,
                   amide_count, is_backbone_cyclic, device):
    """Run the peptide sub-classifier on hand-crafted features."""
    ac = np.log1p(float(amide_count))
    bc = float(is_backbone_cyclic)
    feat = np.concatenate([func_groups, elem_ratios, [ac, bc]]).astype(np.float32)
    feat_norm = (feat - norm_mean) / norm_std
    feat_t = torch.from_numpy(feat_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sub_model(feat_t)
        probs = torch.softmax(logits, dim=-1).cpu().squeeze().numpy()
    del feat_t, logits
    return SUB_CATEGORIES[probs.argmax()]


def _graph_to_payload(graph):
    """Convert a PyG Data object to a picklable dict of numpy arrays."""
    payload = {}
    for key in ['z', 'pos', 'edge_index', 'edge_attr', 'edge_type',
                'node_type', 'ligand_mask',
                'func_groups', 'elem_ratios']:
        val = getattr(graph, key, None)
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            payload[key] = val.cpu().numpy()
        else:
            payload[key] = val
    payload['amide_count'] = int(graph.amide_count.item()) if hasattr(graph, 'amide_count') else 0
    payload['is_backbone_cyclic'] = int(graph.is_backbone_cyclic.item()) if hasattr(graph, 'is_backbone_cyclic') else 0
    return payload


def _proxy_predict(task_type, payload):
    """Send prediction request to GPU daemon and wait for response."""
    from ligandexplorer.workflow import ModelContainer
    req_id = str(uuid.uuid4())
    reply_q = ModelContainer.worker_reply_queue
    ModelContainer.request_queue.put((req_id, task_type, payload, reply_q))
    resp_id, result = reply_q.get(timeout=120)
    assert resp_id == req_id, f"Response ID mismatch: {resp_id} != {req_id}"
    return result


def load_model_and_pred(input_pdb, LGBM_Model_package=None):
    """Classify molecule type (two-stage).

    Stage 1: GNN + chemical rules -> 8 classes
    Stage 2: if peptide group, sub-model refines -> peptide/peptide_like/cyclic_peptide
    Ions are detected by pre-classification rule (graph isomorphism).
    """
    try:
        from ligandexplorer.workflow import ModelContainer

        graph = structure_to_graph(input_pdb)
        if graph is None or graph.z.size(0) == 0:
            print(f"[WARNING] Failed to build graph for {input_pdb}, defaulting to organic")
            return "organic"

        n_atoms = graph.z.size(0)

        if n_atoms <= IONS_MAX_ATOMS:
            if match_ion_template(graph):
                return "ions"

        if ModelContainer.device == 'cuda_proxy':
            return _proxy_predict('mol', _graph_to_payload(graph))

        # CPU direct mode
        model = ModelContainer.mol_classifier
        device = ModelContainer.device
        if model is None:
            raise RuntimeError("mol_classifier not loaded")

        graph = graph.to(device)
        graph.batch = torch.zeros(graph.z.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(graph)
            probs = torch.softmax(out, dim=-1).cpu().squeeze().numpy()

        amide_count = int(graph.amide_count.cpu().item())
        elem_ratios = graph.elem_ratios.cpu().numpy()
        func_groups = graph.func_groups.cpu().numpy()

        ruled_probs = _apply_chemical_rules(probs, amide_count, elem_ratios, func_groups)
        pred = ruled_probs.argmax()
        pred_label = CATEGORIES[pred]

        if pred in PEPTIDE_INDICES and ModelContainer.peptide_sub_classifier is not None:
            is_cyclic = int(graph.is_backbone_cyclic.cpu().item())
            pred_label = _run_sub_model(
                ModelContainer.peptide_sub_classifier,
                ModelContainer.peptide_sub_norm_mean,
                ModelContainer.peptide_sub_norm_std,
                func_groups, elem_ratios, amide_count, is_cyclic, device)

        del graph
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return pred_label
    except Exception as e:
        print(f"[WARNING] GNN mol prediction error for {input_pdb}: {e}")
        return "organic"


def load_model_and_pred_ligand(protein_pdb, ligand_pdb, LGBM_Model_package=None):
    """Binary classification: is this a real ligand?

    Returns 0 (not ligand) or 1 (ligand).
    """
    try:
        from ligandexplorer.workflow import ModelContainer

        graph = complex_to_graph(protein_pdb, ligand_pdb,
                                 pocket_cutoff=6.0, edge_cutoff=5.0,
                                 max_neighbors=32)
        if graph is None or graph.z.size(0) < 3:
            return 0

        if ModelContainer.device == 'cuda_proxy':
            return _proxy_predict('ligand', _graph_to_payload(graph))

        # CPU direct mode
        model = ModelContainer.ligand_classifier
        device = ModelContainer.device
        if model is None:
            return 0

        graph = graph.to(device)
        graph.batch = torch.zeros(graph.z.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(graph)
            pred = out.argmax(dim=-1).item()

        del graph, out
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return pred
    except Exception as e:
        print(f"GNN ligand prediction error: {e}")
        return 0
