"""
Ion template registry for pre-classification rule.

Maintains a library of known ion topological structures. For molecules with
atom count <= IONS_MAX_ATOMS, graph isomorphism is checked against templates
before invoking GNN inference. Matches are directly classified as "ions".

Two-stage matching:
  1. Formula fingerprint (sorted element composition) for O(1) candidate filtering
  2. networkx graph isomorphism with node_match on atomic number
"""
import networkx as nx
from collections import Counter

# ============================================================
# Single-atom ion elements (atomic numbers)
# ============================================================
SINGLE_ATOM_ION_Z = frozenset([
    # Alkali metals
    3, 11, 19, 37, 55,
    # Alkaline earth metals
    4, 12, 20, 38, 56,
    # Transition metals
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    # Post-transition metals
    13, 31, 49, 50, 81, 82, 83,
    # Lanthanides
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    # Halides
    9, 17, 35, 53,
])


def _formula_key(z_list):
    """Create a hashable formula key from a list of atomic numbers."""
    return tuple(sorted(Counter(z_list).items()))


def _build_nx_graph(z_list, edges):
    """Build a networkx graph from atomic numbers and edge list."""
    G = nx.Graph()
    for i, z in enumerate(z_list):
        G.add_node(i, z=z)
    for i, j in edges:
        G.add_edge(i, j)
    return G


def _node_match(n1, n2):
    """Node match function for isomorphism: atomic numbers must be equal."""
    return n1['z'] == n2['z']


# ============================================================
# Multi-atom ion templates (manually defined)
# ============================================================
_MANUAL_TEMPLATES = [
    # Diatomic
    ("CN", [6, 7], [(0, 1)]),
    ("CO", [6, 8], [(0, 1)]),
    ("NO", [7, 8], [(0, 1)]),
    ("O2", [8, 8], [(0, 1)]),

    # Triatomic
    ("SCN", [16, 6, 7], [(0, 1), (1, 2)]),
    ("N3_azide", [7, 7, 7], [(0, 1), (1, 2)]),
    ("NO2", [7, 8, 8], [(0, 1), (0, 2)]),
    ("O3", [8, 8, 8], [(0, 1), (1, 2)]),
    ("CS2", [6, 16, 16], [(0, 1), (0, 2)]),
    ("formate", [6, 8, 8], [(0, 1), (0, 2)]),

    # 4-atom
    ("SO3", [16, 8, 8, 8], [(0, 1), (0, 2), (0, 3)]),
    ("CO3", [6, 8, 8, 8], [(0, 1), (0, 2), (0, 3)]),
    ("NO3", [7, 8, 8, 8], [(0, 1), (0, 2), (0, 3)]),
    ("PO3", [15, 8, 8, 8], [(0, 1), (0, 2), (0, 3)]),
    ("BF3", [5, 9, 9, 9], [(0, 1), (0, 2), (0, 3)]),
    ("ClO3", [17, 8, 8, 8], [(0, 1), (0, 2), (0, 3)]),

    # 5-atom star
    ("SO4", [16, 8, 8, 8, 8], [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("PO4", [15, 8, 8, 8, 8], [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("ClO4", [17, 8, 8, 8, 8], [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("CrO4", [24, 8, 8, 8, 8], [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("MnO4", [25, 8, 8, 8, 8], [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("BF4", [5, 9, 9, 9, 9], [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("SiO4", [14, 8, 8, 8, 8], [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("VO4", [23, 8, 8, 8, 8], [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("WO4", [74, 8, 8, 8, 8], [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("MoO4", [42, 8, 8, 8, 8], [(0, 1), (0, 2), (0, 3), (0, 4)]),

    # 7-atom octahedral
    ("PF6", [15, 9, 9, 9, 9, 9, 9],
     [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]),
    ("SbF6", [51, 9, 9, 9, 9, 9, 9],
     [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]),

    # Pyrophosphate P2O7 (bridged)
    ("P2O7", [15, 15, 8, 8, 8, 8, 8, 8, 8],
     [(0, 2), (0, 3), (0, 4), (0, 8), (1, 5), (1, 6), (1, 7), (1, 8)]),
]


class IonRegistry:
    """Registry of known ion topological templates."""

    def __init__(self):
        self._formula_to_templates = {}
        self._build_manual_templates()

    def _build_manual_templates(self):
        for name, z_list, edges in _MANUAL_TEMPLATES:
            if not z_list or any(z == 0 for z in z_list):
                continue
            self.add_template(z_list, edges)

    def add_template(self, z_list, edges):
        """Add a template to the registry."""
        key = _formula_key(z_list)
        G = _build_nx_graph(z_list, edges)
        if key not in self._formula_to_templates:
            self._formula_to_templates[key] = []
        for existing_G in self._formula_to_templates[key]:
            if nx.is_isomorphic(G, existing_G, node_match=_node_match):
                return
        self._formula_to_templates[key].append(G)

    def add_template_from_pdb(self, pdb_path):
        """Extract ion template from a PDB file and add to registry."""
        from ligandexplorer.utilities.gnn_data import parse_pdb_with_conect, \
            _infer_covalent_pairs
        import numpy as np

        atoms, conect = parse_pdb_with_conect(pdb_path)
        if not atoms or len(atoms) > 10:
            return

        z_list = [a[0] for a in atoms]
        coords = np.array([[a[1], a[2], a[3]] for a in atoms], dtype=np.float64)
        serials = [a[4] for a in atoms]

        serial_to_idx = {s: i for i, s in enumerate(serials)}
        edges = set()
        for s1, s2 in conect:
            if s1 in serial_to_idx and s2 in serial_to_idx:
                i, j = serial_to_idx[s1], serial_to_idx[s2]
                if i != j:
                    edges.add(tuple(sorted((i, j))))
        if not edges and len(z_list) > 1:
            z_np = np.array(z_list)
            edges = _infer_covalent_pairs(z_np, coords)

        self.add_template(z_list, list(edges))

    def match(self, z_list, edge_index_np):
        """Check if a molecule matches any ion template.

        Args:
            z_list: list of atomic numbers
            edge_index_np: (2, E) numpy array of edge indices, or None

        Returns:
            True if the molecule matches a known ion template
        """
        n = len(z_list)
        if n == 0:
            return False

        if n == 1:
            return z_list[0] in SINGLE_ATOM_ION_Z

        # Multi-atom: all single-atom-ion elements with no bonds → ion cluster
        if edge_index_np is None or edge_index_np.shape[1] == 0:
            if all(z in SINGLE_ATOM_ION_Z for z in z_list):
                return True

        # Formula-based lookup
        key = _formula_key(z_list)
        candidates = self._formula_to_templates.get(key)
        if candidates is None:
            if all(z in SINGLE_ATOM_ION_Z for z in z_list):
                if edge_index_np is None or edge_index_np.shape[1] == 0:
                    return True
            return False

        # Extract unique undirected edges from edge_index
        edges = set()
        if edge_index_np is not None and edge_index_np.shape[1] > 0:
            for k in range(edge_index_np.shape[1]):
                i, j = int(edge_index_np[0, k]), int(edge_index_np[1, k])
                edges.add(tuple(sorted((i, j))))

        G_input = _build_nx_graph(z_list, list(edges))

        for G_template in candidates:
            if G_input.number_of_nodes() != G_template.number_of_nodes():
                continue
            if nx.is_isomorphic(G_input, G_template, node_match=_node_match):
                return True

        return False

    @property
    def num_templates(self):
        return sum(len(v) for v in self._formula_to_templates.values())


_REGISTRY = None


def get_registry():
    """Get or create the global ion registry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = IonRegistry()
    return _REGISTRY


def load_templates_from_directory(ions_dir):
    """Load all ion templates from a directory of PDB files."""
    import os
    registry = get_registry()
    for f in sorted(os.listdir(ions_dir)):
        if f.endswith(".pdb"):
            registry.add_template_from_pdb(os.path.join(ions_dir, f))
