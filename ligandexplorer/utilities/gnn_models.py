"""
GNN models for molecule classification.

MoleculeClassifier  (Task A): lightweight 8-class single-molecule classifier
  - 6 interaction layers, hidden=64   (small graphs: 1–500 atoms)
  - mean + max pooling readout

LigandBinaryClassifier (Task B): protein-ligand binary classifier
  - 6 interaction layers, hidden=96   (complex graphs: up to ~850 atoms)
  - dual-channel (ligand + protein) × (mean + sum + max) readout

Both use a SchNet backbone with residue-boundary edge features.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.models.schnet import GaussianSmearing, ShiftedSoftplus


class BoundaryAwareInteraction(nn.Module):
    """SchNet InteractionBlock extended with residue boundary embedding."""

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff,
                 boundary_dim=16):
        super().__init__()
        edge_input_dim = num_gaussians + boundary_dim
        self.mlp_edge = nn.Sequential(
            nn.Linear(edge_input_dim, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.lin1 = nn.Linear(hidden_channels, num_filters)
        self.lin2 = nn.Linear(num_filters, hidden_channels)
        self.act = ShiftedSoftplus()
        self.boundary_emb = nn.Embedding(2, boundary_dim)

    def forward(self, x, edge_index, edge_attr_rbf, residue_boundary):
        bound_emb = self.boundary_emb(residue_boundary)
        edge_feat = torch.cat([edge_attr_rbf, bound_emb], dim=-1)
        W = self.mlp_edge(edge_feat)

        row, col = edge_index
        x_j = self.lin1(x)
        x_j = x_j[col] * W

        out = torch.zeros_like(x)
        out.index_add_(0, row, x_j)

        out = self.act(out)
        out = self.lin2(out)
        return out


# ---------------------------------------------------------------------------
#  Task A — lightweight single-molecule classifier
# ---------------------------------------------------------------------------

class MoleculeClassifier(nn.Module):
    """Lightweight SchNet classifier for single-molecule type prediction.

    Defaults tuned for small graphs (1–500 atoms):
      6 layers, hidden=64, mean+max readout.
    """

    def __init__(self, num_classes=8, hidden_channels=64, num_filters=64,
                 num_interactions=6, num_gaussians=50, cutoff=5.0,
                 boundary_dim=16):
        super().__init__()
        self.cutoff = cutoff
        self.atom_embedding = nn.Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = nn.ModuleList([
            BoundaryAwareInteraction(
                hidden_channels, num_gaussians, num_filters, cutoff, boundary_dim
            )
            for _ in range(num_interactions)
        ])

        pool_dim = hidden_channels * 2  # mean + max
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels),
            ShiftedSoftplus(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            ShiftedSoftplus(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, data):
        z = data.z
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        residue_boundary = data.residue_boundary
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        h = self.atom_embedding(z)
        edge_rbf = self.distance_expansion(edge_attr)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_rbf, residue_boundary)

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_pooled = torch.cat([h_mean, h_max], dim=-1)
        return self.classifier(h_pooled)


# ---------------------------------------------------------------------------
#  Task B — full-size protein-ligand complex classifier
# ---------------------------------------------------------------------------

class LigandBinaryClassifier(nn.Module):
    """SchNet binary classifier with dual-channel 6-pool readout.

    Defaults tuned for complex graphs (up to ~850 atoms):
      6 layers, hidden=96, lig(mean+sum+max) + prot(mean+sum+max).
    """

    def __init__(self, hidden_channels=96, num_filters=96,
                 num_interactions=6, num_gaussians=50, cutoff=5.0,
                 boundary_dim=16, node_type_dim=16):
        super().__init__()
        self.cutoff = cutoff
        self.atom_embedding = nn.Embedding(100, hidden_channels - node_type_dim)
        self.node_type_embedding = nn.Embedding(2, node_type_dim)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = nn.ModuleList([
            BoundaryAwareInteraction(
                hidden_channels, num_gaussians, num_filters, cutoff, boundary_dim
            )
            for _ in range(num_interactions)
        ])

        pool_dim = hidden_channels * 6  # 2 channels × 3 pools
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels * 2),
            ShiftedSoftplus(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels),
            ShiftedSoftplus(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, 2),
        )

    @staticmethod
    def _masked_pools(h, mask, batch, num_graphs):
        """Compute mean, sum, max pools for nodes where *mask* is True."""
        h_masked = h * mask.unsqueeze(1).float()

        pool_sum = global_add_pool(h_masked, batch)

        counts = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch, mask.float())
        pool_mean = pool_sum / counts.clamp(min=1).unsqueeze(1)

        h_for_max = h.clone()
        h_for_max[~mask] = -1e9
        pool_max = global_max_pool(h_for_max, batch)
        pool_max = torch.where(pool_max <= -1e9 + 1,
                               torch.zeros_like(pool_max), pool_max)

        return pool_mean, pool_sum, pool_max

    def forward(self, data):
        z = data.z
        node_type = data.node_type
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        residue_boundary = data.residue_boundary
        ligand_mask = data.ligand_mask
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        h_atom = self.atom_embedding(z)
        h_type = self.node_type_embedding(node_type)
        h = torch.cat([h_atom, h_type], dim=-1)

        edge_rbf = self.distance_expansion(edge_attr)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_rbf, residue_boundary)

        num_graphs = batch.max().item() + 1
        lig_mean, lig_sum, lig_max = self._masked_pools(
            h, ligand_mask, batch, num_graphs)
        prot_mean, prot_sum, prot_max = self._masked_pools(
            h, ~ligand_mask, batch, num_graphs)

        h_pooled = torch.cat([lig_mean, lig_sum, lig_max,
                              prot_mean, prot_sum, prot_max], dim=-1)
        return self.classifier(h_pooled)
