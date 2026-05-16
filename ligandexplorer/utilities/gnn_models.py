"""
GNN models for molecule classification (v3e).

MoleculeClassifier (Task A):
  - Node features: z embedding + degree embedding + ring_count embedding
  - Edge features: GaussianSmearing distance + edge_type embedding
  - Graph features (bond-order aware, 32-dim func_groups):
      amide_count embedding:       64 dims
      elem_ratios MLP (5->64):     64 dims
      func_groups MLP (32->96):    96 dims  (26 bond-order / ring / size features
                                              + 6 backbone topology features from
                                              explicit peptide chain tracing)
      is_backbone_cyclic embedding: 16 dims
      Total graph features:        240 dims
  - pool(256) + graph(240) = 496 -> graph features are ~48% of input
  - Pooling: global mean + max (stable)

LigandBinaryClassifier (Task B):
  - Spatial graph at 5.0A cutoff + covalent edges
  - Node features: z + node_type embedding
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.models.schnet import GaussianSmearing, ShiftedSoftplus


class InteractionBlock(nn.Module):
    """SchNet interaction with binary edge-type embedding."""

    def __init__(self, hidden_channels, num_gaussians, num_filters,
                 num_edge_types=2, edge_type_dim=8):
        super().__init__()
        self.mlp_edge = nn.Sequential(
            nn.Linear(num_gaussians + edge_type_dim, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.lin1 = nn.Linear(hidden_channels, num_filters)
        self.lin2 = nn.Linear(num_filters, hidden_channels)
        self.act = ShiftedSoftplus()
        self.edge_type_emb = nn.Embedding(num_edge_types, edge_type_dim)

    def forward(self, x, edge_index, edge_rbf, edge_type):
        et_emb = self.edge_type_emb(edge_type.clamp(min=0, max=1))
        W = self.mlp_edge(torch.cat([edge_rbf, et_emb], dim=-1))
        row, col = edge_index
        x_j = self.lin1(x)[col] * W
        out = torch.zeros_like(x)
        out.index_add_(0, row, x_j)
        return self.lin2(self.act(out))


class MoleculeClassifier(nn.Module):
    """Task A classifier with bond-order-aware graph-level features.

    Architecture rationale:
      - Node-level features (via message passing + pooling): 256 dims (mean+max)
      - Graph-level features: 208 dims (amide 64 + elem 64 + func 64 + cyclic 16)
      - Total: 464 dims → graph features are ~45% of decision input
    """

    FUNC_DIM = 32

    def __init__(self, num_classes=8, hidden_channels=128, num_filters=128,
                 num_interactions=6, num_gaussians=50, cutoff=3.0,
                 edge_type_dim=8, degree_dim=16, ring_dim=16,
                 amide_emb_dim=64, elem_hidden_dim=64,
                 func_hidden_dim=96, cyclic_emb_dim=16,
                 node_mask_rate=0.0, hidden_drop_rate=0.0):
        super().__init__()
        self.cutoff = cutoff
        self.node_mask_rate = node_mask_rate

        atom_emb_dim = hidden_channels - degree_dim - ring_dim
        self.atom_embedding = nn.Embedding(100, atom_emb_dim)
        self.degree_embedding = nn.Embedding(7, degree_dim)   # 0..6
        self.ring_embedding = nn.Embedding(9, ring_dim)       # 0..8

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters,
                             num_edge_types=2, edge_type_dim=edge_type_dim)
            for _ in range(num_interactions)
        ])
        self.hidden_dropout = nn.Dropout(hidden_drop_rate) if hidden_drop_rate > 0 else None

        # Graph-level feature processing
        self.amide_embedding = nn.Embedding(64, amide_emb_dim)
        self.elem_mlp = nn.Sequential(
            nn.Linear(5, 32),
            ShiftedSoftplus(),
            nn.Linear(32, elem_hidden_dim),
        )
        self.func_mlp = nn.Sequential(
            nn.Linear(self.FUNC_DIM, 64),
            ShiftedSoftplus(),
            nn.Linear(64, func_hidden_dim),
        )
        self.cyclic_embedding = nn.Embedding(2, cyclic_emb_dim)

        pool_dim = hidden_channels * 2  # 256
        graph_feat_dim = amide_emb_dim + elem_hidden_dim + func_hidden_dim + cyclic_emb_dim  # 208
        total_dim = pool_dim + graph_feat_dim  # 464

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 320),
            ShiftedSoftplus(),
            nn.Dropout(0.15),
            nn.Linear(320, 160),
            ShiftedSoftplus(),
            nn.Dropout(0.1),
            nn.Linear(160, num_classes),
        )

    def forward(self, data):
        z = data.z
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        degree = getattr(data, 'degree', None)
        if degree is None:
            degree = torch.zeros_like(z)
        ring_count = getattr(data, 'ring_count', None)
        if ring_count is None:
            ring_count = torch.zeros_like(z)

        h = torch.cat([
            self.atom_embedding(z),
            self.degree_embedding(degree.clamp(0, 6)),
            self.ring_embedding(ring_count.clamp(0, 8)),
        ], dim=-1)

        if self.training and self.node_mask_rate > 0:
            mask = (torch.rand(h.size(0), 1, device=h.device) >= self.node_mask_rate).float()
            h = h * mask

        if edge_index.size(1) > 0:
            edge_rbf = self.distance_expansion(edge_attr)
            edge_type = getattr(data, "edge_type", None)
            if edge_type is None:
                edge_type = torch.zeros(edge_attr.size(0), dtype=torch.long, device=edge_attr.device)
            for interaction in self.interactions:
                delta = interaction(h, edge_index, edge_rbf, edge_type)
                if self.training and self.hidden_dropout is not None:
                    delta = self.hidden_dropout(delta)
                h = h + delta

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        pool_out = torch.cat([h_mean, h_max], dim=-1)

        ng = h_mean.size(0)

        amide_count = getattr(data, 'amide_count', None)
        if amide_count is None:
            amide_count = torch.zeros(ng, dtype=torch.long, device=h.device)
        else:
            if amide_count.dim() == 1:
                if amide_count.size(0) != ng:
                    amide_count = global_add_pool(
                        amide_count.float().unsqueeze(1), batch
                    ).squeeze(1).long()
            else:
                amide_count = amide_count.squeeze(-1)
        amide_emb = self.amide_embedding(amide_count.clamp(0, 63))

        elem_ratios = getattr(data, 'elem_ratios', None)
        if elem_ratios is None:
            elem_ratios = torch.zeros(ng, 5, device=h.device)
        else:
            if elem_ratios.dim() == 1:
                elem_ratios = elem_ratios.view(ng, 5)
        elem_feat = self.elem_mlp(elem_ratios)

        func_groups = getattr(data, 'func_groups', None)
        if func_groups is None:
            func_groups = torch.zeros(ng, self.FUNC_DIM, device=h.device)
        else:
            if func_groups.dim() == 1:
                func_groups = func_groups.view(ng, self.FUNC_DIM)
        func_feat = self.func_mlp(func_groups)

        is_cyclic = getattr(data, 'is_backbone_cyclic', None)
        if is_cyclic is None:
            is_cyclic = torch.zeros(ng, dtype=torch.long, device=h.device)
        else:
            if is_cyclic.dim() > 1:
                is_cyclic = is_cyclic.squeeze(-1)
            if is_cyclic.size(0) != ng:
                is_cyclic = global_add_pool(
                    is_cyclic.float().unsqueeze(1), batch
                ).squeeze(1).clamp(0, 1).long()
        cyclic_emb = self.cyclic_embedding(is_cyclic.clamp(0, 1))

        graph_feat = torch.cat([amide_emb, elem_feat, func_feat, cyclic_emb], dim=-1)
        return self.classifier(torch.cat([pool_out, graph_feat], dim=-1))


class LigandBinaryClassifier(nn.Module):
    """Task B: protein-ligand binary classifier (spatial 5.0A + covalent)."""

    def __init__(self, hidden_channels=96, num_filters=96,
                 num_interactions=6, num_gaussians=50, cutoff=5.0,
                 edge_type_dim=8, node_type_dim=16,
                 node_mask_rate=0.0, hidden_drop_rate=0.0):
        super().__init__()
        self.cutoff = cutoff
        self.node_mask_rate = node_mask_rate
        self.atom_embedding = nn.Embedding(100, hidden_channels - node_type_dim)
        self.node_type_embedding = nn.Embedding(2, node_type_dim)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters,
                             num_edge_types=2, edge_type_dim=edge_type_dim)
            for _ in range(num_interactions)
        ])
        self.hidden_dropout = nn.Dropout(hidden_drop_rate) if hidden_drop_rate > 0 else None
        pool_dim = hidden_channels * 6
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
    def _masked_pools(h, mask, batch, ng):
        h_m = h * mask.unsqueeze(1).float()
        p_sum = global_add_pool(h_m, batch)
        cnt = torch.zeros(ng, device=h.device, dtype=h.dtype)
        cnt.index_add_(0, batch, mask.float())
        p_mean = p_sum / cnt.clamp(min=1).unsqueeze(1)
        h_mx = h.clone(); h_mx[~mask] = -1e9
        p_max = global_max_pool(h_mx, batch)
        p_max = torch.where(p_max <= -1e9 + 1, torch.zeros_like(p_max), p_max)
        return p_mean, p_sum, p_max

    def forward(self, data):
        h = torch.cat([self.atom_embedding(data.z),
                       self.node_type_embedding(data.node_type)], -1)
        if self.training and self.node_mask_rate > 0:
            mask = (torch.rand(h.size(0), 1, device=h.device) >= self.node_mask_rate).float()
            h = h * mask
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(data.z.size(0), dtype=torch.long, device=data.z.device)

        if data.edge_index.size(1) > 0:
            edge_rbf = self.distance_expansion(data.edge_attr)
            edge_type = getattr(data, "edge_type", None)
            if edge_type is None:
                edge_type = torch.zeros(data.edge_attr.size(0), dtype=torch.long, device=data.edge_attr.device)
            for interaction in self.interactions:
                delta = interaction(h, data.edge_index, edge_rbf, edge_type)
                if self.training and self.hidden_dropout is not None:
                    delta = self.hidden_dropout(delta)
                h = h + delta

        ng = batch.max().item() + 1
        lig = data.ligand_mask
        lm, ls, lx = self._masked_pools(h, lig, batch, ng)
        pm, ps, px = self._masked_pools(h, ~lig, batch, ng)
        return self.classifier(torch.cat([lm, ls, lx, pm, ps, px], -1))


class PeptideSubClassifier(nn.Module):
    """Lightweight MLP for peptide/peptide_like/cyclic_peptide refinement.

    Input: 39-dim feature vector (func_groups[32] + elem_ratios[5] + log1p_amide[1] + is_cyclic[1])
    Output: 3-class logits
    """
    FEAT_DIM = 39

    def __init__(self, in_dim=39, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
