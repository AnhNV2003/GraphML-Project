"""Sheaf-based recommender models.

This is a compact PyTorch implementation of neural sheaf diffusion for the
project pipeline. It follows the Sheaf4Rec/NSD formulation at the level needed
for experiments: learned restriction maps, sparse block sheaf Laplacian, and
diffusion updates over a bipartite user-item graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import BPRModelBase


RESTRICTION_ALIASES = {
    "scalar": "scalar",
    "diagonal": "diagonal",
    "attention": "attention",
    "general": "general",
    "identity": "identity",
    "gcn_like": "identity",   # trivial sheaf: fixed I_d, no learning → reduces to plain graph Laplacian
    "gat_like": "attention",
    "full_sheaf": "general",
}


class RestrictionLearner(nn.Module):
    """MLP mapping a directed edge pair [x_v, x_w] to F_{v<e}."""

    def __init__(self, stalk_dim: int, restriction_type: str, dropout: float = 0.0):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.restriction_type = RESTRICTION_ALIASES[restriction_type]

        if self.restriction_type == "identity":
            # No learnable parameters — trivial sheaf (gcn_like reduction)
            return

        if self.restriction_type in {"scalar", "attention"}:
            out_dim = 1
        elif self.restriction_type == "diagonal":
            out_dim = stalk_dim
        elif self.restriction_type == "general":
            out_dim = stalk_dim * stalk_dim
        else:
            raise ValueError(f"Unsupported restriction_type: {restriction_type}")

        hidden_dim = max(8, 2 * stalk_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * stalk_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, source_x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        batch_size = source_x.shape[0]
        eye = torch.eye(self.stalk_dim, device=source_x.device, dtype=source_x.dtype).unsqueeze(0)

        if self.restriction_type == "identity":
            return eye.expand(batch_size, -1, -1)

        raw = self.net(torch.cat([source_x, target_x], dim=-1))
        if self.restriction_type == "scalar":
            scale = F.softplus(raw).view(batch_size, 1, 1) + 1e-6
            return scale * eye
        if self.restriction_type == "attention":
            scale = torch.sigmoid(raw).view(batch_size, 1, 1) + 1e-6
            return scale * eye
        if self.restriction_type == "diagonal":
            diag = F.softplus(raw) + 1e-6
            return torch.diag_embed(diag)
        return raw.view(batch_size, self.stalk_dim, self.stalk_dim)


class Sheaf4Rec(BPRModelBase):
    """Neural sheaf diffusion recommender with learned restriction maps."""

    def __init__(self, config, dataset, restriction_type: str | None = None):
        super().__init__()
        self.n_user, self.m_item = dataset.n_users, dataset.m_items
        self.n_nodes = self.n_user + self.m_item
        self.stalk_dim = int(config.get("sheaf_stalk_dim", 4))
        self.n_layers = int(config.get("sheaf_n_layers", config.get("lightGCN_n_layers", 3)))
        requested_type = restriction_type or config.get("sheaf_restriction_type", "full_sheaf")
        if requested_type not in RESTRICTION_ALIASES:
            raise ValueError(
                f"Unknown sheaf restriction_type {requested_type!r}; "
                f"expected one of {sorted(RESTRICTION_ALIASES)}"
            )
        self.restriction_type = requested_type
        self.dropout = float(config.get("sheaf_dropout", 0.0))
        self.embedding = nn.Embedding(self.n_nodes, self.stalk_dim)
        self.W1 = nn.Linear(self.stalk_dim, self.stalk_dim, bias=False)
        self.learner = RestrictionLearner(self.stalk_dim, requested_type, dropout=self.dropout)
        nn.init.normal_(self.embedding.weight, std=0.1)
        nn.init.eye_(self.W1.weight)

        edge_index = dataset.Graph.indices()
        edge_values = dataset.Graph.values()
        src, dst = edge_index[0], edge_index[1]
        # Use one orientation per bipartite edge; block construction adds both directions.
        mask = src < dst
        self.register_buffer("edge_src", src[mask].long())
        self.register_buffer("edge_dst", dst[mask].long())
        self.register_buffer("edge_weight", edge_values[mask].float())

    def _sheaf_diffuse(self, x_maps: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        """Compute D^{-1/2} δ^T δ D^{-1/2} x_input without materialising the dense Laplacian.

        Using direct edge-level scatter instead of torch.sparse.mm avoids the
        O(n_nodes^2 * d^2) dense gradient that sparse.mm backward would generate.
        """
        d = self.stalk_dim
        src, dst = self.edge_src, self.edge_dst
        n_edges = src.numel()
        if n_edges == 0:
            return torch.zeros_like(x_input)

        f_src = self.learner(x_maps[src], x_maps[dst])   # (n_edges, d, d)
        f_dst = self.learner(x_maps[dst], x_maps[src])   # (n_edges, d, d)
        edge_scale = torch.sqrt(torch.clamp(self.edge_weight.to(x_maps.device, x_maps.dtype), min=1e-8))
        f_src = f_src * edge_scale.view(-1, 1, 1)
        f_dst = f_dst * edge_scale.view(-1, 1, 1)

        # Scalar degree per node: trace(D_v) = sum_e trace(F_{v◁e}^T F_{v◁e})
        # Detach so normalization is treated as a stable constant (NSD convention).
        with torch.no_grad():
            diag_ss = torch.diagonal(torch.bmm(f_src.transpose(1, 2), f_src), dim1=1, dim2=2).sum(1)
            diag_dd = torch.diagonal(torch.bmm(f_dst.transpose(1, 2), f_dst), dim1=1, dim2=2).sum(1)
            degree = torch.zeros(self.n_nodes, device=x_maps.device, dtype=x_maps.dtype)
            degree = degree.scatter_add(0, src, diag_ss).scatter_add(0, dst, diag_dd)
            norm = torch.rsqrt(torch.clamp(degree, min=1e-8))   # (n_nodes,)

        # D^{-1/2} x_input
        xn = x_input * norm.unsqueeze(1)                        # (n_nodes, d)

        # Co-boundary per edge: F_src xn_src − F_dst xn_dst  →  (n_edges, d)
        Fxn_src = torch.bmm(f_src, xn[src].unsqueeze(-1)).squeeze(-1)
        Fxn_dst = torch.bmm(f_dst, xn[dst].unsqueeze(-1)).squeeze(-1)
        coboundary = Fxn_src - Fxn_dst

        # Adjoint: F_v^T coboundary scattered to each node
        FT_src = torch.bmm(f_src.transpose(1, 2), coboundary.unsqueeze(-1)).squeeze(-1)   # (n_edges, d)
        FT_dst = torch.bmm(f_dst.transpose(1, 2), (-coboundary).unsqueeze(-1)).squeeze(-1)

        # Scatter-add (out-of-place, differentiable)
        idx_src = src.unsqueeze(1).expand_as(FT_src)
        idx_dst = dst.unsqueeze(1).expand_as(FT_dst)
        Lx = (
            torch.zeros(self.n_nodes, d, device=x_maps.device, dtype=x_maps.dtype)
            .scatter_add(0, idx_src, FT_src)
            .scatter_add(0, idx_dst, FT_dst)
        )

        # D^{-1/2} again
        return Lx * norm.unsqueeze(1)

    def forward(self):
        x = self.embedding.weight
        states = [x]
        for _ in range(self.n_layers):
            transformed = self.W1(x)                    # (n_nodes, d) — apply W1 within stalk
            delta = self._sheaf_diffuse(x, transformed) # Δ_F (I⊗W1) x
            x = x - torch.tanh(delta)
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
            states.append(x)
        out = torch.stack(states, dim=0).mean(dim=0)
        return torch.split(out, [self.n_user, self.m_item])

