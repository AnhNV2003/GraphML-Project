"""Faithful port of the ORIGINAL Sheaf4Rec model (Purificato et al., TORS 2025).

Source: https://github.com/antoniopurificato/Sheaf4Rec (models.py: SheafConvLayer + RecSysGNN).

This is NOT the simplified reimplementation in `sheaf.py`. It reproduces the original
architecture verbatim so results are directly comparable to the paper:

  * Scalar restriction maps: sheaf_learner = Linear(2*latent, 1, bias=False), then tanh.
  * n x n sheaf Laplacian (not block n*d x n*d).
  * Symmetric normalisation (deg + 1)^{-1/2} with an implicit self-loop.
  * Per-layer shared linear map; diffusion  x <- x - step_size * L @ linear(x).
  * CONCAT read-out over all layers (dim = latent * (num_layers + 1)).

Only two things differ from the upstream file, both mechanical (not architectural):
  * params/edge_index/num_nodes are passed via the constructor instead of a global
    `params.pickle` + `from dataset import *`.
  * `torch_scatter.scatter_add` -> `Tensor.index_add_` (identical result), since
    torch_scatter is not installed here.

The BPR loss / regularisation come from `BPRModelBase`, shared with every other model
in the pipeline, so the comparison isolates the model architecture.
"""

import torch
import torch.nn as nn

from .model_base import BPRModelBase


class SheafConvLayerOfficial(nn.Module):
    """Original SheafConvLayer with a learned scalar sheaf (verbatim math)."""

    def __init__(self, num_nodes: int, edge_index: torch.Tensor, latent_dim: int, step_size: float = 1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer("edge_index", edge_index)
        self.step_size = step_size
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.sheaf_learner = nn.Linear(2 * latent_dim, 1, bias=False)
        left_idx, right_idx = self._compute_left_right_map_index(edge_index)
        self.register_buffer("left_idx", left_idx)
        self.register_buffer("right_idx", right_idx)

    @staticmethod
    def _compute_left_right_map_index(edge_index: torch.Tensor):
        """For each directed edge e=(s,t), find the index of its reverse edge (t,s).

        Vectorised equivalent of the original dict-based loop. Works for any edge_index
        whose edge set is symmetric (every (s,t) has a (t,s)), which holds for the
        bidirectional graph we build below.
        """
        num_nodes = int(edge_index.max()) + 1
        src, dst = edge_index[0], edge_index[1]
        fwd_key = src.to(torch.long) * num_nodes + dst.to(torch.long)
        rev_key = dst.to(torch.long) * num_nodes + src.to(torch.long)
        order = torch.argsort(fwd_key)
        sorted_keys = fwd_key[order]
        pos = torch.searchsorted(sorted_keys, rev_key)
        right_idx = order[pos]
        left_idx = torch.arange(edge_index.size(1), device=edge_index.device)
        return left_idx, right_idx

    def build_laplacian(self, maps: torch.Tensor) -> torch.Tensor:
        row, col = self.edge_index
        left_maps = torch.index_select(maps, index=self.left_idx, dim=0)
        right_maps = torch.index_select(maps, index=self.right_idx, dim=0)
        non_diag_maps = -left_maps * right_maps
        # scatter_add(maps**2, row) -> index_add_ (identical)
        diag_maps = torch.zeros(self.num_nodes, 1, device=maps.device, dtype=maps.dtype)
        diag_maps = diag_maps.index_add(0, row, maps ** 2)

        d_sqrt_inv = (diag_maps + 1).pow(-0.5)
        left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
        norm_maps = left_norm * non_diag_maps * right_norm
        diag = d_sqrt_inv * diag_maps * d_sqrt_inv

        diag_indices = torch.arange(0, self.num_nodes, device=maps.device).view(1, -1).tile(2, 1)
        all_indices = torch.cat([diag_indices, self.edge_index], dim=-1)
        all_values = torch.cat([diag.view(-1), norm_maps.view(-1)])
        return torch.sparse_coo_tensor(all_indices, all_values, size=(self.num_nodes, self.num_nodes))

    def predict_restriction_maps(self, x: torch.Tensor) -> torch.Tensor:
        row, col = self.edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.sheaf_learner(torch.cat([x_row, x_col], dim=1))
        return torch.tanh(maps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.predict_restriction_maps(x)
        laplacian = self.build_laplacian(maps)
        y = self.linear(x)
        return x - self.step_size * torch.sparse.mm(laplacian, y)


class Sheaf4RecOfficial(BPRModelBase):
    """Original RecSysGNN (sheaf path): stacked SheafConvLayer + concat read-out."""

    def __init__(self, config, dataset, step_size: float = 1.0):
        super().__init__()
        self.n_user, self.m_item = dataset.n_users, dataset.m_items
        self.n_nodes = self.n_user + self.m_item
        self.latent_dim = int(config.get("latent_dim_rec", 64))
        self.n_layers = int(config.get("sheaf_n_layers", config.get("lightGCN_n_layers", 3)))

        edge_index = self._build_edge_index(dataset)
        self.embedding = nn.Embedding(self.n_nodes, self.latent_dim)
        self.convs = nn.ModuleList(
            SheafConvLayerOfficial(self.n_nodes, edge_index, self.latent_dim, step_size)
            for _ in range(self.n_layers)
        )
        nn.init.normal_(self.embedding.weight, std=0.1)

    def _build_edge_index(self, dataset) -> torch.Tensor:
        """Bidirectional [u->i, i->u] edge_index from train positives (matches original)."""
        import numpy as np

        lengths = [len(items) for items in dataset.allPos]
        u_arr = np.repeat(np.arange(self.n_user), lengths)
        i_arr = np.concatenate([np.asarray(items) for items in dataset.allPos if len(items)])
        u_t = torch.as_tensor(u_arr, dtype=torch.long)
        i_t = torch.as_tensor(i_arr, dtype=torch.long) + self.n_user
        return torch.stack([torch.cat([u_t, i_t]), torch.cat([i_t, u_t])])

    def forward(self):
        emb = self.embedding.weight
        embs = [emb]
        for conv in self.convs:
            emb = conv(emb)
            embs.append(emb)
        out = torch.cat(embs, dim=-1)  # concat read-out (original sheaf path)
        return torch.split(out, [self.n_user, self.m_item])
