"""Faithful port of TAG-CF (Ju et al., NeurIPS 2024).

Source: https://github.com/snap-research/Test-time-Aggregation-for-CF
Paper: "How Does Message Passing Improve Collaborative Filtering?"

TAG-CF is a *test-time* augmentation: train a plain matrix-factorization model
(no graph during training), then apply a SINGLE message-passing aggregation at
inference over the user-item graph, with tunable degree-normalisation exponents
(m for out-degree, n for in-degree) selected on the validation set.

Faithful details reproduced from the original code:
  * MF embedding init: uniform(-0.05, 0.05)  (BaseMatrixFactorization).
  * Message-passing graph = train edges + reverse edges + self-loops
    (src/utils.py: pre_process_graph -> add_reverse_edges + add_self_loop).
  * Layer op (MessagePassingLayer): h <- D_in^n * A * (D_out^m * h), degrees
    clamped to >=1. On the symmetric (bidirectional+self-loop) graph in/out
    degrees coincide.
  * Read-out: mean over layer states [e^0, e^1, ...]  (node_indices are (N,1),
    so the original cat(dim=1).mean(1) is a per-layer average -> (N, d)).
  * ML-1M config: m=-1.5, n=-0.5, n_layers=1; grid search space [-1.5,-1,-0.5,0].

Training uses the shared BPRModelBase loss (plain dot-product MF); `use_mp` is
toggled on only for evaluation.
"""

import numpy as np
import torch
import torch.nn as nn

from .model_base import BPRModelBase

TAGCF_SEARCH_SPACE = [-1.5, -1.0, -0.5, 0.0]  # from src/constants.py


class MFTagCF(BPRModelBase):
    """Matrix factorization trainable with BPR + optional TAG-CF test-time MP."""

    def __init__(self, config, dataset, m: float = -0.5, n: float = -0.5, n_layers: int = 1):
        super().__init__()
        self.n_user, self.m_item = dataset.n_users, dataset.m_items
        self.n_nodes = self.n_user + self.m_item
        self.latent_dim = int(config.get("latent_dim_rec", 64))
        self.embedding = nn.Embedding(self.n_nodes, self.latent_dim)
        self.embedding.weight.data.uniform_(-0.05, 0.05)  # original MF init

        self.m, self.n, self.n_layers = m, n, n_layers
        self.use_mp = False  # off during training (plain MF), on at eval time

        adj, degree = self._build_graph(dataset)
        self.register_buffer("_adj_indices", adj)
        self.register_buffer("_degree", degree)

    def _build_graph(self, dataset):
        """Bidirectional + self-loop adjacency (indices) and degree (with self-loop)."""
        u_list, i_list = [], []
        for u, items in enumerate(dataset.allPos):
            items = np.asarray(items)
            u_list.append(np.full(len(items), u))
            i_list.append(items + self.n_user)
        u = np.concatenate(u_list)
        i = np.concatenate(i_list)
        self_nodes = np.arange(self.n_nodes)
        # rows/cols for A where A[v,u]=1 means edge u->v; symmetric here.
        rows = np.concatenate([i, u, self_nodes])  # i<-u, u<-i, v<-v
        cols = np.concatenate([u, i, self_nodes])
        indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        degree = torch.zeros(self.n_nodes, dtype=torch.float32)
        degree.index_add_(0, torch.tensor(rows, dtype=torch.long), torch.ones(len(rows)))
        return indices, degree.clamp(min=1.0)

    def _aggregate(self) -> torch.Tensor:
        x = self.embedding.weight
        adj = torch.sparse_coo_tensor(
            self._adj_indices,
            torch.ones(self._adj_indices.size(1), device=x.device, dtype=x.dtype),
            (self.n_nodes, self.n_nodes),
        ).coalesce()
        d_out = self._degree.pow(self.m).unsqueeze(1)  # out-degree norm
        d_in = self._degree.pow(self.n).unsqueeze(1)   # in-degree norm
        results = [x]
        h = x
        for _ in range(self.n_layers):
            h = d_in * torch.sparse.mm(adj, d_out * h)
            results.append(h)
        return torch.stack(results, dim=0).mean(dim=0)

    def forward(self):
        emb = self._aggregate() if self.use_mp else self.embedding.weight
        return torch.split(emb, [self.n_user, self.m_item])
