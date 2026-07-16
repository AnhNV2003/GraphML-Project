"""Simplified additional recommender models used for comparison."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import BPRModelBase
from .sheaf import Sheaf4Rec
from .sheaf_official import Sheaf4RecOfficial
from .ssl_models import DirectAURec, LightGCLRec, NCLRec, SGLRec, SimGCLRec


class PopularityRec(BPRModelBase):
    """Non-personalized top-popular-item recommender baseline."""

    trainable = False

    def __init__(self, config, dataset):
        super().__init__()
        counts = np.zeros(dataset.m_items, dtype=np.float32)
        for pos_items in dataset.allPos:
            counts[pos_items] += 1.0
        self.item_score = nn.Parameter(torch.tensor(counts, dtype=torch.float32), requires_grad=False)

    def getUsersRating(self, users):
        return self.item_score.to(users.device).unsqueeze(0).expand(len(users), -1)


class NGCFRec(BPRModelBase):
    """Faithful port of NGCF (Wang et al., SIGIR'19).

    Source: https://github.com/huangtinglin/NGCF-PyTorch (NGCF/NGCF.py), the de
    facto reference PyTorch implementation (same role as gusye1234/LightGCN-PyTorch
    for LightGCN in this project).

    Differences from the previous (simplified) version in this file, found by
    diffing against the source above:
      * Adjacency: official NGCF propagates over the MEAN-normalized adjacency
        with self-loops, D^-1(A+I) -- NOT the symmetric D^-1/2 A D^-1/2 graph
        shared by LightGCN/Sheaf4Rec/NGCF's old stub in this project. Built
        directly from `dataset.allPos`, same pattern as sheaf_official.py.
      * Depth: the previous version had exactly ONE propagation step; official
        NGCF stacks `n_layers` (paper default 3) layers, each with its OWN
        W_gc/W_bi (+ bias), L2-normalizes the layer output, and reads out via
        CONCATENATION of every layer (dim = latent_dim * (n_layers + 1)), not
        just the last layer.
      * Message dropout: present in the original (default 0.1/layer); left at
        0 here to match this project's convention of disabling dropout for a
        deterministic, apples-to-apples comparison across models.
    """

    def __init__(self, config, dataset, n_layers: int | None = None, mess_dropout: float = 0.0):
        super().__init__()
        self.n_user, self.m_item = dataset.n_users, dataset.m_items
        self.n_nodes = self.n_user + self.m_item
        self.emb_size = config["latent_dim_rec"]
        self.n_layers = n_layers or config.get("lightGCN_n_layers", 3)
        self.mess_dropout = mess_dropout

        self.embedding_user = nn.Embedding(self.n_user, self.emb_size)
        self.embedding_item = nn.Embedding(self.m_item, self.emb_size)
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

        self.W_gc = nn.ModuleList(nn.Linear(self.emb_size, self.emb_size) for _ in range(self.n_layers))
        self.W_bi = nn.ModuleList(nn.Linear(self.emb_size, self.emb_size) for _ in range(self.n_layers))
        for linear in list(self.W_gc) + list(self.W_bi):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        indices, values = self._build_mean_norm_adjacency(dataset)
        self.register_buffer("adj_indices", indices)
        self.register_buffer("adj_values", values)

    def _build_mean_norm_adjacency(self, dataset):
        """D^-1 (A + I): row-normalised bipartite adjacency with self-loops.

        Matches `mean_adj_single(adj_mat + sp.eye(...))` in the official
        load_data.py (the branch actually wired into training, line 144).
        """
        u_list, i_list = [], []
        for u, items in enumerate(dataset.allPos):
            items = np.asarray(items)
            u_list.append(np.full(len(items), u))
            i_list.append(items + self.n_user)
        u = np.concatenate(u_list) if u_list else np.array([], dtype=np.int64)
        i = np.concatenate(i_list) if i_list else np.array([], dtype=np.int64)

        self_loop = np.arange(self.n_nodes)
        rows = np.concatenate([u, i, self_loop])
        cols = np.concatenate([i, u, self_loop])

        degree = np.zeros(self.n_nodes, dtype=np.float64)
        np.add.at(degree, rows, 1.0)
        degree = np.maximum(degree, 1.0)
        values = 1.0 / degree[rows]

        indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float32)
        return indices, values

    def forward(self):
        adj = torch.sparse_coo_tensor(
            self.adj_indices, self.adj_values, (self.n_nodes, self.n_nodes)
        ).coalesce()
        ego = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        all_embeddings = [ego]
        for k in range(self.n_layers):
            side = torch.sparse.mm(adj, ego)
            sum_emb = self.W_gc[k](side)
            bi_emb = self.W_bi[k](ego * side)
            ego = F.leaky_relu(sum_emb + bi_emb, negative_slope=0.2)
            if self.mess_dropout > 0:
                ego = F.dropout(ego, p=self.mess_dropout, training=self.training)
            ego = F.normalize(ego, p=2, dim=1)
            all_embeddings.append(ego)
        out = torch.cat(all_embeddings, dim=1)
        return torch.split(out, [self.n_user, self.m_item])


class GATRec(BPRModelBase):
    def __init__(self, config, dataset):
        super().__init__()
        try:
            import torch_geometric.nn as geom_nn
        except ImportError as exc:
            raise ImportError("GATRec requires torch-geometric. Install torch-geometric first.") from exc

        self.n_user, self.m_item = dataset.n_users, dataset.m_items
        self.latent_dim = config["latent_dim_rec"]
        self.embedding_user = nn.Embedding(self.n_user, self.latent_dim)
        self.embedding_item = nn.Embedding(self.m_item, self.latent_dim)
        self.gat_conv = geom_nn.GATConv(self.latent_dim, self.latent_dim, heads=1)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.edge_index = dataset.Graph.indices()

    def forward(self):
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        res = self.gat_conv(all_emb, self.edge_index)
        return torch.split(res, [self.n_user, self.m_item])


class MFStubRec(BPRModelBase):
    """Plain matrix factorization, NOT a real UltraGCN port.

    This is a placeholder baseline: no graph propagation, no UltraGCN-specific
    loss terms (no beta-score negative weighting, no item-item constraint from
    the original paper). Registered as "UltraGCN-stub" (not "UltraGCN") so
    results tables can't be mistaken for the real algorithm.
    """

    def __init__(self, config, dataset):
        super().__init__()
        self.n_user, self.m_item = dataset.n_users, dataset.m_items
        self.embedding_user = nn.Embedding(self.n_user, config["latent_dim_rec"])
        self.embedding_item = nn.Embedding(self.m_item, config["latent_dim_rec"])
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def forward(self):
        return self.embedding_user.weight, self.embedding_item.weight


def build_extra_models(config, dataset, device, include_gat: bool = True, include_auxiliary: bool = True):
    models = {
        "Popularity": PopularityRec(config, dataset).to(device),
        "NGCF": NGCFRec(config, dataset).to(device),
        "Sheaf4Rec-gcn_like": Sheaf4Rec(config, dataset, restriction_type="gcn_like").to(device),
        "Sheaf4Rec-gat_like": Sheaf4Rec(config, dataset, restriction_type="gat_like").to(device),
        "Sheaf4Rec-full_sheaf": Sheaf4Rec(config, dataset, restriction_type="full_sheaf").to(device),
        "Sheaf4Rec-official": Sheaf4RecOfficial(config, dataset).to(device),
        "SGL": SGLRec(config, dataset).to(device),
        "SimGCL": SimGCLRec(config, dataset).to(device),
        "DirectAU": DirectAURec(config, dataset).to(device),
        "NCL": NCLRec(config, dataset).to(device),
        "LightGCL": LightGCLRec(config, dataset).to(device),
    }
    if include_gat:
        models["GAT"] = GATRec(config, dataset).to(device)
    if include_auxiliary:
        models["UltraGCN-stub"] = MFStubRec(config, dataset).to(device)
    return models
