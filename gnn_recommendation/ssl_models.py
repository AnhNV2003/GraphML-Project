"""Faithful ports of 5 self-supervised graph collaborative-filtering SOTA models.

Sources (all public code, verified before porting):
  * SGL      -- Self-supervised Graph Learning for Recommendation, SIGIR'21
               https://github.com/Coder-Yu/SELFRec (model/graph/SGL.py)
  * SimGCL   -- Are Graph Augmentations Necessary? SIGIR'22
               https://github.com/Coder-Yu/SELFRec (model/graph/SimGCL.py)
  * DirectAU -- Towards Representation Alignment and Uniformity in CF, KDD'22
               https://github.com/Coder-Yu/SELFRec (model/graph/DirectAU.py)
  * NCL      -- Neighborhood-enriched Contrastive Learning, WWW'22
               https://github.com/Coder-Yu/SELFRec (model/graph/NCL.py)
  * LightGCL -- Simple Yet Effective Graph Contrastive Learning, ICLR'23
               https://github.com/HKUDS/LightGCL (model.py)

Integration strategy
---------------------
Every model in this project trains through the SAME loop: LightGCN-PyTorch's
`Procedure.BPR_train_original`, which samples (users, pos, neg) each batch and
calls `model.bpr_loss(users, pos, neg) -> (loss, reg_loss)` (see
`gnn_recommendation/model_base.py`). Each class below overrides `bpr_loss` to
add that paper's self-supervised loss term on top of the batch's BPR loss, so
we get early stopping / patience / multi-seed / timing for free, identical to
how `sheaf_official.py` and `tagcf.py` were integrated.

Two deliberate simplifications relative to the original codebases, both noted
inline where they occur:
  * SGL regenerates its two dropout-augmented views once per epoch in the
    original code; here they are regenerated once per epoch using a batch
    counter (since our shared trainer only exposes a per-batch hook).
  * NCL's ProtoNCE uses `sklearn.cluster.MiniBatchKMeans` instead of the
    original `faiss.Kmeans` (GPU faiss is not installed here); the clustering
    math (E-step every epoch after a warm-up) is otherwise unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import build_normalized_graph
from .model_base import BPRModelBase


def info_nce(view1: torch.Tensor, view2: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE with in-batch negatives (SELFRec util.loss_torch.InfoNCE, verbatim)."""
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


def _bpr_and_reg(user_emb, pos_emb, neg_emb):
    """Standard BPR + L2 reg on a batch, shared by every model below."""
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    rec_loss = torch.mean(nn.Softplus()(neg_scores - pos_scores))
    reg_loss = 0.5 * (
        user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
    ) / user_emb.shape[0]
    return rec_loss, reg_loss


def _build_pairs(dataset) -> list:
    pairs = []
    for u, items in enumerate(dataset.allPos):
        for it in items:
            pairs.append((u, int(it)))
    return pairs


class _LightGCNBase(BPRModelBase):
    """Shared LightGCN-style propagation encoder (embedding + mean-pool over layers)."""

    def __init__(self, config, dataset, n_layers: int):
        super().__init__()
        self.n_user, self.m_item = dataset.n_users, dataset.m_items
        self.n_nodes = self.n_user + self.m_item
        self.latent_dim = int(config.get("latent_dim_rec", 64))
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.n_nodes, self.latent_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.graph = dataset.Graph  # symmetric D^-1/2 A D^-1/2 sparse tensor (graph.py)

    def _propagate(self, emb: torch.Tensor, graph=None, return_all: bool = False):
        graph = graph if graph is not None else self.graph
        embs = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(graph, emb)
            embs.append(emb)
        mean_emb = torch.stack(embs, dim=0).mean(dim=0)
        return (mean_emb, embs) if return_all else mean_emb

    def forward(self):
        emb = self._propagate(self.embedding.weight)
        return torch.split(emb, [self.n_user, self.m_item])


class SimGCLRec(_LightGCNBase):
    """SimGCL (SIGIR'22): noise-perturbed embedding views + InfoNCE, no graph augmentation."""

    def __init__(self, config, dataset, n_layers=2, eps=0.1, cl_rate=0.5, temp=0.2):
        super().__init__(config, dataset, n_layers)
        self.eps, self.cl_rate, self.temp = eps, cl_rate, temp

    def _propagate_perturbed(self) -> torch.Tensor:
        emb = self.embedding.weight
        embs = []
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.graph, emb)
            noise = torch.rand_like(emb)
            emb = emb + torch.sign(emb) * F.normalize(noise, dim=-1) * self.eps
            embs.append(emb)
        return torch.stack(embs, dim=0).mean(dim=0)

    def bpr_loss(self, users, pos, neg):
        user_emb, item_emb = self.forward()
        u, p, n = user_emb[users], item_emb[pos], item_emb[neg]
        rec_loss, reg_loss = _bpr_and_reg(u, p, n)

        view1 = self._propagate_perturbed()
        view2 = self._propagate_perturbed()
        v1u, v1i = torch.split(view1, [self.n_user, self.m_item])
        v2u, v2i = torch.split(view2, [self.n_user, self.m_item])
        u_idx, i_idx = torch.unique(users), torch.unique(pos)
        cl_loss = info_nce(v1u[u_idx], v2u[u_idx], self.temp) + info_nce(v1i[i_idx], v2i[i_idx], self.temp)

        return rec_loss + self.cl_rate * cl_loss, reg_loss


class SGLRec(_LightGCNBase):
    """SGL (SIGIR'21): edge-dropout augmented views + InfoNCE.

    Simplification: the two dropped adjacencies are refreshed once per epoch
    via a batch counter (original refreshes once per epoch directly in its
    own training loop, which we don't control here).
    """

    def __init__(self, config, dataset, n_layers=2, drop_rate=0.1, cl_rate=0.1, temp=0.2):
        super().__init__(config, dataset, n_layers)
        self.drop_rate, self.cl_rate, self.temp = drop_rate, cl_rate, temp
        self._pairs = _build_pairs(dataset)
        batch_size = int(config.get("bpr_batch_size", 2048))
        self._batches_per_epoch = max(1, dataset.trainDataSize // batch_size)
        self._batch_counter = 0
        self._view1 = None
        self._view2 = None

    def _random_dropped_graph(self) -> torch.Tensor:
        device = self.embedding.weight.device
        n = len(self._pairs)
        keep = np.random.rand(n) > self.drop_rate
        if not keep.any():
            keep[0] = True
        kept = [self._pairs[i] for i in range(n) if keep[i]]
        return build_normalized_graph(self.n_user, self.m_item, kept, device=device)

    def _maybe_refresh_views(self):
        if self._batch_counter % self._batches_per_epoch == 0 or self._view1 is None:
            self._view1 = self._random_dropped_graph()
            self._view2 = self._random_dropped_graph()
        self._batch_counter += 1

    def bpr_loss(self, users, pos, neg):
        self._maybe_refresh_views()
        user_emb, item_emb = self.forward()
        u, p, n = user_emb[users], item_emb[pos], item_emb[neg]
        rec_loss, reg_loss = _bpr_and_reg(u, p, n)

        view1 = self._propagate(self.embedding.weight, graph=self._view1)
        view2 = self._propagate(self.embedding.weight, graph=self._view2)
        u_idx = torch.unique(users)
        i_idx = torch.unique(pos)
        v1_sel = torch.cat([view1[u_idx], view1[self.n_user + i_idx]], dim=0)
        v2_sel = torch.cat([view2[u_idx], view2[self.n_user + i_idx]], dim=0)
        cl_loss = info_nce(v1_sel, v2_sel, self.temp)

        return rec_loss + self.cl_rate * cl_loss, reg_loss


class DirectAURec(_LightGCNBase):
    """DirectAU (KDD'22): replaces BPR with alignment + uniformity (no negatives needed)."""

    def __init__(self, config, dataset, n_layers=3, gamma=2.0):
        super().__init__(config, dataset, n_layers)
        self.gamma = gamma

    @staticmethod
    def _alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def _uniformity(x, t=2.0):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def bpr_loss(self, users, pos, neg):
        user_emb, item_emb = self.forward()
        u, p = user_emb[users], item_emb[pos]
        align = self._alignment(u, p)
        uniform = self.gamma * (self._uniformity(u) + self._uniformity(p)) / 2
        loss = align + uniform
        return loss, torch.zeros((), device=loss.device)


class NCLRec(_LightGCNBase):
    """NCL (WWW'22): structural-neighbor SSL + prototype (k-means) contrastive loss.

    Simplification: uses sklearn MiniBatchKMeans instead of the original
    faiss.Kmeans (no GPU faiss installed); clustering cadence (E-step once per
    epoch, after a warm-up) is otherwise unchanged.
    """

    def __init__(
        self, config, dataset, n_layers=3, ssl_reg=1e-6, proto_reg=1e-7,
        tau=0.05, hyper_layers=1, alpha=1.5, num_clusters=None, warmup_epochs=20,
    ):
        super().__init__(config, dataset, n_layers)
        self.ssl_reg, self.proto_reg, self.tau = ssl_reg, proto_reg, tau
        self.hyper_layers, self.alpha = hyper_layers, alpha
        self.num_clusters = num_clusters or min(1000, max(8, min(self.n_user, self.m_item) // 5))
        batch_size = int(config.get("bpr_batch_size", 2048))
        self._batches_per_epoch = max(1, dataset.trainDataSize // batch_size)
        self._warmup_batches = self._batches_per_epoch * warmup_epochs
        self._batch_counter = 0
        self._user_centroids = self._user_2cluster = None
        self._item_centroids = self._item_2cluster = None

    def _run_kmeans(self, x_np: np.ndarray):
        from sklearn.cluster import MiniBatchKMeans

        k = min(self.num_clusters, max(2, x_np.shape[0] - 1))
        km = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=min(4096, x_np.shape[0])).fit(x_np)
        device = self.embedding.weight.device
        centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        assign = torch.tensor(km.labels_, dtype=torch.long, device=device)
        return centroids, assign

    def _maybe_e_step(self):
        # Below warmup_epochs (default 20), centroids stay None and the proto-NCE
        # term in bpr_loss() never fires -- NCL silently degrades to a plain
        # LightGCN-style encoder + structural SSL loss only. Matters for short
        # runs (e.g. --quick with 1-2 epochs): don't read those as NCL's real behavior.
        if self._batch_counter < self._warmup_batches:
            self._batch_counter += 1
            return
        if self._batch_counter % self._batches_per_epoch == 0:
            with torch.no_grad():
                u_np = self.embedding.weight[: self.n_user].detach().cpu().numpy()
                i_np = self.embedding.weight[self.n_user :].detach().cpu().numpy()
            self._user_centroids, self._user_2cluster = self._run_kmeans(u_np)
            self._item_centroids, self._item_2cluster = self._run_kmeans(i_np)
        self._batch_counter += 1

    def _ssl_layer_loss(self, context_emb, initial_emb, user, item):
        cu_all, ci_all = torch.split(context_emb, [self.n_user, self.m_item])
        iu_all, ii_all = torch.split(initial_emb, [self.n_user, self.m_item])

        def side_loss(ctx_all, init_all, idx):
            ctx = F.normalize(ctx_all[idx], dim=-1)
            init = F.normalize(init_all[idx], dim=-1)
            all_init = F.normalize(init_all, dim=-1)
            pos_score = torch.exp((ctx * init).sum(dim=-1) / self.tau)
            ttl_score = torch.exp(ctx @ all_init.T / self.tau).sum(dim=-1)
            return -torch.log(pos_score / ttl_score).sum()

        return self.ssl_reg * (side_loss(cu_all, iu_all, user) + self.alpha * side_loss(ci_all, ii_all, item))

    def _proto_nce_loss(self, initial_emb, users, pos):
        user_emb, item_emb = torch.split(initial_emb, [self.n_user, self.m_item])
        batch_size = users.shape[0]
        u_c = self._user_centroids[self._user_2cluster[users]]
        i_c = self._item_centroids[self._item_2cluster[pos]]
        pu = info_nce(user_emb[users], u_c, self.tau) * batch_size
        pi = info_nce(item_emb[pos], i_c, self.tau) * batch_size
        return self.proto_reg * (pu + pi)

    def bpr_loss(self, users, pos, neg):
        self._maybe_e_step()
        mean_emb, embs = self._propagate(self.embedding.weight, return_all=True)
        user_emb, item_emb = torch.split(mean_emb, [self.n_user, self.m_item])
        u, p, n = user_emb[users], item_emb[pos], item_emb[neg]
        rec_loss, reg_loss = _bpr_and_reg(u, p, n)

        initial_emb = embs[0]
        layer_idx = min(self.hyper_layers * 2, len(embs) - 1)
        context_emb = embs[layer_idx]
        ssl_loss = self._ssl_layer_loss(context_emb, initial_emb, users, pos)

        loss = rec_loss + ssl_loss
        if self._user_centroids is not None:
            loss = loss + self._proto_nce_loss(initial_emb, users, pos)
        return loss, reg_loss


class LightGCLRec(_LightGCNBase):
    """LightGCL (ICLR'23): SVD-reconstructed graph view + InfoNCE against the GNN view.

    SVD of the (n_user x m_item) normalized bipartite matrix is computed once
    at construction time (torch.svd_lowrank), matching the original's fixed,
    non-trainable low-rank reconstruction.
    """

    def __init__(self, config, dataset, n_layers=2, q=5, temp=0.2, lambda_1=0.2):
        super().__init__(config, dataset, n_layers)
        self.temp, self.lambda_1 = temp, lambda_1

        pairs = _build_pairs(dataset)
        rows = np.array([u for u, _ in pairs])
        cols = np.array([i for _, i in pairs])
        deg_u = np.maximum(np.bincount(rows, minlength=self.n_user), 1)
        deg_i = np.maximum(np.bincount(cols, minlength=self.m_item), 1)
        vals = 1.0 / (np.sqrt(deg_u[rows]) * np.sqrt(deg_i[cols]))
        idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        adj_norm = torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), (self.n_user, self.m_item)).coalesce()

        q = min(q, min(self.n_user, self.m_item) - 1)
        svd_u, s, svd_v = torch.svd_lowrank(adj_norm, q=q)
        self.register_buffer("u_mul_s", svd_u @ torch.diag(s))
        self.register_buffer("v_mul_s", svd_v @ torch.diag(s))
        self.register_buffer("ut", svd_u.t().contiguous())
        self.register_buffer("vt", svd_v.t().contiguous())

    def bpr_loss(self, users, pos, neg):
        e_u0 = self.embedding.weight[: self.n_user]
        e_i0 = self.embedding.weight[self.n_user :]

        z_u_list, z_i_list = [e_u0], [e_i0]
        g_u_list, g_i_list = [e_u0], [e_i0]
        for _ in range(self.n_layers):
            full = torch.cat([z_u_list[-1], z_i_list[-1]], dim=0)
            propagated = torch.sparse.mm(self.graph, full)
            z_u, z_i = torch.split(propagated, [self.n_user, self.m_item])
            z_u_list.append(z_u)
            z_i_list.append(z_i)

            g_u_list.append(self.u_mul_s @ (self.vt @ z_i_list[-2]))
            g_i_list.append(self.v_mul_s @ (self.ut @ z_u_list[-2]))

        e_u = sum(z_u_list)
        e_i = sum(z_i_list)
        g_u = sum(g_u_list)
        g_i = sum(g_i_list)

        u, p, n = e_u[users], e_i[pos], e_i[neg]
        rec_loss, reg_loss = _bpr_and_reg(u, p, n)

        u_idx, i_idx = torch.unique(users), torch.unique(pos)
        cl_loss = info_nce(g_u[u_idx], e_u[u_idx], self.temp) + info_nce(g_i[i_idx], e_i[i_idx], self.temp)

        self._e_u, self._e_i = e_u.detach(), e_i.detach()
        return rec_loss + self.lambda_1 * cl_loss, reg_loss

    def forward(self):
        e_u0 = self.embedding.weight[: self.n_user]
        e_i0 = self.embedding.weight[self.n_user :]
        z_u_list, z_i_list = [e_u0], [e_i0]
        for _ in range(self.n_layers):
            full = torch.cat([z_u_list[-1], z_i_list[-1]], dim=0)
            propagated = torch.sparse.mm(self.graph, full)
            z_u, z_i = torch.split(propagated, [self.n_user, self.m_item])
            z_u_list.append(z_u)
            z_i_list.append(z_i)
        return sum(z_u_list), sum(z_i_list)


SSL_MODEL_REGISTRY = {
    "SGL": SGLRec,
    "SimGCL": SimGCLRec,
    "DirectAU": DirectAURec,
    "NCL": NCLRec,
    "LightGCL": LightGCLRec,
}
