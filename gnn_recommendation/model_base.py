"""Shared model API for recommendation models trained with BPR."""

import torch
import torch.nn as nn


class BPRModelBase(nn.Module):
    """Common BPR-compatible API used by LightGCN-PyTorch's BPRLoss wrapper."""

    trainable = True

    def bpr_loss(self, users, pos, neg):
        users_emb, items_emb = self.forward()
        user_e, pos_e, neg_e = users_emb[users], items_emb[pos], items_emb[neg]
        pos_scores = torch.mul(user_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(user_e, neg_e).sum(dim=1)
        loss = torch.mean(nn.Softplus()(neg_scores - pos_scores))
        reg_loss = (
            (1 / 2)
            * (user_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2))
            / float(len(users))
        )
        return loss, reg_loss

    def getUsersRating(self, users):
        user_emb, item_emb = self.forward()
        return torch.matmul(user_emb[users], item_emb.t())

