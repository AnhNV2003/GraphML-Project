"""Shared runtime configuration."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import random

import numpy as np
import torch

SEED = 42
PRIMARY_DATASET = "movielens_1m"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_ROOT = PROJECT_ROOT / "assets"
DATA_ROOT = ASSETS_ROOT / "data"
EXTERNAL_REPOS_ROOT = ASSETS_ROOT / "external_repos"

MIN_INTERACTIONS = 5
EMBED_DIM = 64
N_LAYERS = 3
N_EPOCHS = 30
N_EPOCHS_MULTI = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(embed_dim: int = EMBED_DIM, n_layers: int = N_LAYERS) -> dict:
    return {
        "bpr_batch_size": 2048,
        "latent_dim_rec": embed_dim,
        "lightGCN_n_layers": n_layers,
        "dropout": 0,
        "keep_prob": 0.6,
        "A_split": False,
        "pretrain": 0,
        "decay": 1e-4,
        "lr": 1e-3,
        "test_u_batch_size": 100,
        "multicore": 0,
        "A_n_fold": 100,
        "bigdata": False,
        "sheaf_stalk_dim": 4,
        "sheaf_n_layers": 3,
        "sheaf_restriction_type": "full_sheaf",
        "sheaf_dropout": 0.0,
    }
