"""Adapter for the official-linked LightGCN-PyTorch implementation."""

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

from .config import DEVICE, EXTERNAL_REPOS_ROOT, PRIMARY_DATASET, SEED, make_config


def ensure_official_repo(repo_dir: str = "LightGCN-PyTorch") -> Path:
    """Clone LightGCN-PyTorch if missing and add its code directory to sys.path.

    `repo_dir` may be a bare repo name (resolved under assets/external_repos/)
    or an absolute/relative path pointing directly at a checkout.
    """
    path = Path(repo_dir)
    if not path.is_absolute() and not path.exists():
        path = EXTERNAL_REPOS_ROOT / repo_dir
    if not path.exists():
        result = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/gusye1234/LightGCN-PyTorch.git",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Could not clone LightGCN-PyTorch: {result.stderr[-500:]}")

    code_dir = path / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    return path


def install_world_stub(config: dict | None = None, dataset_name: str = PRIMARY_DATASET):
    """Install the minimal `world` module expected by LightGCN-PyTorch."""
    stub_world = types.ModuleType("world")
    stub_world.cprint = lambda words: print(f"[world] {words}")
    stub_world.config = config or make_config()
    stub_world.device = DEVICE
    stub_world.dataset = dataset_name
    stub_world.model_name = "lgn"
    stub_world.seed = SEED
    stub_world.tensorboard = False
    stub_world.topks = [10, 20]
    sys.modules["world"] = stub_world
    return stub_world


def load_official_modules(repo_dir: str = "LightGCN-PyTorch", config: dict | None = None):
    ensure_official_repo(repo_dir)
    world = install_world_stub(config=config)
    from dataloader import BasicDataset
    from model import LightGCN, PureMF
    import Procedure
    import utils

    return world, BasicDataset, PureMF, LightGCN, utils, Procedure


class GraphRecDataset:
    """Factory wrapper; call `create` after official BasicDataset has been imported.

    Subclasses BasicDataset inside `create()` (not at module level) because
    BasicDataset only exists after `ensure_official_repo()` has added the
    vendored repo's `code/` dir to sys.path -- it can't be imported/subclassed
    at this module's own import time.
    """

    @staticmethod
    def create(BasicDataset, n_users, n_items, train_pairs, test_pairs, graph, valid_pairs=None):
        class _GraphRecDataset(BasicDataset):
            def __init__(self):
                super().__init__()
                self._n_users = n_users
                self._m_items = n_items
                self.Graph = graph
                valid_pairs_local = valid_pairs or []

                user_items = {}
                for u, i in train_pairs:
                    user_items.setdefault(u, []).append(i)
                self._allPos = [
                    np.array(user_items.get(u, []), dtype=np.int64) for u in range(n_users)
                ]

                self._testDict = {}
                for u, i in test_pairs:
                    self._testDict.setdefault(u, []).append(i)
                self._validDict = {}
                for u, i in valid_pairs_local:
                    self._validDict.setdefault(u, []).append(i)
                self._trainDataSize = len(train_pairs)

            @property
            def n_users(self):
                return self._n_users

            @property
            def m_items(self):
                return self._m_items

            @property
            def trainDataSize(self):
                return self._trainDataSize

            @property
            def testDict(self):
                return self._testDict

            @property
            def validDict(self):
                return self._validDict

            @property
            def allPos(self):
                return self._allPos

            def getUserPosItems(self, users):
                return [self._allPos[u] for u in users]

            def getSparseGraph(self):
                return self.Graph

        return _GraphRecDataset()
