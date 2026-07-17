"""Dataset loaders."""

import io
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .config import DATA_ROOT


def _standardize_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Return the minimal schema consumed by the recommendation pipeline."""
    if "parent_asin" in df.columns and "item_id" not in df.columns:
        df = df.rename(columns={"parent_asin": "item_id"})
    required = ["user_id", "item_id", "rating", "timestamp"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required interaction columns: {missing}")
    out = df[required].copy()
    out["user_id"] = out["user_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    return out


def amazon_benchmark_dir(
    core: str = "5core",
    split: str = "timestamp",
    data_root: Path = DATA_ROOT,
) -> Path:
    return data_root / "amazon_reviews_2023" / "benchmark" / core / split


def load_amazon_category_raw(category: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load local Amazon Reviews 2023 raw reviews for a given category."""
    path = data_root / "amazon_reviews_2023" / "raw" / "review_categories" / f"{category}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_json(path, lines=True, convert_dates=False)
    return _standardize_interactions(df)


def load_amazon_category_benchmark(
    category: str,
    core: str = "5core",
    split: str = "timestamp",
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    """Load local Amazon benchmark train/valid/test CSVs and concatenate them."""
    base_dir = amazon_benchmark_dir(core=core, split=split, data_root=data_root)
    frames = []
    for part in ("train", "valid", "test"):
        path = base_dir / f"{category}.{part}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["benchmark_split"] = part
        frames.append(frame)
    return _standardize_interactions(pd.concat(frames, ignore_index=True))


def load_amazon_category_real(
    category: str,
    source: str = "raw",
    core: str = "5core",
    split: str = "timestamp",
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    """Load an Amazon Reviews 2023 category from local files.

    source="raw" reads the local JSONL review dump.
    source="benchmark" reads local published train/valid/test CSV splits.
    """
    if source == "raw":
        return load_amazon_category_raw(category, data_root=data_root)
    if source == "benchmark":
        return load_amazon_category_benchmark(category, core=core, split=split, data_root=data_root)
    raise ValueError(f"Unknown Amazon source: {source!r}. Expected 'raw' or 'benchmark'.")


def load_amazon_beauty_real(**kwargs) -> pd.DataFrame:
    return load_amazon_category_real("All_Beauty", **kwargs)


def load_amazon_video_games_real(**kwargs) -> pd.DataFrame:
    return load_amazon_category_real("Video_Games", **kwargs)


def load_movielens_1m_local(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load MovieLens 1M from local ratings.dat."""
    path = data_root / "datamovielens-1m" / "ratings.dat"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return _standardize_interactions(df)


def load_movielens_1m_from_url() -> pd.DataFrame:
    """Load MovieLens 1M from the official GroupLens URL as a last-resort fallback."""
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    with zf.open("ml-1m/ratings.dat") as f:
        df = pd.read_csv(
            f,
            sep="::",
            engine="python",
            encoding="latin-1",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
    return _standardize_interactions(df)


def load_movielens_1m_real(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load MovieLens 1M locally first; use the official URL only if local data is missing."""
    try:
        return load_movielens_1m_local(data_root=data_root)
    except FileNotFoundError:
        return load_movielens_1m_from_url()


DATASET_REGISTRY: dict[str, dict[str, str | Callable[[], pd.DataFrame]]] = {
    "amazon_beauty": {
        "display_name": "Amazon Beauty (All Beauty)",
        "loader": load_amazon_beauty_real,
        "source": "raw",
        "benchmark_dir": amazon_benchmark_dir(),
        "benchmark_core": "5core",
        "benchmark_split": "timestamp",
    },
    "amazon_video_games": {
        "display_name": "Amazon Video Games",
        "loader": load_amazon_video_games_real,
        "source": "raw",
        "benchmark_dir": amazon_benchmark_dir(),
        "benchmark_core": "5core",
        "benchmark_split": "timestamp",
    },
    "movielens_1m": {
        "display_name": "MovieLens 1M",
        "loader": load_movielens_1m_real,
    },
}


def load_dataset_by_name(name: str) -> pd.DataFrame:
    cfg = DATASET_REGISTRY[name]
    df = cfg["loader"]()
    print(f"[{name}] Loaded real data, shape = {df.shape}")
    return df
