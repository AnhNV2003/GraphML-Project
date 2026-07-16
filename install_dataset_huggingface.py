import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
DATA_ROOT = Path(__file__).resolve().parent / "assets" / "data" / "amazon_reviews_2023"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an Amazon Reviews 2023 category from Hugging Face.")
    parser.add_argument(
        "--category",
        default="All_Beauty",
        help="Category name as used in the dataset's file names, e.g. All_Beauty, Video_Games, Movies_and_TV.",
    )
    return parser.parse_args()


def _already_downloaded_categories(output_dir: Path) -> set[str]:
    """Category names already present locally, so a fresh download doesn't wipe them.

    `snapshot_download(local_dir=...)` treats local_dir as a full mirror of
    `allow_patterns` and DELETES any local file that no longer matches on each
    call. Without this, downloading a second category (e.g. Video_Games) would
    silently remove a previously downloaded one (e.g. All_Beauty).
    """
    raw_dir = output_dir / "raw" / "review_categories"
    if not raw_dir.exists():
        return set()
    return {path.stem for path in raw_dir.glob("*.jsonl")}


def main() -> None:
    args = parse_args()
    category = args.category
    output_dir = DATA_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = _already_downloaded_categories(output_dir) | {category}
    allow_patterns = [f"**/*{cat}*" for cat in categories] + ["README.md"]

    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision="main",
        local_dir=output_dir,

        # Union of every category downloaded so far, so this stays additive
        # instead of pruning categories fetched in earlier runs.
        allow_patterns=allow_patterns,

        max_workers=8,
    )

    print(f"Download completed for category: {category}")
    print(f"Dataset path: {snapshot_path}")


if __name__ == "__main__":
    main()