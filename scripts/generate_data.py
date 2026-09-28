import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_pipeline.ingestion import ClickStreamGenerator, BatchDataLoader
from src.data_pipeline.preprocessing import (
    ClickStreamPreprocessor, TimeFeatureExtractor, TextPreprocessor,
)
from src.feature_store.feature_engineering import FeaturePipeline
from src.training.dataset import train_test_split_by_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CTR data")
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} synthetic impressions...")
    gen = ClickStreamGenerator(seed=args.seed)
    df = gen.generate(args.num_samples)

    print("Preprocessing...")
    df = ClickStreamPreprocessor().clean(df)
    df = TimeFeatureExtractor().transform(df)
    df = TextPreprocessor().transform(df)

    print("Building features...")
    df = FeaturePipeline().build_features(df)

    print("Splitting train / val / test (temporal)...")
    train_df, val_df, test_df = train_test_split_by_time(df)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(args.output_dir, f"{name}.csv")
        BatchDataLoader.save(split, path)
        pos_rate = split["clicked"].mean()
        print(f"  {name}: {len(split)} rows, CTR={pos_rate:.4f} -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
