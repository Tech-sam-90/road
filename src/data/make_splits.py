"""
Splits data/Train.csv into train_split.csv / val_split.csv (90/10) — we have
no real ground truth for Test.csv (it's what we're predicting), so this is
the local stand-in for measuring weighted WER/CER during development.

Stratified by binned Target character length so val mirrors the full length
distribution rather than being randomly skewed toward short or long lines.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
N_BINS = 5
VAL_FRACTION = 0.1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def length_distribution_report(df, name):
    lengths = df["_length"]
    stats = lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    print(f"\n--- {name} (n={len(df)}) ---")
    print(stats.to_string())
    print("length_bin proportions:")
    print(df["length_bin"].value_counts(normalize=True).sort_index().to_string())


def main():
    train_csv = os.path.join(DATA_DIR, "Train.csv")
    df = pd.read_csv(train_csv)

    df["_length"] = df["Target"].astype(str).str.len()
    df["length_bin"] = pd.qcut(df["_length"], q=N_BINS, labels=False, duplicates="drop")

    n_bins_actual = df["length_bin"].nunique()
    if n_bins_actual != N_BINS:
        print(f"[WARN] requested {N_BINS} quantile bins, got {n_bins_actual} (duplicate bin edges from ties)")

    train_split, val_split = train_test_split(
        df,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=df["length_bin"],
    )

    length_distribution_report(train_split, "train_split")
    length_distribution_report(val_split, "val_split")

    helper_cols = ["_length", "length_bin"]
    train_out = train_split.drop(columns=helper_cols).sort_index()
    val_out = val_split.drop(columns=helper_cols).sort_index()

    train_path = os.path.join(DATA_DIR, "train_split.csv")
    val_path = os.path.join(DATA_DIR, "val_split.csv")
    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)

    print(f"\nSaved {len(train_out)} rows to {train_path}")
    print(f"Saved {len(val_out)} rows to {val_path}")


if __name__ == "__main__":
    main()
