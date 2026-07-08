"""One-off verification of the raw competition data after copying into data/.

Checks: expected row counts, every ID has a matching image file, no empty
Target values in Train.csv. Prints a short pass/fail report.
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EXPECTED_TRAIN_ROWS = 4098
EXPECTED_TEST_ROWS = 1374


def main() -> int:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    image_ids = {p.stem for p in (DATA_DIR / "images").glob("*.jpg")}

    ok = True
    print(f"Train rows: {len(train)} (expected {EXPECTED_TRAIN_ROWS})")
    if len(train) != EXPECTED_TRAIN_ROWS:
        ok = False

    print(f"Test rows: {len(test)} (expected {EXPECTED_TEST_ROWS})")
    if len(test) != EXPECTED_TEST_ROWS:
        ok = False

    missing_train_images = sorted(set(train["ID"]) - image_ids)
    missing_test_images = sorted(set(test["ID"]) - image_ids)
    print(f"Train IDs missing an image: {len(missing_train_images)}")
    print(f"Test IDs missing an image: {len(missing_test_images)}")
    if missing_train_images or missing_test_images:
        ok = False
        print("  examples:", (missing_train_images + missing_test_images)[:5])

    empty_targets = train["Target"].isna() | (train["Target"].astype(str).str.strip() == "")
    print(f"Empty Target values in Train.csv: {int(empty_targets.sum())}")
    if empty_targets.any():
        ok = False
        print("  example IDs:", train.loc[empty_targets, "ID"].head(5).tolist())

    print(f"Total images on disk: {len(image_ids)}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
