"""
Converts one of our ID,Target CSVs (train_split.csv, val_split.csv, or the
full Train.csv) into Kraken's `path`-format training data: each line image
gets a same-stem `.gt.txt` sidecar file containing its transcription, and a
manifest.txt listing the image paths — this is exactly what
`ketos train -f path -t manifest.txt` expects.

Adapted from starters/Kraken-OCR/train.py's prepare_training_data(), with
the record-ID/text lookup simplified to our fixed ID/Target schema.
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def clean_target(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line)


def build_manifest(csv_path, base_image_dir, output_dir, max_samples=None):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.txt")
    added, missing, empty = 0, 0, 0

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for _, row in df.iterrows():
            record_id = str(row["ID"]).strip()
            label = clean_target(row["Target"])

            if not record_id or not label:
                empty += 1
                continue

            image_path = os.path.join(base_image_dir, f"{record_id}.jpg")
            if not os.path.exists(image_path):
                missing += 1
                continue

            target_image = os.path.join(output_dir, f"{record_id}.jpg")
            target_label = os.path.join(output_dir, f"{record_id}.gt.txt")

            if not os.path.exists(target_image):
                try:
                    os.symlink(os.path.abspath(image_path), target_image)
                except OSError:
                    # Windows without dev mode / admin rights can't symlink — copy instead
                    import shutil
                    shutil.copy2(image_path, target_image)

            Path(target_label).write_text(label, encoding="utf-8")

            manifest.write(f"{target_image}\n")
            added += 1
            if max_samples and added >= max_samples:
                break

    if missing:
        print(f"[WARN] skipped {missing} row(s) with no matching image")
    if empty:
        print(f"[WARN] skipped {empty} row(s) with empty ID/Target")
    print(f"[DATA] prepared {added} samples in {output_dir} (manifest: {manifest_path})")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Build a Kraken path-format manifest from an ID,Target CSV.")
    parser.add_argument("--csv", required=True, help="e.g. data/train_split.csv or data/val_split.csv")
    parser.add_argument("--base_image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    build_manifest(args.csv, args.base_image_dir, args.output_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
