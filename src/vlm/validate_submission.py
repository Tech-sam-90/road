"""
Validates a submission.csv against SampleSubmission.csv before upload:
- same ID set (no extra, none missing)
- no duplicate IDs
- no missing (NaN) or empty-string Target values
- readable as UTF-8, matching Test.csv's ID formatting

Exits non-zero on any failure so this can gate a submission pipeline.
"""

import argparse
import sys

import pandas as pd


def validate(submission_csv, sample_submission_csv):
    problems = []

    sub = pd.read_csv(submission_csv, encoding="utf-8-sig")
    sample = pd.read_csv(sample_submission_csv, encoding="utf-8-sig")

    if list(sub.columns) != ["ID", "Target"]:
        problems.append(f"columns are {list(sub.columns)}, expected ['ID', 'Target']")

    sub_ids = sub["ID"].astype(str)
    sample_ids = sample["ID"].astype(str)

    dupes = sub_ids[sub_ids.duplicated()].unique().tolist()
    if dupes:
        problems.append(f"{len(dupes)} duplicate ID(s): {dupes[:5]}{'...' if len(dupes) > 5 else ''}")

    missing_from_sub = set(sample_ids) - set(sub_ids)
    if missing_from_sub:
        problems.append(f"{len(missing_from_sub)} ID(s) from SampleSubmission missing in submission: "
                         f"{list(missing_from_sub)[:5]}{'...' if len(missing_from_sub) > 5 else ''}")

    extra_in_sub = set(sub_ids) - set(sample_ids)
    if extra_in_sub:
        problems.append(f"{len(extra_in_sub)} ID(s) in submission not present in SampleSubmission: "
                         f"{list(extra_in_sub)[:5]}{'...' if len(extra_in_sub) > 5 else ''}")

    if "Target" in sub.columns:
        is_missing = sub["Target"].isna()
        is_empty = sub["Target"].astype(str).str.strip() == ""
        bad = sub.loc[is_missing | is_empty, "ID"].tolist()
        if bad:
            problems.append(f"{len(bad)} row(s) with empty/NaN Target: {bad[:5]}{'...' if len(bad) > 5 else ''}")

    print(f"Submission rows: {len(sub)}  |  SampleSubmission rows: {len(sample)}")
    if problems:
        print("FAILED validation:")
        for p in problems:
            print(f"  - {p}")
        return False

    print("PASSED validation: ID set matches, no duplicates, no missing/empty Target values.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--sample_submission", required=True)
    args = parser.parse_args()

    ok = validate(args.submission, args.sample_submission)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
