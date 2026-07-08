"""
Weighted WER / CER scoring for the R.O.A.D. Barbados handwriting challenge.

Zindi's final leaderboard score is 0.5 * weighted_WER + 0.5 * weighted_CER.
For each metric, per sample i:
    edits_i  = Levenshtein edit distance between ref_i and hyp_i
               (word-level for WER, char-level for CER)
    L_i      = reference length (word count for WER, char count for CER)
    W_i      = L_i ** 0.5

    weighted_metric = sum(edits_i * W_i) / sum(L_i * W_i)

Note on this denominator: the raw form sum(edits_i * W_i) / sum(W_i) is
degenerate whenever edits_i happens to be constant across samples (it
collapses to that constant regardless of weighting, since it factors out of
the sum) — verified against a worked toy example where every edits_i was 1
and the naive form forced a value of exactly 1.0 no matter the weights.
Normalizing by weighted reference length instead (sum(L_i * W_i)) gives a
proper bounded error rate: it's a weighted mean of each sample's own error
rate (edits_i / L_i), weighted by L_i * W_i = L_i ** 1.5 — samples with
longer references count for more, same direction as W_i alone, just applied
consistently to numerator and denominator.

A missing prediction ID, or an empty-string prediction (after role-tag
stripping and whitespace normalization), is scored against the full
reference length as if the hypothesis were "" — i.e. edits_i = L_i, the
maximum possible edit distance for that sample. It is never dropped.
"""

import argparse

import pandas as pd

# Order matters: this mirrors starters/VLM/inference.py's clean_output,
# which is the closest thing to a documented reference for how Zindi's own
# scorer strips chat-template artifacts from VLM outputs before comparing
# them to plain-text ground truth.
ROLE_TAGS = ["assistant", "user", "<|assistant|>", "<|user|>"]


def _to_str(value):
    if pd.isna(value):
        return ""
    return str(value)


def strip_role_tags(text):
    """Keep only what follows the last occurrence of each role tag, in turn."""
    text = _to_str(text)
    for tag in ROLE_TAGS:
        if tag in text:
            text = text.split(tag)[-1]
    return text


def normalize_text(text):
    """Normalize line endings and collapse repeated whitespace. Never
    lowercases, strips punctuation, or otherwise touches spelling — this
    competition's ground truth deliberately preserves archaic/non-standard
    spelling and normalizing it away would hide real errors.
    str.split() with no args already splits on every whitespace variant
    (space, tab, \\n, \\r, \\r\\n) and drops empty runs, so join-on-split
    handles both line-ending normalization and whitespace collapse in one
    step."""
    return " ".join(_to_str(text).split())


def levenshtein_distance(seq_a, seq_b):
    """Standard DP edit distance (substitutions + deletions + insertions,
    unit cost). Works on any indexable, comparable sequence — a list of
    words or a string of characters."""
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev = list(range(len_b + 1))
    curr = [0] * (len_b + 1)

    for i in range(1, len_a + 1):
        curr[0] = i
        a_i = seq_a[i - 1]
        for j in range(1, len_b + 1):
            cost = 0 if a_i == seq_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[len_b]


def _require_columns(df, name):
    if "ID" not in df.columns or "Target" not in df.columns:
        raise ValueError(f"{name} must have 'ID' and 'Target' columns, got {list(df.columns)}")


def _score(gt_df, pred_df, level):
    _require_columns(gt_df, "gt_df")
    _require_columns(pred_df, "pred_df")

    gt = gt_df[["ID", "Target"]].copy()
    gt["ID"] = gt["ID"].astype(str)
    gt = gt.drop_duplicates(subset="ID", keep="last")

    pred = pred_df[["ID", "Target"]].copy()
    pred["ID"] = pred["ID"].astype(str)
    pred = pred.drop_duplicates(subset="ID", keep="last")
    pred_map = dict(zip(pred["ID"], pred["Target"]))

    matched = len(set(gt["ID"]) & set(pred["ID"]))

    rows = []
    for _, row in gt.iterrows():
        gt_id = row["ID"]
        ref = normalize_text(row["Target"])

        raw_hyp = pred_map.get(gt_id)  # None if the ID is simply missing
        hyp = normalize_text(strip_role_tags(raw_hyp)) if raw_hyp is not None else ""

        if level == "word":
            ref_tokens, hyp_tokens = ref.split(), hyp.split()
        else:  # "char"
            ref_tokens, hyp_tokens = list(ref), list(hyp)

        edits = levenshtein_distance(ref_tokens, hyp_tokens)
        length = len(ref_tokens)
        weight = length ** 0.5

        rows.append({
            "ID": gt_id,
            "edits": edits,
            "length": length,
            "weight": weight,
            "contribution": edits * weight,
        })

    per_sample = pd.DataFrame(rows, columns=["ID", "edits", "length", "weight", "contribution"])

    denom = float((per_sample["length"] * per_sample["weight"]).sum())
    score = float(per_sample["contribution"].sum() / denom) if denom > 0 else 0.0

    return {
        "score": score,
        "matched": matched,
        "total_reference": len(gt),
        "per_sample": per_sample,
    }


def compute_weighted_wer(gt_df, pred_df):
    """Word-level weighted WER. See module docstring for the formula."""
    return _score(gt_df, pred_df, level="word")


def compute_weighted_cer(gt_df, pred_df):
    """Char-level weighted CER. See module docstring for the formula."""
    return _score(gt_df, pred_df, level="char")


def compute_final_score(gt_df, pred_df, cer_weight=0.5, wer_weight=0.5):
    """Combines weighted WER and CER at the competition's 0.5/0.5 blend."""
    wer_result = compute_weighted_wer(gt_df, pred_df)
    cer_result = compute_weighted_cer(gt_df, pred_df)
    final = cer_weight * cer_result["score"] + wer_weight * wer_result["score"]

    per_sample_report = wer_result["per_sample"].merge(
        cer_result["per_sample"], on="ID", suffixes=("_wer", "_cer")
    )

    return {
        "cer": cer_result["score"],
        "wer": wer_result["score"],
        "final": final,
        "matched": wer_result["matched"],
        "total_reference": wer_result["total_reference"],
        "per_sample_report": per_sample_report,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score a submission against ground truth using the "
                     "R.O.A.D. Barbados weighted WER/CER metric."
    )
    parser.add_argument("--gt", required=True, help="Path to ground-truth CSV (ID, Target)")
    parser.add_argument("--pred", required=True, help="Path to prediction/submission CSV (ID, Target)")
    parser.add_argument("--save-report", default=None, help="Optional path to save the per-sample breakdown CSV")
    args = parser.parse_args(argv)

    gt_df = pd.read_csv(args.gt)
    pred_df = pd.read_csv(args.pred)

    result = compute_final_score(gt_df, pred_df)

    print(f"Matched:  {result['matched']}/{result['total_reference']}")
    print(f"CER:      {result['cer']:.4f}")
    print(f"WER:      {result['wer']:.4f}")
    print(f"FINAL:    {result['final']:.4f}  (0.5*CER + 0.5*WER)")

    if args.save_report:
        result["per_sample_report"].to_csv(args.save_report, index=False)
        print(f"Per-sample report saved to {args.save_report}")

    return result


if __name__ == "__main__":
    main()
