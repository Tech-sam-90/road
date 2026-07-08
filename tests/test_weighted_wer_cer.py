import pandas as pd
import pytest

from src.metrics.weighted_wer_cer import (
    compute_final_score,
    compute_weighted_cer,
    compute_weighted_wer,
    levenshtein_distance,
    normalize_text,
    strip_role_tags,
)


# --- levenshtein_distance -----------------------------------------------

def test_levenshtein_identical():
    assert levenshtein_distance("kitten", "kitten") == 0
    assert levenshtein_distance(["a", "b"], ["a", "b"]) == 0


def test_levenshtein_classic_example():
    # textbook case, widely cited expected distance
    assert levenshtein_distance("kitten", "sitting") == 3


def test_levenshtein_empty_sequences():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance([], []) == 0


def test_levenshtein_words():
    ref = "the quick brown fox".split()
    hyp = "the quick brown".split()
    assert levenshtein_distance(ref, hyp) == 1  # one deletion


# --- role tag stripping / normalization ----------------------------------

def test_strip_role_tags_keeps_text_after_last_tag():
    assert strip_role_tags("user\nTranscribe this\nassistant\nBy this publique Act") == "\nBy this publique Act"


def test_strip_role_tags_no_tags_present():
    assert strip_role_tags("By this publique Act and Instrument") == "By this publique Act and Instrument"


def test_normalize_text_collapses_whitespace_and_line_endings():
    assert normalize_text("By this  publique\r\nAct   and\ttoday") == "By this publique Act and today"


def test_normalize_text_does_not_touch_spelling_or_punctuation():
    # archaic spelling / punctuation must survive untouched — never "fixed"
    assert normalize_text("Publiqve Instrvment, ye olde towne.") == "Publiqve Instrvment, ye olde towne."


def test_normalize_text_handles_nan():
    assert normalize_text(float("nan")) == ""
    assert normalize_text(None) == ""


# --- compute_weighted_wer / compute_weighted_cer -------------------------

def _gt(ids, targets):
    return pd.DataFrame({"ID": ids, "Target": targets})


def test_perfect_match_scores_zero():
    gt = _gt(["a", "b"], ["the quick brown fox", "go home"])
    pred = _gt(["a", "b"], ["the quick brown fox", "go home"])

    wer = compute_weighted_wer(gt, pred)
    cer = compute_weighted_cer(gt, pred)

    assert wer["score"] == 0.0
    assert cer["score"] == 0.0
    assert wer["matched"] == 2
    assert wer["total_reference"] == 2
    assert (wer["per_sample"]["edits"] == 0).all()


def test_completely_empty_predictions_score_max_penalty():
    gt = _gt(["a", "b"], ["the quick brown fox", "go home"])
    pred = _gt(["a", "b"], ["", ""])

    wer = compute_weighted_wer(gt, pred)
    cer = compute_weighted_cer(gt, pred)

    # every edits_i == L_i (full deletion), so the length-normalized ratio
    # is forced to exactly 1.0 regardless of weighting
    assert wer["score"] == pytest.approx(1.0)
    assert cer["score"] == pytest.approx(1.0)
    assert list(wer["per_sample"]["edits"]) == list(wer["per_sample"]["length"])


def test_missing_prediction_ids_treated_as_empty_not_dropped():
    gt = _gt(["a", "b", "c"], ["the quick brown fox", "go home", "hello"])
    pred = _gt(["a"], ["the quick brown fox"])  # b and c never predicted

    wer = compute_weighted_wer(gt, pred)

    assert wer["total_reference"] == 3
    assert wer["matched"] == 1
    assert len(wer["per_sample"]) == 3  # b and c still scored, not dropped

    row_b = wer["per_sample"].loc[wer["per_sample"]["ID"] == "b"].iloc[0]
    row_c = wer["per_sample"].loc[wer["per_sample"]["ID"] == "c"].iloc[0]
    assert row_b["edits"] == row_b["length"] == 2  # "go home" -> "" is 2 deletions
    assert row_c["edits"] == row_c["length"] == 1  # "hello" -> "" is 1 deletion

    # missing ID scored the same as an explicit empty-string prediction
    gt2 = _gt(["b"], ["go home"])
    pred_missing = _gt([], [])
    pred_empty = _gt(["b"], [""])
    assert compute_weighted_wer(gt2, pred_missing)["score"] == pytest.approx(
        compute_weighted_wer(gt2, pred_empty)["score"]
    )


def test_role_tags_stripped_before_scoring():
    gt = _gt(["a"], ["By this publique Act"])
    pred = _gt(["a"], ["user\nTranscribe\nassistant\nBy this publique Act"])

    wer = compute_weighted_wer(gt, pred)
    assert wer["score"] == 0.0


def test_per_sample_columns():
    gt = _gt(["a"], ["go home"])
    pred = _gt(["a"], ["go"])
    wer = compute_weighted_wer(gt, pred)
    assert list(wer["per_sample"].columns) == ["ID", "edits", "length", "weight", "contribution"]


# --- worked example -------------------------------------------------------
# Two-sample toy case: ref/hyp differ by exactly one trailing word each.
# Under the literal formula sum(edits_i * W_i) / sum(W_i), this is
# degenerate (both edits_i == 1, so the result is forced to exactly 1.0
# regardless of weighting) — verified by hand and confirmed not to match
# Zindi's documented ~0.3090 for this input under any weighting scheme we
# could find. Per explicit direction, this module instead normalizes by
# weighted reference length: sum(edits_i * W_i) / sum(L_i * W_i), a proper
# bounded weighted mean of per-sample error rates. That formula evaluates
# to ~0.13284 for this input (verified independently against the DP edit
# distance + formula above) — not 0.3090, since the two formulas are not
# equivalent. This test locks in the length-normalized formula's own
# correct value so a future accidental change to the formula is caught.

def test_worked_toy_example_length_normalized_wer():
    gt = _gt(
        ["a", "b"],
        ["the quick brown fox jumps over the lazy dog today", "go home"],
    )
    pred = _gt(
        ["a", "b"],
        ["the quick brown fox jumps over the lazy dog", "go"],
    )

    wer = compute_weighted_wer(gt, pred)
    assert wer["score"] == pytest.approx(0.13284, abs=1e-4)


# --- compute_final_score --------------------------------------------------

def test_compute_final_score_blends_50_50():
    gt = _gt(["a"], ["go home"])
    pred = _gt(["a"], ["go"])

    result = compute_final_score(gt, pred)
    assert result["final"] == pytest.approx(0.5 * result["cer"] + 0.5 * result["wer"])
    assert set(["ID", "edits_wer", "length_wer", "edits_cer", "length_cer"]).issubset(
        result["per_sample_report"].columns
    )
