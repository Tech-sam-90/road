"""
Lightweight lexicon + n-gram rescoring pass over Kraken's decoded output.

Why word-level post-hoc rescoring rather than lattice-level LM integration:
kraken's rpred() already returns a fully CTC-decoded string per line — it
doesn't expose the per-timestep lattice/logits needed for true lattice-level
LM rescoring without reaching into kraken.lib.models' internal decode path,
which is version-fragile and, with no trained checkpoint to validate
against yet, premature to build. This instead applies a noisy-channel-style
correction: for each word in the decoded output that ISN'T in the training
vocabulary, look for a nearby (edit-distance-bounded) in-vocabulary word
whose bigram probability given the previous (already-accepted) word is
higher, and substitute only then. In-vocabulary words are never touched,
so this can only affect words the model already got wrong by spelling.

Not yet validated against a trained model (none exists at time of writing —
training happens on Colab). Compare val_split scores with/without
--lm_rescore once a checkpoint exists before trusting this by default.
"""

import argparse
import collections
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.metrics.weighted_wer_cer import levenshtein_distance  # reuse, don't duplicate

START = "<s>"


class LexiconLM:
    def __init__(self, texts, max_edit_distance=2):
        self.max_edit_distance = max_edit_distance
        self.unigrams = collections.Counter()
        self.bigrams = collections.Counter()
        for text in texts:
            words = str(text).split()
            if not words:
                continue
            self.unigrams.update(words)
            self.bigrams.update(zip([START] + words, words))
        self.vocab = set(self.unigrams)
        self.total_unigrams = sum(self.unigrams.values())

    def word_log_prob(self, word, prev_word):
        """Bigram probability given prev_word, Laplace-smoothed, backing off
        to unigram probability when the bigram was never seen."""
        prev_count = self.total_unigrams if prev_word == START else self.unigrams.get(prev_word, 0)
        bigram_count = self.bigrams.get((prev_word, word), 0)
        if bigram_count > 0 and prev_count > 0:
            return math.log((bigram_count + 1) / (prev_count + len(self.vocab)))
        unigram_count = self.unigrams.get(word, 0)
        return math.log((unigram_count + 1) / (self.total_unigrams + len(self.vocab)))

    def candidates(self, word):
        """In-vocabulary words within max_edit_distance of `word`. O(|vocab|)
        per call — fine for a train-set-sized lexicon (thousands of words),
        not meant for scale."""
        out = []
        for cand in self.vocab:
            if abs(len(cand) - len(word)) > self.max_edit_distance:
                continue
            d = levenshtein_distance(word, cand)
            if d <= self.max_edit_distance:
                out.append((cand, d))
        return out

    def rescore_sentence(self, text):
        words = str(text).split()
        prev = START
        out_words = []
        for word in words:
            if word in self.vocab:
                out_words.append(word)
                prev = word
                continue

            candidates = self.candidates(word)
            if not candidates:
                out_words.append(word)  # nothing close enough in-vocab — leave as-is
                prev = word
                continue

            best_word, best_dist = max(
                candidates, key=lambda c: (self.word_log_prob(c[0], prev), -c[1])
            )
            out_words.append(best_word)
            prev = best_word
        return " ".join(out_words)


def build_lexicon_lm(train_csv, max_edit_distance=2):
    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    return LexiconLM(df["Target"].astype(str).tolist(), max_edit_distance=max_edit_distance)


def main():
    parser = argparse.ArgumentParser(
        description="Rescore a predictions CSV's Target column against a train-set lexicon."
    )
    parser.add_argument("--train_csv", required=True, help="CSV to build the lexicon/bigram LM from (ID, Target)")
    parser.add_argument("--pred_csv", required=True, help="CSV with raw predictions to rescore (ID, Target)")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_edit_distance", type=int, default=2)
    args = parser.parse_args()

    lm = build_lexicon_lm(args.train_csv, max_edit_distance=args.max_edit_distance)
    pred_df = pd.read_csv(args.pred_csv, encoding="utf-8-sig")
    pred_df["Target"] = pred_df["Target"].astype(str).apply(lm.rescore_sentence)
    pred_df.to_csv(args.output_csv, index=False)
    print(f"[done] rescored {len(pred_df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
