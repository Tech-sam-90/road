from src.kraken.lm_rescore import LexiconLM


def test_in_vocabulary_words_are_never_touched():
    lm = LexiconLM(["the quick brown fox", "the lazy dog"])
    assert lm.rescore_sentence("the quick brown fox") == "the quick brown fox"


def test_oov_word_corrected_to_close_lexicon_word():
    # "hte" (1 edit away from "the") should get corrected given "the" is
    # both in-vocabulary and a much more probable opener
    lm = LexiconLM(["the quick brown fox", "the lazy dog", "the quick fox runs"])
    assert lm.rescore_sentence("hte quick brown fox") == "the quick brown fox"


def test_oov_word_with_no_close_candidate_is_left_alone():
    lm = LexiconLM(["the quick brown fox"], max_edit_distance=1)
    # "zzzzzzzzzz" isn't within edit distance 1 of anything in the lexicon
    assert lm.rescore_sentence("the zzzzzzzzzz fox") == "the zzzzzzzzzz fox"


def test_empty_sentence():
    lm = LexiconLM(["the quick brown fox"])
    assert lm.rescore_sentence("") == ""


def test_bigram_context_breaks_ties_between_equidistant_candidates():
    # "cot" is edit-distance 1 from both "cat" and "cog"; "cat" should win
    # because it's the far more probable continuation after "the".
    lm = LexiconLM(["the cat sat"] * 5 + ["a cog turned"])
    assert lm.rescore_sentence("the cot sat") == "the cat sat"


def test_max_edit_distance_is_respected():
    lm = LexiconLM(["hello world"], max_edit_distance=1)
    candidates = lm.candidates("hellox")  # edit distance 1 from "hello"
    assert ("hello", 1) in candidates
    far_candidates = lm.candidates("xyzxyzxyz")
    assert far_candidates == []
