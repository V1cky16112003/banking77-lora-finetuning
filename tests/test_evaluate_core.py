import json
import math

import pytest

from src.evaluate_core import (
    aurc,
    bootstrap_confidence_interval,
    brier_score,
    compute_confusion_ranking,
    compute_macro_f1,
    compute_per_class_f1,
    coverage_at_risk,
    expected_calibration_error,
    format_label_for_display,
    load_resumable_jsonl,
    parse_label,
    risk_coverage_curve,
    softmax_scores,
)

LABELS = ["card_lost", "balance_not_updated", "wrong_exchange_rate_for_cash_withdrawal"]


# --- parse_label ---

def test_exact_match():
    assert parse_label("card_lost", LABELS) == "card_lost"


def test_exact_match_case_and_whitespace_insensitive():
    assert parse_label("  Card_Lost  ", LABELS) == "card_lost"


def test_fuzzy_match_resolves_near_miss():
    # near-miss typo/variation of a real label
    assert parse_label("card_losst", LABELS) == "card_lost"


def test_unresolvable_returns_none_not_dropped():
    result = parse_label("completely unrelated gibberish text", LABELS)
    assert result is None


def test_denominator_invariant_unresolved_still_counted():
    # Regression-shaped: an unresolved prediction must still occupy a slot
    # in y_pred so macro-F1's denominator is never silently shrunk.
    y_true = ["card_lost", "balance_not_updated"]
    y_pred_raw = ["garbage output", "balance_not_updated"]
    y_pred = [parse_label(p, LABELS) for p in y_pred_raw]

    assert len(y_pred) == len(y_true)
    assert y_pred[0] is None
    assert y_pred[1] == "balance_not_updated"


# --- compute_macro_f1 / compute_per_class_f1 ---

def test_macro_f1_perfect_predictions():
    # Macro-F1 averages over the full configured label set (LABELS), so a
    # label with zero support in this slice (wrong_exchange_rate...) scores
    # 0 and pulls the average down — correct behavior for a real eval where
    # all classes appear, but for a "perfect prediction" unit test we score
    # only over the labels actually exercised here.
    y_true = ["card_lost", "balance_not_updated", "card_lost"]
    y_pred = ["card_lost", "balance_not_updated", "card_lost"]
    used_labels = ["card_lost", "balance_not_updated"]

    assert compute_macro_f1(y_true, y_pred, used_labels) == 1.0


def test_macro_f1_with_unresolved_prediction_is_penalized():
    y_true = ["card_lost", "balance_not_updated"]
    y_pred = ["card_lost", None]  # unresolved

    score = compute_macro_f1(y_true, y_pred, LABELS)
    assert 0.0 < score < 1.0


def test_per_class_f1_hand_crafted_matrix():
    y_true = ["card_lost", "card_lost", "balance_not_updated"]
    y_pred = ["card_lost", "balance_not_updated", "balance_not_updated"]

    scores = compute_per_class_f1(y_true, y_pred, LABELS)

    assert set(scores.keys()) == set(LABELS)
    # card_lost: 1 TP, 1 FN, 0 FP -> F1 = 2*1/(2*1+1+0) = 0.666...
    assert round(scores["card_lost"], 3) == 0.667
    # wrong_exchange_rate never appears -> F1 = 0
    assert scores["wrong_exchange_rate_for_cash_withdrawal"] == 0.0


# --- compute_confusion_ranking ---

def test_confusion_ranking_orders_by_total_misclassification_involvement():
    y_true = ["card_lost", "card_lost", "card_lost", "balance_not_updated"]
    y_pred = [
        "balance_not_updated",
        "balance_not_updated",
        "card_lost",  # correct, not a confusion pair
        "card_lost",
    ]

    ranking = compute_confusion_ranking(y_true, y_pred, top_n=5)

    assert ranking[0] == ("card_lost", "balance_not_updated", 2)
    assert ("balance_not_updated", "card_lost", 1) in ranking


def test_confusion_ranking_respects_top_n():
    y_true = ["card_lost"] * 3 + ["balance_not_updated"] * 3
    y_pred = ["balance_not_updated"] * 3 + ["wrong_exchange_rate_for_cash_withdrawal"] * 3

    ranking = compute_confusion_ranking(y_true, y_pred, top_n=1)

    assert len(ranking) == 1


# --- bootstrap_confidence_interval ---

def test_bootstrap_ci_bounds_contain_point_estimate():
    y_true = ["card_lost", "balance_not_updated"] * 20  # 40 elements
    y_pred = ["card_lost", "balance_not_updated"] * 19 + ["card_lost", "card_lost"]  # 40 elements

    point, lo, hi = bootstrap_confidence_interval(y_true, y_pred, LABELS, n_resamples=200)

    assert lo <= point <= hi


def test_bootstrap_ci_is_deterministic_given_seed():
    y_true = ["card_lost", "balance_not_updated"] * 10
    y_pred = ["card_lost", "card_lost"] * 10

    result_a = bootstrap_confidence_interval(y_true, y_pred, LABELS, n_resamples=100, seed=7)
    result_b = bootstrap_confidence_interval(y_true, y_pred, LABELS, n_resamples=100, seed=7)

    assert result_a == result_b


# --- format_label_for_display ---

def test_format_label_for_display_underscores_to_title_case():
    assert format_label_for_display("card_payment_wrong_exchange_rate") == "Card Payment Wrong Exchange Rate"


def test_format_label_for_display_single_word():
    assert format_label_for_display("card_lost") == "Card Lost"


# --- calibration metrics (Pointwise Phase 0) ---


def test_softmax_scores_sums_to_one_and_survives_large_magnitudes():
    probs = softmax_scores([-1000.0, -1001.0, -1002.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]
    assert probs[0] == pytest.approx(1 / (1 + math.e**-1 + math.e**-2))


def test_brier_perfect_and_worst():
    assert brier_score([[1.0, 0.0]], [0]) == 0.0
    assert brier_score([[0.0, 1.0]], [0]) == 2.0


def test_brier_uniform_three_way():
    # (2/3)^2 + (1/3)^2 + (1/3)^2 = 2/3
    assert brier_score([[1 / 3, 1 / 3, 1 / 3]], [0]) == pytest.approx(2 / 3)


def test_ece_zero_when_confidence_matches_accuracy():
    # 10 predictions at 0.8 confidence, 8 correct -> perfectly calibrated
    conf = [0.8] * 10
    correct = [True] * 8 + [False] * 2
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)


def test_ece_overconfident():
    # always says 0.99 but is right half the time -> ECE 0.49
    conf = [0.99] * 4
    correct = [True, False, True, False]
    assert expected_calibration_error(conf, correct) == pytest.approx(0.49)


def test_risk_coverage_orders_by_confidence():
    conf = [0.9, 0.2, 0.6]
    correct = [True, False, True]
    assert risk_coverage_curve(conf, correct) == [
        (1 / 3, 0.0),
        (2 / 3, 0.0),
        (1.0, 1 / 3),
    ]


def test_aurc_perfect_ranking_beats_inverted_ranking():
    correct = [True, True, False, False]
    good = aurc([0.9, 0.8, 0.2, 0.1], correct)
    bad = aurc([0.1, 0.2, 0.8, 0.9], correct)
    assert good < bad


def test_coverage_at_risk():
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [True, True, False, True]
    assert coverage_at_risk(conf, correct, max_risk=0.0) == 0.5
    assert coverage_at_risk(conf, correct, max_risk=0.25) == 1.0


def test_load_resumable_jsonl_missing_file_is_empty(tmp_path):
    assert load_resumable_jsonl(tmp_path / "none.jsonl") == []


def test_load_resumable_jsonl_drops_torn_last_line_and_truncates(tmp_path):
    path = tmp_path / "preds.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "pro', encoding="utf-8")
    assert load_resumable_jsonl(path) == [{"a": 1}, {"a": 2}]
    # File now ends on a whole record, so appending the next one stays valid JSONL.
    with path.open("a", encoding="utf-8") as f:
        f.write('{"a": 3}\n')
    assert load_resumable_jsonl(path) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_load_resumable_jsonl_raises_on_corruption_before_last_line(tmp_path):
    path = tmp_path / "preds.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_resumable_jsonl(path)
