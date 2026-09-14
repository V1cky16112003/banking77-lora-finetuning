from src.evaluate_core import (
    bootstrap_confidence_interval,
    compute_confusion_ranking,
    compute_macro_f1,
    compute_per_class_f1,
    format_label_for_display,
    parse_label,
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
