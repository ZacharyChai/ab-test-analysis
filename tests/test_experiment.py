"""Unit tests for src/experiment.py, plus one frozen-value regression test.

Ordering mirrors the module docstring's pipeline: load/clean -> SRM -> power
-> primary test -> novelty -> segments -> guardrail. Synthetic-data tests check
each function's logic in isolation; the final test replays the pipeline on the
real committed dataset and checks the headline numbers haven't silently moved.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportion_confint

import experiment as ex

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = REPO_ROOT / "data" / "raw" / "ab_data.csv"
FROZEN_RESULTS = REPO_ROOT / "data" / "processed" / "results.json"


# --------------------------------------------------------------------------- #
# clean_experiment
# --------------------------------------------------------------------------- #
def test_clean_experiment_drops_mismatched_pairs_and_duplicate_users():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime(
                ["2017-01-02", "2017-01-02", "2017-01-02", "2017-01-03", "2017-01-02"]
            ),
            "group": ["control", "treatment", "control", "control", "control"],
            "landing_page": ["old_page", "new_page", "new_page", "old_page", "old_page"],
            # user 5 is a same-day duplicate of user 1's arm, distinct id -> not a dup;
            # instead duplicate user_id 1 appears twice via a second row below.
        }
    )
    # user_id 1 visits twice (once on day 2) -> should keep only the earlier row.
    dup_row = pd.DataFrame(
        {
            "user_id": [1],
            "timestamp": pd.to_datetime(["2017-01-03"]),
            "group": ["control"],
            "landing_page": ["old_page"],
        }
    )
    df = pd.concat([df, dup_row], ignore_index=True)

    clean, report = ex.clean_experiment(df)

    assert report.n_raw == 6
    assert report.n_mismatched_pairs == 1  # user_id 3: control/new_page is invalid
    assert report.n_duplicate_users == 1  # user_id 1's later visit
    assert report.n_clean == 4
    assert len(clean) == report.n_clean
    assert clean["user_id"].is_unique
    # the surviving row for user 1 must be the earlier timestamp
    kept = clean.loc[clean["user_id"] == 1, "timestamp"].iloc[0]
    assert kept == pd.Timestamp("2017-01-02")


def test_clean_experiment_report_dict_matches_fields():
    df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "timestamp": pd.to_datetime(["2017-01-02", "2017-01-02"]),
            "group": ["control", "treatment"],
            "landing_page": ["old_page", "new_page"],
        }
    )
    _, report = ex.clean_experiment(df)
    d = report.as_dict()
    assert d["rows_raw"] == 2
    assert d["rows_clean"] == 2
    assert d["pct_dropped"] == 0.0


# --------------------------------------------------------------------------- #
# srm_check
# --------------------------------------------------------------------------- #
def test_srm_check_balanced_split_is_not_flagged():
    result = ex.srm_check({"control": 50_000, "treatment": 50_050})
    assert result.p_value > 0.01
    assert result.is_mismatch is False


def test_srm_check_flags_a_real_mismatch():
    # 45k vs 55k on an intended 50/50 split is a large, obvious SRM.
    result = ex.srm_check({"control": 45_000, "treatment": 55_000})
    assert result.p_value < 0.01
    assert result.is_mismatch is True


def test_srm_check_matches_hand_computed_chi2():
    counts = {"control": 480, "treatment": 520}
    result = ex.srm_check(counts)
    total = 1000
    expected_each = total * 0.5
    hand_chi2 = sum((counts[k] - expected_each) ** 2 / expected_each for k in counts)
    assert result.chi2 == pytest.approx(hand_chi2, rel=1e-9)
    assert result.ratio_observed == pytest.approx(520 / 480, rel=1e-9)


# --------------------------------------------------------------------------- #
# power / sample size
# --------------------------------------------------------------------------- #
def test_required_sample_size_shrinks_as_mde_grows():
    n_small_effect = ex.required_sample_size(baseline_rate=0.12, mde_absolute=0.003)
    n_large_effect = ex.required_sample_size(baseline_rate=0.12, mde_absolute=0.02)
    assert n_large_effect < n_small_effect


def test_mde_for_sample_size_roundtrips_required_sample_size():
    baseline = 0.12
    target_mde = 0.006
    n = ex.required_sample_size(baseline, target_mde)
    achievable_mde = ex.mde_for_sample_size(baseline, n)
    # required_sample_size ceil()s up, so the achievable MDE at that n should be
    # at or just below what was asked for, not meaningfully larger.
    assert achievable_mde == pytest.approx(target_mde, abs=5e-4)
    assert achievable_mde <= target_mde + 5e-4


# --------------------------------------------------------------------------- #
# two_proportion_test
# --------------------------------------------------------------------------- #
def _synthetic_arms(n_control, conv_control, n_treatment, conv_treatment):
    rows = []
    rows += [{"group": "control", "converted": 1}] * conv_control
    rows += [{"group": "control", "converted": 0}] * (n_control - conv_control)
    rows += [{"group": "treatment", "converted": 1}] * conv_treatment
    rows += [{"group": "treatment", "converted": 0}] * (n_treatment - conv_treatment)
    return pd.DataFrame(rows)


def test_two_proportion_test_rates_and_lift_are_correct():
    df = _synthetic_arms(n_control=100, conv_control=20, n_treatment=100, conv_treatment=30)
    t = ex.two_proportion_test(df)

    assert t.rate_control == pytest.approx(0.20)
    assert t.rate_treatment == pytest.approx(0.30)
    assert t.abs_lift == pytest.approx(0.10)
    assert t.rel_lift == pytest.approx(0.50)
    # pooled two-proportion z for this exact table, computed by hand.
    assert t.z_stat == pytest.approx(1.6330, abs=1e-3)
    assert t.p_value == pytest.approx(0.1025, abs=1e-3)


def test_two_proportion_test_chi2_equals_z_squared_for_2x2():
    # For an uncorrected 2x2 table, chi-square == z^2. Cheap cross-check that
    # the two statistics in the dataclass agree with each other.
    df = _synthetic_arms(n_control=300, conv_control=40, n_treatment=300, conv_treatment=55)
    t = ex.two_proportion_test(df)
    assert t.chi2_stat == pytest.approx(t.z_stat ** 2, rel=1e-6)


def test_two_proportion_test_wald_ci_is_symmetric_around_the_lift():
    df = _synthetic_arms(n_control=500, conv_control=60, n_treatment=500, conv_treatment=75)
    t = ex.two_proportion_test(df)
    lo, hi = t.ci_abs
    assert (t.abs_lift - lo) == pytest.approx(hi - t.abs_lift, rel=1e-6)
    assert lo < t.abs_lift < hi


def test_two_proportion_test_no_lift_gives_high_p_value():
    df = _synthetic_arms(n_control=1000, conv_control=120, n_treatment=1000, conv_treatment=121)
    t = ex.two_proportion_test(df)
    assert t.p_value > 0.5


# --------------------------------------------------------------------------- #
# wilson_ci
# --------------------------------------------------------------------------- #
def test_wilson_ci_matches_statsmodels_reference():
    for successes, n in [(50, 100), (12, 145274), (0, 50), (100, 100)]:
        lo, hi = ex.wilson_ci(successes, n)
        ref_lo, ref_hi = proportion_confint(successes, n, alpha=0.05, method="wilson")
        assert lo == pytest.approx(ref_lo, abs=1e-9)
        assert hi == pytest.approx(ref_hi, abs=1e-9)


def test_wilson_ci_handles_zero_n():
    lo, hi = ex.wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------- #
# daily_effect
# --------------------------------------------------------------------------- #
def test_daily_effect_computes_expected_daily_and_cumulative_lift():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2017-01-02"] * 20 + ["2017-01-03"] * 20
            ),
            "group": (["control"] * 10 + ["treatment"] * 10) * 2,
            "converted": (
                [1] * 2 + [0] * 8  # day 1 control: 2/10 = 0.20
                + [1] * 4 + [0] * 6  # day 1 treatment: 4/10 = 0.40
                + [1] * 3 + [0] * 7  # day 2 control: 3/10 = 0.30
                + [1] * 3 + [0] * 7  # day 2 treatment: 3/10 = 0.30
            ),
        }
    )
    out = ex.daily_effect(df).sort_values("date").reset_index(drop=True)

    assert out.loc[0, "rate_control"] == pytest.approx(0.20)
    assert out.loc[0, "rate_treatment"] == pytest.approx(0.40)
    assert out.loc[0, "daily_lift"] == pytest.approx(0.20)

    assert out.loc[1, "rate_control"] == pytest.approx(0.30)
    assert out.loc[1, "rate_treatment"] == pytest.approx(0.30)
    assert out.loc[1, "daily_lift"] == pytest.approx(0.0)

    # cumulative after day 2: control (2+3)/20=0.25, treatment (4+3)/20=0.35
    assert out.loc[1, "cum_lift"] == pytest.approx(0.35 - 0.25)


# --------------------------------------------------------------------------- #
# segment_effects
# --------------------------------------------------------------------------- #
def test_segment_effects_surfaces_a_sign_flip():
    # Aggregate lift is positive, but segment "B" alone is negative -> a real
    # Simpson's-paradox-style reversal that segment_effects must not hide.
    seg_a = _synthetic_arms(n_control=100, conv_control=10, n_treatment=100, conv_treatment=40)
    seg_a["region"] = "A"
    seg_b = _synthetic_arms(n_control=100, conv_control=40, n_treatment=100, conv_treatment=10)
    seg_b["region"] = "B"
    df = pd.concat([seg_a, seg_b], ignore_index=True)

    overall = ex.two_proportion_test(df)
    by_region = ex.segment_effects(df, "region")

    row_a = by_region.loc[by_region["region"] == "A", "abs_lift_pp"].iloc[0]
    row_b = by_region.loc[by_region["region"] == "B", "abs_lift_pp"].iloc[0]
    assert row_a > 0
    assert row_b < 0
    # confirms the two segments really do disagree in sign with each other
    assert np.sign(row_a) != np.sign(row_b)
    assert overall.abs_lift == pytest.approx(0.0, abs=1e-9)  # they cancel in aggregate


def test_segment_effects_skips_a_segment_with_only_one_arm():
    df = _synthetic_arms(n_control=50, conv_control=5, n_treatment=50, conv_treatment=8)
    df["region"] = "A"
    control_only = pd.DataFrame({"group": ["control"], "converted": [1]})
    control_only["region"] = "B"
    df = pd.concat([df, control_only], ignore_index=True)

    out = ex.segment_effects(df, "region")
    assert set(out["region"]) == {"A"}  # region B had no treatment arm, so it's dropped


# --------------------------------------------------------------------------- #
# guardrail_delivery_error_rate
# --------------------------------------------------------------------------- #
def _raw_with_delivery_errors(n_control, err_control, n_treatment, err_treatment):
    rows = []
    rows += [{"group": "control", "landing_page": "old_page"}] * (n_control - err_control)
    rows += [{"group": "control", "landing_page": "new_page"}] * err_control
    rows += [{"group": "treatment", "landing_page": "new_page"}] * (n_treatment - err_treatment)
    rows += [{"group": "treatment", "landing_page": "old_page"}] * err_treatment
    return pd.DataFrame(rows)


def test_guardrail_flags_a_real_instrumentation_imbalance():
    raw = _raw_with_delivery_errors(
        n_control=5000, err_control=50, n_treatment=5000, err_treatment=250
    )
    g = ex.guardrail_delivery_error_rate(raw)
    assert g.value_control == pytest.approx(0.01)
    assert g.value_treatment == pytest.approx(0.05)
    assert bool(g.worse_in_treatment) is True
    assert g.p_value < 0.05


def test_guardrail_does_not_flag_a_balanced_error_rate():
    raw = _raw_with_delivery_errors(
        n_control=5000, err_control=60, n_treatment=5000, err_treatment=65
    )
    g = ex.guardrail_delivery_error_rate(raw)
    assert bool(g.worse_in_treatment) is False


# --------------------------------------------------------------------------- #
# Integration / frozen-value regression test
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not REAL_DATA.exists(), reason="data/raw/ab_data.csv not present in this checkout"
)
def test_pipeline_headline_matches_committed_results_json():
    """Guards against silently changing the estimator's output.

    Re-runs the real pipeline on the committed dataset and checks the numbers
    against data/processed/results.json (frozen at the time the README's
    headline figures were written). A change here means either a real bug fix
    (update the frozen file deliberately) or a regression (fix the code) --
    this test is what tells the two apart.
    """
    import json

    frozen = json.loads(FROZEN_RESULTS.read_text())

    raw = ex.load_experiment(REAL_DATA.parent)
    clean, report = ex.clean_experiment(raw)
    assert report.as_dict() == frozen["cleaning"]

    srm = ex.srm_check(clean["group"].value_counts())
    assert srm.is_mismatch == frozen["srm"]["is_mismatch"]
    assert srm.p_value == pytest.approx(frozen["srm"]["p_value"], abs=1e-6)

    t = ex.two_proportion_test(clean)
    assert t.abs_lift * 100 == pytest.approx(frozen["primary"]["abs_lift_pp"], abs=1e-6)
    assert t.p_value == pytest.approx(frozen["primary"]["p_value"], abs=1e-6)
    assert t.rate_control == pytest.approx(frozen["primary"]["rate_control"], abs=1e-9)
    assert t.rate_treatment == pytest.approx(frozen["primary"]["rate_treatment"], abs=1e-9)
