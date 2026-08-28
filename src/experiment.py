"""Core experimental-analysis routines for the landing-page A/B test.

Everything the notebook and the tests rely on lives here as plain functions, so
the same code that produces the memo is the code under test. Nothing in this
module reads global state or plots — callers pass data in and get numbers back.

Pipeline order is deliberate and matches `analysis/ab_test_analysis.ipynb`:

    load_experiment() -> clean_experiment() -> srm_check()   [gate]
                      -> two_proportion_test() / effect CI
                      -> daily_effect()  (novelty / primacy)
                      -> segment_effects() (Simpson's-paradox check)
                      -> guardrail_repeat_visits()

The SRM check comes *before* any outcome analysis on purpose: if randomisation
is broken, none of the downstream numbers mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    proportions_ztest,
)

# The randomisation was intended to be a 50/50 split of traffic between the
# existing page (control) and the new page (treatment).
INTENDED_SPLIT = {"control": 0.5, "treatment": 0.5}

# treatment must see new_page, control must see old_page. Any other pairing is a
# delivery bug, not a real assignment.
VALID_PAIRS = {("control", "old_page"), ("treatment", "new_page")}

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# --------------------------------------------------------------------------- #
# Load & clean
# --------------------------------------------------------------------------- #
def load_experiment(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Read the raw event log and join on country. One row per page view."""
    data_dir = Path(data_dir)
    events = pd.read_csv(data_dir / "ab_data.csv", parse_dates=["timestamp"])
    countries = pd.read_csv(data_dir / "countries.csv")
    return events.merge(countries, on="user_id", how="left")


@dataclass
class CleaningReport:
    n_raw: int
    n_mismatched_pairs: int          # group/landing_page disagree
    n_duplicate_users: int           # users with >1 retained row after pair fix
    n_clean: int

    def as_dict(self) -> dict:
        return {
            "rows_raw": self.n_raw,
            "rows_dropped_mismatched_pair": self.n_mismatched_pairs,
            "rows_dropped_duplicate_user": self.n_duplicate_users,
            "rows_clean": self.n_clean,
            "pct_dropped": round(100 * (1 - self.n_clean / self.n_raw), 3),
        }


def clean_experiment(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop rows where assignment and page disagree, then de-duplicate users.

    Returns the cleaned frame and a report of exactly what was removed, so the
    notebook can show it rather than silently shrinking the dataset.
    """
    n_raw = len(df)

    paired = df.apply(
        lambda r: (r["group"], r["landing_page"]) in VALID_PAIRS, axis=1
    )
    df_paired = df[paired].copy()
    n_mismatched = n_raw - len(df_paired)

    # A handful of users appear twice. Keep their first observed visit only.
    df_paired = df_paired.sort_values("timestamp")
    dupes = df_paired["user_id"].duplicated(keep="first").sum()
    df_clean = df_paired.drop_duplicates(subset="user_id", keep="first")

    report = CleaningReport(
        n_raw=n_raw,
        n_mismatched_pairs=int(n_mismatched),
        n_duplicate_users=int(dupes),
        n_clean=len(df_clean),
    )
    return df_clean.reset_index(drop=True), report


# --------------------------------------------------------------------------- #
# Step 1 — Sample Ratio Mismatch (run before looking at conversion)
# --------------------------------------------------------------------------- #
@dataclass
class SRMResult:
    observed: dict[str, int]
    expected: dict[str, float]
    chi2: float
    p_value: float
    ratio_observed: float            # treatment / control

    @property
    def is_mismatch(self) -> bool:
        """Convention: flag SRM at p < 0.01 (Twitman / Kohavi guidance)."""
        return self.p_value < 0.01

    def summary(self) -> str:
        verdict = "SRM DETECTED — do not trust downstream results" if self.is_mismatch \
            else "no SRM — split is consistent with the intended ratio"
        return (f"chi2={self.chi2:.3f}, p={self.p_value:.4f}  ->  {verdict}")


def srm_check(group_counts: pd.Series | dict,
              intended: dict[str, float] = INTENDED_SPLIT) -> SRMResult:
    """Chi-square goodness-of-fit of arm sizes against the intended split."""
    observed = {k: int(group_counts[k]) for k in intended}
    total = sum(observed.values())
    expected = {k: total * w for k, w in intended.items()}

    obs = np.array([observed[k] for k in intended])
    exp = np.array([expected[k] for k in intended])
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, df=len(intended) - 1))

    return SRMResult(
        observed=observed,
        expected=expected,
        chi2=chi2,
        p_value=p,
        ratio_observed=observed["treatment"] / observed["control"],
    )


# --------------------------------------------------------------------------- #
# Step 2 — Power / sample size ("what we should have specified going in")
# --------------------------------------------------------------------------- #
def required_sample_size(baseline_rate: float,
                         mde_absolute: float,
                         alpha: float = 0.05,
                         power: float = 0.80,
                         two_sided: bool = True) -> int:
    """Per-arm sample size to detect an absolute lift of `mde_absolute`.

    Uses the arcsine-transformed effect size (Cohen's h) via statsmodels, which
    is the standard approach for a two-proportion z-test power calculation.
    """
    effect = proportion_effectsize(baseline_rate + mde_absolute, baseline_rate)
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect),
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided" if two_sided else "larger",
    )
    return int(np.ceil(n))


def mde_for_sample_size(baseline_rate: float,
                        n_per_arm: int,
                        alpha: float = 0.05,
                        power: float = 0.80,
                        two_sided: bool = True) -> float:
    """Inverse of the above: smallest absolute lift detectable at given n."""
    analysis = NormalIndPower()
    h = analysis.solve_power(
        effect_size=None,
        nobs1=n_per_arm,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided" if two_sided else "larger",
    )
    # invert Cohen's h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2)) for p1
    phi2 = 2 * np.arcsin(np.sqrt(baseline_rate))
    p1 = np.sin((h + phi2) / 2) ** 2
    return float(p1 - baseline_rate)


# --------------------------------------------------------------------------- #
# Step 3 — Primary test + effect size with CI
# --------------------------------------------------------------------------- #
@dataclass
class ProportionTest:
    conv_control: int
    n_control: int
    conv_treatment: int
    n_treatment: int
    rate_control: float
    rate_treatment: float
    abs_lift: float                  # treatment - control (percentage points, as a proportion)
    rel_lift: float                  # abs_lift / rate_control
    z_stat: float
    p_value: float                   # two-sided, pooled
    ci_abs: tuple[float, float]      # 95% CI on the absolute lift
    chi2_stat: float
    chi2_p: float

    def summary(self) -> str:
        lo, hi = self.ci_abs
        sig = "significant" if self.p_value < 0.05 else "not significant"
        return (
            f"control {self.rate_control:.4f}  vs  treatment {self.rate_treatment:.4f}\n"
            f"absolute lift {self.abs_lift*100:+.3f} pp "
            f"(95% CI {lo*100:+.3f} to {hi*100:+.3f} pp)\n"
            f"relative lift {self.rel_lift*100:+.2f}%\n"
            f"two-proportion z = {self.z_stat:.3f}, p = {self.p_value:.4f} ({sig} at alpha=0.05)"
        )


def two_proportion_test(df: pd.DataFrame,
                        group_col: str = "group",
                        outcome_col: str = "converted",
                        alpha: float = 0.05) -> ProportionTest:
    """Pooled two-proportion z-test + chi-square + Wald CI on the lift."""
    g = df.groupby(group_col)[outcome_col].agg(["sum", "count"])
    c_c, n_c = int(g.loc["control", "sum"]), int(g.loc["control", "count"])
    c_t, n_t = int(g.loc["treatment", "sum"]), int(g.loc["treatment", "count"])

    p_c, p_t = c_c / n_c, c_t / n_t

    z_stat, p_val = proportions_ztest([c_t, c_c], [n_t, n_c], alternative="two-sided")

    lo, hi = confint_proportions_2indep(
        c_t, n_t, c_c, n_c, method="wald", compare="diff", alpha=alpha
    )

    chi2, chi2_p, _, _ = stats.chi2_contingency(
        [[c_t, n_t - c_t], [c_c, n_c - c_c]], correction=False
    )

    return ProportionTest(
        conv_control=c_c, n_control=n_c,
        conv_treatment=c_t, n_treatment=n_t,
        rate_control=p_c, rate_treatment=p_t,
        abs_lift=p_t - p_c,
        rel_lift=(p_t - p_c) / p_c,
        z_stat=float(z_stat), p_value=float(p_val),
        ci_abs=(float(lo), float(hi)),
        chi2_stat=float(chi2), chi2_p=float(chi2_p),
    )


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a single proportion (used for per-arm bars)."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


# --------------------------------------------------------------------------- #
# Step 4 — Novelty / primacy: treatment effect over time
# --------------------------------------------------------------------------- #
def daily_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day conversion by arm, the daily lift, and the cumulative lift.

    A new-page effect that decays (novelty) or grows (primacy / learning) across
    the window is a different recommendation than a flat effect.
    """
    d = df.copy()
    d["date"] = d["timestamp"].dt.floor("D")
    daily = (
        d.groupby(["date", "group"])["converted"]
        .agg(["sum", "count"])
        .unstack("group")
    )
    out = pd.DataFrame(index=daily.index)
    out["conv_control"] = daily[("sum", "control")]
    out["n_control"] = daily[("count", "control")]
    out["conv_treatment"] = daily[("sum", "treatment")]
    out["n_treatment"] = daily[("count", "treatment")]
    out["rate_control"] = out["conv_control"] / out["n_control"]
    out["rate_treatment"] = out["conv_treatment"] / out["n_treatment"]
    out["daily_lift"] = out["rate_treatment"] - out["rate_control"]

    cum_c = out["conv_control"].cumsum() / out["n_control"].cumsum()
    cum_t = out["conv_treatment"].cumsum() / out["n_treatment"].cumsum()
    out["cum_lift"] = cum_t - cum_c
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Step 5 — Segment cuts (Simpson's-paradox live check)
# --------------------------------------------------------------------------- #
def segment_effects(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Run the primary test within each level of `by` (country, weekday, ...).

    If the aggregate says one thing and a segment reverses it, that surfaces
    here as a sign flip in `abs_lift_pp`.
    """
    rows = []
    for level, sub in df.groupby(by, observed=True):
        if sub["group"].nunique() < 2:
            continue
        t = two_proportion_test(sub)
        rows.append({
            by: level,
            "n": len(sub),
            "rate_control": t.rate_control,
            "rate_treatment": t.rate_treatment,
            "abs_lift_pp": t.abs_lift * 100,
            "rel_lift_pct": t.rel_lift * 100,
            "p_value": t.p_value,
            "ci_low_pp": t.ci_abs[0] * 100,
            "ci_high_pp": t.ci_abs[1] * 100,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def add_weekday(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["weekday"] = pd.Categorical(
        d["timestamp"].dt.day_name(),
        categories=["Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    return d


# --------------------------------------------------------------------------- #
# Step 6 — Guardrail metric
# --------------------------------------------------------------------------- #
@dataclass
class GuardrailResult:
    metric: str
    value_control: float
    value_treatment: float
    delta: float
    p_value: float
    worse_in_treatment: bool
    note: str

    def summary(self) -> str:
        direction = "WORSE in treatment" if self.worse_in_treatment else "not worse in treatment"
        return (f"{self.metric}: control {self.value_control:.4f} vs "
                f"treatment {self.value_treatment:.4f} "
                f"(delta {self.delta:+.4f}, p={self.p_value:.3f}) -> {direction}")


def guardrail_delivery_error_rate(raw_df: pd.DataFrame) -> GuardrailResult:
    """Guardrail: page-delivery error rate by assigned arm.

    This dataset carries no engagement field (time-on-page, bounce,
    pages/session, order value), so a true business guardrail can't be computed
    here -- see the memo for the metrics a real ship decision would require.

    What we *can* guard is instrumentation: ~1.3% of sessions were served the
    wrong page for their assignment. If that delivery bug were caused by the new
    page, it would land disproportionately on the treatment arm. We test whether
    the error rate is worse in treatment.
    """
    d = raw_df.copy()
    d["delivery_error"] = ~d.apply(
        lambda r: (r["group"], r["landing_page"]) in VALID_PAIRS, axis=1
    )
    g = d.groupby("group")["delivery_error"].agg(["sum", "count"])
    s_c, n_c = int(g.loc["control", "sum"]), int(g.loc["control", "count"])
    s_t, n_t = int(g.loc["treatment", "sum"]), int(g.loc["treatment", "count"])
    r_c, r_t = s_c / n_c, s_t / n_t

    _, p = proportions_ztest([s_t, s_c], [n_t, n_c], alternative="two-sided")

    return GuardrailResult(
        metric="page-delivery error rate",
        value_control=r_c,
        value_treatment=r_t,
        delta=r_t - r_c,
        p_value=float(p),
        worse_in_treatment=(r_t > r_c) and (p < 0.05),
        note=("no engagement guardrail (time-on-page, bounce, pages/session, AOV) "
              "exists in this dataset; this checks instrumentation only"),
    )
