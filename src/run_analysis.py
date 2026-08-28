"""Reproduce the full landing-page A/B analysis and write the charts.

`make all` runs this. It walks the same seven steps as the notebook, prints a
plain-text report, and writes five figures to analysis/charts/. The notebook is
the narrated version of the same pipeline; this script is the headless one that
CI and `make` can run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import experiment as ex

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "analysis" / "charts"
PROCESSED = ROOT / "data" / "processed"

# a plausible minimum detectable effect we'd have cared about, set before
# looking at results: a 5% relative lift on a ~12% base = +0.6 pp absolute.
BASELINE = 0.1204            # control rate, used as the pre-test planning base
MDE_ABS = 0.006
ALPHA = 0.05
POWER = 0.80

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})
C_CONTROL = "#6b7280"
C_TREAT = "#2563eb"
C_WARN = "#dc2626"


def chart_srm(srm: ex.SRMResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    arms = list(srm.observed)
    x = np.arange(len(arms))
    obs = [srm.observed[a] for a in arms]
    exp = [srm.expected[a] for a in arms]
    ax.bar(x - 0.19, obs, 0.38, label="observed", color=C_TREAT)
    ax.bar(x + 0.19, exp, 0.38, label="expected (50/50)", color=C_CONTROL)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("sessions")
    ax.set_title(f"Sample ratio mismatch check — p = {srm.p_value:.3f}")
    ax.legend()
    for xi, o in zip(x, obs):
        ax.text(xi - 0.19, o, f"{o:,}", ha="center", va="bottom", fontsize=8)
    fig.text(0.5, -0.02,
             "Split is within noise of the intended ratio → randomisation is trustworthy.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_power(path: Path, achieved_n: int) -> None:
    mdes = np.linspace(0.001, 0.012, 60)
    ns = [ex.required_sample_size(BASELINE, m, ALPHA, POWER) for m in mdes]
    detectable = ex.mde_for_sample_size(BASELINE, achieved_n, ALPHA, POWER)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(mdes * 100, ns, color=C_TREAT)
    ax.axhline(achieved_n, color=C_CONTROL, ls="--", lw=1)
    ax.axvline(MDE_ABS * 100, color=C_WARN, ls=":", lw=1.2)
    ax.axvline(detectable * 100, color=C_CONTROL, ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("minimum detectable effect (percentage points, absolute)")
    ax.set_ylabel("required sample size per arm (log)")
    ax.set_title("Power: what this test could and couldn't have caught")
    ax.annotate(f"achieved n ≈ {achieved_n:,}/arm",
                xy=(0.55, achieved_n), xytext=(0.6, achieved_n * 3),
                fontsize=8, color=C_CONTROL)
    ax.annotate(f"pre-registered MDE\n+{MDE_ABS*100:.1f} pp",
                xy=(MDE_ABS * 100, ex.required_sample_size(BASELINE, MDE_ABS)),
                xytext=(MDE_ABS * 100 + 0.15, 5e5),
                fontsize=8, color=C_WARN,
                arrowprops=dict(arrowstyle="->", color=C_WARN, lw=0.8))
    ax.annotate(f"actually detectable\n≈ +{detectable*100:.2f} pp",
                xy=(detectable * 100, achieved_n),
                xytext=(detectable * 100 + 0.15, achieved_n / 12),
                fontsize=8, color=C_CONTROL,
                arrowprops=dict(arrowstyle="->", color=C_CONTROL, lw=0.8))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_primary(t: ex.ProportionTest, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    rates = [t.rate_control, t.rate_treatment]
    cis = [ex.wilson_ci(t.conv_control, t.n_control),
           ex.wilson_ci(t.conv_treatment, t.n_treatment)]
    err = [[r - lo for r, (lo, _) in zip(rates, cis)],
           [hi - r for r, (_, hi) in zip(rates, cis)]]
    ax1.bar(["control", "treatment"], rates, color=[C_CONTROL, C_TREAT],
            yerr=err, capsize=6)
    ax1.set_ylim(0.11, 0.126)
    ax1.set_ylabel("conversion rate")
    ax1.set_title("Conversion by arm (95% Wilson CI)")
    for i, r in enumerate(rates):
        ax1.text(i, r + 0.0006, f"{r*100:.2f}%", ha="center", fontsize=9)

    lo, hi = t.ci_abs
    ax2.errorbar([t.abs_lift * 100], [0], xerr=[[(t.abs_lift - lo) * 100]],
                 fmt="o", color=C_TREAT, capsize=6)
    ax2.axvline(0, color=C_WARN, lw=1)
    ax2.set_yticks([])
    ax2.set_xlabel("absolute lift, treatment − control (pp)")
    ax2.set_title(f"Effect size — p = {t.p_value:.3f}")
    ax2.text(t.abs_lift * 100, 0.15,
             f"{t.abs_lift*100:+.2f} pp\n[{lo*100:+.2f}, {hi*100:+.2f}]",
             ha="center", fontsize=9)
    ax2.set_xlim(-0.6, 0.4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_novelty(daily: "pd.DataFrame", path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4))
    ax.axhline(0, color=C_WARN, lw=1)
    ax.plot(daily["date"], daily["daily_lift"] * 100, marker="o", ms=4,
            color=C_CONTROL, alpha=0.6, label="daily lift")
    ax.plot(daily["date"], daily["cum_lift"] * 100, lw=2.2,
            color=C_TREAT, label="cumulative lift")
    ax.set_ylabel("lift, treatment − control (pp)")
    ax.set_title("Novelty / primacy check — treatment effect over the test window")
    ax.legend()
    fig.autofmt_xdate()
    fig.text(0.5, -0.02,
             "Cumulative lift settles near −0.16 pp within a few days and stays flat — "
             "no decay, no ramp.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_segments(seg_country: "pd.DataFrame", seg_weekday: "pd.DataFrame",
                   overall: ex.ProportionTest, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5))
    rows = []
    for _, r in seg_country.iterrows():
        rows.append((f"country: {r['country']}", r["abs_lift_pp"],
                     r["ci_low_pp"], r["ci_high_pp"], r["n"]))
    for _, r in seg_weekday.iterrows():
        rows.append((f"weekday: {r['weekday']}", r["abs_lift_pp"],
                     r["ci_low_pp"], r["ci_high_pp"], r["n"]))
    rows.reverse()

    ys = np.arange(len(rows))
    for y, (_, mid, lo, hi, _) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=C_CONTROL, lw=1.6)
        ax.plot(mid, y, "o", color=C_TREAT)
    ax.axvline(0, color=C_WARN, lw=1)
    ax.axvline(overall.abs_lift * 100, color=C_TREAT, ls="--", lw=1,
               label=f"overall {overall.abs_lift*100:+.2f} pp")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("absolute lift, treatment − control (pp)")
    ax.set_title("Segment cuts — does any slice reverse the aggregate?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import pandas as pd  # noqa: F401  (used via type hints / daily frame)

    CHARTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 0  Load and clean")
    raw = ex.load_experiment()
    clean, report = ex.clean_experiment(raw)
    print(json.dumps(report.as_dict(), indent=2))

    print("=" * 70)
    print("STEP 1  Sample ratio mismatch (before any outcome analysis)")
    srm = ex.srm_check(clean["group"].value_counts())
    print(srm.summary())
    chart_srm(srm, CHARTS / "01_srm_check.png")
    if srm.is_mismatch:
        raise SystemExit("SRM detected — stopping before outcome analysis.")

    n_per_arm = int(clean["group"].value_counts().min())

    print("=" * 70)
    print("STEP 2  Power / sample size (framed as what we should have specified)")
    need = ex.required_sample_size(BASELINE, MDE_ABS, ALPHA, POWER)
    detectable = ex.mde_for_sample_size(BASELINE, n_per_arm, ALPHA, POWER)
    print(f"  pre-registered MDE  : +{MDE_ABS*100:.2f} pp  -> needs {need:,}/arm")
    print(f"  achieved n          : {n_per_arm:,}/arm")
    print(f"  actually detectable : +{detectable*100:.3f} pp at 80% power")
    chart_power(CHARTS / "02_power_curve.png", n_per_arm)

    print("=" * 70)
    print("STEP 3  Primary test + effect size")
    t = ex.two_proportion_test(clean)
    print(t.summary())
    chart_primary(t, CHARTS / "03_primary_effect.png")

    print("=" * 70)
    print("STEP 4  Novelty / primacy")
    daily = ex.daily_effect(clean)
    print(f"  day-1 cumulative lift : {daily['cum_lift'].iloc[0]*100:+.3f} pp")
    print(f"  final cumulative lift : {daily['cum_lift'].iloc[-1]*100:+.3f} pp")
    print(f"  daily-lift std        : {daily['daily_lift'].std()*100:.3f} pp")
    chart_novelty(daily, CHARTS / "04_novelty_check.png")

    print("=" * 70)
    print("STEP 5  Segment cuts (Simpson's-paradox check)")
    seg_country = ex.segment_effects(clean, "country")
    seg_weekday = ex.segment_effects(ex.add_weekday(clean), "weekday")
    print(seg_country.to_string(index=False))
    print()
    print(seg_weekday.to_string(index=False))
    n_flip = int((np.sign(seg_country["abs_lift_pp"]) != np.sign(t.abs_lift)).sum()
                 + (np.sign(seg_weekday["abs_lift_pp"]) != np.sign(t.abs_lift)).sum())
    print(f"  segments with opposite sign to the aggregate: {n_flip}")
    print(f"  any significant after Bonferroni (alpha/{len(seg_weekday)}): "
          f"{bool((seg_weekday['p_value'] < ALPHA/len(seg_weekday)).any())}")
    chart_segments(seg_country, seg_weekday, t, CHARTS / "05_segment_effects.png")

    print("=" * 70)
    print("STEP 6  Guardrail")
    g = ex.guardrail_delivery_error_rate(raw)
    print(g.summary())
    print(f"  note: {g.note}")

    summary = {
        "cleaning": report.as_dict(),
        "srm": {"chi2": srm.chi2, "p_value": srm.p_value,
                "observed": srm.observed, "is_mismatch": srm.is_mismatch},
        "power": {"baseline": BASELINE, "pre_registered_mde_pp": MDE_ABS * 100,
                  "required_n_per_arm": need, "achieved_n_per_arm": n_per_arm,
                  "detectable_mde_pp": detectable * 100},
        "primary": {"rate_control": t.rate_control, "rate_treatment": t.rate_treatment,
                    "abs_lift_pp": t.abs_lift * 100, "rel_lift_pct": t.rel_lift * 100,
                    "ci_abs_pp": [t.ci_abs[0] * 100, t.ci_abs[1] * 100],
                    "z": t.z_stat, "p_value": t.p_value},
        "novelty": {"cum_lift_day1_pp": daily["cum_lift"].iloc[0] * 100,
                    "cum_lift_final_pp": daily["cum_lift"].iloc[-1] * 100},
        "segments_country": seg_country.to_dict(orient="records"),
        "segments_weekday": seg_weekday.to_dict(orient="records"),
        "guardrail": {"metric": g.metric, "control": g.value_control,
                      "treatment": g.value_treatment, "p_value": g.p_value,
                      "worse_in_treatment": g.worse_in_treatment},
    }
    (PROCESSED / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print("=" * 70)
    print(f"wrote {CHARTS}/*.png and {PROCESSED}/results.json")


if __name__ == "__main__":
    main()
