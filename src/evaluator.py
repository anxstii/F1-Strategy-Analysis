# src/evaluator.py
"""
Phase 5 — Evaluation & statistical testing.
Research question: Did Red Bull's strategy produce a statistically
significant pace advantage over the rest of the field?

Test chosen: Mann-Whitney U (non-parametric)
Justification: Unequal group sizes (2 drivers vs 18), non-normal
distributions confirmed by visual inspection, ordinal comparison
of pace deltas is sufficient for our research question.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from pathlib import Path

REPORT_DIR = Path("reports")

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F9F9F7",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "monospace",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.titleweight":  "bold",
})


def run_statistical_test(df: pd.DataFrame) -> dict:
    """
    Mann-Whitney U test comparing Red Bull pace deltas vs field pace deltas.
    Also computes effect size (rank-biserial correlation r).

    H₀: The pace delta distributions of Red Bull and the rest of the
        field are drawn from the same population.
    H₁: Red Bull's pace deltas are systematically lower (faster).
    """
    rb_deltas    = df[df["IsRedBull"]]["PaceDelta"].dropna().values
    field_deltas = df[~df["IsRedBull"]]["PaceDelta"].dropna().values

    # One-sided test: we hypothesise RB is faster (lower delta)
    stat, p_value = stats.mannwhitneyu(
        rb_deltas, field_deltas, alternative="less"
    )

    # Effect size: rank-biserial correlation
    # r = 1 - (2U / n1*n2), ranges from -1 to +1
    n1, n2   = len(rb_deltas), len(field_deltas)
    r_effect = 1 - (2 * stat) / (n1 * n2)

    # Common language effect size: probability RB lap is faster than random field lap
    cles = stat / (n1 * n2)

    results = {
        "rb_mean_delta":    np.mean(rb_deltas),
        "field_mean_delta": np.mean(field_deltas),
        "rb_median_delta":  np.median(rb_deltas),
        "field_median_delta": np.median(field_deltas),
        "mean_advantage":   np.mean(field_deltas) - np.mean(rb_deltas),
        "u_statistic":      stat,
        "p_value":          p_value,
        "r_effect":         r_effect,
        "cles":             cles,
        "n_rb":             n1,
        "n_field":          n2,
    }
    return results


def print_statistical_report(results: dict, regression: dict) -> None:
    """Print a structured evaluation report to the terminal."""

    sig  = results["p_value"] < 0.05
    conc = "REJECT H₀ — statistically significant advantage detected." if sig \
           else "FAIL TO REJECT H₀ — no significant advantage detected."

    print("\n" + "═" * 60)
    print("  PHASE 5 — STATISTICAL EVALUATION REPORT")
    print("═" * 60)

    print("\n  I. Hypothesis Test — Mann-Whitney U")
    print("     H₀: RB and field pace deltas from same distribution")
    print("     H₁: RB pace deltas systematically lower (faster)")
    print(f"\n     U statistic       : {results['u_statistic']:,.0f}")
    print(f"     p-value (one-tail): {results['p_value']:.2e}")
    print("     Significance α    : 0.05")
    print(f"\n     Conclusion: {conc}")

    print("\n  II. Effect Size")
    print(f"     Rank-biserial r   : {results['r_effect']:.4f}")
    print("     Interpretation    : ", end="")
    r = abs(results["r_effect"])
    if r >= 0.5:
        print("Large effect (r ≥ 0.50)")
    elif r >= 0.3:
        print("Medium effect (r ≥ 0.30)")
    else:
        print("Small effect (r < 0.30)")
    print(f"     CLES              : {results['cles']:.3f}")
    print("     (Probability that a random RB lap beats a random field lap)")

    print("\n  III. Descriptive Pace Statistics")
    print(f"     Red Bull mean delta   : {results['rb_mean_delta']:+.3f} s")
    print(f"     Field mean delta      : {results['field_mean_delta']:+.3f} s")
    print(f"     Mean pace advantage   : {results['mean_advantage']:+.3f} s/lap")
    print(f"     Over 57 laps          : ~{results['mean_advantage']*57:+.1f} s total")

    print("\n  IV. Degradation Analysis")
    for compound, res in regression.items():
        direction = "improving (track evolution)" if res["slope"] < 0 else "degrading"
        print(f"\n     {compound}")
        print(f"       Slope   : {res['slope']:+.4f} s/lap ({direction})")
        print(f"       R²      : {res['r_squared']:.4f}")
        print(f"       p-value : {res['p_value']:.4f}")

    print("\n" + "═" * 60)


def plot_evaluation_summary(df: pd.DataFrame, results: dict) -> None:
    """
    Two-panel evaluation figure:
      Left:  Box plots of pace delta distributions (RB vs field, per compound)
      Right: Cumulative distribution functions — visualises the stochastic
             dominance that Mann-Whitney detects
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Phase 5 — Evaluation: Red Bull vs Field Pace Distribution",
        fontsize=14, fontweight="bold"
    )

    # ── Left: Box plots by group and compound ─────────────────────────────
    ax = axes[0]

    plot_data   = []
    plot_labels = []
    plot_colors = []
    color_map   = {
        ("Red Bull", "SOFT"): "#C0392B",
        ("Red Bull", "HARD"): "#7F8C8D",
        ("Field",    "SOFT"): "#E8A09A",
        ("Field",    "HARD"): "#D5D8DC",
    }

    for is_rb, group_label in [(True, "Red Bull"), (False, "Field")]:
        for compound in ["SOFT", "HARD"]:
            subset = df[(df["IsRedBull"] == is_rb) &
                        (df["Compound"]  == compound)]["PaceDelta"].dropna()
            if len(subset) > 5:
                plot_data.append(subset.values)
                plot_labels.append(f"{group_label}\n{compound}")
                plot_colors.append(color_map[(group_label, compound)])

    bp = ax.boxplot(
        plot_data,
        labels=plot_labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1, linestyle="--"),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        boxprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Pace delta vs field median (s)")
    ax.set_title("Distribution by group and compound", pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # ── Right: Empirical CDFs ──────────────────────────────────────────────
    ax2 = axes[1]

    rb_deltas    = np.sort(df[df["IsRedBull"]]["PaceDelta"].dropna().values)
    field_deltas = np.sort(df[~df["IsRedBull"]]["PaceDelta"].dropna().values)

    rb_cdf    = np.arange(1, len(rb_deltas) + 1) / len(rb_deltas)
    field_cdf = np.arange(1, len(field_deltas) + 1) / len(field_deltas)

    ax2.plot(rb_deltas, rb_cdf,
             color="#1E3A8A", linewidth=2.5, label=f"Red Bull (n={len(rb_deltas)})")
    ax2.plot(field_deltas, field_cdf,
             color="#888888", linewidth=1.8, label=f"Field (n={len(field_deltas)})", alpha=0.8)

    # Annotate the horizontal gap at CDF = 0.5 (median gap)
    rb_med    = np.median(rb_deltas)
    field_med = np.median(field_deltas)
    ax2.annotate(
        "",
        xy=(field_med, 0.5), xytext=(rb_med, 0.5),
        arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.5)
    )
    ax2.text(
        (rb_med + field_med) / 2, 0.53,
        f"Δ = {field_med - rb_med:.2f} s",
        ha="center", fontsize=9, color="#C0392B",
        fontfamily="monospace"
    )

    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Pace delta vs field median (s)")
    ax2.set_ylabel("Cumulative probability")
    ax2.set_title("Empirical CDF — stochastic dominance test", pad=8)
    ax2.legend(fontsize=9)
    ax2.grid(linestyle="--", alpha=0.25)

    plt.tight_layout()
    out = REPORT_DIR / "fig_D_evaluation.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from visualiser import plot_degradation_curves
    from data_loader import load_session, extract_lap_dataFrame
    from data_cleaner import clean_pipeline

    df = pd.read_csv("data/processed/bahrain_2024_clean.csv")

    # Re-derive regression results for the report
    from scipy.stats import linregress
    regression_results = {}
    for compound in ["SOFT", "HARD"]:
        comp   = df[df["Compound"] == compound]
        median = comp.groupby("TyreLife")["LapTimeSeconds"].median()
        valid  = comp.groupby("TyreLife").size()
        median = median[valid[valid >= 3].index].sort_index()
        x, y   = median.index.values, median.values
        slope, intercept, r, p, _ = linregress(x, y)
        regression_results[compound] = {
            "slope": slope, "intercept": intercept,
            "r_squared": r**2, "p_value": p
        }

    results = run_statistical_test(df)
    print_statistical_report(results, regression_results)
    plot_evaluation_summary(df, results)