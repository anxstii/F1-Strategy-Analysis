# src/visualiser.py
"""
Phase 4 — Portfolio visualisations.
  Fig A: Stint Strategy Gantt chart
  Fig B: Tire Degradation Curves (with linear regression fits)
  Fig C: Pace Delta — winning team vs field
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
COMPOUND_COLORS = {
    "SOFT":  "#E8383D",
    "HARD":  "#C8C8C8",
}

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F9F9F7",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "monospace",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})


# ══════════════════════════════════════════════════════════════════════════
# FIGURE A — Stint Strategy Gantt Chart
# ══════════════════════════════════════════════════════════════════════════

def plot_stint_gantt(df: pd.DataFrame) -> None:
    """
    Visualise every driver's stint structure as a horizontal Gantt chart.
    Bar colour encodes tire compound. Bar width encodes stint length in laps.
    Drivers are sorted by finishing position.
    """
    # Build one row per stint
    stint_data = (
        df.groupby(["Driver", "Team", "Stint", "Compound"])
        .agg(
            StintStart=("LapNumber", "min"),
            StintEnd=("LapNumber", "max"),
            IsRedBull=("IsRedBull", "first"),
        )
        .reset_index()
    )
    stint_data["StintLength"] = stint_data["StintEnd"] - stint_data["StintStart"] + 1

    # Sort drivers by final position (lowest lap number = finished furthest ahead
    # is not reliable — use final position from the raw data instead)
    final_pos = (
        df.sort_values("LapNumber", ascending=False)
        .drop_duplicates("Driver")[["Driver", "Position"]]
        .sort_values("Position")
    )
    driver_order = final_pos["Driver"].tolist()

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        "2024 Bahrain Grand Prix — Stint Strategy Overview",
        fontsize=15, fontweight="bold", x=0.5, y=1.01
    )

    y_positions = {driver: i for i, driver in enumerate(driver_order)}

    for _, row in stint_data.iterrows():
        driver  = row["Driver"]
        y       = y_positions.get(driver, 0)
        color   = COMPOUND_COLORS.get(row["Compound"], "#888888")
        alpha   = 1.0 if row["IsRedBull"] else 0.6

        ax.barh(
            y=y,
            width=row["StintLength"],
            left=row["StintStart"],
            height=0.65,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.8,
        )

        # Label the compound inside the bar if wide enough
        if row["StintLength"] >= 6:
            ax.text(
                row["StintStart"] + row["StintLength"] / 2,
                y,
                row["Compound"][0],          # 'S' or 'H'
                ha="center", va="center",
                fontsize=7, color="black", alpha=0.7,
                fontfamily="monospace",
            )

    # Driver labels on y-axis — bold Red Bull drivers
    ax.set_yticks(range(len(driver_order)))
    y_labels = []
    for d in driver_order:
        team_row = df[df["Driver"] == d]["Team"].iloc[0] if len(df[df["Driver"] == d]) > 0 else ""
        is_rb    = "Red Bull" in str(team_row)
        y_labels.append(f"► {d}" if is_rb else f"  {d}")
    ax.set_yticklabels(y_labels, fontsize=9)

    ax.set_xlabel("Lap number")
    ax.set_xlim(1, df["LapNumber"].max() + 2)
    ax.set_title("Sorted by finishing position — Red Bull drivers highlighted (►)",
                 fontsize=10, fontweight="normal", pad=6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.6)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COMPOUND_COLORS["SOFT"], label="Soft"),
        mpatches.Patch(color=COMPOUND_COLORS["HARD"], label="Hard"),
        mpatches.Patch(color="#AAAAAA", alpha=0.4, label="Other teams"),
        mpatches.Patch(color="#AAAAAA", alpha=1.0, label="Red Bull"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = REPORT_DIR / "fig_A_stint_gantt.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE B — Tire Degradation Curves
# ══════════════════════════════════════════════════════════════════════════

def plot_degradation_curves(df: pd.DataFrame) -> dict:
    """
    Plot lap time vs tire age for each compound with:
      - Individual driver stint traces (low alpha)
      - Median trend line per compound
      - Linear regression fit with slope annotation

    Returns a dict of regression results for use in Phase 5.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        "2024 Bahrain GP — Tire Degradation Curves",
        fontsize=15, fontweight="bold"
    )

    regression_results = {}

    for ax, compound in zip(axes, ["SOFT", "HARD"]):
        comp_df = df[df["Compound"] == compound].copy()
        color   = COMPOUND_COLORS[compound]

        # Individual stint traces — one line per (Driver, Stint) pair
        for (driver, stint), grp in comp_df.groupby(["Driver", "Stint"]):
            grp_sorted = grp.sort_values("TyreLife")
            is_rb      = grp["IsRedBull"].iloc[0]
            ax.plot(
                grp_sorted["TyreLife"],
                grp_sorted["LapTimeSeconds"],
                color=color,
                alpha=0.55 if is_rb else 0.18,
                linewidth=1.8 if is_rb else 0.8,
                zorder=3 if is_rb else 2,
            )

        # Median trend line (binned by tire age)
        median_trend = (
            comp_df.groupby("TyreLife")["LapTimeSeconds"]
            .median()
            .reset_index()
            .sort_values("TyreLife")
        )
        # Only plot where we have ≥3 data points for stability
        valid_bins = comp_df.groupby("TyreLife").size()
        valid_ages = valid_bins[valid_bins >= 3].index
        median_trend = median_trend[median_trend["TyreLife"].isin(valid_ages)]

        ax.plot(
            median_trend["TyreLife"],
            median_trend["LapTimeSeconds"],
            color=color,
            linewidth=2.8,
            alpha=0.95,
            zorder=5,
            label="Field median",
        )

        # Linear regression on median trend
        x = median_trend["TyreLife"].values
        y = median_trend["LapTimeSeconds"].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit, y_fit,
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
            zorder=6,
            label="Linear fit",
        )

        regression_results[compound] = {
            "slope":     slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "p_value":   p_value,
        }

        # Slope annotation box
        ax.text(
            0.97, 0.07,
            f"Degradation: +{slope:.3f} s/lap\n$R^2$ = {r_value**2:.3f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color, alpha=0.9),
        )

        ax.set_title(f"{compound.capitalize()} compound", pad=8)
        ax.set_xlabel("Tire age (laps)")
        ax.legend(fontsize=9, loc="upper left")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6)

    axes[0].set_ylabel("Lap time (s)")

    # Red Bull label in figure margin
    fig.text(
        0.5, -0.02,
        "Red Bull driver stints drawn at higher opacity",
        ha="center", fontsize=9, color="#555555", style="italic"
    )

    plt.tight_layout()
    out = REPORT_DIR / "fig_B_degradation_curves.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")
    return regression_results


# ══════════════════════════════════════════════════════════════════════════
# FIGURE C — Pace Delta: Red Bull vs Field
# ══════════════════════════════════════════════════════════════════════════

def plot_pace_delta(df: pd.DataFrame) -> None:
    """
    Plot the race-long pace delta (lap time minus field median on same lap)
    for Red Bull vs the rest of the field.
    A negative delta = faster than the median field.
    """
    rb_df    = df[df["IsRedBull"]].copy()
    field_df = df[~df["IsRedBull"]].copy()

    # Smooth with a rolling window to reduce lap-to-lap noise
    def rolling_median_delta(group_df, window=4):
        return (
            group_df.groupby("LapNumber")["PaceDelta"]
            .median()
            .rolling(window=window, center=True, min_periods=2)
            .mean()
        )

    rb_smooth    = rolling_median_delta(rb_df)
    field_smooth = rolling_median_delta(field_df)

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(
        "2024 Bahrain GP — Red Bull Race Pace Advantage",
        fontsize=15, fontweight="bold"
    )

    # Shaded field band (±1 std dev of field deltas per lap)
    field_std = field_df.groupby("LapNumber")["PaceDelta"].std()
    field_med = field_df.groupby("LapNumber")["PaceDelta"].median()

    ax.fill_between(
        field_std.index,
        field_med - field_std,
        field_med + field_std,
        alpha=0.15, color="#888888", label="Field ±1σ band"
    )
    ax.plot(
        field_smooth.index, field_smooth.values,
        color="#888888", linewidth=1.5, alpha=0.8, label="Field median delta"
    )
    ax.plot(
        rb_smooth.index, rb_smooth.values,
        color="#1E3A8A", linewidth=2.5, label="Red Bull median delta"
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Lap number")
    ax.set_ylabel("Pace delta vs field median (s)")
    ax.set_title(
        "Negative delta = faster than field median on that lap",
        fontsize=10, fontweight="normal"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6)

    plt.tight_layout()
    out = REPORT_DIR / "fig_C_pace_delta.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    df = pd.read_csv("data/processed/bahrain_2024_clean.csv")
    print(f"Loaded clean data: {df.shape}")

    print("\n── Figure A: Stint Gantt ─────────────────────────────")
    plot_stint_gantt(df)

    print("\n── Figure B: Degradation Curves ─────────────────────")
    regression_results = plot_degradation_curves(df)

    print("\n── Regression summary ────────────────────────────────")
    for compound, res in regression_results.items():
        print(f"\n  {compound}")
        print(f"    Slope      : +{res['slope']:.4f} s/lap")
        print(f"    R²         : {res['r_squared']:.4f}")
        print(f"    p-value    : {res['p_value']:.4f}")

    print("\n── Figure C: Pace Delta ──────────────────────────────")
    plot_pace_delta(df)