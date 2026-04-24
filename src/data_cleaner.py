# src/data_cleaner.py
"""
Phase 3 - Data cleaning feature engineering.
Every filter decision is explicitly justified by Phase 2 EDA findings.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -- Filtering thresholds --
LAP_TIME_UPPER_BOUND = 105.0  # s - everything above this post-lap is anomalous
LAP_TIME_LOWER_BOUND = 88.0  # s - below this is physically impossible at Bahrain
CLEAN_TRACK_STATUS = "1"  # FastF1 '1' = fully clear track
VALID_COMPOUNDS = {"SOFT", "HARD"}  # Only compunds present in this race


def remove_outlier_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 - Remove structural outliers that are not representative of steady-state tire performance.
    """
    original_len = len(df)
    mask = pd.Series(True, index=df.index)

    # Lap 1: formation phase, cold tires, racing incidents
    mask &= df["LapNumber"] > 1

    # Pit-put laps: cold tires, slow acceleration from pit exit
    mask &= df["IsAccurate"]

    # Lap time bounds - evidence-based from histogram
    mask &= df["LapTimeSeconds"] >= LAP_TIME_LOWER_BOUND
    mask &= df["LapTimeSeconds"] <= LAP_TIME_UPPER_BOUND

    df_clean = df[mask].copy()
    removed = original_len - len(df_clean)
    print(
        f"Outlier removal: {removed} laps removed ({removed / original_len * 100:.1f}%)"
    )
    print(f"Remaining: {len(df_clean)} laps")
    return df_clean


def filter_track_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Step 2 - Keep only laps run under fully clear track conditions.
    Yellow flags and VSC artificially inflate lap times and corrupt
    the degradation signal.
    """

    original_len = len(df)

    # TrackStatus can be stores as string or int depending on FastF1 version
    df["TrackStatus"] = df["TrackStatus"].astype(str).str.strip()

    df_clean = df[df["TrackStatus"] == CLEAN_TRACK_STATUS].copy()
    removed = original_len - len(df_clean)
    print(f"Track condition filter: {removed} laps removed")
    print(f"Remaining: {len(df_clean)} laps")
    return df_clean


def filter_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3 - Retain only the compounds present in this race.
    Defensive filter: if any UNKNOWN laps survived previous steps, remove them.
    """

    original_len = len(df)
    df_clean = df[df["Compound"].isin(VALID_COMPOUNDS)].copy()
    removed = original_len - len(df_clean)
    print(f"Compound filter: {removed} laps removed (UNKNOWN or unexpected)")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4 - Feature engineering
    Create derived columns needed for degradation curve and the
    statistical pace comparison.
    """

    df = df.copy()

    # Stint-relative tire age (should match TyreLife, but we derive it
    # independently as a sanity check and to check and to handle edge cases)

    df["StintLap"] = (
        df.groupby(["Driver", "Stint"])["LapNumber"].rank(method="first").astype(int)
    )

    # Pace delta vs field median on the same lap number.
    # This is our key analytical variable - it isolates a drivers relative pace
    # from track evolution effects (rubber, temperature shifts across the race).
    lap_median = df.groupby("LapNumber")["LapTimeSeconds"].transform("median")
    df["PaceDelta"] = df["LapTimeSeconds"] - lap_median

    # Boolean flags for convenient grouping
    df["IsWinner"] = df["Driver"] == "VER"  # Verstappen won the '24 Bahrain GP
    df["IsRedBull"] = df["Team"].str.contains("Red Bull", case=False, na=False)

    # Compound ordinal encoding for regression {if needed later}
    compound_order = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
    df["CompundOrdinal"] = df["Compound"].map(compound_order)

    print(f"Fearures enigineered. Final shape: {df.shape}")
    print("new columns: StintLap, PaceDelta, IsWinner, IsRedBull, CompoundOridnal")
    return df


def clean_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline - runs all cleaning and engineering steps in order.
    Returns the analysis-ready DataFrame
    """

    print("\n -- Phase 3: Data Preparation --")
    df = remove_outlier_laps(df_raw)
    df = filter_track_conditions(df)
    df = filter_compounds(df)
    df = engineer_features(df)

    print("\n-- Cleaning summary --")
    print(f"Raw laps: {len(df_raw):>5}")
    print(f"Clean laps: {len(df):>5}")
    print(f"Retention: {len(df) / len(df_raw) * 100:.1f}%")
    print("\nCompound breakdown (clean data):")
    print(df["Compound"].value_counts())
    print("\nNull check (clean data):")
    print(
        df[["LapTimeSeconds", "Compound", "TyreLife", "Stint", "PaceDelta"]]
        .isnull()
        .sum()
    )

    return df


if __name__ == "__main__":
    from data_loader import load_session, extract_lap_dataFrame

    session = load_session()
    df_raw = extract_lap_dataFrame(session)
    df_clean = clean_pipeline(df_raw)

    # Save the analysis-ready dataset
    out_path = Path("data/processed/bahrain_2024_clean.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"\nClean data soved -> {out_path}")

    # Phase 3 checkpoint visualisation
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 3 checkpoint - cleaned data", fontweight="bold")

    COMPOUND_COLORS = {"SOFT": "#E8383D", "HARD": "#AAAAAA"}

    for compound, grp in df_clean.groupby("Compound"):
        axes[0].hist(
            grp["LapTimeSeconds"],
            bins=40,
            alpha=0.7,
            color=COMPOUND_COLORS[compound],
            label=compound,
            edgecolor="none",
        )

        axes[0].set_xlabel("Lap time (s)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Lap time distribution by compound (clean)")
        axes[0].legend()

        for compound, grp in df_clean.groupby("Compound"):
            axes[1].scatter(
                grp["TyreLife"],
                grp["LapTimeSeconds"],
                alpha=0.25,
                s=12,
                color=COMPOUND_COLORS[compound],
                label=compound,
            )

        axes[1].set_xlabel("Tire age {laps}")
        axes[1].set_ylabel("Lap time (s)")
        axes[1].set_title("Early degradation signal - tire age vs lap time")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(
            "reports/fig_04_phase3_checkpoint.png", dpi=150, bbox_inches="tight"
        )
        plt.show()
