import fastf1
import pandas as pd
from pathlib import Path

# -- Cache configuration --
CACHE_DIR = Path("data/raw/fastf1_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# -- Session constants --
YEAR = 2024
ROUND = 1  # Bahrain GP
SESSION = "R"  # Race


def load_session() -> fastf1.coreSession:
    """Load and return the full race session object."""
    session = fastf1.get_session(YEAR, ROUND, SESSION)
    session.load(
        laps=True,
        telemetry=False,  # Telemetry is large ; we don't need it for pace analysis
        weather=False,
        messages=False,
    )
    print(f"Session loaded: {session.event['EventName']} {YEAR}")
    print(f"Total drivers: {session.laps['Driver'].nunique()}")
    return session


def extract_lap_dataFrame(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract a raw lap-level DataFrame from the session.
    Retains inly the colmns relevant to pace and strategy analysis.
    No cleaning is performed here"""
    laps = session.laps.copy()

    # FastF1 returns timedelta for LaptTime.
    # Convert to seconds (float) for numerical analysis.
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    # Retain the columns that matter for our research question
    cols = [
        "Driver",  # Three-letter code (e.g., 'VER')
        "Team",  # Constructor name
        "LapNumber",  # Sequential lap number in the race
        "LapTimeSeconds",  # Lap time in seconds (our target variable)
        "Compound",  # Tire compound: SOFT / MEDIUM / HARD / INTERMEDIATE / WET
        "TyreLife",  # Laps completed on current tire set
        "Stint",  # Stint number (increments after each pit stop)
        "PitOutTime",  # Timedelta — non-null on out-laps
        "PitInTime",  # Timedelta — non-null on in-laps
        "TrackStatus",  # '1' = clear, '2' = yellow, '4' = SC, '6' = VSC, '7' = red
        "IsAccurate",  # FastF1 quality flag — False for laps it considers unreliable
        "Position",  # Race position at end of lap
    ]

    # Keep inly columns that exist (defensive for different FastF1 versions)
    cols = [col for col in cols if col in laps.columns]
    df = laps[cols].reset_index(drop=True)
    print(f"\nRaw DataFrame shape: {df.shape}")
    return df


if __name__ == "__main__":
    session = load_session()
    df = extract_lap_dataFrame(session)

    # Phase 2 checkpoint - first look at the data
    print("\n-- Head --")
    print(df.head(10).to_string())

    print("\n-- Data Types --")
    print(df.dtypes)

    print("\n-- Compound distribution --")
    print(df.isnull().sum())

    print("\n--TrackStatus distribution --")
    print(df["LapTimeSeconds"].describe())

    # Save raw extract for reference - do NOT overwrite with cleaned version!
    raw_path = Path("data/raw/bahrain_2024_laps_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw lap data saved to -> {raw_path}")
