"""
Preprocess road accident data:
- Reads data/raw/accidents_raw.csv
- Cleans datatypes, normalizes text, derives features
- Writes data/processed/accidents_clean.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/accidents_raw.csv")
OUT_PATH = Path("data/processed/accidents_clean.csv")

def parse_datetime(df):
    # Try common date/time column names; adjust as needed
    # Create unified 'datetime', 'year', 'month', 'day', 'hour'
    date_cols = [c for c in df.columns if c.lower() in ["date", "accident_date", "crash_date"]]
    time_cols = [c for c in df.columns if c.lower() in ["time", "accident_time", "crash_time"]]
    if date_cols:
        df["date_std"] = pd.to_datetime(df[date_cols[0]], errors="coerce", dayfirst=True)
    else:
        df["date_std"] = pd.NaT
    if time_cols:
        # If time exists, combine to full datetime
        try:
            t = pd.to_datetime(df[time_cols[0]].astype(str), errors="coerce").dt.time
            df["datetime"] = pd.to_datetime(df["date_std"].dt.date.astype(str) + " " + df[time_cols[0]].astype(str), errors="coerce")
        except Exception:
            df["datetime"] = df["date_std"]
    else:
        df["datetime"] = df["date_std"]

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()

    # Time-of-day bins (Night/Early/Day/Evening)
    bins = [-1, 5, 11, 17, 21, 24]
    labels = ["Night (0-5)", "Morning (6-11)", "Afternoon (12-17)", "Evening (18-21)", "Late (22-24)"]
    df["time_of_day"] = pd.cut(df["hour"].fillna(-1), bins=bins, labels=labels)

    return df

def normalize_text(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip().str.lower()
    # Standardize key columns if present
    rename_map = {
        "state/ut": "state",
        "district": "city",
        "vehicle": "vehicle_type",
        "light": "light_conditions",
        "road": "road_type",
        "weather_condition": "weather",
        "severity_of_accident": "severity"
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df

def clean_severity(df):
    if "severity" in df.columns:
        df["severity"] = (
            df["severity"].replace({
                "fatal": "high",
                "grievous": "medium",
                "serious": "medium",
                "minor": "low",
                "slight": "low"
            })
        )
        # Keep only Low/Medium/High
        df["severity"] = df["severity"].where(df["severity"].isin(["low","medium","high"]), np.nan)
    return df

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Put your raw CSV at {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    # Basic cleaning
    df = normalize_text(df)
    df = parse_datetime(df)
    df = clean_severity(df)

    # Drop obvious duplicates
    df = df.drop_duplicates()

    # Minimal expected columns (create if missing)
    for col in ["state","city","vehicle_type","weather","road_type","severity"]:
        if col not in df.columns:
            df[col] = np.nan

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data to {OUT_PATH} with shape {df.shape}")

if __name__ == "__main__":
    main()
