"""
build_app_data.py - Builds the lightweight, commit-friendly datasets used by the Streamlit app.

The raw hourly station CSVs (~32 MB) are gitignored, so Streamlit Cloud cannot see them.
This script condenses them into two small files under data/processed/:
  * beijing_daily.csv            - station x day means (PM2.5, co-pollutants, weather, dominant wind)
  * beijing_diurnal_profile.csv  - station x month x hour mean PM2.5 (for the diurnal heatmap)

Run once locally:  python src/build_app_data.py
"""

import os
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw", "PRSA_Data_20130301-20170228")
OUT_DIR = os.path.join(ROOT, "data", "processed")


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if len(files) != 12:
        raise FileNotFoundError(f"Expected 12 station CSVs in {RAW_DIR}, found {len(files)}. Run the notebook's Step 2 first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["date"] = df["datetime"].dt.normalize()

    num_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
    grp = df.groupby(["station", "date"])
    daily = grp[num_cols].mean()
    daily["RAIN"] = grp["RAIN"].sum()
    daily["PM25_max"] = grp["PM2.5"].max()
    daily["wd"] = df.dropna(subset=["wd"]).groupby(["station", "date"])["wd"].agg(lambda s: s.mode().iat[0])
    daily = daily.reset_index().round(2)

    profile = df.groupby(["station", "month", "hour"])["PM2.5"].mean().round(2).reset_index()

    os.makedirs(OUT_DIR, exist_ok=True)
    daily.to_csv(os.path.join(OUT_DIR, "beijing_daily.csv"), index=False)
    profile.to_csv(os.path.join(OUT_DIR, "beijing_diurnal_profile.csv"), index=False)
    print(f"daily rows: {len(daily):,} | profile rows: {len(profile):,}")


if __name__ == "__main__":
    main()
