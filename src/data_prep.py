"""
data_prep.py - Beijing Air Quality Data Ingestion & Step 3 Feature Engineering

Pipeline Module:
- Automated UCI Cloud Fetch & Extraction
- 12 Station Concatenation & DateTime Construction
- Missing Value Handling (Station-wise short-gap linear interpolation, limit=3)
- Step 3 Feature Engineering (BEFORE Split - Domain Creation):
  * temp_dew_spread (TEMP - DEWP)
  * season (Spring, Summer, Autumn, Winter)
  * time_of_day (Night, Morning_Rush, Afternoon, Evening_Rush)
  * wd_sector (NW, NE, SW, SE, CV per Zhang et al., 2017)
  * zone_type (Urban vs Rural_Suburban)
  * is_urban, is_rural dummy variables
"""

import os
import glob
import urllib.request
import zipfile
import pandas as pd
import numpy as np

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"

URBAN_STATIONS = {
    'Aotizhongxin', 'Dongsi', 'Guanyuan', 'Gucheng',
    'Nongzhanguan', 'Tiantan', 'Wanliu', 'Wanshouxigong'
}

RURAL_SUBURBAN_STATIONS = {
    'Changping', 'Dingling', 'Huairou', 'Shunyi'
}

WD_SECTOR_MAP = {
    'N': 'NW', 'NNW': 'NW', 'NW': 'NW', 'WNW': 'NW',
    'ENE': 'NE', 'NE': 'NE', 'NNE': 'NE',
    'W': 'SW', 'WSW': 'SW', 'SW': 'SW', 'SSW': 'SW',
    'S': 'SE', 'SSE': 'SE', 'SE': 'SE', 'ESE': 'SE', 'E': 'SE',
    'cv': 'CV', 'CV': 'CV'
}


def ensure_dataset_ready(raw_dir: str = "data/raw") -> str:
    """
    Checks if 12 station CSV files exist locally; if not, fetches from UCI cloud and extracts.
    """
    station_dir = os.path.join(raw_dir, "PRSA_Data_20130301-20170228")
    if os.path.isdir(station_dir):
        csv_files = glob.glob(os.path.join(station_dir, "*.csv"))
        if len(csv_files) == 12:
            return station_dir

    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "beijing_air_quality.zip")
    
    if not os.path.exists(zip_path):
        print(f"Dataset not found locally. Fetching directly from UCI ML Repository cloud...")
        urllib.request.urlretrieve(UCI_DATA_URL, zip_path)
        print("Download complete.")
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(raw_dir)
        
    inner_zip = os.path.join(raw_dir, "PRSA2017_Data_20130301-20170228.zip")
    if os.path.exists(inner_zip):
        with zipfile.ZipFile(inner_zip, 'r') as z:
            z.extractall(raw_dir)
            
    return station_dir


def get_season(month: int) -> str:
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'


def get_time_of_day(hour: int) -> str:
    if 0 <= hour <= 5:
        return 'Night'
    elif 6 <= hour <= 10:
        return 'Morning_Rush'
    elif 11 <= hour <= 16:
        return 'Afternoon'
    else:
        return 'Evening_Rush'


def load_and_preprocess_data(station_dir: str) -> pd.DataFrame:
    """
    Loads 12 station CSVs, merges them, interpolates short gaps, and creates engineered features.
    """
    csv_files = glob.glob(os.path.join(station_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {station_dir}")
        
    dfs = []
    for f in csv_files:
        df_station = pd.read_csv(f)
        dfs.append(df_station)
        
    df = pd.concat(dfs, ignore_index=True)
    
    # 1. Drop row index 'No' (avoids index leakage)
    if 'No' in df.columns:
        df = df.drop(columns=['No'])
        
    # 2. Construct chronological datetime
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.sort_values(by=['station', 'datetime']).reset_index(drop=True)
    
    # 3. Missing Value Handling: Short-gap linear interpolation (limit=3 hrs) per station
    weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    
    # Interpolate weather and target PM2.5 for short sensor dropouts (<= 3 hrs)
    interp_cols = weather_cols + ['PM2.5']
    df[interp_cols] = df.groupby('station')[interp_cols].transform(
        lambda s: s.interpolate(method='linear', limit=3)
    )
    
    # Drop rows that still have NA in PM2.5 or continuous weather variables
    df = df.dropna(subset=['PM2.5'] + weather_cols).reset_index(drop=True)
    
    # Fill remaining wind direction 'wd' forward/backward per station
    df['wd'] = df.groupby('station')['wd'].transform(lambda s: s.ffill().bfill())
    
    # 4. Step 3 Feature Engineering (Domain Creation BEFORE split)
    # A. Atmospheric Moisture / Condensation spread
    df['temp_dew_spread'] = df['TEMP'] - df['DEWP']
    
    # B. Temporal features
    df['season'] = df['month'].apply(get_season)
    df['time_of_day'] = df['hour'].apply(get_time_of_day)
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # C. Grouped Wind Direction Sectors (Zhang et al., 2017)
    df['wd_sector'] = df['wd'].map(WD_SECTOR_MAP).fillna('CV')
    
    # D. Spatial Zone & Urban/Rural Dummies
    df['zone_type'] = df['station'].apply(lambda st: 'Urban' if st in URBAN_STATIONS else 'Rural_Suburban')
    df['is_urban'] = (df['zone_type'] == 'Urban').astype(int)
    df['is_rural'] = (df['zone_type'] == 'Rural_Suburban').astype(int)
    
    return df
