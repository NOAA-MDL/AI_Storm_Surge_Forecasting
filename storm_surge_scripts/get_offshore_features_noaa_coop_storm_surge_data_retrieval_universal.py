#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
########################################################################
# Direct Hourly Observation Data Retrieval Engine - Universal Baseline #
# Full Hourly Grid Generation with 99999 Missing Value Imputation      #
########################################################################

import os, sys, time, requests
import pandas as pd, numpy as np
from datetime import datetime

os.environ["TF_USE_LEGACY_KERAS"] = "1"
API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# --- Station & Output Configuration ---
STATION_ID = "8729108"
STATION_NAME = "Panama_City_FL"
years = ['2021', '2022', '2023', '2024', '2025', '2026']

MISSING_VAL = 99999.000

output_dir = '../storm_surge/'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'obs_{STATION_ID}_2021_2026.csv')

def fetch_noaa_hourly_month(stn, prod, yr, month, datum=None):
    """Fetches monthly observation stream and filters top-of-hour (minute 0) records."""
    start_str = f"{yr}{month:02d}01"

    if month in [1, 3, 5, 7, 8, 10, 12]:
        last_day = "31"
    elif month == 2:
        is_leap = (yr % 400 == 0) or (yr % 4 == 0 and yr % 100 != 0)
        last_day = "29" if is_leap else "28"
    else:
        last_day = "30"

    end_str = f"{yr}{month:02d}{last_day}"

    params = {
        "begin_date": start_str,
        "end_date": end_str,
        "station": stn,
        "product": prod,
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "web_services"
    }
    if datum and prod in ["water_level", "preliminary_water_level", "hourly_height"]:
        params["datum"] = datum

    try:
        r = requests.get(API_URL, params=params, timeout=10)
        if r.status_code == 200 and 'application/json' in r.headers.get('Content-Type', ''):
            data = r.json()
            if "data" in data and len(data["data"]) > 0:
                df = pd.DataFrame(data["data"])
                t_col = 't' if 't' in df.columns else 'time'
                if t_col in df.columns:
                    df['t'] = pd.to_datetime(df[t_col])
                    df.set_index('t', inplace=True)
                    for col in df.columns:
                        if col != t_col:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Filter Top-of-Hour Records (Minute 00 Only)
                    hourly_df = df[df.index.minute == 0].copy()
                    return hourly_df
    except Exception:
        pass
    return pd.DataFrame()

# Clean previous output before running
if os.path.exists(output_file):
    os.remove(output_file)

print("\n" + "="*80)
print(f"🏭 RETRIEVING HOURLY NOAA OBSERVATIONS FOR: {STATION_ID} ({STATION_NAME})")
print(f"   Missing values padded with: {MISSING_VAL}")
print("="*80)

total_saved_rows = 0

for yr_str in years:
    yr = int(yr_str)
    print(f"\n📅 --- Processing Year: {yr_str} ---")

    for month in range(1, 13):
        print(f"  > Fetching Hourly Block: {yr_str}-{month:02d}...", end='', flush=True)

        start_dt = datetime(yr, month, 1, 0, 0)
        last_d = 31 if month in [1,3,5,7,8,10,12] else (29 if (month==2 and ((yr%400==0) or (yr%4==0 and yr%100!=0))) else (28 if month==2 else 30))
        end_dt = datetime(yr, month, last_d, 23, 0)

        # Full monthly hourly grid base
        month_index = pd.date_range(start=start_dt, end=end_dt, freq='h')
        df_month_base = pd.DataFrame(index=month_index)

        # 1. Water Level Cascade
        df_wl = fetch_noaa_hourly_month(STATION_ID, "water_level", yr, month, datum="MLLW")
        if df_wl.empty or ('v' not in df_wl.columns and 'water_level' not in df_wl.columns):
            df_wl = fetch_noaa_hourly_month(STATION_ID, "preliminary_water_level", yr, month, datum="MLLW")
        if df_wl.empty or ('v' not in df_wl.columns and 'water_level' not in df_wl.columns):
            df_wl = fetch_noaa_hourly_month(STATION_ID, "hourly_height", yr, month, datum="MLLW")
        time.sleep(0.05)

        # 2. Meteorological Parameter Calls
        df_wind = fetch_noaa_hourly_month(STATION_ID, "wind", yr, month, datum=None)
        time.sleep(0.05)
        df_at = fetch_noaa_hourly_month(STATION_ID, "air_temperature", yr, month, datum=None)
        time.sleep(0.05)
        df_ap = fetch_noaa_hourly_month(STATION_ID, "air_pressure", yr, month, datum=None)
        time.sleep(0.05)

        def extract_col(df, col_candidates, target_name):
            if isinstance(df, pd.DataFrame) and not df.empty:
                for c in col_candidates:
                    if c in df.columns:
                        return df[[c]].rename(columns={c: target_name})
            return pd.DataFrame(columns=[target_name])

        wl = extract_col(df_wl, ['v', 'water_level'], 'water_level')
        at = extract_col(df_at, ['v', 'air_temp'], 'air_temp')
        ap = extract_col(df_ap, ['v', 'air_pressure'], 'air_pressure')

        if isinstance(df_wind, pd.DataFrame) and not df_wind.empty and set(['s', 'd', 'g']).issubset(df_wind.columns):
            w = df_wind[['s', 'd', 'g']].rename(columns={'s': 'wind_speed', 'd': 'wind_direction', 'g': 'wind_gust'})
        else:
            w = pd.DataFrame(columns=['wind_speed', 'wind_direction', 'wind_gust'])

        # Join features onto full monthly hourly grid
        df_month = df_month_base.join([wl, at, ap, w], how='left')

        # Fill missing values with 99999.000 across all expected columns
        expected_cols = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'water_level']
        for c in expected_cols:
            if c not in df_month.columns:
                df_month[c] = MISSING_VAL
            else:
                df_month[c] = df_month[c].fillna(MISSING_VAL)

        # Format output dataframe
        df_month.reset_index(names='timestamp', inplace=True)
        df_month['timestamp'] = df_month['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        export_cols = ['timestamp'] + expected_cols
        df_month = df_month[export_cols]

        write_header = not os.path.exists(output_file)
        df_month.to_csv(output_file, index=False, mode='a', header=write_header, float_format='%.3f')

        valid_wl = len(df_month[df_month['water_level'] != MISSING_VAL])
        valid_met = len(df_month[df_month['wind_speed'] != MISSING_VAL])
        total_saved_rows += len(df_month)

        print(f" Found! ({len(df_month)} rows | {valid_wl} valid WL | {valid_met} valid Met)")

print(f"\n✅ Successfully exported {total_saved_rows} hourly rows (missing padded with 99999.000) to: {output_file}")
print("\n🏁 Direct Hourly NOAA CO-OPS Data Retrieval Complete.")
