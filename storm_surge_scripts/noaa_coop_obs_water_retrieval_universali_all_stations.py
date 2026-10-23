#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Direct Hourly NOAA CO-OPS Water Level Data Retrieval Engine
Uses Monthly 6-Minute Stream Resampled to Top-of-Hour
Datum Standardized: MLLW & GMT (Direct ETSS Alignment)
Output Format per station: ../storm_surge/obs_<STATION_ID>_wl_only.csv
'''

import os, sys, time, requests
import pandas as pd, numpy as np
from datetime import datetime

os.environ["TF_USE_LEGACY_KERAS"] = "1"
API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

years = ['2021', '2022', '2023', '2024', '2025', '2026']

output_dir = '../storm_surge/'
os.makedirs(output_dir, exist_ok=True)

CONFIG_FILE = 'stations_config.csv'
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = os.path.join(output_dir, 'stations_config.csv')

if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: Master station configuration matrix missing at: {CONFIG_FILE}")
    sys.exit(1)

df_config = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})
df_config['station_id'] = df_config['station_id'].str.strip()
df_config['station_name'] = df_config['station_name'].str.strip().str.replace('"', '').str.replace(',', '').str.replace(' ', '_')

now_dt = datetime.now()

def fetch_noaa_monthly_mllw(stn, yr, month):
    """Fetches monthly water level stream (6-min resampled to top-of-hour) on MLLW & GMT."""
    start_str = f"{yr}{month:02d}01"
    last_day = "31" if month in [1, 3, 5, 7, 8, 10, 12] else ("29" if (month == 2 and ((yr % 400 == 0) or (yr % 4 == 0 and yr % 100 != 0))) else ("28" if month == 2 else "30"))
    end_str = f"{yr}{month:02d}{last_day}"

    # Try preliminary/water_level first, then hourly_height
    for prod in ["water_level", "preliminary_water_level", "hourly_height"]:
        params = {
            "begin_date": start_str,
            "end_date": end_str,
            "station": stn,
            "product": prod,
            "datum": "MLLW",       # MLLW Datum
            "units": "metric",
            "time_zone": "gmt",    # GMT Time
            "format": "json",
            "application": "web_services"
        }

        try:
            r = requests.get(API_URL, params=params, timeout=12)
            if r.status_code == 200 and 'application/json' in r.headers.get('Content-Type', ''):
                data = r.json()
                if "data" in data and len(data["data"]) > 0:
                    df = pd.DataFrame(data["data"])
                    t_col = 't' if 't' in df.columns else 'time'
                    val_col = 'v' if 'v' in df.columns else 'water_level'

                    if t_col in df.columns and val_col in df.columns:
                        df['t'] = pd.to_datetime(df[t_col])
                        df.set_index('t', inplace=True)
                        df['water_level'] = pd.to_numeric(df[val_col], errors='coerce')

                        # Top-of-Hour Minute 0 Filter (Legacy Match)
                        hourly_df = df[df.index.minute == 0][['water_level']].dropna().copy()
                        if not hourly_df.empty:
                            return hourly_df
        except Exception:
            pass

    return pd.DataFrame()

# =====================================================================
# 🚀 MASTER RETRIEVAL LOOP ACROSS ALL REGISTRY STATIONS
# =====================================================================
print("\n" + "="*80)
print(f"🏭 RUNNING NOAA WATER LEVEL RETRIEVAL FOR {len(df_config)} STATIONS")
print(f"   Baseline Datum: MLLW & GMT (Direct ETSS Alignment)")
print("="*80)

for idx, row in df_config.iterrows():
    STATION_ID = str(row['station_id']).strip()
    STATION_NAME = str(row['station_name']).strip()
    output_file = os.path.join(output_dir, f'obs_{STATION_ID}_wl_only.csv')

    print(f"\n📍 [{idx+1}/{len(df_config)}] Processing Station: {STATION_NAME} ({STATION_ID})")

    if os.path.exists(output_file):
        os.remove(output_file)

    stop_retrieval = False
    station_dfs = []

    for yr_str in years:
        if stop_retrieval:
            break
        yr = int(yr_str)

        for month in range(1, 13):
            start_dt = datetime(yr, month, 1, 0, 0)
            if start_dt > now_dt:
                stop_retrieval = True
                break

            df_wl = fetch_noaa_monthly_mllw(STATION_ID, yr, month)
            time.sleep(0.05)

            if not df_wl.empty and 'water_level' in df_wl.columns:
                # Metric -> Feet Conversion (Legacy Match)
                df_wl['water_level'] = df_wl['water_level'] * 3.28084
                df_month = df_wl.dropna(subset=['water_level']).copy()
                df_month = df_month[df_month.index <= now_dt]

                if not df_month.empty:
                    df_month.reset_index(names='timestamp', inplace=True)
                    df_month['timestamp'] = df_month['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    station_dfs.append(df_month[['timestamp', 'water_level']])

    if station_dfs:
        final_df = pd.concat(station_dfs, ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        final_df.to_csv(output_file, index=False, float_format='%.3f')
        print(f"  ✅ Saved {len(final_df)} hourly MLLW records to {output_file}")
    else:
        print(f"  ⚠️ Zero valid MLLW records found for station {STATION_ID}.")
