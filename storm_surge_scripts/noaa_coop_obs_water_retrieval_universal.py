#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Developed with the assistance of the Gemini AI code assistant.
Direct Hourly NOAA CO-OPS Water Level Data Retrieval Engine - Universal Baseline
Full Hourly Grid Generation (Capped at Current Date) with 99999 Missing Value Imputation
'''

import os, sys, time, requests
import pandas as pd, numpy as np
from datetime import datetime

os.environ["TF_USE_LEGACY_KERAS"] = "1"
API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# --- Station & Output Configuration ---
STATION_ID = "8726724"
STATION_NAME = "St_Petersburg_FL"
years = ['2021', '2022', '2023', '2024', '2025', '2026']

MISSING_VAL = 99999.000

output_dir = '../storm_surge/'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'obs_{STATION_ID}_2021_2026.csv')

# Get current system execution date & time
now_dt = datetime.now()

def fetch_noaa_hourly_water_level(stn, prod, yr, month, datum="MLLW"):
    """Fetches monthly water level stream and filters top-of-hour (minute 0) records."""
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
        "datum": datum,
        "units": "metric",
        "time_zone": "gmt",
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

                    # Filter Top-of-Hour Records (Minute 00 Only)
                    hourly_df = df[df.index.minute == 0][['water_level']].copy()
                    return hourly_df
    except Exception:
        pass
    return pd.DataFrame()

# Clean previous output before running
if os.path.exists(output_file):
    os.remove(output_file)

print("\n" + "="*80)
print(f"🏭 RETRIEVING HOURLY NOAA WATER LEVEL FOR: {STATION_ID} ({STATION_NAME})")
print(f"   Retrieval capped at current date: {now_dt.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Missing values padded with: {MISSING_VAL}")
print("="*80)

total_saved_rows = 0
stop_retrieval = False

for yr_str in years:
    if stop_retrieval:
        break

    yr = int(yr_str)
    print(f"\n📅 --- Processing Year: {yr_str} ---")

    for month in range(1, 13):
        start_dt = datetime(yr, month, 1, 0, 0)
        
        # 🌟 STRICT FUTURE BOUNDARY CHECK
        if start_dt > now_dt:
            print(f"  > Reached future boundary ({start_dt.strftime('%Y-%m')}). Halting retrieval loop.")
            stop_retrieval = True
            break

        last_d = 31 if month in [1,3,5,7,8,10,12] else (29 if (month==2 and ((yr%400==0) or (yr%4==0 and yr%100!=0))) else (28 if month==2 else 30))
        end_dt = datetime(yr, month, last_d, 23, 0)

        # Cap the end_dt to current hour if processing current month
        if end_dt > now_dt:
            end_dt = now_dt.replace(minute=0, second=0, microsecond=0)

        print(f"  > Fetching Hourly Block: {yr_str}-{month:02d}...", end='', flush=True)

        # Full monthly hourly grid base up to capped end_dt boundary
        month_index = pd.date_range(start=start_dt, end=end_dt, freq='h')
        df_month_base = pd.DataFrame(index=month_index)

        # Water Level Product Cascade (Verified -> Preliminary -> Hourly Height)
        df_wl = fetch_noaa_hourly_water_level(STATION_ID, "water_level", yr, month, datum="MLLW")
        if df_wl.empty:
            df_wl = fetch_noaa_hourly_water_level(STATION_ID, "preliminary_water_level", yr, month, datum="MLLW")
        if df_wl.empty:
            df_wl = fetch_noaa_hourly_water_level(STATION_ID, "hourly_height", yr, month, datum="MLLW")
        time.sleep(0.05)

        # Convert meters to feet for valid readings
        if not df_wl.empty and 'water_level' in df_wl.columns:
            df_wl['water_level'] = df_wl['water_level'] * 3.28084

        # Join onto full monthly hourly grid
        df_month = df_month_base.join(df_wl, how='left')

        # 🌟 SAFE COLUMN INITIALIZATION FIX
        if 'water_level' not in df_month.columns:
            df_month['water_level'] = MISSING_VAL
        else:
            df_month['water_level'] = df_month['water_level'].fillna(MISSING_VAL)

        # Format output dataframe
        df_month.reset_index(names='timestamp', inplace=True)
        df_month['timestamp'] = df_month['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        export_cols = ['timestamp', 'water_level']
        df_month = df_month[export_cols]

        write_header = not os.path.exists(output_file)
        df_month.to_csv(output_file, index=False, mode='a', header=write_header, float_format='%.3f')

        valid_wl = len(df_month[df_month['water_level'] != MISSING_VAL])
        total_saved_rows += len(df_month)

        print(f" Found! ({len(df_month)} rows | {valid_wl} valid water level records)")

print(f"\n✅ Successfully exported {total_saved_rows} hourly rows (missing padded with {MISSING_VAL}) to: {output_file}")
print("\n🏁 Direct Hourly NOAA CO-OPS Water Level Data Retrieval Complete.")
