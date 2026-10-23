#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Developed with the assistance of the Gemini AI code assistant.
Multi-Station NOAA CO-OPS Data Retrieval Engine
Loops over stations in ../storm_surge/stations_config.csv with full weather features.
'''

import noaa_coops as nc
from noaa_coops.station import COOPSAPIError
import os
# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pandas as pd
import time

# --- Output & Config Directories ---
output_dir = '../storm_surge/'
os.makedirs(output_dir, exist_ok=True)

CONFIG_FILE = os.path.join(output_dir, 'stations_config.csv')
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = 'stations_config.csv'

if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: Master station configuration matrix missing at: {CONFIG_FILE}")
    sys.exit(1)

print(f"📖 Parsing station list from registry: {CONFIG_FILE}")
df_config = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})
df_config['station_id'] = df_config['station_id'].str.strip()
df_config['station_name'] = df_config['station_name'].str.strip().str.replace('"', '').str.replace(',', '').str.replace(' ', '_')

# Years to process
years_to_run = ['2021', '2022', '2023', '2024', '2025', '2026']
months_to_run = [f"{i:02d}" for i in range(1, 13)]

# Helper function to find the right column name (v/s/d/g or full names)
def get_col(df, pref):
    for c in [pref, 'v', 's', 'd', 'g']:
        if c in df.columns: return df[[c]]
    return None

# =====================================================================
# 🚀 MASTER RETRIEVAL LOOP ACROSS ALL STATIONS IN CONFIG
# =====================================================================
print("\n" + "="*80)
print(f"🏭 STARTING MULTI-STATION RETRIEVAL FOR {len(df_config)} STATIONS")
print("="*80)

for idx, row in df_config.iterrows():
    stn_id = str(row['station_id']).strip()
    stn_name = str(row['station_name']).strip()
    
    # Target output file per station in ../storm_surge/
    output_file_name = os.path.join(output_dir, f"obs_training_{stn_id}.csv")

    print(f"\n📍 [{idx+1}/{len(df_config)}] Starting retrieval for station {stn_name} ({stn_id})...")
    print(f"   Target Output File: {output_file_name}")

    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    # Use a retry loop for the initial connection
    connected = False
    conn_attempts = 0
    stn = None

    while not connected and conn_attempts < 5:
        try:
            stn = nc.Station(stn_id)
            connected = True
        except Exception as e:
            conn_attempts += 1
            print(f"Connection failed: {e}. Retrying in 30 seconds (Attempt {conn_attempts}/5)...")
            time.sleep(30)

    if not connected or stn is None:
        print(f"❌ Could not connect to station {stn_id}. Skipping.")
        continue

    for year_str in years_to_run:
        for month_str in months_to_run:
            beginDate = f"{year_str}{month_str}01"

            # Last day calculation
            if month_str in ['01', '03', '05', '07', '08', '10', '12']:
                last_day = "31"
            elif month_str == '02':
                y = int(year_str)
                last_day = "29" if (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0) else "28"
            else:
                last_day = "30"

            endDate = f"{year_str}{month_str}{last_day}"
            print(f"--- Processing: {beginDate} to {endDate} ---")

            try:
                # 1. Sequential API calls with progress feedback
                print("  > Pulling Water Level...", end='', flush=True)
                df_wl = stn.get_data(begin_date=beginDate, end_date=endDate, product="water_level", datum="MLLW", units="metric", time_zone="gmt")
                time.sleep(0.5) # Short pause to avoid API rate limiting

                print(" Air Temp...", end='', flush=True)
                df_at = stn.get_data(begin_date=beginDate, end_date=endDate, product="air_temperature", units="metric", time_zone="gmt")
                time.sleep(0.5)

                print(" Wind...", end='', flush=True)
                df_wind = stn.get_data(begin_date=beginDate, end_date=endDate, product="wind", units="metric", time_zone="gmt")
                time.sleep(0.5)

                print(" Pressure...", end='', flush=True)
                df_ap = stn.get_data(begin_date=beginDate, end_date=endDate, product="air_pressure", units="metric", time_zone="gmt")
                print(" Done.")

                # 2. Extract and Rename
                wl = get_col(df_wl, 'water_level').rename(columns={'v':'water_level', 'water_level':'water_level'})
                at = get_col(df_at, 'air_temp').rename(columns={'v':'air_temp', 'air_temp':'air_temp'})
                ap = get_col(df_ap, 'air_pressure').rename(columns={'v':'air_pressure', 'air_pressure':'air_pressure'})

                # Wind has 3 columns
                w = df_wind[['s', 'd', 'g']].rename(columns={'s':'wind_speed', 'd':'wind_direction', 'g':'wind_gust'})

                # 3. Conversion: Metric -> Feet (The ETSS Match)
                wl['water_level'] = wl['water_level'] * 3.28084

                # 4. Merge and Filter
                df_combined = pd.concat([wl, at, ap, w], axis=1)
                df_hourly = df_combined[df_combined.index.minute == 0].dropna().copy()

                if not df_hourly.empty:
                    df_hourly.reset_index(names='timestamp', inplace=True)
                    df_hourly['timestamp'] = df_hourly['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                    output_cols = ['timestamp', 'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'water_level']
                    write_header = not os.path.exists(output_file_name)
                    df_hourly[output_cols].to_csv(output_file_name, index=False, mode='a', header=write_header, float_format='%.3f')
                    print(f"  [OK] Saved {len(df_hourly)} rows.")
                else:
                    print("  [Skip] No complete hourly rows found.")

            except Exception as e:
                print(f"\n[ERROR] Skipping {beginDate}: {e}")
                continue

    print(f"Task Finished for station {stn_id}. Output saved to: {output_file_name}")

print("\n" + "="*80)
print("🎉 Task Finished for all stations in config matrix.")
print("="*80)
