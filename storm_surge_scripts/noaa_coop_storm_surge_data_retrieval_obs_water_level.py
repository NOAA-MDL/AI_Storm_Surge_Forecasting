#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Developed with the assistance of the Gemini AI code assistant.
'''

import noaa_coops as nc
from noaa_coops.station import COOPSAPIError
import os
# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pandas as pd
import time
from datetime import datetime, timedelta

# --- 1. Load Station Configuration ---
config_file = '../storm_surge/stations_config.csv'
if not os.path.exists(config_file):
    print(f"Error: Configuration file {config_file} not found.")
    exit(1)

stations_df = pd.read_csv(config_file)

# --- 2. Date Configuration ---
years_to_run = ['2021', '2022', '2023', '2024', '2025', '2026']
months_to_run = [i for i in range(1, 13)]

# --- 3. Main Loop ---
for index, row in stations_df.iterrows():
    stn_id = str(row['station_id'])
    if stn_id != "8658120": continue  # Isolated for Wilmington, NC
    stn_name = str(row['station_name']).replace(", ", "_").replace(" ", "_").replace(".", "")

    try:
        stn = nc.Station(stn_id)
    except Exception as e:
        print(f"Could not initialize station {stn_id}: {e}")
        continue

    # --- Clean Slate Logic ---
    output_file_name = f"../storm_surge/obs_{stn_id}_2021_2026.csv"

    if os.path.exists(output_file_name):
        print(f"🗑️ Found existing file: {output_file_name}. Deleting to ensure clean ISO formatting...")
        os.remove(output_file_name)

    print(f"\n" + "="*50)
    print(f"Starting Pure Water Level retrieval for station: {stn_id} ({stn_name})")
    print(f"="*50)

    for year_str in years_to_run:
        y = int(year_str)
        for month in months_to_run:
            start_dt = datetime(y, month, 1)

            if month == 12:
                end_dt = datetime(y + 1, 1, 1)
            else:
                end_dt = datetime(y, month + 1, 1)

            beginDate = start_dt.strftime("%Y%m%d")
            endDate = end_dt.strftime("%Y%m%d")

            print(f"--- Processing: {start_dt.strftime('%B %Y')} (API Range: {beginDate} to {endDate}) ---")

            try:
                # --- PURE WATER LEVEL API ONLY ---
                print("  > Pulling Water Level...", end='', flush=True)
                df_wl = stn.get_data(begin_date=beginDate, end_date=endDate, product="water_level", datum="MLLW", units="metric", time_zone="gmt")
                print(" Done.")
                time.sleep(0.5)

                # Clean extraction block
                if 'v' in df_wl.columns:
                    wl = df_wl[['v']].rename(columns={'v': 'water_level'})
                else:
                    print("  [Skip] No valid values ('v') column found in data payload.")
                    continue

                # Conversion to feet (NOAA standard matching your verified chart)
                wl['water_level'] = wl['water_level'] * 3.28084

                # Filter for top-of-the-hour data blocks
                df_hourly = wl[wl.index.minute == 0].copy()
                df_hourly = df_hourly[df_hourly.index < end_dt]

                if not df_hourly.empty:
                    df_hourly.reset_index(names='timestamp', inplace=True)

                    # Standard ISO formatting layer conversion
                    df_hourly['timestamp'] = df_hourly['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                    output_cols = ['timestamp', 'water_level']

                    write_header = not os.path.exists(output_file_name)
                    df_hourly[output_cols].to_csv(output_file_name, index=False, mode='a', header=write_header, float_format='%.3f')
                    print(f"    [OK] Saved {len(df_hourly)} rows.")
                else:
                    print("    [Skip] No complete hourly water level points found.")

            except Exception as e:
                print(f"\n[ERROR] Skipping {beginDate}: {e}")
                continue

print(f"\n🏁 Finished. Pure Water Level data stored cleanly in: {output_file_name}")
