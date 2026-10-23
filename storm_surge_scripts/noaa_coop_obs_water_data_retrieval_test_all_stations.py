#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Developed with the assistance of the Gemini AI code assistant.
NOAA CO-OPS Water Level Data Retrieval Engine with 99999 Missing Value Imputation.
'''

import noaa_coops as nc
from noaa_coops.station import COOPSAPIError
import os

# 🌟 FORCE KERAS 2 DESERIALIZER
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np
import time
from datetime import datetime

# --- 1. Load Station Configuration ---
config_file = '../storm_surge/stations_config.csv'
if not os.path.exists(config_file):
    print(f"Error: Configuration file {config_file} not found.")
    exit(1)

stations_df = pd.read_csv(config_file)

# --- 2. Date Configuration ---
years_to_run = ['2021', '2022', '2023', '2024', '2025', '2026']
months_to_run = [i for i in range(1, 13)]  # Full annual 12-month cycle

MISSING_VAL = 99999.000

# --- 3. Main Loop ---
for index, row in stations_df.iterrows():
    stn_id = str(row['station_id']).strip()
    if stn_id != "8729108": 
        continue
    stn_name = str(row['station_name']).replace(", ", "_").replace(" ", "_").replace(".", "")

    try:
        stn = nc.Station(stn_id)
    except Exception as e:
        print(f"Could not initialize station {stn_id}: {e}")
        continue

    output_file_name = f"../storm_surge/obs_{stn_id}_2021_2026.csv"

    # Reset existing output file to rebuild cleanly
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    print(f"\n" + "="*60)
    print(f"Starting Water Level retrieval for station: {stn_id} ({stn_name})")
    print(f"Missing values padded with: {MISSING_VAL}")
    print(f"="*60)

    for year_str in years_to_run:
        y = int(year_str)
        for month in months_to_run:
            # Days in month calculation
            if month in [1, 3, 5, 7, 8, 10, 12]:
                last_d = 31
            elif month == 2:
                is_leap = (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)
                last_d = 29 if is_leap else 28
            else:
                last_d = 30

            start_dt = datetime(y, month, 1, 0, 0)
            end_dt = datetime(y, month, last_d, 23, 0)

            beginDate = f"{y}{month:02d}01"
            endDate = f"{y}{month:02d}{last_d}"

            print(f"--- Processing: {start_dt.strftime('%B %Y')} (API Range: {beginDate} to {endDate}) ---")

            # Initialize full monthly hourly grid
            month_hourly_index = pd.date_range(start=start_dt, end=end_dt, freq='h')
            df_hourly = pd.DataFrame(index=month_hourly_index)

            df_wl = pd.DataFrame()

            try:
                print("  > Pulling Water Level...", end='', flush=True)
                
                # Cascade through water level products if primary is unpopulated
                for prod in ["water_level", "preliminary_water_level", "hourly_height"]:
                    try:
                        df_wl = stn.get_data(
                            begin_date=beginDate, 
                            end_date=endDate, 
                            product=prod, 
                            datum="MLLW", 
                            units="metric", 
                            time_zone="gmt"
                        )
                        if isinstance(df_wl, pd.DataFrame) and not df_wl.empty and ('v' in df_wl.columns or 'water_level' in df_wl.columns):
                            break
                    except Exception:
                        pass

                time.sleep(0.3)
                print(" Done.")

                # Extract and Clean Top-of-Hour Records (Minute 00 Only)
                if isinstance(df_wl, pd.DataFrame) and not df_wl.empty:
                    val_col = 'v' if 'v' in df_wl.columns else 'water_level'
                    if val_col in df_wl.columns:
                        res = df_wl[[val_col]].copy()
                        res.index = pd.to_datetime(res.index)
                        res = res[res.index.minute == 0]
                        res['water_level'] = pd.to_numeric(res[val_col], errors='coerce')
                        
                        # Convert Meters to Feet for valid observations
                        res['water_level'] = res['water_level'] * 3.28084
                        
                        # Join extracted water level onto full monthly hourly grid
                        df_hourly = df_hourly.join(res[['water_level']], how='left')

                # Fill unpopulated / missing observation cells with 99999.000
                if 'water_level' not in df_hourly.columns:
                    df_hourly['water_level'] = MISSING_VAL
                else:
                    df_hourly['water_level'] = df_hourly['water_level'].fillna(MISSING_VAL)

                # Format timestamp string column
                df_hourly.reset_index(names='timestamp', inplace=True)
                df_hourly['timestamp'] = df_hourly['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                output_cols = ['timestamp', 'water_level']
                write_header = not os.path.exists(output_file_name)
                df_hourly[output_cols].to_csv(output_file_name, index=False, mode='a', header=write_header, float_format='%.3f')

                valid_rows = len(df_hourly[df_hourly['water_level'] != MISSING_VAL])
                print(f"  [OK] Saved {len(df_hourly)} rows ({valid_rows} valid water level records, missing padded with {MISSING_VAL}).")

            except Exception as e:
                print(f"\n[ERROR] Skipping {beginDate}: {e}")
                continue

print(f"\n🏁 Finished. Water level data stored in: {output_file_name}")
