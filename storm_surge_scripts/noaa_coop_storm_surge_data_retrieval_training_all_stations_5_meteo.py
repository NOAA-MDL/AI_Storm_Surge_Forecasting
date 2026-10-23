#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Developed with the assistance of the Gemini AI code assistant.
NOAA CO-OPS Historical Retrieval Engine with 99999 Missing Value Imputation.
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
years_to_run = ['2021', '2022', '2023', '2024', '2025']
months_to_run = [i for i in range(1, 13)]  # Full 12-month annual cycle

MISSING_VAL = 99999.0

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

    output_file_name = f"../storm_surge/training_obs_{stn_id}_{stn_name}_2021_2025.csv"

    # Reset existing file to rebuild cleanly
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    print(f"\n" + "="*60)
    print(f"Starting retrieval for station: {stn_id} ({stn_name})")
    print(f"="*60)

    for year_str in years_to_run:
        y = int(year_str)
        for month in months_to_run:
            # Calculate total days in month
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
            df_base = pd.DataFrame(index=month_hourly_index)

            df_wl, df_at, df_wind, df_ap = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            try:
                print("  > Pulling Water Level...", end='', flush=True)
                try:
                    df_wl = stn.get_data(begin_date=beginDate, end_date=endDate, product="water_level", datum="MLLW", units="metric", time_zone="gmt")
                except Exception:
                    pass
                time.sleep(0.3)

                print(" Air Temp...", end='', flush=True)
                try:
                    df_at = stn.get_data(begin_date=beginDate, end_date=endDate, product="air_temperature", units="metric", time_zone="gmt")
                except Exception:
                    pass
                time.sleep(0.3)

                print(" Wind...", end='', flush=True)
                try:
                    df_wind = stn.get_data(begin_date=beginDate, end_date=endDate, product="wind", units="metric", time_zone="gmt")
                except Exception:
                    pass
                time.sleep(0.3)

                print(" Pressure...", end='', flush=True)
                try:
                    df_ap = stn.get_data(begin_date=beginDate, end_date=endDate, product="air_pressure", units="metric", time_zone="gmt")
                except Exception:
                    pass
                print(" Done.")

                # Clean & Filter Top-of-Hour Records
                def get_clean_col(df, pref, target_col):
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        for c in [pref, 'v', 's', 'd', 'g']:
                            if c in df.columns:
                                res = df[[c]].copy()
                                res.index = pd.to_datetime(res.index)
                                res = res[res.index.minute == 0]
                                res[c] = pd.to_numeric(res[c], errors='coerce')
                                return res.rename(columns={c: target_col})
                    return pd.DataFrame(columns=[target_col])

                wl = get_clean_col(df_wl, 'v', 'water_level')
                at = get_clean_col(df_at, 'v', 'air_temp')
                ap = get_clean_col(df_ap, 'v', 'air_pressure')

                if isinstance(df_wind, pd.DataFrame) and not df_wind.empty and set(['s', 'd', 'g']).issubset(df_wind.columns):
                    w = df_wind[['s', 'd', 'g']].copy()
                    w.index = pd.to_datetime(w.index)
                    w = w[w.index.minute == 0]
                    for c in ['s', 'd', 'g']:
                        w[c] = pd.to_numeric(w[c], errors='coerce')
                    w = w.rename(columns={'s': 'wind_speed', 'd': 'wind_direction', 'g': 'wind_gust'})
                else:
                    w = pd.DataFrame(columns=['wind_speed', 'wind_direction', 'wind_gust'])

                # Convert Meters to Feet for valid water levels
                if not wl.empty and 'water_level' in wl.columns:
                    wl['water_level'] = wl['water_level'] * 3.28084

                # Join parameter streams on the full hourly index grid
                df_hourly = df_base.join([wl, at, ap, w], how='left')

                # Fill all unpopulated / missing observation cells with 99999.0
                output_cols = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'water_level']
                for col in output_cols:
                    if col not in df_hourly.columns:
                        df_hourly[col] = MISSING_VAL
                    else:
                        df_hourly[col] = df_hourly[col].fillna(MISSING_VAL)

                # Format timestamp string column
                df_hourly.reset_index(names='timestamp', inplace=True)
                df_hourly['timestamp'] = df_hourly['timestamp'].dt.strftime('%Y-%m-%d %H %M %S')

                final_cols = ['timestamp'] + output_cols
                write_header = not os.path.exists(output_file_name)
                df_hourly[final_cols].to_csv(output_file_name, index=False, mode='a', header=write_header, float_format='%.3f')

                valid_rows = len(df_hourly[df_hourly['water_level'] != MISSING_VAL])
                print(f"  [OK] Saved {len(df_hourly)} rows ({valid_rows} valid water level records, missing padded with {int(MISSING_VAL)}).")

            except Exception as e:
                print(f"\n[ERROR] Skipping {beginDate}: {e}")
                continue

print(f"\n🏁 Finished. Data stored in: {output_file_name}")
