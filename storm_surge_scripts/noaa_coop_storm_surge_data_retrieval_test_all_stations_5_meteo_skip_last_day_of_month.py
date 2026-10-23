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

# --- 1. Load Station Configuration ---
# Updated to a generic station list file
config_file = '../storm_surge/stations_config.csv' 
if not os.path.exists(config_file):
    print(f"Error: Configuration file {config_file} not found.")
    exit(1)

stations_df = pd.read_csv(config_file)

# --- 2. Date Configuration ---
# Set to 2021-2025 per your requirements
years_to_run = ['2026']
months_to_run = [f"{i:02d}" for i in range(1, 13)]

# --- 3. Main Loop: Iterate through each station ---
for index, row in stations_df.iterrows():
    stn_id = str(row['station_id'])
    if stn_id != "9468756": continue
    
    print("********* ",stn_id)
    # Sanitize station name for filename
    stn_name = str(row['station_name']).replace(", ", "_").replace(" ", "_").replace(".", "")
    print(stn_name)

    # Initialize station object
    try:
        stn = nc.Station(stn_id)
    except Exception as e:
        print(f"Could not initialize station {stn_id}: {e}")
        continue

    # Generic output naming (removed 'case' and 'meteo' specificities if desired)
    output_file_name = f"../storm_surge/obs_{stn_id}_{stn_name}_2021_2026.csv"

    print(f"\n" + "="*50)
    print(f"Starting retrieval for station: {stn_id} ({stn_name})")
    print(f"="*50)

    for year_str in years_to_run:
        y = int(year_str)
        print('********************* ', year_str)
        # Leap year check
        is_leap = (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)

        for month_str in months_to_run:
            beginDate = f"{year_str}{month_str}01"

            # Determine last day of the month
            if month_str in ['01', '03', '05', '07', '08', '10', '12']:
                last_day = "31"
            elif month_str == '02':
                last_day = "29" if is_leap else "28"
            else:
                last_day = "30"

            endDate = f"{year_str}{month_str}{last_day}"
            print(f"--- Processing: {beginDate} to {endDate} ---")

            try:
                # API calls
                print("  > Pulling Water Level...", end='', flush=True)
                df_wl = stn.get_data(begin_date=beginDate, end_date=endDate, product="water_level", datum="MLLW", units="metric", time_zone="gmt")
                time.sleep(0.4)

                print(" Air Temp...", end='', flush=True)
                df_at = stn.get_data(begin_date=beginDate, end_date=endDate, product="air_temperature", units="metric", time_zone="gmt")
                time.sleep(0.4)

                print(" Wind...", end='', flush=True)
                df_wind = stn.get_data(begin_date=beginDate, end_date=endDate, product="wind", units="metric", time_zone="gmt")
                time.sleep(0.4)

                print(" Pressure...", end='', flush=True)
                df_ap = stn.get_data(begin_date=beginDate, end_date=endDate, product="air_pressure", units="metric", time_zone="gmt")
                print(" Done.")

                # Clean and Rename logic
                def get_clean_col(df, pref):
                    for c in [pref, 'v', 's', 'd', 'g']:
                        if c in df.columns: return df[[c]]
                    return None

                wl = get_clean_col(df_wl, 'v').rename(columns={'v':'water_level'})
                at = get_clean_col(df_at, 'v').rename(columns={'v':'air_temp'})
                ap = get_clean_col(df_ap, 'v').rename(columns={'v':'air_pressure'})
                w = df_wind[['s', 'd', 'g']].rename(columns={'s':'wind_speed', 'd':'wind_direction', 'g':'wind_gust'})

                # Conversion: Metric -> Feet
                wl['water_level'] = wl['water_level'] * 3.28084

                # Merge and isolate hourly (minute 00)
                df_combined = pd.concat([wl, at, ap, w], axis=1)
                df_hourly = df_combined[df_combined.index.minute == 0].dropna().copy()

                if not df_hourly.empty:
                    df_hourly.reset_index(names='timestamp', inplace=True)
                    df_hourly['timestamp'] = df_hourly['timestamp'].dt.strftime('%Y-%m-%d %H %M %S')

                    output_cols = ['timestamp', 'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'water_level']
                    
                    # Append mode logic
                    write_header = not os.path.exists(output_file_name)
                    df_hourly[output_cols].to_csv(output_file_name, index=False, mode='a', header=write_header, float_format='%.3f')
                    print(f"  [OK] Saved {len(df_hourly)} rows.")
                else:
                    print("  [Skip] No complete hourly rows found.")

            except Exception as e:
                print(f"\n[ERROR] Skipping {beginDate}: Data unavailable or {e}")
                continue

print(f"\n🏁 Finished. Data stored in ../storm_surge/")
