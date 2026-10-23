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
config_file = '../storm_surge/stations_case_study_config.csv'
if not os.path.exists(config_file):
    print(f"Error: Configuration file {config_file} not found.")
    exit(1)

# Read the config containing station_id, station_name, lat, lon, etc.
stations_df = pd.read_csv(config_file)

# --- 2. Date Configuration ---
# Retrieval covers training period + the 2026 case study months
years_to_run = [2021, 2022, 2023, 2024, 2025, 2026]
months_to_run = [f"{i:02d}" for i in range(1, 13)]

# --- 3. Main Loop: Iterate through each station in config ---
for index, row in stations_df.iterrows():
    stn_id = str(row['station_id'])
    print('station_id ', stn_id)
    if stn_id == '8651370' or '8557380' : continue
    
    # Sanitize station name for filename (e.g., "Atlantic City_NJ")
    stn_name = str(row['station_name']).replace(", ", "_").replace(" ", "_").replace(".", "")

    # Initialize station object
    try:
        stn = nc.Station(stn_id)
    except Exception as e:
        print(f"Could not initialize station {stn_id}: {e}")
        continue

    # Create unique output file for each station in the case study
    output_file_name = f"../storm_surge/obs_{stn_id}_2021_2026.csv"

    print(f"\n" + "="*60)
    print(f"Retriever: {stn_id} | {row['station_name']}")
    print(f"File: {output_file_name}")
    print(f"="*60)

    for y in years_to_run:
        # Check for leap year
        is_leap = (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)

        for month_str in months_to_run:
            # Current date logic: Don't pull data past February 2026\nif y == 2026 and int(month_str) > 2:
                continue
                
            beginDate = f"{y}{month_str}01"

            if month_str in ['01', '03', '05', '07', '08', '10', '12']:
                last_day = "31"
            elif month_str == '02':
                last_day = "29" if is_leap else "28"
            else:
                last_day = "30"

            endDate = f"{y}{month_str}{last_day}"
            print(f"📅 {y}-{month_str} | Requesting {beginDate} to {endDate}...")

            try:
                # API calls with small delays to respect NOAA servers
                df_wl = stn.get_data(begin_date=beginDate, end_date=endDate, product="water_level", datum="MLLW", units="metric", time_zone="gmt")
                time.sleep(0.3)

                df_at = stn.get_data(begin_date=beginDate, end_date=endDate, product="air_temperature", units="metric", time_zone="gmt")
                time.sleep(0.3)

                df_wind = stn.get_data(begin_date=beginDate, end_date=endDate, product="wind", units="metric", time_zone="gmt")
                time.sleep(0.3)

                df_ap = stn.get_data(begin_date=beginDate, end_date=endDate, product="air_pressure", units="metric", time_zone="gmt")

                # Mapping helper to handle inconsistent column names in noaa_coops
                def get_col(df, pref):
                    for c in [pref, 'v', 's', 'd', 'g']:
                        if c in df.columns: return df[[c]]
                    return None

                # Extract and format
                wl = get_col(df_wl, 'water_level').rename(columns={'v':'water_level', 'water_level':'water_level'})
                at = get_col(df_at, 'air_temp').rename(columns={'v':'air_temp', 'air_temp':'air_temp'})
                ap = get_col(df_ap, 'air_pressure').rename(columns={'v':'air_pressure', 'air_pressure':'air_pressure'})
                w = df_wind[['s', 'd', 'g']].rename(columns={'s':'wind_speed', 'd':'wind_direction', 'g':'wind_gust'})

                # Conversion: Metric (Meters) -> Feet
                wl['water_level'] = wl['water_level'] * 3.28084

                # Combine and filter to top-of-the-hour observations
                df_combined = pd.concat([wl, at, ap, w], axis=1)
                df_hourly = df_combined[df_combined.index.minute == 0].dropna().copy()

                if not df_hourly.empty:
                    df_hourly.reset_index(names='timestamp', inplace=True)
                    df_hourly['timestamp'] = df_hourly['timestamp'].dt.strftime('%Y-%m-%d %H %M %S')

                    output_cols = ['timestamp', 'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'water_level']
                    write_header = not os.path.exists(output_file_name)
                    df_hourly[output_cols].to_csv(output_file_name, index=False, mode='a', header=write_header, float_format='%.3f')
                    print(f"   [OK] Appended {len(df_hourly)} rows.")
                else:
                    print("   [Skip] No complete hourly data.")

            except Exception as e:
                print(f"   [ERROR] Month {month_str} failed: {e}")
                continue

print(f"\n🏁 Finished. Observed data files for all configuration stations are in ../storm_surge/")
