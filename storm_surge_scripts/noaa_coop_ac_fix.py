#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Developed with the assistance of the Gemini AI code assistant.
'''

import noaa_coops as nc
import os
# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pandas as pd
import time
from datetime import datetime

# --- 1. Setup ---
STN_ID_WL = "8534720"  # Steel Pier (Water Level)
STN_ID_MET = "8534730" # Marina (Wind/Temp/Pressure)
OUTPUT_FILE = f"../storm_surge/obs_{STN_ID_WL}_2020_2026.csv"

years = ['2020', '2022', '2023', '2024', '2025', '2026']
months = [1, 6]

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# Initialize the primary station object
try:
    # Use the first station to get the library context
    main_stn = nc.Station(STN_ID_WL)
    # We will use the Marina ID directly in the get_data calls for MET
    print(f"✅ Stations Initialized: {STN_ID_WL} (WL) and {STN_ID_MET} (MET)")
except Exception as e:
    print(f"❌ Critical Error: {e}")
    exit(1)

# --- 2. Retrieval Loop ---
for yr in years:
    y = int(yr)
    # 2025/2026 data is Preliminary
    wl_prod = "preliminary_water_level" if y >= 2025 else "water_level"
    
    for month in months:
        start_dt = datetime(y, month, 1)
        end_dt = datetime(y, month + 1, 1) if month < 12 else datetime(y + 1, 1, 1)
        
        beginStr = start_dt.strftime("%Y%m%d")
        endStr = end_dt.strftime("%Y%m%d")
        
        print(f"--- Processing {start_dt.strftime('%B %Y')} ---")
        
        try:
            # 1. Pull Water Level from Steel Pier
            df_wl = main_stn.get_data(begin_date=beginStr, end_date=endStr, product=wl_prod, 
                                     datum="MLLW", units="metric", time_zone="gmt")
            
            # 2. Pull Meteo from Marina (passing the different station ID to the get_data method)
            # Some versions of noaa_coops allow passing 'station' inside get_data. 
            # If your version doesn't, we initialize a temporary object for the Marina.\nstn_marina = nc.Station(STN_ID_MET)
            
            df_at = stn_marina.get_data(begin_date=beginStr, end_date=endStr, product="air_temperature", 
                                        units="metric", time_zone="gmt")
            df_wind = stn_marina.get_data(begin_date=beginStr, end_date=endStr, product="wind", 
                                          units="metric", time_zone="gmt")
            df_ap = stn_marina.get_data(begin_date=beginStr, end_date=endStr, product="air_pressure", 
                                        units="metric", time_zone="gmt")

            # --- Data Processing ---
            def clean(df, col_name):
                # Map 'v' to the desired column name
                return df[['v']].rename(columns={'v': col_name}) if 'v' in df.columns else None

            wl = clean(df_wl, 'water_level')
            at = clean(df_at, 'air_temp')
            ap = clean(df_ap, 'air_pressure')
            
            # Wind has s, d, g
            w = df_wind[['s', 'd', 'g']].rename(columns={'s':'wind_speed', 'd':'wind_direction', 'g':'wind_gust'})

            # Conversion and Merging
            wl['water_level'] = wl['water_level'] * 3.28084
            df_combined = pd.concat([wl, at, ap, w], axis=1)
            
            # Filter Hourly
            df_hourly = df_combined[df_combined.index.minute == 0].dropna().copy()
            df_hourly = df_hourly[df_hourly.index < end_dt]

            if not df_hourly.empty:
                df_hourly.reset_index(names='timestamp', inplace=True)
                df_hourly['timestamp'] = df_hourly['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                output_cols = ['timestamp', 'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'water_level']
                write_header = not os.path.exists(OUTPUT_FILE)
                df_hourly[output_cols].to_csv(OUTPUT_FILE, index=False, mode='a', header=write_header, float_format='%.3f')
                print(f"  [OK] Success: {len(df_hourly)} rows saved.")
            else:
                print("  [SKIP] No overlapping hourly data found.")

        except Exception as e:
            print(f"  [ERROR] at {beginStr}: {e}")
            continue
        
        time.sleep(1) # Be kind to NOAA's server\nprint(f"\n🏁 Finished. Check {OUTPUT_FILE}")
