#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import os, sys
import pandas as pd
import numpy as np

OBS_FILE = '../storm_surge/obs_9468756_2021_2026.csv'
GFS_FILE = '../storm_surge/test_gfs_etss_spatial_data_9468756_2026.csv'
OUTPUT_FILE = '../storm_surge/obs_9468756_2021_2026.csv'

print("🛠️ BUILDING HYBRID IMPUTED OBSERVATION DATASET FOR NOME, AK (9468756)...")

# 1. Load Raw Observations
if not os.path.exists(OBS_FILE):
    print(f"❌ Error: Raw observation file not found at {OBS_FILE}")
    sys.exit(1)

df_obs = pd.read_csv(OBS_FILE)
df_obs['dt'] = pd.to_datetime(df_obs['timestamp'])
df_obs.set_index('dt', inplace=True)

# 2. Load GFS / ETSS Reference Stream
if os.path.exists(GFS_FILE):
    df_gfs = pd.read_csv(GFS_FILE)
    t_col = 'timestamp_gui' if 'timestamp_gui' in df_gfs.columns else 'timestamp'
    df_gfs['dt'] = pd.to_datetime(df_gfs[t_col])
    df_gfs = df_gfs.sort_values('cycle').drop_duplicates(subset=['dt'], keep='last').set_index('dt')
else:
    df_gfs = pd.DataFrame()

# 3. Create Continuous Hourly Datetime Index from 2021 to 2026
full_index = pd.date_range(start="2021-01-01 00:00:00", end="2026-03-31 23:00:00", freq="h")
df_hybrid = pd.DataFrame(index=full_index)

# Merge raw observations onto the continuous time skeleton
df_hybrid = df_hybrid.join(df_obs[['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'water_level']], how='left')

# 4. Fill Meteorological Gaps using GFS Model Data
if not df_gfs.empty:
    if 'air_temp' in df_hybrid.columns and 'air_temp_gui' in df_gfs.columns:
        df_hybrid['air_temp'] = df_hybrid['air_temp'].fillna(df_gfs['air_temp_gui'])
    if 'air_pressure' in df_hybrid.columns and 'air_pressure_gui' in df_gfs.columns:
        df_hybrid['air_pressure'] = df_hybrid['air_pressure'].fillna(df_gfs['air_pressure_gui'])
    if 'wind_speed' in df_hybrid.columns and 'wind_speed_gui' in df_gfs.columns:
        df_hybrid['wind_speed'] = df_hybrid['wind_speed'].fillna(df_gfs['wind_speed_gui'])
    if 'wind_direction' in df_hybrid.columns and 'wind_direction_gui' in df_gfs.columns:
        df_hybrid['wind_direction'] = df_hybrid['wind_direction'].fillna(df_gfs['wind_direction_gui'])
    if 'wind_gust' in df_hybrid.columns and 'wind_gust_gui' in df_gfs.columns:
        df_hybrid['wind_gust'] = df_hybrid['wind_gust'].fillna(df_gfs['wind_gust_gui'] if 'wind_gust_gui' in df_gfs.columns else df_gfs['wind_speed_gui'] * 1.2)

    # 5. Fill Water Level Gaps using ETSS Physics Baseline
    if 'etss_water_level' in df_gfs.columns:
        etss_wl = df_gfs['etss_water_level']
    elif 'etss_surge' in df_gfs.columns and 'etss_tide' in df_gfs.columns:
        etss_wl = df_gfs['etss_surge'] + df_gfs['etss_tide']
    else:
        etss_wl = pd.Series(index=df_hybrid.index, dtype=float)

    df_hybrid['water_level'] = df_hybrid['water_level'].fillna(etss_wl)

# 6. Apply Interpolation / Forward-Back Fill for Any Remaining Edge Gaps
df_hybrid = df_hybrid.ffill().bfill()

# 7. Format and Export Output
df_hybrid.reset_index(names='timestamp', inplace=True)
df_hybrid['timestamp'] = df_hybrid['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

df_hybrid.to_csv(OUTPUT_FILE, index=False, float_format='%.3f')
print(f"✅ SUCCESSFULLY EXPORTED CONTINUOUS IMPUTED DATASET TO: {OUTPUT_FILE}")
