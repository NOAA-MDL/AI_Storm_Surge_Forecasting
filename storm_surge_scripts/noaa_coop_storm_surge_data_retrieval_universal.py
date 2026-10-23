#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Developed with the assistance of the Gemini AI code assistant.
'''

import os, pandas as pd, requests, time
# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from datetime import datetime

# --- 1. Configuration ---
STN_WL = "8729108"  # Steel Pier (Primary Water Level)
# List of Meteo Stations in order of preference (Marina, Cape May, Sandy Hook)
MET_STATIONS = ["8729108"] 

OUTPUT_FILE = f"../storm_surge/obs_{STN_WL}_2021_2026.csv"
API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

years = ['2021', '2022', '2023', '2024', '2025', '2026']
months = [4, 10]

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

def get_noaa_data(stn, prod, start, end, datum=None):
    params = {
        "begin_date": start, "end_date": end, "station": stn,
        "product": prod, "units": "metric", "time_zone": "gmt",
        "format": "json", "application": "web_services"
    }
    if datum: params["datum"] = datum
    
    try:
        r = requests.get(API_URL, params=params, timeout=15)
        data = r.json()
        if "data" in data:
            df = pd.DataFrame(data["data"])
            df['t'] = pd.to_datetime(df['t'])
            df.set_index('t', inplace=True)
            for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            # Using '1h' to avoid FutureWarning
            return df.resample('1h').mean()
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- 2. Retrieval Loop ---
for yr in years:
    y = int(yr)
    for month in months:
        start_str = f"{y}{month:02d}01"
        # Correctly calculate end of month
        end_dt = datetime(y, month + 1, 1) if month < 12 else datetime(y + 1, 1, 1)
        end_str = end_dt.strftime("%Y%m%d")
        
        print(f"--- Processing {yr}-{month:02d} ---")
        
        # 1. Pull Water Level from Steel Pier (Try Verified, then Preliminary)
        df_wl = get_noaa_data(STN_WL, "water_level", start_str, end_str, datum="MLLW")
        if df_wl.empty:
            df_wl = get_noaa_data(STN_WL, "preliminary_water_level", start_str, end_str, datum="MLLW")

        if df_wl.empty:
            print(f"  [CRITICAL] No Water Level data for {STN_WL}. Skipping month.")
            continue

        # 2. Try Meteo Stations in order until we get Wind
        df_wind = pd.DataFrame()
        active_met_stn = None
        
        for met_stn in MET_STATIONS:
            print(f"  > Trying Meteo from {met_stn}...", end='', flush=True)
            df_wind = get_noaa_data(met_stn, "wind", start_str, end_str)
            if not df_wind.empty:
                active_met_stn = met_stn
                df_at = get_noaa_data(met_stn, "air_temperature", start_str, end_str)
                df_ap = get_noaa_data(met_stn, "air_pressure", start_str, end_str)
                print(" Found!")
                break
            else:
                print(" No Wind.")

        # 3. Standardize and Save
        try:
            if df_wind.empty:
                print(f"  [SKIP] Could not find Wind data at ANY station for {yr}-{month:02d}")
                continue

            wl = df_wl[['v']].rename(columns={'v': 'water_level'}) * 3.28084
            at = df_at[['v']].rename(columns={'v': 'air_temp'}) if not df_at.empty else pd.DataFrame()
            ap = df_ap[['v']].rename(columns={'v': 'air_pressure'}) if not df_ap.empty else pd.DataFrame()
            w = df_wind[['s', 'd', 'g']].rename(columns={'s':'wind_speed', 'd':'wind_direction', 'g':'wind_gust'})

            # Join
            df_final = pd.concat([wl, at, ap, w], axis=1).dropna(subset=['water_level', 'wind_speed'])

            if not df_final.empty:
                df_final.reset_index(names='timestamp', inplace=True)
                df_final['timestamp'] = df_final['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                cols = ['timestamp', 'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'water_level']
                write_header = not os.path.exists(OUTPUT_FILE)
                df_final[cols].to_csv(OUTPUT_FILE, index=False, mode='a', header=write_header, float_format='%.3f')
                print(f"  [OK] Saved {len(df_final)} rows using Meteo from {active_met_stn}.")
            
            time.sleep(1)

        except Exception as e:
            print(f"  [ERROR] {e}")

print(f"\n🏁 Finished. Dataset: {OUTPUT_FILE}")
