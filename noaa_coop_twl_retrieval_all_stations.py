#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Operational NOAA CO-OPS Real-Time Total Water Level (TWL) Retrieval Engine.
Iterates over all configured stations, dynamically fetches recent verified or preliminary
water levels, converts all datums (STND/NAVD) uniformly to MLLW, and saves to obs_twl_<id>.csv.
'''

import os, sys, time, json, argparse, warnings
import urllib.request, urllib.parse
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

BASE_PATH = '/contrib/Mamoudou.Ba/storm_surge'
CONFIG_FILE = os.path.join(BASE_PATH, 'stations_config.csv')
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = os.path.join(BASE_PATH, 'all_stations_config.csv')
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = 'stations_config.csv'

NOAA_API_BASE = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'

# Hardcoded fallback offsets (STND - MLLW) in case the datums endpoint is unreachable
KNOWN_STND_TO_MLLW = {
    '8725520': 4.95,   # Fort Myers, FL
    '8726520': 4.63,   # St. Petersburg, FL
    '8726607': 5.21,   # Old Port Tampa, FL
    '8726724': 4.78,   # Clearwater Beach, FL
    '8729108': 4.12,   # Panama City, FL
    '8670870': 0.00,   # Fort Pulaski, GA (Defaults MLLW)
    '8665530': 0.00,   # Charleston, SC (Defaults MLLW)
}

def parse_args():
    parser = argparse.ArgumentParser(description="NOAA CO-OPS TWL Retrieval Engine (Uniform MLLW)")
    parser.add_argument('--config', type=str, default=CONFIG_FILE, help="Path to stations configuration CSV")
    parser.add_argument('--station', type=str, default=None, help="Optional single station ID to process")
    parser.add_argument('--days_back', type=int, default=35, help="Number of past days to fetch (default: 35)")
    parser.add_argument('--window_days', type=int, default=15, help="Chunk window in days (max 16 for CO-OPS)")
    return parser.parse_args()

def get_mllw_conversion_offset(station_id, source_datum):
    """
    Returns the scalar offset in feet to add to source_datum to get MLLW.
    formula: val_mllw = val_source + offset
    """
    source_datum = source_datum.upper()
    if source_datum == 'MLLW':
        return 0.0

    url = (
        f"{NOAA_API_BASE}?station={station_id}&product=datums"
        f"&units=english&epoch=current&time_zone=gmt&application=NOAA_AI_Surge&format=json"
    )
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'NOAA_AI_Surge'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        datums = {item['name'].upper(): float(item['value']) for item in data.get('datums', [])}
        
        mllw_elev = datums.get('MLLW')
        if mllw_elev is not None:
            if source_datum == 'STND':
                # STND values are higher than MLLW by mllw_elev
                return -mllw_elev
            elif source_datum in ['NAVD', 'NAVD88']:
                navd_elev = datums.get('NAVD88', datums.get('NAVD'))
                if navd_elev is not None:
                    return navd_elev - mllw_elev
    except Exception as e:
        pass

    # Use station fallback table if dynamic query fails
    if source_datum == 'STND' and station_id in KNOWN_STND_TO_MLLW:
        return -KNOWN_STND_TO_MLLW[station_id]

    return 0.0

def fetch_twl_chunk(station_id, b_date, e_date):
    """
    Queries NOAA CO-OPS for a single chunk window.
    Tries hourly_height across datums, then falls back to preliminary 6-minute water_level.
    Returns (DataFrame, datum_used) or (None, None).
    """
    datums = ['MLLW', 'NAVD', 'STND']
    products = ['hourly_height', 'water_level']

    for prod in products:
        for datum in datums:
            params = {
                'begin_date': b_date,
                'end_date': e_date,
                'station': station_id,
                'product': prod,
                'datum': datum,
                'time_zone': 'gmt',
                'units': 'english',
                'application': 'NOAA_AI_Surge',
                'format': 'json'
            }
            url = f"{NOAA_API_BASE}?{urllib.parse.urlencode(params)}"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'NOAA_AI_Surge'})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    payload = json.loads(resp.read().decode())
                
                if 'data' in payload and len(payload['data']) > 0:
                    df = pd.DataFrame(payload['data'])
                    if 't' in df.columns and 'v' in df.columns:
                        if prod == 'water_level':
                            df['dt_temp'] = pd.to_datetime(df['t'])
                            df = df[df['dt_temp'].dt.minute == 0].drop(columns=['dt_temp'])
                        
                        df = df.rename(columns={'t': 'dt', 'v': 'water_level'})
                        df['water_level'] = pd.to_numeric(df['water_level'], errors='coerce')
                        df = df.dropna(subset=['water_level'])
                        if not df.empty:
                            return df[['dt', 'water_level']], datum
            except Exception:
                continue

    return None, None

def retrieve_station_twl(station_id, station_name, days_back=35, window_days=15):
    """
    Iterates backward from current time in discrete chunks, fetches TWL,
    converts to MLLW, deduplicates, and saves to obs_twl_<station_id>.csv.
    """
    end_dt = pd.Timestamp.utcnow().floor('h')
    start_dt = end_dt - pd.Timedelta(days=days_back)
    
    # Generate chunk windows
    window_edges = pd.date_range(start=start_dt, end=end_dt, freq=f'{window_days}D')
    if window_edges[-1] < end_dt:
        window_edges = window_edges.append(pd.DatetimeIndex([end_dt]))
    
    all_chunks = []
    datums_used = set()

    for i in range(len(window_edges) - 1):
        b_str = window_edges[i].strftime('%Y%m%d %H:%M')
        e_str = window_edges[i+1].strftime('%Y%m%d %H:%M')
        
        df_chunk, used_datum = fetch_twl_chunk(station_id, b_str, e_str)
        if df_chunk is not None and not df_chunk.empty:
            # Apply datum conversion to MLLW
            offset = get_mllw_conversion_offset(station_id, used_datum)
            if offset != 0.0:
                df_chunk['water_level'] += offset
            
            datums_used.add(f"{used_datum}->MLLW({offset:+.2f}ft)" if offset != 0.0 else used_datum)
            all_chunks.append(df_chunk)
        time.sleep(0.3)  # Respect API rate limits

    if not all_chunks:
        print(f"   [FAIL] Could not retrieve any valid water level data for {station_name} ({station_id})")
        return False

    final_df = pd.concat(all_chunks, ignore_index=True)
    final_df['dt'] = pd.to_datetime(final_df['dt'])
    final_df = final_df.drop_duplicates(subset=['dt']).sort_values('dt').reset_index(drop=True)
    
    # Enforce standard string format for timestamps
    final_df['dt'] = final_df['dt'].dt.strftime('%Y-%m-%d %H:%M:%S')

    out_obs = os.path.join(BASE_PATH, f"obs_twl_{station_id}.csv")
    out_wl = os.path.join(BASE_PATH, f"wl_{station_id}.csv")

    final_df.to_csv(out_obs, index=False)
    final_df.to_csv(out_wl, index=False)

    min_t, max_t = final_df['dt'].iloc[0], final_df['dt'].iloc[-1]
    datum_info = ", ".join(datums_used)
    print(f"   [DONE] {station_name} ({station_id}): {len(final_df)} records | [{min_t} -> {max_t}] | Datums: [{datum_info}]")
    return True

def main():
    args = parse_args()
    print("=" * 80)
    print("NOAA CO-OPS OPERATIONAL WATER LEVEL RETRIEVAL (UNIFORM MLLW ENGINE)")
    print(f"Config File : {args.config}")
    print(f"Lookback    : Past {args.days_back} days")
    print(f"Target File : obs_twl_<id>.csv & wl_<id>.csv (strictly MLLW)")
    print("=" * 80)

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    df_cfg = pd.read_csv(args.config, dtype={'station_id': str})
    station_list = []
    for _, r in df_cfg.iterrows():
        sid = str(r['station_id']).strip()
        sname = str(r['station_name']).strip().replace('"', '').replace(',', '').replace('.', '').replace(' ', '_') if 'station_name' in r else f"Station_{sid}"
        station_list.append((sid, sname))

    if args.station:
        station_list = [(sid, sname) for sid, sname in station_list if sid == args.station.strip()]

    print(f"Found {len(station_list)} station(s) to process.\n")
    for idx, (sid, sname) in enumerate(station_list, 1):
        print(f"[{idx}/{len(station_list)}] Fetching data for {sname} ({sid})...")
        retrieve_station_twl(sid, sname, days_back=args.days_back, window_days=args.window_days)

    print("\nAll station retrievals completed successfully.")

if __name__ == '__main__':
    main()
