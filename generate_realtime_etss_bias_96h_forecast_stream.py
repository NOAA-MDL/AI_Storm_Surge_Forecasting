#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Operational Real-Time ETSS Bias / Surge Residual 96-Hour Forecast Stream Generator.
Batch Multi-Station Engine utilizing trained 4-Quadrant Spatial GRU Models ([None, 36, 26]).

Feature Matrix Schema (Exact 26-column alignment):
  1-5.   Meteorology Guidance (air_temp, air_pressure, wind_speed, wind_gust, etss_surge_guidance)
  6-25.  4-Quadrant Spatial GFS Fields (c1..c4: wind_u, wind_v, pressure, temp, wind_speed)
  26.    Target Pivot Track (surge_residual = water_level - etss_tide)
'''

import os, sys, gc, glob, argparse, warnings
warnings.filterwarnings('ignore')

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import joblib

# --- DIRECTORY CONFIGURATIONS ---
MODELS_PATH = '../storm_surge/'
STREAMS_DIR = '/contrib/Mamoudou.Ba/storm_surge/realtime_forecast_streams'
os.makedirs(STREAMS_DIR, exist_ok=True)

CONFIG_FILE = os.path.join(MODELS_PATH, 'all_stations_config.csv')
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = 'all_stations_config.csv'
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = os.path.join(MODELS_PATH, 'stations_config.csv')
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = 'stations_config.csv'

TARGET_SCALER = "ROBUST"
N_FUTURE_TRAINED = 12
LOOKBACKS = [24, 48]
N_OFFSHORE_CELLS = 4

# Construct exact 20 4-quadrant feature column names
GFS_4QUAD_COLS = []
for c in range(1, N_OFFSHORE_CELLS + 1):
    GFS_4QUAD_COLS.extend([
        f'c{c}_wind_u_gui', f'c{c}_wind_v_gui',
        f'c{c}_air_pressure_gui', f'c{c}_air_temp_gui',
        f'c{c}_wind_speed_gui'
    ])

OBS_MET_BASE = ['air_temp', 'air_pressure', 'wind_speed', 'wind_gust', 'etss_surge_guidance']
GFS_MET_BASE = ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_gust_gui', 'etss_surge_guidance']

OBS_WEATHER_COLS = OBS_MET_BASE + GFS_4QUAD_COLS
GFS_GUIDANCE_COLS = GFS_MET_BASE + GFS_4QUAD_COLS
TARGET_COL = 'surge_residual'

def parse_args():
    parser = argparse.ArgumentParser(description="Operational 96h Real-Time ETSS Bias Stream Generator (4-Quadrant Spatial 26-Feature Architecture)")
    parser.add_argument('--config', type=str, default=CONFIG_FILE, help="Path to stations configuration CSV file")
    parser.add_argument('--station', type=str, default=None, help="Optional single station ID to process")
    parser.add_argument('--mode', type=str, default='BIAS', choices=['BIAS', 'RESID'], help="Operational model mode")
    parser.add_argument('--horizon', type=int, default=96, help="Prospective rollout horizon in hours (default 96)")
    return parser.parse_args()

def load_and_parse(file_path):
    """Loads CSV, strips sentinel missing values, and standardizes datetime stamps."""
    df = pd.read_csv(file_path)
    sentinels = [9999.0, 9999.00, 99999.0, 99999.000, -9999.0, -99999.0]
    df = df.replace(sentinels, np.nan)

    time_candidates = ['timestamp_gui', 'timestamp', 'dt', 'datetime', 'date_time', 'valid_time', 'time', 'date']
    t_col = None
    for c in time_candidates:
        if c in df.columns:
            t_col = c
            break
    if not t_col:
        matched = [c for c in df.columns if any(k in c.lower() for k in ['time', 'date', 'dt'])]
        if matched:
            t_col = matched[0]

    if not t_col:
        raise KeyError(f"No datetime column recognized in {file_path}. Columns: {list(df.columns)}")

    df[t_col] = df[t_col].astype(str).str.replace(r'(\d{2}) (\d{2}) (\d{2})$', r'\1:\2:\3', regex=True)
    df['dt'] = pd.to_datetime(df[t_col], errors='coerce', format='mixed')
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_localize(None)
    df['dt'] = df['dt'].dt.round('h')
    return df.dropna(subset=['dt'])

def resolve_station_files(station_id):
    """Resolves GFS spatial test file and Observation ground truth file."""
    gfs_path = os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{station_id}_2026.csv')
    if not os.path.exists(gfs_path):
        if os.path.exists(f'test_gfs_etss_spatial_data_{station_id}_2026.csv'):
            gfs_path = f'test_gfs_etss_spatial_data_{station_id}_2026.csv'
        else:
            gfs_path = None

    obs_candidates = [
        os.path.join(MODELS_PATH, f'obs_training_{station_id}.csv'),
        os.path.join(MODELS_PATH, f'obs_training_{station_id}_2026.csv'),
        os.path.join(MODELS_PATH, f'realtime_obs_{station_id}_2026.csv'),
        f'obs_training_{station_id}.csv'
    ]
    obs_path = next((f for f in obs_candidates if os.path.exists(f)), None)

    return gfs_path, obs_path

def load_station_models(station_id, station_name, mode):
    """Loads trained 4-quadrant GRU models and corresponding 26-feature scalers."""
    cached_models = {}
    cached_scalers = {}

    name_clean = station_name.strip().replace('"', '').replace(',', '').replace('.', '').replace(' ', '_')
    name_no_state = name_clean.split('_')[0]
    prefixes = [name_clean, station_id, name_no_state]

    for n_past in LOOKBACKS:
        key = f"PIVOT_{n_past}_{TARGET_SCALER}"
        m_json, m_h5, sf_path, st_path = None, None, None, None

        for prefix in prefixes:
            base_model = f"{prefix}_{N_FUTURE_TRAINED}_ETSS_{mode}_PIVOT_ROBUST_{n_past}past_CHRONO"
            j_cand = os.path.join(MODELS_PATH, f"{base_model}.json")
            h_cand = os.path.join(MODELS_PATH, f"{base_model}.weights.h5")
            if not os.path.exists(h_cand):
                h_cand = os.path.join(MODELS_PATH, f"{base_model}.h5")

            sf_cand = os.path.join(MODELS_PATH, f"{prefix}_scaler_etss_bias_pivot_robust_{n_past}h_CHRONO.joblib")
            st_cand = os.path.join(MODELS_PATH, f"{prefix}_scaler_etss_bias_pivot_target_robust_{n_past}h_CHRONO.joblib")

            if os.path.exists(j_cand) and os.path.exists(h_cand) and os.path.exists(sf_cand) and os.path.exists(st_cand):
                m_json, m_h5, sf_path, st_path = j_cand, h_cand, sf_cand, st_cand
                break

        # Fallback globs
        if not (m_json and os.path.exists(m_h5) and sf_path and st_path):
            for ident in [name_clean, station_id, name_no_state]:
                if not m_json:
                    j_matches = glob.glob(os.path.join(MODELS_PATH, f"*{ident}*{mode}*{n_past}past*.json"))
                    if j_matches:
                        m_json = j_matches[0]
                        stem = m_json[:-5]
                        m_h5 = f"{stem}.weights.h5" if os.path.exists(f"{stem}.weights.h5") else f"{stem}.h5"

                if not sf_path:
                    sf_m = glob.glob(os.path.join(MODELS_PATH, f"*{ident}*scaler*etss*{mode.lower()}*robust_{n_past}h*.joblib"))
                    sf_m = [f for f in sf_m if 'target' not in os.path.basename(f)]
                    if sf_m: sf_path = sf_m[0]

                if not st_path:
                    st_m = glob.glob(os.path.join(MODELS_PATH, f"*{ident}*scaler*etss*{mode.lower()}*target*robust_{n_past}h*.joblib"))
                    if st_m: st_path = st_m[0]

                if m_json and os.path.exists(m_h5) and sf_path and st_path:
                    break

        if m_json and os.path.exists(m_h5) and sf_path and st_path:
            try:
                with open(m_json, 'r') as f:
                    model = model_from_json(f.read())
                model.load_weights(m_h5)
                cached_models[key] = model
                cached_scalers[f"{key}_f"] = joblib.load(sf_path)
                cached_scalers[f"{key}_t"] = joblib.load(st_path)
            except Exception as e:
                print(f"      [!] Failed to load model/scaler for {key}: {e}")
                return None, None
        else:
            missing = []
            if not m_json: missing.append(f"JSON model ({n_past}past)")
            elif not os.path.exists(m_h5): missing.append(f"Weights ({m_h5})")
            if not sf_path: missing.append(f"Feature scaler ({n_past}h)")
            if not st_path: missing.append(f"Target scaler ({n_past}h)")
            print(f"      [!] Missing components for {key}: {', '.join(missing)}")
            return None, None

    return cached_models, cached_scalers

def predict_single_chunk(cached_models, cached_scalers, data_full, chunk_origin, op_gui, anchor_target):
    """
    Performs 12-hour spatial inference using exact 26-feature sequences:
      [hybrid_met (25 features), hybrid_surge (1 feature)] -> 26 features.
    """
    target_len = len(op_gui)
    local_horizons = {}

    for n_past in LOOKBACKS:
        key = f"PIVOT_{n_past}_{TARGET_SCALER}"
        if key not in cached_models:
            local_horizons[n_past] = np.full(target_len, anchor_target)
            continue

        model = cached_models[key]
        sc_f = cached_scalers[f"{key}_f"]
        sc_t = cached_scalers[f"{key}_t"]

        # 1. Past Weather Track (25 columns): slice last n_past hours up to chunk_origin
        past_hist_df = data_full.loc[:chunk_origin].tail(n_past)
        if len(past_hist_df) < n_past:
            pad_n = n_past - len(past_hist_df)
            pad_block = pd.concat([past_hist_df.iloc[[0]]] * pad_n)
            past_hist_df = pd.concat([pad_block, past_hist_df])

        past_met = past_hist_df[OBS_WEATHER_COLS].values  # Shape: (n_past, 25)

        # 2. Future Weather Guidance Track (25 columns): from op_gui up to N_FUTURE_TRAINED
        fut_guidance_df = op_gui[GFS_GUIDANCE_COLS].copy()
        if len(fut_guidance_df) < N_FUTURE_TRAINED:
            pad_n = N_FUTURE_TRAINED - len(fut_guidance_df)
            pad_block = pd.concat([fut_guidance_df.iloc[[-1]]] * pad_n)
            fut_guidance_df = pd.concat([fut_guidance_df, pad_block])
        elif len(fut_guidance_df) > N_FUTURE_TRAINED:
            fut_guidance_df = fut_guidance_df.iloc[:N_FUTURE_TRAINED]

        future_met = fut_guidance_df.values  # Shape: (N_FUTURE_TRAINED, 25)
        hybrid_met = np.vstack([past_met, future_met])  # Shape: (n_past + 12, 25)

        # 3. Water Track (1 column: surge_residual with pivot anchor)
        past_surge = past_hist_df[[TARGET_COL]].values  # Shape: (n_past, 1)
        future_surge_lkv = np.full((N_FUTURE_TRAINED, 1), anchor_target)
        hybrid_surge = np.vstack([past_surge, future_surge_lkv])  # Shape: (n_past + 12, 1)

        # 4. Form Combined 26-Feature Array
        combined = np.ascontiguousarray(np.hstack([hybrid_met, hybrid_surge]))  # Shape: (36, 26)

        # Scale features
        if hasattr(sc_f, 'transform'):
            scaled_inp = sc_f.transform(combined)
        elif hasattr(sc_f, 'center_') and hasattr(sc_f, 'scale_'):
            scaled_inp = (combined - sc_f.center_) / sc_f.scale_
        else:
            scaled_inp = combined

        # Predict with GRU
        inp_tensor = tf.convert_to_tensor(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, 26), dtype=tf.float32)
        raw_pred = model(inp_tensor, training=False).numpy().flatten()

        # Invert target scaling
        aligned_pred = np.zeros(N_FUTURE_TRAINED)
        fill_len = min(len(raw_pred), N_FUTURE_TRAINED)
        aligned_pred[:fill_len] = raw_pred[:fill_len]

        if hasattr(sc_t, 'inverse_transform'):
            pred_delta = sc_t.inverse_transform(aligned_pred.reshape(-1, 1)).flatten()
        elif hasattr(sc_t, 'center_') and hasattr(sc_t, 'scale_'):
            pred_delta = (aligned_pred * sc_t.scale_[0]) + sc_t.center_[0]
        else:
            pred_delta = aligned_pred

        # Re-anchor predicted delta trajectory to current pivot (C0 Continuity)
        local_horizons[n_past] = (pred_delta[:target_len] + anchor_target)

    # Blend 24h & 48h horizons with sigmoid transition
    timeline = np.arange(target_len)
    weight_24h = 1.0 / (1.0 + np.exp((timeline - 36) / 8))
    h24 = local_horizons.get(24, np.full(target_len, anchor_target))
    h48 = local_horizons.get(48, np.full(target_len, anchor_target))
    return ((h24[:target_len] * weight_24h) + (h48[:target_len] * (1.0 - weight_24h)))[:target_len]

def process_station(station_id, station_name, mode, horizon):
    """Executes 96h stream extraction for an individual station."""
    gfs_path, obs_path = resolve_station_files(station_id)
    if not gfs_path:
        print(f"   [SKIP] Missing GFS spatial test file: test_gfs_etss_spatial_data_{station_id}_2026.csv")
        return False
    if not obs_path:
        print(f"   [SKIP] Missing Obs training file: obs_training_{station_id}.csv")
        return False

    cached_models, cached_scalers = load_station_models(station_id, station_name, mode)
    if not cached_models:
        print(f"   [SKIP] Missing trained model or scaler assets in {MODELS_PATH}")
        return False

    try:
        df_g = load_and_parse(gfs_path)
        df_o = load_and_parse(obs_path)
    except Exception as e:
        print(f"   [SKIP] Error parsing data files: {e}")
        return False

    # Derive 4-quadrant wind speeds if U and V exist
    for c in range(1, N_OFFSHORE_CELLS + 1):
        u_col, v_col, ws_col = f'c{c}_wind_u_gui', f'c{c}_wind_v_gui', f'c{c}_wind_speed_gui'
        if u_col in df_g.columns and v_col in df_g.columns:
            df_g[ws_col] = np.sqrt(df_g[u_col]**2 + df_g[v_col]**2)
        elif ws_col not in df_g.columns:
            df_g[ws_col] = 0.0

    # Ensure all required GFS 4-quadrant guidance columns exist
    for col in GFS_4QUAD_COLS:
        if col not in df_g.columns:
            df_g[col] = 0.0

    df_o = df_o.drop_duplicates(subset=['dt']).set_index('dt').sort_index()
    df_g['cycle_int'] = pd.to_numeric(df_g['cycle'] if 'cycle' in df_g.columns else 0, errors='coerce').fillna(0).astype(int)

    for col in ['etss_surge', 'etss_tide', 'etss_bias', 'etss_water_level']:
        if col in df_g.columns:
            df_g[col] = pd.to_numeric(df_g[col], errors='coerce').interpolate(method='linear').ffill().bfill()

    # Determine latest continuous prospective forecast block
    df_g_work = df_g.copy().sort_values(['cycle_int', 'dt'])
    df_g_work['run_block'] = ((df_g_work['dt'].diff() != pd.Timedelta(hours=1)) |
                              (df_g_work['cycle_int'].diff() != 0)).cumsum()

    blocks = [b for _, b in df_g_work.groupby('run_block') if len(b) >= 24]
    if not blocks:
        print("   [SKIP] No contiguous forecast guidance block >= 24h found.")
        return False

    selected_block = blocks[-1].sort_values('dt')
    t0 = selected_block.iloc[0]['dt']
    forecast_window = selected_block.head(horizon).set_index('dt')
    rollout_len = len(forecast_window)

    if rollout_len < 12:
        print(f"   [SKIP] Forecast window too short ({rollout_len}h).")
        return False

    # Harmonize spatial historical guidance and observations
    df_g_unique = df_g.sort_values(['dt', 'cycle_int']).drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()
    data_full = pd.DataFrame(index=df_g_unique.index.union(df_o.index)).sort_index()

    # Copy all GFS columns into data_full
    for col in df_g_unique.columns:
        data_full[col] = df_g_unique[col]

    # Harmonize observation water level
    data_full['water_level'] = df_o['water_level'] if 'water_level' in df_o.columns else np.nan
    data_full['water_level'] = data_full['water_level'].interpolate(method='linear').ffill().bfill()

    # ETSS guidance fields
    t_cols = [c for c in df_g_unique.columns if 'tide' in c.lower()]
    data_full['etss_tide'] = df_g_unique[t_cols[0]] if t_cols else 0.0

    s_cols = [c for c in ['etss_surge', 'SURGE'] if c in df_g_unique.columns]
    data_full['etss_surge'] = df_g_unique[s_cols[0]] if s_cols else 0.0
    data_full['etss_surge_guidance'] = data_full['etss_surge']

    # Station observation meteorology (fallback to GFS guidance if missing from obs)
    for met_base in ['air_temp', 'air_pressure', 'wind_speed', 'wind_gust']:
        gui_col = f"{met_base}_gui"
        if met_base in df_o.columns and df_o[met_base].notna().sum() > 10:
            data_full[met_base] = df_o[met_base].interpolate(method='linear').ffill().bfill()
        elif gui_col in data_full.columns:
            data_full[met_base] = data_full[gui_col]
        else:
            data_full[met_base] = 0.0

        if gui_col not in data_full.columns:
            data_full[gui_col] = data_full[met_base]

    # Target metric: surge_residual = water_level - etss_tide
    data_full[TARGET_COL] = data_full['water_level'] - data_full['etss_tide']

    # Ensure all required numeric model columns are strictly float and interpolated
    num_cols = list(set(OBS_WEATHER_COLS + GFS_GUIDANCE_COLS + [TARGET_COL, 'etss_tide', 'etss_surge', 'water_level']))
    for col in num_cols:
        if col in data_full.columns:
            data_full[col] = pd.to_numeric(data_full[col], errors='coerce').interpolate(method='linear').ffill().bfill()
        else:
            data_full[col] = 0.0

    # Ensure forecast window has all required guidance columns
    forecast_window['etss_surge_guidance'] = forecast_window['etss_surge']
    for col in GFS_GUIDANCE_COLS:
        if col not in forecast_window.columns:
            forecast_window[col] = data_full[col].reindex(forecast_window.index).ffill().bfill()

    # Prospective Rollout Loop (12-hour chunk rollout)
    predicted_surge_residual = []
    current_anchor_time = t0

    if current_anchor_time in data_full.index:
        current_anchor_val = data_full.loc[current_anchor_time, TARGET_COL]
    else:
        nearest_idx = data_full.index.get_indexer([current_anchor_time], method='nearest')[0]
        current_anchor_val = data_full.iloc[nearest_idx][TARGET_COL]
    if isinstance(current_anchor_val, pd.Series):
        current_anchor_val = current_anchor_val.iloc[0]

    num_chunks = int(np.ceil(rollout_len / 12.0))
    for c_idx in range(num_chunks):
        c_start = c_idx * 12
        c_end = min(c_start + 12, rollout_len)
        chunk_gui = forecast_window.iloc[c_start:c_end]
        if chunk_gui.empty:
            break

        c_pred = predict_single_chunk(cached_models, cached_scalers, data_full, current_anchor_time, chunk_gui, current_anchor_val)
        predicted_surge_residual.extend(c_pred)

        # Update pivot anchor for next chunk
        current_anchor_val = c_pred[-1]
        current_anchor_time = chunk_gui.index[-1]

    predicted_surge_residual = np.array(predicted_surge_residual)[:rollout_len]
    stream_idx = forecast_window.index[:rollout_len]

    tide_vals = forecast_window['etss_tide'].values[:rollout_len] if 'etss_tide' in forecast_window.columns else np.zeros(rollout_len)
    etss_surge_vals = forecast_window['etss_surge'].values[:rollout_len] if 'etss_surge' in forecast_window.columns else np.zeros(rollout_len)
    etss_wl_vals = etss_surge_vals + tide_vals

    # Convert predicted surge residual to corrected surge, bias, and total water level
    corrected_surge = predicted_surge_residual
    predicted_bias = corrected_surge - etss_surge_vals
    corrected_wl = corrected_surge + tide_vals

    export_df = pd.DataFrame({
        'dt': stream_idx,
        'station_id': station_id,
        'station_name': station_name,
        'forecast_origin': t0.strftime('%Y-%m-%d %H:%M:%S'),
        'lead_hour': np.arange(1, rollout_len + 1),
        'etss_tide_ft': tide_vals,
        'etss_surge_ft': etss_surge_vals,
        'etss_water_level_raw_ft': etss_wl_vals,
        'predicted_etss_bias_ft': predicted_bias,
        'etss_corrected_surge_ft': corrected_surge,
        'etss_corrected_twl_ft': corrected_wl,
        'pivot_surge_forecast_ft': corrected_surge,
        'pivot_total_water_level_ft': corrected_wl
    })

    # Add meteorological variables to stream output
    for met_base in ['wind_speed', 'wind_direction', 'wind_gust', 'air_temp', 'air_pressure']:
        gui_col = f"{met_base}_gui"
        val_col = gui_col if gui_col in forecast_window.columns else (met_base if met_base in forecast_window.columns else None)
        if val_col:
            unit_col = f"{met_base}_kts" if 'wind' in met_base and 'direction' not in met_base else (
                f"{met_base}_deg" if 'direction' in met_base else (
                    f"{met_base}_C" if 'temp' in met_base else f"{met_base}_hPa"
                )
            )
            export_df[unit_col] = forecast_window[val_col].values[:rollout_len]
            export_df[met_base] = forecast_window[val_col].values[:rollout_len]

    t0_tag = t0.strftime('%Y%m%d_%H%M')
    csv_fname = f"realtime_etss_bias_96h_forecast_{station_id}_{t0_tag}.csv"
    out_csv_path = os.path.join(STREAMS_DIR, csv_fname)
    export_df.to_csv(out_csv_path, index=False)

    print(f"   [DONE] Generated {rollout_len}h stream (t0: {t0.strftime('%Y-%m-%d %H:%MZ')}) -> {csv_fname}")

    del cached_models, cached_scalers, data_full, df_g, df_o, df_g_work, export_df
    K.clear_session()
    gc.collect()
    return True

def main():
    args = parse_args()
    print("=" * 80)
    print("OPERATIONAL 96-HOUR ETSS BIAS FORECAST STREAM GENERATOR (26-FEATURE 4-QUAD)")
    print("=" * 80)
    print(f"Target Configuration File : {args.config}")
    print(f"Forecast Rollout Horizon  : +{args.horizon} Hours")
    print(f"Target Model Mode         : {args.mode}")
    print(f"Output Destination        : {STREAMS_DIR}")
    print("=" * 80)

    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)

    df_cfg = pd.read_csv(args.config, dtype={'station_id': str})
    if 'station_id' not in df_cfg.columns:
        print(f"Error: 'station_id' column not found in {args.config}.")
        sys.exit(1)

    station_list = []
    for _, r in df_cfg.iterrows():
        sid = str(r['station_id']).strip()
        sname = str(r['station_name']).strip().replace('"', '').replace(',', '').replace('.', '').replace(' ', '_') if 'station_name' in r else f"Station_{sid}"
        station_list.append((sid, sname))

    if args.station:
        station_list = [(sid, sname) for sid, sname in station_list if sid == args.station.strip()]
        if not station_list:
            print(f"Warning: Station {args.station} not found in configuration file. Adding as fallback.")
            station_list = [(args.station.strip(), f"Station_{args.station.strip()}")]

    print(f"Found {len(station_list)} candidate station(s) in configuration.")
    print("-" * 80)

    success_count = 0
    skipped_count = 0

    for idx, (sid, sname) in enumerate(station_list, 1):
        print(f"[{idx}/{len(station_list)}] Processing Station {sname} ({sid})...")
        ok = process_station(sid, sname, args.mode, args.horizon)
        if ok:
            success_count += 1
        else:
            skipped_count += 1

    print("=" * 80)
    print(f"BATCH RUN COMPLETE: {success_count} Generated | {skipped_count} Skipped")
    print(f"Output Streams Directory: {STREAMS_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
