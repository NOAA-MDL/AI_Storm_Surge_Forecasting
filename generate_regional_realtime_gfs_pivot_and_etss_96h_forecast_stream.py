#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Real-Time Operational GFS Weather + ETSS Forecast Stream Generator.
Multi-Station Regional Edition:
Ingests 4-Quadrant GFS meteorology + ETSS surge guidance, rolls the trained
multi-station/regional CHRONO AI models recursively out to 96-102 hours, computes the
pure pivot surge forecast, ETSS bias, and exports seamless, anchored CSV streams.
'''

import os
import sys
import gc
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

# Force Keras legacy deserializer and headless CPU execution
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import joblib

# --- DIRECTORY AND CONFIGURATION SETTINGS ---
MODELS_PATH = '/contrib/Mamoudou.Ba/storm_surge/'
CONFIG_FILE = os.path.join(MODELS_PATH, 'all_stations_config.csv')
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = os.path.join(MODELS_PATH, 'stations_config.csv')
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = 'all_stations_config.csv'

OUTPUT_DIR = '/contrib/Mamoudou.Ba/storm_surge/realtime_forecast_streams'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FUTURE_TRAINED = 12       # Block size model was trained on (12h)
DEFAULT_HORIZON = 96        # Default operational rollout (96 hours)
HINDCAST_LOOKBACK = 48      # Hindcast lookback hours (48h)
N_OFFSHORE_CELLS = 4        # 4 Quadrants (c1, c2, c3, c4)
LOOKBACKS = [24, 48]
MAX_ALLOWED_GAP_HOURS = 2.0 # Max raw guidance gap before discarding cycle

# Regional tag mapping: associates stations with their regional multi-station model
REGIONAL_CLUSTERS = {
    '8725520': 'FL_WEST_COAST',  # Fort Myers, FL
    '8726520': 'FL_WEST_COAST',  # St. Petersburg, FL
    '8726607': 'FL_WEST_COAST',  # Old Port Tampa, FL
    '8726724': 'FL_WEST_COAST',  # Clearwater Beach, FL
}

def parse_args():
    parser = argparse.ArgumentParser(description="Operational Regional Stream Generator")
    parser.add_argument('--config', type=str, default=CONFIG_FILE, help="Stations config CSV")
    parser.add_argument('--station', type=str, default=None, help="Optional single station ID (e.g. 8725520)")
    parser.add_argument('--t0', type=str, default=None, help="Explicit forecast origin (YYYY-MM-DD HH:MM or YYYYMMDD_HHMM)")
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON, help="Horizon hours (e.g. 96 or 102)")
    parser.add_argument('--lookback', type=int, default=HINDCAST_LOOKBACK, help="Hindcast lookback hours (48)")
    return parser.parse_args()

def load_and_parse(file_path):
    df = pd.read_csv(file_path)
    sentinels = [9999.0, 9999.00, 99999.0, 99999.000, -9999.0, -99999.0]
    df = df.replace(sentinels, np.nan)

    time_candidates = ['timestamp_gui', 'timestamp', 'dt', 'datetime', 'date_time', 'valid_time', 'time', 'date']
    t_col = next((c for c in time_candidates if c in df.columns), None)
    if not t_col:
        matched = [c for c in df.columns if any(k in c.lower() for k in ['time', 'date', 'dt'])]
        if matched:
            t_col = matched[0]
    if not t_col:
        raise KeyError(f"No datetime column in {file_path}")

    df[t_col] = df[t_col].astype(str).str.replace(r'(\d{2}) (\d{2}) (\d{2})$', r'\1:\2:\3', regex=True)
    df['dt'] = pd.to_datetime(df[t_col], errors='coerce', format='mixed')
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_localize(None)
    df['dt'] = df['dt'].dt.round('h')
    return df.dropna(subset=['dt'])

def clean_numeric_df(df, cols, fill_val=0.0):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_val
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].interpolate(method='linear').ffill().bfill().fillna(fill_val)
    return df

def compute_wind_components(speed, direction_deg):
    rad = np.radians(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v

def resolve_station_files(station_id):
    gfs_candidates = [
        os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{station_id}_2026.csv'),
        os.path.join(MODELS_PATH, f'test_gfs_etss_data_{station_id}_2026.csv'),
        os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{station_id}.csv'),
    ]
    gfs_candidates.extend(glob.glob(os.path.join(MODELS_PATH, f'*{station_id}*spatial*.csv')))
    df_g_path = next((f for f in gfs_candidates if os.path.exists(f)), None)

    obs_candidates = [
        os.path.join(MODELS_PATH, f'obs_twl_{station_id}.csv'),
        os.path.join(MODELS_PATH, f'obs_training_{station_id}.csv'),
        os.path.join(MODELS_PATH, f'obs_training_{station_id}_2026.csv'),
        os.path.join(MODELS_PATH, f'obs_{station_id}_wl_only.csv'),
        os.path.join(MODELS_PATH, f'water_level_{station_id}_2021_2026.csv')
    ]
    df_o_path = next((f for f in obs_candidates if os.path.exists(f)), None)
    return df_g_path, df_o_path

def load_regional_or_station_models(station_id, station_name):
    cached_models, cached_scalers = {}, {}
    region_tag = REGIONAL_CLUSTERS.get(station_id, None)

    name_clean = station_name
    name_no_state = name_clean.split('_')[0]

    prefixes = []
    if region_tag:
        prefixes.append(region_tag)
    prefixes.extend([name_clean, station_id, name_no_state])

    for n_past in LOOKBACKS:
        key = f"{n_past}_CHRONO"
        m_json_path, m_h5_path, sf_path, st_path = None, None, None, None

        for prefix in prefixes:
            m_cand = f"{prefix}_{N_FUTURE_TRAINED}_GFS_WEATHER_4QUADRANT_GFS_WEATHER_OBS_ETSS_WL_PIVOT_ROBUST_{n_past}past_CHRONO"
            j_cand = os.path.join(MODELS_PATH, f"{m_cand}.json")
            h_cand = os.path.join(MODELS_PATH, f"{m_cand}.weights.h5")
            if not os.path.exists(h_cand):
                h_cand = os.path.join(MODELS_PATH, f"{m_cand}.h5")

            sf_cand = os.path.join(MODELS_PATH, f"{prefix}_scaler_gfs_weather_4quadrant_gfs_weather_obs_etss_wl_pivot_robust_{n_past}h_CHRONO.joblib")
            st_cand = os.path.join(MODELS_PATH, f"{prefix}_scaler_gfs_weather_4quadrant_gfs_weather_obs_etss_wl_pivot_target_robust_{n_past}h_CHRONO.joblib")

            if os.path.exists(j_cand) and os.path.exists(h_cand) and os.path.exists(sf_cand) and os.path.exists(st_cand):
                m_json_path, m_h5_path, sf_path, st_path = j_cand, h_cand, sf_cand, st_cand
                if prefix == region_tag:
                    print(f"   [Bound] Lookback {n_past}h -> Multi-Station Regional Model: {region_tag}")
                else:
                    print(f"   [Bound] Lookback {n_past}h -> Dedicated Single-Station Model: {prefix}")
                break

        if not (m_json_path and os.path.exists(m_h5_path) and sf_path and st_path):
            return None, None

        try:
            with open(m_json_path, 'r') as f:
                model = model_from_json(f.read())
            model.load_weights(m_h5_path)
            cached_models[key] = model
            cached_scalers[f"{key}_f"] = joblib.load(sf_path)
            cached_scalers[f"{key}_t"] = joblib.load(st_path)
        except Exception as e:
            print(f"❌ Failed loading {key} for {station_name}: {e}")
            return None, None

    return cached_models, cached_scalers

def main():
    args = parse_args()
    print("=" * 80)
    print("OPERATIONAL MULTI-STATION REGIONAL STREAM GENERATOR")
    print(f"Config File    : {args.config}")
    print(f"Horizon/Look   : +{args.horizon}h / -{args.lookback}h")
    if args.t0:
        print(f"Target Origin  : {args.t0}")
    print("=" * 80)

    if not os.path.exists(args.config):
        print(f"❌ Error: Station configuration matrix missing at: {args.config}")
        sys.exit(1)

    df_config = pd.read_csv(args.config, dtype={'station_id': str})
    df_config['station_id'] = df_config['station_id'].str.strip()
    df_config['station_name'] = df_config['station_name'].str.strip().str.replace('"', '').str.replace(',', '').str.replace(' ', '_')

    station_list = []
    for _, r in df_config.iterrows():
        sid = str(r['station_id']).strip()
        sname = str(r['station_name']).strip()
        station_list.append((sid, sname))

    if args.station:
        station_list = [(sid, sname) for sid, sname in station_list if sid == args.station.strip()]

    for idx_stn, (STATION_ID, STATION_NAME) in enumerate(station_list, 1):
        df_g_path, df_o_path = resolve_station_files(STATION_ID)
        if not (df_g_path and df_o_path):
            continue

        print(f"\n[{idx_stn}/{len(station_list)}] Processing Station {STATION_NAME} ({STATION_ID})...")

        try:
            df_g = load_and_parse(df_g_path)
            df_o = load_and_parse(df_o_path)

            df_o_unique = df_o.sort_values('dt').drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()

            if 'cycle' in df_g.columns:
                df_g['cycle_int'] = pd.to_numeric(df_g['cycle'], errors='coerce').fillna(0).astype(int)
                df_g_unique = df_g.sort_values('cycle_int').drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()
            else:
                df_g_unique = df_g.sort_values('dt').drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()

            data_full = pd.DataFrame(index=df_g_unique.index.union(df_o_unique.index)).sort_index()

            # 1. Harmonic Tide Baseline
            if 'etss_tide' in df_g_unique.columns:
                data_full['etss_tide'] = df_g_unique['etss_tide'].reindex(data_full.index)
            else:
                t_col = [c for c in df_g_unique.columns if 'tide' in c.lower()]
                data_full['etss_tide'] = df_g_unique[t_col[0]].reindex(data_full.index) if t_col else 0.0
            data_full['etss_tide'] = pd.to_numeric(data_full['etss_tide'], errors='coerce').interpolate(method='linear').ffill().bfill().fillna(0.0)

            # 2. Water Level Observations
            wl_col = next((c for c in ['water_level', 'Water Level', 'wl', 'v'] if c in df_o_unique.columns), None)
            if wl_col:
                data_full['water_level'] = df_o_unique[wl_col].reindex(data_full.index)
            else:
                data_full['water_level'] = data_full['etss_tide']
            data_full['water_level'] = pd.to_numeric(data_full['water_level'], errors='coerce').interpolate(method='linear').ffill().bfill().fillna(0.0)

            # 3. Clean Surge Residual Baseline
            data_full['surge_residual'] = data_full['water_level'] - data_full['etss_tide']
            data_full['surge_residual'] = data_full['surge_residual'].interpolate(method='linear').ffill().bfill().fillna(0.0)

            # 4. ETSS Surge Guidance
            if 'etss_surge' in df_g_unique.columns:
                data_full['etss_surge_guidance'] = df_g_unique['etss_surge'].reindex(data_full.index)
            elif 'etss_water_level' in df_g_unique.columns:
                data_full['etss_surge_guidance'] = (df_g_unique['etss_water_level'] - data_full['etss_tide']).reindex(data_full.index)
            else:
                data_full['etss_surge_guidance'] = 0.0
            data_full['etss_surge_guidance'] = pd.to_numeric(data_full['etss_surge_guidance'], errors='coerce').interpolate(method='linear').ffill().bfill().fillna(0.0)

            # 5. Extract 4-Quadrant Features (20 Features)
            gfs_4quad_cols = []
            for c in range(1, N_OFFSHORE_CELLS + 1):
                u_col = f'c{c}_wind_u_gui'
                v_col = f'c{c}_wind_v_gui'
                p_col = f'c{c}_air_pressure_gui'
                t_col = f'c{c}_air_temp_gui'
                ws_col = f'c{c}_wind_speed_gui'

                if u_col in df_g_unique.columns and v_col in df_g_unique.columns:
                    u_vals = pd.to_numeric(df_g_unique[u_col], errors='coerce').fillna(0.0)
                    v_vals = pd.to_numeric(df_g_unique[v_col], errors='coerce').fillna(0.0)
                    df_g_unique[ws_col] = np.sqrt(u_vals**2 + v_vals**2)
                elif ws_col not in df_g_unique.columns:
                    df_g_unique[ws_col] = 0.0

                gfs_4quad_cols.extend([u_col, v_col, p_col, t_col, ws_col])

            for col in gfs_4quad_cols:
                if col in df_g_unique.columns:
                    data_full[col] = df_g_unique[col].reindex(data_full.index)
                else:
                    data_full[col] = 0.0
                data_full[col] = pd.to_numeric(data_full[col], errors='coerce').interpolate(method='linear').ffill().bfill().fillna(0.0)

            # 6. Surface Meteorological Guidance (5 Features)
            for col in ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui']:
                if col in df_g_unique.columns:
                    data_full[col] = df_g_unique[col].reindex(data_full.index)
                else:
                    data_full[col] = 0.0
                data_full[col] = pd.to_numeric(data_full[col], errors='coerce').interpolate(method='linear').ffill().bfill().fillna(0.0)

            if 'wind_speed_gui' in data_full.columns and 'wind_direction_gui' in data_full.columns:
                data_full['wind_u_gui'], data_full['wind_v_gui'] = compute_wind_components(
                    data_full['wind_speed_gui'].values, data_full['wind_direction_gui'].values
                )
            else:
                data_full['wind_u_gui'], data_full['wind_v_gui'] = 0.0, 0.0

            if 'wind_gust_gui' not in data_full.columns or (data_full['wind_gust_gui'] == 0.0).all():
                data_full['wind_gust_gui'] = data_full['wind_speed_gui'] * 1.2

            gfs_weather_vec = ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_gust_gui', 'etss_surge_guidance'] + gfs_4quad_cols
            data_full = clean_numeric_df(data_full, gfs_weather_vec + ['surge_residual', 'water_level', 'etss_tide'])

        except Exception as data_err:
            print(f"❌ Input parse failed on {STATION_NAME}: {data_err}")
            continue

        cached_models, cached_scalers = load_regional_or_station_models(STATION_ID, STATION_NAME)
        if not cached_models:
            print(f"⚠️ Model assets absent for {STATION_NAME}. Skipping station.")
            continue

        OBS_FEAT = gfs_weather_vec
        GUI_FEAT = gfs_weather_vec
        TARGET = 'surge_residual'

        def run_recursive_inference(fc_origin, op_gui, anchor_surge):
            local_horizons = {}
            block_variances = {24: [], 48: []}
            target_len = len(op_gui)

            for n_past in LOOKBACKS:
                key = f"{n_past}_CHRONO"
                model = cached_models[key]
                sc_f = cached_scalers[f"{key}_f"]
                sc_t = cached_scalers[f"{key}_t"]

                past_slice = data_full.loc[:fc_origin][OBS_FEAT + [TARGET]].copy()
                past_slice = clean_numeric_df(past_slice, OBS_FEAT + [TARGET])
                raw_past = np.nan_to_num(past_slice.values.astype(np.float32), nan=0.0)

                past_block = raw_past[-n_past:] if len(raw_past) >= n_past else np.vstack([np.repeat(raw_past[:1, :], n_past - len(raw_past), axis=0), raw_past])

                current_pivot = anchor_surge
                full_pred_surge = []

                for step_start in range(0, target_len, N_FUTURE_TRAINED):
                    step_end = min(step_start + N_FUTURE_TRAINED, target_len)
                    chunk_len = step_end - step_start

                    gui_chunk = op_gui.iloc[step_start:step_end]

                    fut_met_list = []
                    for col in GUI_FEAT:
                        if col in gui_chunk.columns:
                            val = pd.to_numeric(gui_chunk[col], errors='coerce').fillna(0.0).values.reshape(-1, 1)
                        elif col in data_full.columns:
                            val = pd.to_numeric(data_full.loc[gui_chunk.index, col], errors='coerce').fillna(0.0).values.reshape(-1, 1)
                        else:
                            val = np.zeros((len(gui_chunk), 1), dtype=np.float32)
                        fut_met_list.append(val)

                    fut_met_profile = np.nan_to_num(np.hstack(fut_met_list).astype(np.float32), nan=0.0)

                    pad_size = N_FUTURE_TRAINED - chunk_len
                    if pad_size > 0:
                        pad_met = np.repeat(fut_met_profile[-1:, :], pad_size, axis=0)
                        fut_met_profile = np.vstack([fut_met_profile, pad_met])

                    last_known_surge = past_block[-1, -1]
                    fut_fb = np.linspace(last_known_surge, current_pivot, N_FUTURE_TRAINED, dtype=np.float32).reshape(-1, 1)

                    op_fut_block = np.hstack([fut_met_profile, fut_fb])
                    inp = np.ascontiguousarray(np.vstack([past_block, op_fut_block]), dtype=np.float32)
                    inp = np.nan_to_num(inp, nan=0.0)

                    scaled_inp = sc_f.transform(inp) if hasattr(sc_f, 'transform') else inp
                    scaled_inp = np.nan_to_num(scaled_inp, nan=0.0)

                    inp_tensor = tf.convert_to_tensor(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, -1), dtype=tf.float32)
                    raw_pred = model(inp_tensor, training=False).numpy().flatten()
                    raw_pred = np.nan_to_num(raw_pred, nan=0.0)

                    pred_delta = sc_t.inverse_transform(raw_pred.reshape(-1, 1)).flatten() if hasattr(sc_t, 'inverse_transform') else raw_pred
                    pred_delta = np.nan_to_num(pred_delta, nan=0.0)

                    relative_delta = np.clip(pred_delta - pred_delta[0], -1.5, 1.5)
                    pred_surge = relative_delta + current_pivot
                    pred_surge_trimmed = pred_surge[:chunk_len]

                    full_pred_surge.extend(pred_surge_trimmed)

                    chunk_var = np.var(pred_surge_trimmed) if len(pred_surge_trimmed) > 1 else 1e-4
                    block_variances[n_past].extend([chunk_var] * chunk_len)

                    current_pivot = pred_surge_trimmed[-1]
                    chunk_to_append = np.hstack([fut_met_profile[:chunk_len], pred_surge_trimmed.reshape(-1, 1)])
                    past_block = np.vstack([past_block, chunk_to_append])[-n_past:]

                local_horizons[n_past] = np.array(full_pred_surge)[:target_len].flatten()
                block_variances[n_past] = np.array(block_variances[n_past])[:target_len]

            eps = 1e-6
            v24 = np.maximum(np.array(block_variances[24][:target_len], dtype=np.float64), eps)
            v48 = np.maximum(np.array(block_variances[48][:target_len], dtype=np.float64), eps)

            with np.errstate(divide='ignore', invalid='ignore'):
                inv_var_24 = 1.0 / v24
                inv_var_48 = 1.0 / v48
                total_inv_var = inv_var_24 + inv_var_48
                raw_w_24h = inv_var_24 / total_inv_var

            w_24h = np.nan_to_num(raw_w_24h, nan=0.5, posinf=0.5, neginf=0.5)
            w_24h = pd.Series(w_24h).rolling(window=12, min_periods=1, center=True).mean().values
            w_24h = np.clip(w_24h, 0.0, 1.0)
            w_48h = 1.0 - w_24h

            blended_track = (local_horizons[24] * w_24h) + (local_horizons[48] * w_48h)
            return np.nan_to_num(blended_track.flatten(), nan=anchor_surge)

        # --- RESOLVE FORECAST ORIGIN ---
        if args.t0:
            clean_t0 = args.t0.strip()
            # Handle YYYYMMDD_HHMM / YYYYMMDD_HH / standard datetime strings
            if '_' in clean_t0 and '-' not in clean_t0:
                parts = clean_t0.split('_')
                d_part, t_part = parts[0], parts[1]
                if len(t_part) == 2:
                    t_part += "00"
                fc_origin = pd.to_datetime(f"{d_part[:4]}-{d_part[4:6]}-{d_part[6:8]} {t_part[:2]}:{t_part[2:4]}:00")
            else:
                fc_origin = pd.to_datetime(clean_t0).round('h')

            if fc_origin not in data_full.index:
                # Snap to nearest hourly index within 1 hour if slight minute mismatch exists
                nearby = data_full.index[abs((data_full.index - fc_origin).total_seconds()) <= 3600]
                if len(nearby) > 0:
                    fc_origin = nearby[0]
                else:
                    print(f"⚠️ Specified t0 {args.t0} ({fc_origin}) not found in timeline for {STATION_NAME}. Skipping.")
                    continue
        else:
            # Automatic latest cycle search fallback
            dt_series = pd.Series(df_g_unique.index)
            breaks = (dt_series.diff().dt.total_seconds() / 3600.0 > MAX_ALLOWED_GAP_HOURS).cumsum()
            df_g_unique['block_id'] = breaks.values

            candidate_cycles = [t for t in df_g_unique.index if t in data_full.index]
            valid_cycles = sorted(candidate_cycles)

            if not valid_cycles:
                print(f"⚠️ No valid cycles found for {STATION_NAME}.")
                continue

            fc_origin = None
            for cand in reversed(valid_cycles):
                blk = df_g_unique.loc[cand, 'block_id']
                future_window = df_g_unique.loc[cand + pd.Timedelta(hours=1) : cand + pd.Timedelta(hours=args.horizon)]
                if len(future_window) >= args.horizon and (future_window['block_id'] == blk).all():
                    fc_origin = cand
                    break

            if fc_origin is None:
                for cand in reversed(valid_cycles):
                    future_window = df_g_unique.loc[cand + pd.Timedelta(hours=1) : cand + pd.Timedelta(hours=args.horizon)]
                    if len(future_window) >= 12:
                        fc_origin = cand
                        break

        if fc_origin is None:
            print(f"⚠️ No valid forecast window for {STATION_NAME}.")
            continue

        anchor_surge = float(data_full.loc[fc_origin, 'surge_residual']) if fc_origin in data_full.index else 0.0
        if np.isnan(anchor_surge):
            anchor_surge = 0.0

        print(f"🎯 Forecast Origin Anchor: {fc_origin} ({fc_origin.strftime('%Y-%m-%d %H:%M:%S')}) | Ground Anchor Surge: {anchor_surge:+.3f} ft")

        op_gui = df_g_unique.loc[
            fc_origin + pd.Timedelta(hours=1) : fc_origin + pd.Timedelta(hours=args.horizon)
        ].copy()

        actual_steps = len(op_gui)
        if actual_steps < 12:
            print(f"⚠️ Insufficient guidance points ({actual_steps}h) for {STATION_NAME} starting at {fc_origin}.")
            continue

        pred_pivot_surge_forward = run_recursive_inference(fc_origin, op_gui, anchor_surge)[:actual_steps]

        tide_guidance = data_full.loc[op_gui.index, 'etss_tide'].values[:actual_steps]
        etss_surge_guidance = data_full.loc[op_gui.index, 'etss_surge_guidance'].values[:actual_steps]

        etss_bias_forward = etss_surge_guidance - pred_pivot_surge_forward
        pred_twl_forward = pred_pivot_surge_forward + tide_guidance

        # Prospective Forecast Frame
        df_forecast = pd.DataFrame({
            'station_id': STATION_ID,
            'station_name': STATION_NAME,
            'forecast_origin': fc_origin.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': op_gui.index.strftime('%Y-%m-%d %H:%M:%S'),
            'dt': op_gui.index,
            'lead_hour': np.arange(1, actual_steps + 1),
            'observed_water_level_ft': np.nan,
            'observed_surge_residual_ft': np.nan,
            'air_temp_degC': op_gui['air_temp_gui'].values[:actual_steps] if 'air_temp_gui' in op_gui.columns else np.nan,
            'air_pressure_hPa': op_gui['air_pressure_gui'].values[:actual_steps] if 'air_pressure_gui' in op_gui.columns else np.nan,
            'wind_speed_kts': op_gui['wind_speed_gui'].values[:actual_steps] if 'wind_speed_gui' in op_gui.columns else np.nan,
            'wind_direction_deg': op_gui['wind_direction_gui'].values[:actual_steps] if 'wind_direction_gui' in op_gui.columns else np.nan,
            'wind_gust_kts': op_gui['wind_gust_gui'].values[:actual_steps] if 'wind_gust_gui' in op_gui.columns else np.nan,
            'etss_tide_ft': np.round(tide_guidance, 3),
            'etss_surge_ft': np.round(etss_surge_guidance, 3),
            'etss_water_level_raw_ft': np.round(etss_surge_guidance + tide_guidance, 3),
            'pure_gfs_surge_ft': np.round(pred_pivot_surge_forward, 3),
            'pure_gfs_twl_ft': np.round(pred_twl_forward, 3),
            'pivot_surge_forecast_ft': np.round(pred_pivot_surge_forward, 3),
            'pivot_total_water_level_ft': np.round(pred_twl_forward, 3),
            'etss_corrected_surge_ft': np.round(pred_pivot_surge_forward, 3),
            'etss_corrected_twl_ft': np.round(pred_twl_forward, 3),
            'etss_bias_ft': np.round(etss_bias_forward, 3),
            'predicted_etss_bias_ft': np.round(etss_bias_forward, 3)
        })

        # Trailing Hindcast Frame
        t_hindcast_start = fc_origin - pd.Timedelta(hours=args.lookback)
        hindcast_dates = pd.date_range(start=t_hindcast_start, end=fc_origin, freq='h')
        df_hindcast_data = data_full.loc[data_full.index.intersection(hindcast_dates)].copy()

        h_len = len(hindcast_dates)
        h_lead_hours = np.linspace(-h_len + 1, 0, h_len, dtype=int)

        h_obs_wl = df_hindcast_data['water_level'].reindex(hindcast_dates).ffill().bfill().values
        h_tide = df_hindcast_data['etss_tide'].reindex(hindcast_dates).ffill().bfill().values
        h_surge = df_hindcast_data['etss_surge_guidance'].reindex(hindcast_dates).ffill().bfill().values
        h_obs_res = h_obs_wl - h_tide

        df_hindcast = pd.DataFrame({
            'station_id': STATION_ID,
            'station_name': STATION_NAME,
            'forecast_origin': fc_origin.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': hindcast_dates.strftime('%Y-%m-%d %H:%M:%S'),
            'dt': hindcast_dates,
            'lead_hour': h_lead_hours,
            'observed_water_level_ft': np.round(h_obs_wl, 3),
            'observed_surge_residual_ft': np.round(h_obs_res, 3),
            'air_temp_degC': np.nan,
            'air_pressure_hPa': np.nan,
            'wind_speed_kts': np.nan,
            'wind_direction_deg': np.nan,
            'wind_gust_kts': np.nan,
            'etss_tide_ft': np.round(h_tide, 3),
            'etss_surge_ft': np.round(h_surge, 3),
            'etss_water_level_raw_ft': np.round(h_surge + h_tide, 3),
            'pure_gfs_surge_ft': np.nan,
            'pure_gfs_twl_ft': np.nan,
            'pivot_surge_forecast_ft': np.nan,
            'pivot_total_water_level_ft': np.nan,
            'etss_corrected_surge_ft': np.nan,
            'etss_corrected_twl_ft': np.nan,
            'etss_bias_ft': np.nan,
            'predicted_etss_bias_ft': np.nan
        })

        t0_twl = anchor_surge + h_tide[-1]
        t0_bias = h_surge[-1] - anchor_surge

        df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'pure_gfs_surge_ft'] = np.round(anchor_surge, 3)
        df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'pure_gfs_twl_ft'] = np.round(t0_twl, 3)
        df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'pivot_surge_forecast_ft'] = np.round(anchor_surge, 3)
        df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'pivot_total_water_level_ft'] = np.round(t0_twl, 3)
        df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'etss_corrected_surge_ft'] = np.round(anchor_surge, 3)
        df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'etss_corrected_twl_ft'] = np.round(t0_twl, 3)
        df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'etss_bias_ft'] = np.round(t0_bias, 3)
        df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'predicted_etss_bias_ft'] = np.round(t0_bias, 3)

        stream_df = pd.concat([df_hindcast, df_forecast], ignore_index=True)
        stream_df = stream_df.sort_values('dt').reset_index(drop=True)

        t0_tag = fc_origin.strftime('%Y%m%d_%H%M')
        out_csv = os.path.join(OUTPUT_DIR, f"realtime_gfs_etss_96h_forecast_regional_{STATION_ID}_{t0_tag}.csv")
        stream_df.to_csv(out_csv, index=False)
        print(f"✅ Exported {len(stream_df)}h stream (t0: {t0_tag}, anchor: {anchor_surge:+.3f} ft) to: {out_csv}")

        K.clear_session()
        gc.collect()

    print("\n🎉 Regional real-time forecast generation completed.")

if __name__ == '__main__':
    main()
