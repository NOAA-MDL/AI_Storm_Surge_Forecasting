#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Operational Real-Time ETSS Bias / Surge Residual 96-Hour Forecast Stream Generator.
Batch Multi-Station Engine utilizing trained 4-Quadrant Spatial GRU Models ([None, 36, 26]).
Guarantees C0 continuous chunk junctions and eliminates all 12-hour staircase notches.
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

MODELS_PATH = '/contrib/Mamoudou.Ba/storm_surge/'
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
N_FORECAST_HORIZON = 96
HINDCAST_LOOKBACK = 48
N_OFFSHORE_CELLS = 4
LOOKBACKS = [24, 48]

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

def clean_numeric_df(df, cols, fill_val=0.0):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_val
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].interpolate(method='linear').ffill().bfill().fillna(fill_val)
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Operational 96h Stream Generator")
    parser.add_argument('--config', type=str, default=CONFIG_FILE, help="Stations config CSV")
    parser.add_argument('--station', type=str, default=None, help="Optional single station ID")
    parser.add_argument('--mode', type=str, default='BIAS', choices=['BIAS', 'RESID'], help="Model mode")
    parser.add_argument('--horizon', type=int, default=N_FORECAST_HORIZON, help="Horizon hours (96)")
    parser.add_argument('--lookback', type=int, default=HINDCAST_LOOKBACK, help="Hindcast lookback hours (48)")
    parser.add_argument('--t0', type=str, default=None, help="Explicit forecast origin (YYYY-MM-DD HH:MM)")
    parser.add_argument('--max_gap', type=float, default=2.0, help="Max raw data gap allowed in hours (default: 2.0)")
    return parser.parse_args()

def load_and_parse(file_path):
    df = pd.read_csv(file_path)
    sentinels = [9999.0, 9999.00, 99999.0, 99999.000, -9999.0, -99999.0]
    df = df.replace(sentinels, np.nan)

    time_candidates = ['timestamp_gui', 'timestamp', 'dt', 'datetime', 'date_time', 'valid_time', 'time', 'date']
    t_col = next((c for c in time_candidates if c in df.columns), None)
    if not t_col:
        matched = [c for c in df.columns if any(k in c.lower() for k in ['time', 'date', 'dt'])]
        if matched: t_col = matched[0]
    if not t_col: raise KeyError(f"No datetime column in {file_path}")

    df[t_col] = df[t_col].astype(str).str.replace(r'(\d{2}) (\d{2}) (\d{2})$', r'\1:\2:\3', regex=True)
    df['dt'] = pd.to_datetime(df[t_col], errors='coerce', format='mixed')
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_localize(None)
    df['dt'] = df['dt'].dt.round('h')
    return df.dropna(subset=['dt'])

def resolve_station_files(station_id):
    gfs_candidates = [
        os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{station_id}_2026.csv'),
        os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{station_id}.csv'),
        f'test_gfs_etss_spatial_data_{station_id}_2026.csv'
    ]
    gfs_path = next((f for f in gfs_candidates if os.path.exists(f)), None)

    obs_candidates = [
        os.path.join(MODELS_PATH, f'obs_training_{station_id}.csv'),
        os.path.join(MODELS_PATH, f'obs_training_{station_id}_2026.csv'),
        os.path.join(MODELS_PATH, f'realtime_obs_{station_id}_2026.csv'),
        os.path.join(MODELS_PATH, f'obs_{station_id}.csv'),
        f'obs_training_{station_id}.csv'
    ]
    obs_path = next((f for f in obs_candidates if os.path.exists(f)), None)
    return gfs_path, obs_path

def load_station_models(station_id, station_name, mode):
    cached_models, cached_scalers = {}, {}
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
            if not os.path.exists(h_cand): h_cand = os.path.join(MODELS_PATH, f"{base_model}.h5")
            sf_cand = os.path.join(MODELS_PATH, f"{prefix}_scaler_etss_bias_pivot_robust_{n_past}h_CHRONO.joblib")
            st_cand = os.path.join(MODELS_PATH, f"{prefix}_scaler_etss_bias_pivot_target_robust_{n_past}h_CHRONO.joblib")

            if os.path.exists(j_cand) and os.path.exists(h_cand) and os.path.exists(sf_cand) and os.path.exists(st_cand):
                m_json, m_h5, sf_path, st_path = j_cand, h_cand, sf_cand, st_cand
                break

        if not (m_json and os.path.exists(m_h5) and sf_path and st_path):
            for ident in [name_clean, station_id, name_no_state]:
                if not m_json:
                    j_m = glob.glob(os.path.join(MODELS_PATH, f"*{ident}*{mode}*{n_past}past*.json"))
                    if j_m:
                        m_json = j_m[0]
                        stem = m_json[:-5]
                        m_h5 = f"{stem}.weights.h5" if os.path.exists(f"{stem}.weights.h5") else f"{stem}.h5"
                if not sf_path:
                    sf_m = glob.glob(os.path.join(MODELS_PATH, f"*{ident}*scaler*etss*{mode.lower()}*robust_{n_past}h*.joblib"))
                    sf_m = [f for f in sf_m if 'target' not in os.path.basename(f)]
                    if sf_m: sf_path = sf_m[0]
                if not st_path:
                    st_m = glob.glob(os.path.join(MODELS_PATH, f"*{ident}*scaler*etss*{mode.lower()}*target*robust_{n_past}h*.joblib"))
                    if st_m: st_path = st_m[0]
                if m_json and os.path.exists(m_h5) and sf_path and st_path: break

        if m_json and os.path.exists(m_h5) and sf_path and st_path:
            with open(m_json, 'r') as f:
                model = model_from_json(f.read())
            model.load_weights(m_h5)
            cached_models[key] = model
            cached_scalers[f"{key}_f"] = joblib.load(sf_path)
            cached_scalers[f"{key}_t"] = joblib.load(st_path)
        else:
            return None, None

    return cached_models, cached_scalers

def predict_single_chunk(cached_models, cached_scalers, data_full, chunk_origin, op_gui, feedback_val):
    target_len = len(op_gui)
    local_horizons = {}

    for n_past in LOOKBACKS:
        key = f"PIVOT_{n_past}_{TARGET_SCALER}"
        if key not in cached_models:
            local_horizons[n_past] = np.full(target_len, 0.0, dtype=np.float32)
            continue

        model = cached_models[key]
        sc_f = cached_scalers[f"{key}_f"]
        sc_t = cached_scalers[f"{key}_t"]

        past_hist_df = data_full.loc[:chunk_origin].tail(n_past).copy()
        if len(past_hist_df) < n_past:
            pad_n = n_past - len(past_hist_df)
            past_hist_df = pd.concat([pd.concat([past_hist_df.iloc[[0]]] * pad_n), past_hist_df])

        past_met = np.nan_to_num(past_hist_df[OBS_WEATHER_COLS].values.astype(np.float32), nan=0.0)

        fut_guidance_df = op_gui[GFS_GUIDANCE_COLS].copy()
        if len(fut_guidance_df) < N_FUTURE_TRAINED:
            pad_n = N_FUTURE_TRAINED - len(fut_guidance_df)
            fut_guidance_df = pd.concat([fut_guidance_df, pd.concat([fut_guidance_df.iloc[[-1]]] * pad_n)])
        elif len(fut_guidance_df) > N_FUTURE_TRAINED:
            fut_guidance_df = fut_guidance_df.iloc[:N_FUTURE_TRAINED]

        future_met = np.nan_to_num(fut_guidance_df.values.astype(np.float32), nan=0.0)
        hybrid_met = np.vstack([past_met, future_met])

        past_surge = np.nan_to_num(past_hist_df[[TARGET_COL]].values.astype(np.float32), nan=0.0)
        future_surge_lkv = np.full((N_FUTURE_TRAINED, 1), feedback_val, dtype=np.float32)
        hybrid_surge = np.vstack([past_surge, future_surge_lkv])

        combined = np.ascontiguousarray(np.hstack([hybrid_met, hybrid_surge]), dtype=np.float32)
        combined = np.nan_to_num(combined, nan=0.0)

        scaled_inp = sc_f.transform(combined) if hasattr(sc_f, 'transform') else combined
        scaled_inp = np.nan_to_num(scaled_inp, nan=0.0)

        inp_tensor = tf.convert_to_tensor(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, 26), dtype=tf.float32)
        raw_pred = model(inp_tensor, training=False).numpy().flatten()
        raw_pred = np.nan_to_num(raw_pred, nan=0.0)

        aligned_pred = np.zeros(N_FUTURE_TRAINED, dtype=np.float32)
        fill_len = min(len(raw_pred), N_FUTURE_TRAINED)
        aligned_pred[:fill_len] = raw_pred[:fill_len]

        pred_bias = sc_t.inverse_transform(aligned_pred.reshape(-1, 1)).flatten() if hasattr(sc_t, 'inverse_transform') else aligned_pred
        local_horizons[n_past] = np.nan_to_num(pred_bias[:target_len], nan=0.0)

    timeline = np.arange(target_len)
    weight_24h = 1.0 / (1.0 + np.exp((timeline - 36) / 8))
    h24 = local_horizons.get(24, np.full(target_len, 0.0))
    h48 = local_horizons.get(48, np.full(target_len, 0.0))
    combined_bias = ((h24[:target_len] * weight_24h) + (h48[:target_len] * (1.0 - weight_24h)))[:target_len]
    return np.nan_to_num(combined_bias, nan=0.0)

def process_station(station_id, station_name, mode, horizon, lookback, user_t0, max_gap_hours=2.0):
    gfs_path, obs_path = resolve_station_files(station_id)
    if not gfs_path or not obs_path:
        print(f"   [SKIP] Missing files for {station_id}")
        return False

    cached_models, cached_scalers = load_station_models(station_id, station_name, mode)
    if not cached_models:
        print(f"   [SKIP] Missing models for {station_id}")
        return False

    try:
        df_g = load_and_parse(gfs_path)
        df_o = load_and_parse(obs_path)
    except Exception as e:
        print(f"   [SKIP] Data parse error: {e}")
        return False

    for c in range(1, N_OFFSHORE_CELLS + 1):
        u_col, v_col, ws_col = f'c{c}_wind_u_gui', f'c{c}_wind_v_gui', f'c{c}_wind_speed_gui'
        if u_col in df_g.columns and v_col in df_g.columns:
            df_g[ws_col] = np.sqrt(pd.to_numeric(df_g[u_col], errors='coerce')**2 + 
                                   pd.to_numeric(df_g[v_col], errors='coerce')**2)
        elif ws_col not in df_g.columns:
            df_g[ws_col] = 0.0

    for col in GFS_4QUAD_COLS:
        if col not in df_g.columns: df_g[col] = 0.0

    for col in ['etss_surge', 'etss_tide', 'etss_bias', 'etss_water_level', 'SURGE']:
        if col in df_g.columns:
            df_g[col] = pd.to_numeric(df_g[col], errors='coerce')

    df_g_unique = df_g.sort_values('dt').drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()
    df_o_unique = df_o.sort_values('dt').drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()

    dt_series = pd.Series(df_g_unique.index)
    breaks = (dt_series.diff().dt.total_seconds() / 3600.0 > max_gap_hours).cumsum()
    df_g_unique['block_id'] = breaks.values

    clean_blocks = []
    for bid, grp in df_g_unique.groupby('block_id'):
        span_hours = (grp.index.max() - grp.index.min()).total_seconds() / 3600.0
        if span_hours >= horizon:
            clean_blocks.append(grp)

    if user_t0:
        t0 = pd.to_datetime(user_t0)
    elif clean_blocks:
        chosen_block = clean_blocks[-1]
        t0 = chosen_block.index[0]
    else:
        print(f"   [SKIP] Station {station_name} ({station_id}) has no continuous block >= {horizon}h.")
        return False

    t0 = pd.to_datetime(t0)

    # Full timeline for hindcast observations
    data_full = pd.DataFrame(index=df_g_unique.index.union(df_o_unique.index)).sort_index()
    for col in df_g_unique.columns:
        if col != 'block_id': data_full[col] = df_g_unique[col]

    data_full['water_level'] = pd.to_numeric(df_o_unique['water_level'] if 'water_level' in df_o_unique.columns else np.nan, errors='coerce')
    data_full['water_level'] = data_full['water_level'].interpolate(method='linear').ffill().bfill().fillna(0.0)

    t_cols = [c for c in df_g_unique.columns if 'tide' in c.lower()]
    data_full['etss_tide'] = df_g_unique[t_cols[0]] if t_cols else 0.0

    s_cols = [c for c in ['etss_surge', 'SURGE'] if c in df_g_unique.columns]
    data_full['etss_surge'] = df_g_unique[s_cols[0]] if s_cols else 0.0
    data_full['etss_surge_guidance'] = data_full['etss_surge']

    for met_base in ['air_temp', 'air_pressure', 'wind_speed', 'wind_gust']:
        gui_col = f"{met_base}_gui"
        if gui_col not in data_full.columns: data_full[gui_col] = 0.0
        if met_base in df_o_unique.columns and df_o_unique[met_base].notna().sum() > 10:
            data_full[met_base] = pd.to_numeric(df_o_unique[met_base], errors='coerce')
        else:
            data_full[met_base] = data_full[gui_col]

    data_full[TARGET_COL] = data_full['water_level'] - data_full['etss_tide']

    all_needed_cols = list(set(OBS_WEATHER_COLS + GFS_GUIDANCE_COLS + [TARGET_COL, 'etss_tide', 'etss_surge', 'water_level']))
    for col in all_needed_cols:
        if col not in data_full.columns: data_full[col] = 0.0
        data_full[col] = pd.to_numeric(data_full[col], errors='coerce').interpolate(method='linear').ffill().bfill().fillna(0.0)

    # 1. Anchor Values at t0
    obs_res_t0 = float(data_full.loc[t0, TARGET_COL]) if t0 in data_full.index else float(data_full.loc[:t0].iloc[-1][TARGET_COL])
    etss_surge_t0 = float(data_full.loc[t0, 'etss_surge']) if t0 in data_full.index else float(data_full.loc[:t0].iloc[-1]['etss_surge'])
    tide_t0 = float(data_full.loc[t0, 'etss_tide']) if t0 in data_full.index else float(data_full.loc[:t0].iloc[-1]['etss_tide'])

    bias_anchor_t0 = obs_res_t0 - etss_surge_t0

    # 2. Prospective Window [t0 + 1h, t0 + horizon]
    fcst_dates = pd.date_range(start=t0 + pd.Timedelta(hours=1), periods=horizon, freq='h')
    forecast_window = pd.DataFrame(index=fcst_dates)
    for col in GFS_GUIDANCE_COLS + ['etss_tide', 'etss_surge']:
        if col in data_full.columns:
            forecast_window[col] = data_full.loc[data_full.index.intersection(fcst_dates), col]
        forecast_window[col] = pd.to_numeric(forecast_window[col], errors='coerce').interpolate(method='linear').ffill().bfill().fillna(0.0)

    forecast_window['etss_surge_guidance'] = forecast_window['etss_surge']
    rollout_len = len(forecast_window)

    # 3. Continuous Rollout with C0 Junction Stitching
    continuous_predicted_bias = []
    current_anchor_time = t0
    running_feedback = obs_res_t0
    last_chunk_endpoint = bias_anchor_t0  # Stitch point starts directly at t0 bias

    num_chunks = int(np.ceil(rollout_len / 12.0))
    for c_idx in range(num_chunks):
        c_start = c_idx * 12
        c_end = min(c_start + 12, rollout_len)
        chunk_gui = forecast_window.iloc[c_start:c_end]
        if chunk_gui.empty: break

        raw_c_bias = predict_single_chunk(cached_models, cached_scalers, data_full, current_anchor_time, chunk_gui, running_feedback)
        
        # Stitch chunk start to previous endpoint to prevent drops/jumps
        # Transition offset decays across chunk length
        step_offset = last_chunk_endpoint - raw_c_bias[0]
        decay = np.exp(-np.arange(len(raw_c_bias)) / 8.0)
        stitched_chunk_bias = raw_c_bias + (step_offset * decay)

        continuous_predicted_bias.extend(stitched_chunk_bias)

        # Autoregressive update: feed generated surge residual into data_full history
        chunk_surge_pred = chunk_gui['etss_surge'].values + stitched_chunk_bias
        for dt_val, s_val in zip(chunk_gui.index, chunk_surge_pred):
            data_full.loc[dt_val, TARGET_COL] = s_val

        last_chunk_endpoint = float(stitched_chunk_bias[-1])
        running_feedback = float(chunk_surge_pred[-1])
        current_anchor_time = chunk_gui.index[-1]

    final_predicted_bias = np.array(continuous_predicted_bias)[:rollout_len]
    etss_surge_vals = forecast_window['etss_surge'].values[:rollout_len]
    tide_vals = forecast_window['etss_tide'].values[:rollout_len]

    corrected_surge = etss_surge_vals + final_predicted_bias
    corrected_wl = corrected_surge + tide_vals

    lead_hours = np.arange(1, rollout_len + 1)
    df_forecast = pd.DataFrame({
        'dt': forecast_window.index[:rollout_len],
        'station_id': station_id,
        'station_name': station_name,
        'forecast_origin': t0.strftime('%Y-%m-%d %H:%M:%S'),
        'lead_hour': lead_hours,
        'observed_water_level_ft': np.nan,
        'observed_surge_residual_ft': np.nan,
        'etss_tide_ft': tide_vals,
        'etss_surge_ft': etss_surge_vals,
        'etss_water_level_raw_ft': etss_surge_vals + tide_vals,
        'predicted_etss_bias_ft': final_predicted_bias,
        'etss_corrected_surge_ft': corrected_surge,
        'etss_corrected_twl_ft': corrected_wl,
        'pivot_surge_forecast_ft': corrected_surge,
        'pivot_total_water_level_ft': corrected_wl
    })

    # Hindcast window [t0 - lookback, t0]
    t_hindcast_start = t0 - pd.Timedelta(hours=lookback)
    hindcast_dates = pd.date_range(start=t_hindcast_start, end=t0, freq='h')
    df_hindcast_data = data_full.loc[data_full.index.intersection(hindcast_dates)].copy()

    h_len = len(hindcast_dates)
    h_lead_hours = np.linspace(-h_len + 1, 0, h_len, dtype=int)

    h_obs_wl = df_hindcast_data['water_level'].reindex(hindcast_dates).ffill().bfill().values
    h_tide = df_hindcast_data['etss_tide'].reindex(hindcast_dates).ffill().bfill().values
    h_surge = df_hindcast_data['etss_surge'].reindex(hindcast_dates).ffill().bfill().values
    h_obs_res = h_obs_wl - h_tide

    df_hindcast = pd.DataFrame({
        'dt': hindcast_dates,
        'station_id': station_id,
        'station_name': station_name,
        'forecast_origin': t0.strftime('%Y-%m-%d %H:%M:%S'),
        'lead_hour': h_lead_hours,
        'observed_water_level_ft': h_obs_wl,
        'observed_surge_residual_ft': h_obs_res,
        'etss_tide_ft': h_tide,
        'etss_surge_ft': h_surge,
        'etss_water_level_raw_ft': h_surge + h_tide,
        'predicted_etss_bias_ft': np.nan,
        'etss_corrected_surge_ft': np.nan,
        'etss_corrected_twl_ft': np.nan,
        'pivot_surge_forecast_ft': np.nan,
        'pivot_total_water_level_ft': np.nan
    })

    # Precise t0 anchor pinning
    df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'predicted_etss_bias_ft'] = bias_anchor_t0
    df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'etss_corrected_surge_ft'] = obs_res_t0
    df_hindcast.loc[df_hindcast['lead_hour'] == 0, 'etss_corrected_twl_ft'] = obs_res_t0 + tide_t0

    export_df = pd.concat([df_hindcast, df_forecast], ignore_index=True)
    export_df = export_df.sort_values('dt').reset_index(drop=True)

    t0_tag = t0.strftime('%Y%m%d_%H%M')
    csv_fname = f"realtime_etss_bias_96h_forecast_{station_id}_{t0_tag}.csv"
    out_csv_path = os.path.join(STREAMS_DIR, csv_fname)
    export_df.to_csv(out_csv_path, index=False)

    obs_count = export_df.loc[export_df['lead_hour'] <= 0, 'observed_water_level_ft'].notna().sum()
    fcst_count = export_df.loc[export_df['lead_hour'] > 0, 'etss_corrected_surge_ft'].notna().sum()
    print(f"   [DONE] {station_name} ({station_id}): Anchor Obs={obs_res_t0:+.2f}ft, Bias={bias_anchor_t0:+.2f}ft | (Obs: {obs_count}, Fcst: {fcst_count}) -> {csv_fname}")

    del cached_models, cached_scalers, data_full, df_g, df_o, export_df
    K.clear_session()
    gc.collect()
    return True

def main():
    args = parse_args()
    print("=" * 80)
    print("OPERATIONAL 96-HOUR ETSS BIAS STREAM GENERATOR (SEAMLESS JUNCTIONS)")
    print(f"Config File    : {args.config}")
    print(f"Horizon/Look   : +{args.horizon}h / -{args.lookback}h")
    print(f"Max Raw Gap    : {args.max_gap}h")
    if args.t0:
        print(f"Specified t0   : {args.t0}")
    print("=" * 80)

    df_cfg = pd.read_csv(args.config, dtype={'station_id': str})
    station_list = []
    for _, r in df_cfg.iterrows():
        sid = str(r['station_id']).strip()
        sname = str(r['station_name']).strip().replace('"', '').replace(',', '').replace('.', '').replace(' ', '_') if 'station_name' in r else f"Station_{sid}"
        station_list.append((sid, sname))

    if args.station:
        station_list = [(sid, sname) for sid, sname in station_list if sid == args.station.strip()]

    for idx, (sid, sname) in enumerate(station_list, 1):
        print(f"[{idx}/{len(station_list)}] Processing Station {sname} ({sid})...")
        process_station(sid, sname, args.mode, args.horizon, args.lookback, args.t0, args.max_gap)

if __name__ == '__main__':
    main()
