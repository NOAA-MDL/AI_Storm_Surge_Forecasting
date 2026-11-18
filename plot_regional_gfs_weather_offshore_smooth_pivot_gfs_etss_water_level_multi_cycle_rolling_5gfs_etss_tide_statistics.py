#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Script developed with the assistance of the Gemini AI code assistant.
Operational GFS 4-Quadrant + ETSS Hybrid Pivot Multi-Station Validation Engine (Non-DOY + Tide Predictor).
Evaluates multi-cycle unified CHRONO regional models across configured stations using historical test streams.
Generates 3-panel monthly performance charts (Total Water Level, Pure Surge Residual, Side Stream Card)
and automatically exports all comparative AI vs ETSS performance metrics to a summary CSV file.
Dynamic Inverse-Variance Weighting Integration: Dynamically blends 24h & 48h lookback ensemble members.
Config File: Reads full registry from '../storm_surge/stations_config.csv'.
'''

import os, sys, gc, warnings
warnings.filterwarnings('ignore')

# 🌟 FORCE KERAS 2 DESERIALIZER
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# 🌟 SUPPRESS ALL TENSORFLOW NOISE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Clean CPU execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import joblib

# Force headless Matplotlib execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 🌟 SYSTEM & MODEL DEFINITIONS
TARGET_SCALER = "ROBUST_TIDE"
EVAL_START = "2026-01-01 00:00:00"
EVAL_END = "2026-10-31 23:00:00"

MODELS_PATH = '../storm_surge/'
if not os.path.exists(MODELS_PATH):
    MODELS_PATH = '/contrib/Mamoudou.Ba/storm_surge/'

CONFIG_FILE = os.path.join(MODELS_PATH, 'stations_config.csv')
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = os.path.join(MODELS_PATH, 'all_stations_config.csv')
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = 'stations_config.csv'

OUTPUT_DIR = '/contrib/Mamoudou.Ba/storm_surge/verification_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regional cluster mapping
REGIONAL_CLUSTERS = {
    '8725520': 'FL_WEST_COAST',  # Fort Myers, FL
    '8726520': 'FL_WEST_COAST',  # St. Petersburg, FL
    '8726607': 'FL_WEST_COAST',  # Old Port Tampa, FL
    '8726724': 'FL_WEST_COAST',  # Clearwater Beach, FL
}

# 🌟 CONFIGURATION PARAMETERS
N_FUTURE_TRAINED = 12   # Trained step horizon: 12h
N_FUTURE_OP = 12        # Operational evaluation horizon: 12h
N_OFFSHORE_CELLS = 4    # 4 Quadrants (c1, c2, c3, c4)
lookbacks = [24, 48]


def load_and_parse(file_path):
    df = pd.read_csv(file_path)

    # Detect headerless 2-column observation CSVs (e.g. '2025-12-31 12:00:00,-0.098')
    if len(df.columns) == 2 and pd.notna(pd.to_datetime(df.columns[0], errors='coerce')):
        df = pd.read_csv(file_path, header=None, names=['timestamp', 'water_level'])

    sentinels = [9999.0, 9999.00, 99999.0, 99999.000, -9999.0, -99999.0, -99.0, 99.0]
    df = df.replace(sentinels, np.nan)

    time_candidates = ['timestamp_gui', 'timestamp', 'dt', 'datetime', 'date_time', 'valid_time', 'time', 'date', 'Date Time', 't']
    t_col = next((c for c in time_candidates if c in df.columns), None)
    if not t_col:
        matched = [c for c in df.columns if any(k in c.lower() for k in ['time', 'date', 'dt', 't'])]
        t_col = matched[0] if matched else df.columns[0]

    df[t_col] = df[t_col].astype(str).str.replace(r'(\d{2}) (\d{2}) (\d{2})$', r'\1:\2:\3', regex=True)
    df['dt'] = pd.to_datetime(df[t_col], errors='coerce', format='mixed')
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_localize(None)
    df['dt'] = df['dt'].dt.round('h')
    return df.dropna(subset=['dt'])


def compute_wind_components(speed, direction_deg):
    rad = np.radians(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


def load_regional_pure_pivot_models(station_id, station_name):
    cached_models = {}
    cached_scalers = {}
    region_tag = REGIONAL_CLUSTERS.get(str(station_id).strip(), None)

    name_clean = station_name.strip().replace('"', '').replace(',', '').replace('.', '').replace(' ', '_')
    name_no_state = name_clean.split('_')[0]

    prefixes = []
    if region_tag:
        prefixes.append(region_tag)
    prefixes.extend([name_clean, str(station_id).strip(), name_no_state])

    search_dirs = [MODELS_PATH, os.path.join('..', 'storm_surge')]

    for n_past in lookbacks:
        key = f"CHRONO_{n_past}"
        m_json_path, m_h5_path, sf_path, st_path = None, None, None, None

        for prefix in prefixes:
            m_name = f"{prefix}_TIDE_{N_FUTURE_TRAINED}_GFS_WEATHER_4QUADRANT_GFS_WEATHER_OBS_ETSS_WL_PIVOT_ROBUST_{n_past}past_CHRONO"
            s_f_name = f"{prefix}_tide_scaler_gfs_weather_4quadrant_gfs_weather_obs_etss_wl_pivot_robust_{n_past}h_CHRONO.joblib"
            s_t_name = f"{prefix}_tide_scaler_gfs_weather_4quadrant_gfs_weather_obs_etss_wl_pivot_target_robust_{n_past}h_CHRONO.joblib"

            for sdir in search_dirs:
                j_c = os.path.join(sdir, f"{m_name}.json")
                h_c = os.path.join(sdir, f"{m_name}.weights.h5")
                if not os.path.exists(h_c):
                    h_c = os.path.join(sdir, f"{m_name}.h5")

                sf_c = os.path.join(sdir, s_f_name)
                st_c = os.path.join(sdir, s_t_name)

                if os.path.exists(j_c) and os.path.exists(h_c) and os.path.exists(sf_c):
                    m_json_path = j_c
                    m_h5_path = h_c
                    sf_path = sf_c
                    st_path = st_c if os.path.exists(st_c) else sf_c
                    break

            if m_json_path and os.path.exists(m_h5_path) and sf_path:
                break

        if not (m_json_path and os.path.exists(m_h5_path) and sf_path):
            return None, None

        try:
            with open(m_json_path, 'r') as f:
                model = model_from_json(f.read())
            model.load_weights(m_h5_path)
            cached_models[key] = model
            cached_scalers[f"{key}_f"] = joblib.load(sf_path)
            cached_scalers[f"{key}_t"] = joblib.load(st_path)
            print(f"      [Pure GFS + Tide Regional] Loaded {os.path.basename(m_h5_path)}")
        except Exception as e:
            print(f"❌ Error loading assets for {key}: {e}")
            return None, None

    return cached_models, cached_scalers


# --- 1. DYNAMIC CONFIGURATION LOADER ---
if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: Master station configuration matrix missing at: {CONFIG_FILE}")
    sys.exit(1)

print(f"📖 Parsing master CSV registry tracks from {CONFIG_FILE}...")
df_config = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})
df_config['station_id'] = df_config['station_id'].str.strip()
df_config['station_name'] = df_config['station_name'].str.strip().str.replace('"', '').str.replace(',', '').str.replace(' ', '_')

# --- 🌟 INITIALIZE ACCUMULATOR FOR CSV STATISTICS ---
summary_stats_list = []
CSV_SUMMARY_PATH = os.path.join(OUTPUT_DIR, f"regional_gfs_etss_hybrid_4quadrant_TIDE_performance_statistics_{N_FUTURE_OP}h.csv")

# --- 🚀 MASTER PLOTTER OUTER LOOP ACROSS ALL STATIONS IN REGISTRY ---
for idx_stn, row in df_config.iterrows():
    STATION_ID = str(row['station_id']).strip()
    if STATION_ID not in REGIONAL_CLUSTERS:
        continue
    STATION_NAME = str(row['station_name']).strip()

    gfs_cands = [
        os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{STATION_ID}_2026.csv'),
    ]
    df_g_path = next((f for f in gfs_cands if os.path.exists(f)), None)

    obs_cands = [
        os.path.join(MODELS_PATH, f'obs_twl_{STATION_ID}.csv'),
    ]
    found_obs_files = [f for f in obs_cands if os.path.exists(f)]

    if not (df_g_path and found_obs_files):
        print(f"⚠️ [Skipping] Data files absent for station: {STATION_NAME} ({STATION_ID})")
        continue

    print("\n" + "="*80)
    print(f"🎨 COMPILING MONTHLY REGIONAL 4-QUADRANT + TIDE SPATIAL PLOTS FOR: {STATION_NAME} ({STATION_ID})")
    print(f"   GFS Source: {df_g_path}")
    print(f"   OBS Source: {found_obs_files[0]}")
    print(f"   📅 Target Evaluation Window: {EVAL_START} to {EVAL_END}")
    print("="*80)

    try:
        df_g = load_and_parse(df_g_path)

        obs_dfs = [load_and_parse(f) for f in found_obs_files]
        df_o = pd.concat(obs_dfs, ignore_index=True).drop_duplicates(subset=['dt']).sort_values('dt')

        t_start = pd.to_datetime(EVAL_START)
        t_end = pd.to_datetime(EVAL_END)
        t_min = t_start - pd.Timedelta(hours=48)
        t_max = t_end + pd.Timedelta(hours=12)

        df_g = df_g[(df_g['dt'] >= t_min) & (df_g['dt'] <= t_max)].copy()
        df_o = df_o[(df_o['dt'] >= t_min) & (df_o['dt'] <= t_max)].copy()

        if df_o.empty:
            print(f"⚠️ [Skipping] Observation dataset has no records between {t_min} and {t_max} for {STATION_NAME}.")
            continue

        if df_g.empty:
            print(f"⚠️ [Skipping] GFS guidance dataset has no records between {t_min} and {t_max} for {STATION_NAME}.")
            continue

        df_o = df_o.set_index('dt').sort_index()

        cycle_col = 'cycle' if 'cycle' in df_g.columns else ('cycle_int' if 'cycle_int' in df_g.columns else None)
        if cycle_col:
            df_g_unique = df_g.sort_values(cycle_col).drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()
        else:
            df_g_unique = df_g.sort_values('dt').drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()

        data_full = pd.DataFrame(index=df_g_unique.index.union(df_o.index)).sort_index()

        # Extract Tide Baseline & Hourly Rate of Change
        if 'etss_tide' in df_g_unique.columns:
            data_full['etss_tide'] = pd.to_numeric(df_g_unique['etss_tide'], errors='coerce')
        else:
            t_col = [c for c in df_g_unique.columns if 'tide' in c.lower()]
            data_full['etss_tide'] = pd.to_numeric(df_g_unique[t_col[0]], errors='coerce') if t_col else 0.0

        data_full['etss_tide'] = data_full['etss_tide'].interpolate(method='linear', limit=3).ffill().bfill()
        data_full['etss_tide_diff'] = data_full['etss_tide'].diff().fillna(0.0)

        # Extract Water Level with fallback to obs_water_level inside GFS CSV
        wl_col = next((c for c in ['water_level', 'Water Level', 'wl', 'twl', 'v', 'observed_water_level_ft'] if c in df_o.columns), None)
        if wl_col:
            data_full['water_level'] = pd.to_numeric(df_o[wl_col], errors='coerce')
            data_full['water_level'] = data_full['water_level'].interpolate(method='linear', limit=12).ffill().bfill()
        elif 'obs_water_level' in df_g_unique.columns:
            data_full['water_level'] = pd.to_numeric(df_g_unique['obs_water_level'], errors='coerce')
            data_full['water_level'] = data_full['water_level'].interpolate(method='linear', limit=12).ffill().bfill()
        else:
            data_full['water_level'] = data_full['etss_tide']

        raw_surge = data_full['water_level'] - data_full['etss_tide']
        # 24h rolling mean to attenuate mixed diurnal/semi-diurnal tide leakage
        data_full['surge_residual'] = pd.Series(raw_surge.values, index=raw_surge.index)\
                                        .rolling(window=24, center=True, min_periods=1).mean().ffill().bfill()

        # Robust ETSS Water Level Baseline Resolver
        if 'etss_surge' in df_g_unique.columns:
            data_full['etss_surge_guidance'] = df_g_unique['etss_surge']
            data_full['etss_water_level_baseline'] = data_full['etss_surge_guidance'] + data_full['etss_tide']
        elif 'etss_water_level' in df_g_unique.columns:
            data_full['etss_water_level_baseline'] = df_g_unique['etss_water_level']
            data_full['etss_surge_guidance'] = data_full['etss_water_level_baseline'] - data_full['etss_tide']
        else:
            data_full['etss_surge_guidance'] = 0.0
            data_full['etss_water_level_baseline'] = data_full['etss_tide']

        data_full['etss_water_level_baseline'] = data_full['etss_water_level_baseline'].ffill().bfill()
        data_full['etss_surge_guidance'] = data_full['etss_surge_guidance'].ffill().bfill()

        # 4-Quadrant GFS Guidance Fields
        gfs_4quad_cols = []
        for c in range(1, N_OFFSHORE_CELLS + 1):
            gfs_4quad_cols.extend([f'c{c}_wind_u_gui', f'c{c}_wind_v_gui', f'c{c}_air_pressure_gui', f'c{c}_air_temp_gui'])
            u_col, v_col, ws_col = f'c{c}_wind_u_gui', f'c{c}_wind_v_gui', f'c{c}_wind_speed_gui'
            if u_col in df_g_unique.columns and v_col in df_g_unique.columns:
                df_g_unique[ws_col] = np.sqrt(pd.to_numeric(df_g_unique[u_col], errors='coerce')**2 + pd.to_numeric(df_g_unique[v_col], errors='coerce')**2)
                gfs_4quad_cols.append(ws_col)

        for col in gfs_4quad_cols:
            data_full[col] = pd.to_numeric(df_g_unique[col], errors='coerce') if col in df_g_unique.columns else 0.0

        for col in ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui']:
            if col in df_g_unique.columns:
                data_full[col] = pd.to_numeric(df_g_unique[col], errors='coerce')

        if 'wind_speed_gui' in data_full.columns and 'wind_direction_gui' in data_full.columns:
            data_full['wind_u_gui'], data_full['wind_v_gui'] = compute_wind_components(
                data_full['wind_speed_gui'].values, data_full['wind_direction_gui'].values
            )
        else:
            data_full['wind_u_gui'], data_full['wind_v_gui'] = 0.0, 0.0

        if 'wind_gust_gui' not in data_full.columns and 'wind_speed_gui' in data_full.columns:
            data_full['wind_gust_gui'] = data_full['wind_speed_gui'] * 1.2

        data_full = data_full.ffill().bfill()

    except Exception as e:
        print(f"❌ Plotter Data Alignment Exception on {STATION_NAME}: {e}")
        continue

    candidate_cycles = [t for t in df_g_unique.index if t_start <= t <= t_end and t in data_full.index]
    synoptic_cycles = [t for t in candidate_cycles if t.hour in [0, 6, 12, 18]]
    available_cycles = sorted(synoptic_cycles if len(synoptic_cycles) >= 4 else candidate_cycles[::6])

    if not available_cycles:
        print(f"⚠️ No forecast validation anchor cycles found within specified window for {STATION_NAME}.")
        continue

    cached_models, cached_scalers = load_regional_pure_pivot_models(STATION_ID, STATION_NAME)
    if not cached_models:
        print(f"❌ Missing regional Pure GFS + Tide assets for {STATION_NAME}. Skipping.")
        continue

    # Feature vector must match exact 27-feature order used during non-DOY + Tide model training (+ 1 target = 28)
    gfs_weather_vec = [
        'air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_gust_gui',
        'etss_surge_guidance', 'etss_tide', 'etss_tide_diff'
    ] + gfs_4quad_cols
    OBS_FEAT = gfs_weather_vec
    GUI_FEAT = gfs_weather_vec
    TARGET = 'surge_residual'

    def generate_isolated_engine_blend(fc_origin, op_gui, anchor_surge):
        local_horizons = {}
        block_variances = {24: [], 48: []}
        target_len = len(op_gui)

        for n_past in lookbacks:
            key = f"CHRONO_{n_past}"
            model = cached_models[key]
            sc_f = cached_scalers[f"{key}_f"]
            sc_t = cached_scalers[f"{key}_t"]

            raw_past = data_full.loc[:fc_origin][OBS_FEAT + [TARGET]].copy().values
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
                        val = np.zeros((len(gui_chunk), 1))
                    fut_met_list.append(val)

                fut_met_profile = np.hstack(fut_met_list)

                if chunk_len < N_FUTURE_TRAINED:
                    pad_size = N_FUTURE_TRAINED - chunk_len
                    pad_met = np.repeat(fut_met_profile[-1:, :], pad_size, axis=0)
                    fut_met_profile = np.vstack([fut_met_profile, pad_met])
                    fb_len = N_FUTURE_TRAINED
                else:
                    fb_len = N_FUTURE_TRAINED

                last_known_surge = past_block[-1, -1]
                fut_fb = np.linspace(last_known_surge, current_pivot, fb_len).reshape(-1, 1)

                op_fut_block = np.hstack([fut_met_profile, fut_fb])

                inp = np.ascontiguousarray(np.vstack([past_block, op_fut_block]))
                scaled_inp = sc_f.transform(inp) if hasattr(sc_f, 'transform') else inp

                inp_tensor = tf.convert_to_tensor(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, -1), dtype=tf.float32)
                raw_pred = model(inp_tensor, training=False).numpy().flatten()
                pred_delta = sc_t.inverse_transform(raw_pred.reshape(-1, 1)).flatten() if hasattr(sc_t, 'inverse_transform') else raw_pred

                relative_delta = pred_delta - pred_delta[0]
                relative_delta = np.clip(relative_delta, -1.5, 1.5)

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

        # Raise eps floor to 0.01 to eliminate weight flipping during calm regimes
        eps = 0.01
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
        return blended_track.flatten()

    # --- 5. Chronological Operational Stitching Ingestion Engine ---
    time_axis_records = []

    for fc_origin in available_cycles:
        try:
            op_gui = df_g_unique.loc[
                fc_origin + pd.Timedelta(hours=1) : fc_origin + pd.Timedelta(hours=N_FUTURE_OP)
            ].copy()

            if len(op_gui) < N_FUTURE_OP:
                continue
            target_len = len(op_gui)

            tide_vals = data_full.loc[op_gui.index, 'etss_tide'].values.flatten()[:target_len]

            obs_wl_series = data_full.loc[op_gui.index, 'water_level'].values[:target_len]
            obs_surge_filtered = data_full.loc[op_gui.index, 'surge_residual'].values[:target_len]

            anchor_surge = float(data_full.loc[fc_origin, 'surge_residual'])

            piv_track = generate_isolated_engine_blend(fc_origin, op_gui, anchor_surge).flatten()[:target_len]

            raw_etss_s = (data_full.loc[op_gui.index, 'etss_water_level_baseline'] - tide_vals).values.flatten()[:target_len]
            etss_surge_filtered = pd.Series(raw_etss_s).rolling(window=12, center=True, min_periods=1).mean().values

            time_axis_records.append(pd.DataFrame({
                'obs_tide': tide_vals,
                'obs_water_level': obs_wl_series,
                'obs_surge': obs_surge_filtered,
                'etss_surge': etss_surge_filtered,
                'etss_water_level': etss_surge_filtered + tide_vals,
                'pivot_surge': piv_track
            }, index=op_gui.index[:target_len]))
        except Exception as loop_err:
            continue

    if not time_axis_records:
        print(f"⚠️ Warning: Verification vector stack completely empty for {STATION_NAME}. No matching times.")
        continue

    master_plot_df = pd.concat(time_axis_records).groupby(level=0).mean()
    master_plot_df['year_month'] = master_plot_df.index.to_period('M')

    # 6h rolling mean preserves sub-daily tide-surge modulation in TWL while smoothing rollout transitions
    master_plot_df['pivot_surge'] = master_plot_df['pivot_surge'].rolling(window=6, center=True, min_periods=1).mean()
    master_plot_df['pivot_wl'] = master_plot_df['pivot_surge'] + master_plot_df['obs_tide']

    master_plot_df['obs_water_level'] = master_plot_df['obs_water_level'].ffill().bfill()
    master_plot_df['obs_surge'] = master_plot_df['obs_surge'].ffill().bfill()

    # --- 6. Monthly Dual-Panel Plot Generating Layer ---
    for current_month in master_plot_df['year_month'].unique():
        required_cols = ['obs_water_level', 'pivot_surge', 'etss_surge', 'obs_surge', 'pivot_wl']
        m_slice = master_plot_df[master_plot_df['year_month'] == current_month].dropna(subset=required_cols)

        if len(m_slice) < 1:
            print(f"    ⚠️ [Skipping Month {current_month}] Zero valid records available after dropping NaN rows.")
            continue

        month_str = current_month.strftime('%B %Y')
        cluster_tag = REGIONAL_CLUSTERS.get(STATION_ID, "NONE")

        obs_surge = m_slice['obs_surge'].values
        piv_surge = m_slice['pivot_surge'].values
        etss_surge = m_slice['etss_surge'].values

        rmse_piv_s = np.sqrt(np.mean((piv_surge - obs_surge) ** 2))
        rmse_etss_s = np.sqrt(np.mean((etss_surge - obs_surge) ** 2))
        bias_piv_s = np.mean(piv_surge - obs_surge)
        bias_etss_s = np.mean(etss_surge - obs_surge)

        obs_wl = m_slice['obs_water_level'].values
        piv_wl = m_slice['pivot_wl'].values
        etss_wl = m_slice['etss_water_level'].values

        rmse_piv_w = np.sqrt(np.mean((piv_wl - obs_wl) ** 2))
        rmse_etss_w = np.sqrt(np.mean((etss_wl - obs_wl) ** 2))

        bias_piv_w = np.mean(piv_wl - obs_wl)
        bias_etss_w = np.mean(etss_wl - obs_wl)

        skill_wl = ((rmse_etss_w - rmse_piv_w) / rmse_etss_w) * 100.0 if rmse_etss_w > 0 else 0.0
        skill_surge = ((rmse_etss_s - rmse_piv_s) / rmse_etss_s) * 100.0 if rmse_etss_s > 0 else 0.0

        summary_stats_list.append({
            "station_id": STATION_ID,
            "station_name": STATION_NAME,
            "cluster": cluster_tag,
            "month_year": current_month.strftime('%m/%Y'),
            "surrogate_mode": "REGIONAL-PURE-GFS-TIDE-PIVOT",
            "scaler": TARGET_SCALER,
            "ai_wl_rmse_ft": round(rmse_piv_w, 3),
            "ai_wl_bias_ft": round(bias_piv_w, 3),
            "etss_wl_rmse_ft": round(rmse_etss_w, 3),
            "etss_wl_bias_ft": round(bias_etss_w, 3),
            "ai_surge_rmse_ft": round(rmse_piv_s, 3),
            "ai_surge_bias_ft": round(bias_piv_s, 3),
            "etss_surge_rmse_ft": round(rmse_etss_s, 3),
            "etss_surge_bias_ft": round(bias_etss_s, 3),
            "ai_wl_error_reduction_pct": round(skill_wl, 1),
            "ai_surge_error_reduction_pct": round(skill_surge, 1)
        })

        fig = plt.figure(figsize=(21, 12))
        gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[0.75, 0.25], hspace=0.15, wspace=0.08)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax_stat = fig.add_subplot(gs[:, 1])
        ax_stat.axis('off')

        # --- PANEL 1: Total Water Level Space ---
        ax1.plot(m_slice.index, m_slice['obs_water_level'], label='Observed Total Water (Truth)', color='#2c3e50', linewidth=2.5)
        ax1.plot(m_slice.index, m_slice['pivot_wl'], label='Regional Pure GFS+Tide Pivot Rolling TWL', color='#7570b3', linewidth=2.0)
        ax1.plot(m_slice.index, m_slice['etss_water_level'], label='ETSS Physics Baseline WL', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)
        ax1.plot(m_slice.index, m_slice['obs_tide'], label='Astronomical Tide Baseline', color='#7f8c8d', linestyle='-', linewidth=1.0, alpha=0.3)

        ax1.set_title(f"Regional Pure GFS+Tide Weather Pivot Suite ({cluster_tag} - {N_FUTURE_OP}h Horizon): {STATION_NAME} ({month_str})", fontsize=13, fontweight='bold', pad=12)
        ax1.set_ylabel("Total Water Level Scale (ft)", fontsize=10, labelpad=10)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
        ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

        # --- PANEL 2: Cleaned Meteorological Surge Residual Space ---
        ax2.plot(m_slice.index, m_slice['obs_surge'], label='Pristine Observed Surge (Filtered)', color='#2c3e50', linewidth=2.5)
        ax2.plot(m_slice.index, m_slice['pivot_surge'], label='Regional Pure GFS+Tide Pivot Rolling Surge', color='#7570b3', linewidth=2.0)
        ax2.plot(m_slice.index, m_slice['etss_surge'], label='ETSS Physics Baseline Model', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)

        ax2.set_xlabel("Timeline Partition (Hourly Increments)", fontsize=10, labelpad=10)
        ax2.set_ylabel("Surge Elevation Scale (ft)", fontsize=10, labelpad=10)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
        ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

        fig.autofmt_xdate()

        # --- SIDE PANEL: STREAM CARD ---
        stat_box_text = (
            f"   REGIONAL STREAM CARD\n"
            f"=========================\n"
            f"  Month: {current_month.strftime('%m/%Y')}\n"
            f"  Station: {STATION_NAME} ({STATION_ID})\n"
            f"  Cluster: {cluster_tag}\n"
            f"  Architecture: Pure GFS+Tide Pivot\n"
            f"  Forced Scaler: {TARGET_SCALER}\n\n"
            f"  [1] TOTAL WATER LEVEL (WL)\n"
            f"  -------------------------\n"
            f"  * REGIONAL AI {N_FUTURE_OP}H ROLLOUT:\n"
            f"    > RMSE : {rmse_piv_w:.3f} ft\n"
            f"    > Bias : {bias_piv_w:+.3f} ft\n"
            f"  * ETSS PHYSICS BASELINE:\n"
            f"    > RMSE : {rmse_etss_w:.3f} ft\n"
            f"    > Bias : {bias_etss_w:+.3f} ft\n\n"
            f"  [2] PURE SURGE RESIDUAL\n"
            f"  -------------------------\n"
            f"  * REGIONAL AI {N_FUTURE_OP}H ROLLOUT:\n"
            f"    > RMSE : {rmse_piv_s:.3f} ft\n"
            f"    > Bias : {bias_piv_s:+.3f} ft\n"
            f"  * ETSS PHYSICS BASELINE:\n"
            f"    > RMSE : {rmse_etss_s:.3f} ft\n"
            f"    > Bias : {bias_etss_s:+.3f} ft\n\n"
            f"  OPERATIONAL PERFORMANCE\n"
            f"  -------------------------\n"
            f"  AI WL Error Reduction:\n"
            f"    * {skill_wl:+.1f}% Skill\n"
            f"  AI Surge Error Reduction:\n"
            f"    * {skill_surge:+.1f}% Skill"
        )

        ax_stat.text(0.05, 0.95, stat_box_text, transform=ax_stat.transAxes,
                     fontsize=9.5, verticalalignment='top', fontname='monospace',
                     bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#eceff1', linewidth=1.0))

        plot_out = os.path.join(OUTPUT_DIR, f"regional_gfs_weather_obs_wl_etss_tide_4quadrant_TIDE_chrono_rolling_forecast_{N_FUTURE_TRAINED}h_{STATION_ID}_{current_month.strftime('%Y_%m')}.png")
        plt.savefig(plot_out, dpi=300, bbox_inches='tight')
        print(f"    💾 Operational rolling chart successfully exported ➔ {os.path.basename(plot_out)}")

        plt.close('all')
        fig.clear()

    K.clear_session()
    gc.collect()

# --- 🌟 EXPORT ACCUMULATED PERFORMANCE STATISTICS TO CSV ---
if summary_stats_list:
    df_summary = pd.DataFrame(summary_stats_list)
    df_summary.to_csv(CSV_SUMMARY_PATH, index=False)
    print("\n" + "="*80)
    print(f"📊 SUMMARY PERFORMANCE CSV EXPORT COMPLETE ➔ {CSV_SUMMARY_PATH}")
    print("="*80)

print("\n🎉 Unified Multi-Cycle Prediction Engine Complete Across All Targets.")
