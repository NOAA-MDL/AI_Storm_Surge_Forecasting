#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Developed with the assistance of the Gemini AI code assistant.
Operational ETSS Bias Correction Validation Engine (Custom Evaluation Window).
Evaluates Pure ETSS GRU Stitched CHRONO Models targeting "etss_bias" (= OB - ETSS_WL_raw).
Matches 12-hour rolling pivot strategy across all cycles.
Dynamically maps stations from ../storm_surge/stations_config.csv.
Full Sentinel Data Cleaning Prior to Anchor Block Segmentation.
'''

import os, sys, gc, warnings
warnings.filterwarnings('ignore')

# 🌟 FORCE KERAS 2 DESERIALIZER & CPU EXECUTION
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

# Force headless Matplotlib execution BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATION_ID = sys.argv[1].strip() if len(sys.argv) > 1 else "8557380"

# 🌟 OPERATIONAL MODE: Pass "BIAS" or "RESID" as second argument
MODE = sys.argv[2].strip().upper() if len(sys.argv) > 2 else "BIAS"
if MODE not in ["RESID", "BIAS"]:
    print(f"⚠️ Unknown mode '{MODE}'. Defaulting to BIAS.")
    MODE = "BIAS"

# 🌟 TARGET SCALER SELECTION & EVALUATION WINDOW
TARGET_SCALER = "ROBUST"
EVAL_START = "2026-01-01 01:00:00"
EVAL_END = "2026-03-31 23:00:00"

MODELS_PATH = '../storm_surge/'
OUTPUT_DIR = '/contrib/Mamoudou.Ba/storm_surge/verification_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FUTURE_TRAINED = 12
N_FUTURE = 12

# 🌟 DYNAMICALLY LOAD STATION CONFIGURATION MATRIX
CONFIG_FILE = 'stations_config.csv'
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = os.path.join(MODELS_PATH, 'stations_config.csv')

STN_MAP = {}
if os.path.exists(CONFIG_FILE):
    df_cfg = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})
    for _, r in df_cfg.iterrows():
        st_id = str(r['station_id']).strip()
        st_name = str(r['station_name']).strip().replace('"', '').replace(',', '').replace('.', '').replace(' ', '_')
        STN_MAP[st_id] = st_name

STATION_NAME = STN_MAP.get(STATION_ID, f"Station_{STATION_ID}")
print(f"📍 Station Identifier Parsed: ID={STATION_ID} | Resolved Name={STATION_NAME}")
print(f"🗓 Evaluation Window Configured: {EVAL_START} -> {EVAL_END}")

def load_and_parse(file_path):
    """Loads CSV, cleans sentinel values, and normalizes space-formatted timestamps ("YYYY-MM-DD HH MM SS" -> "YYYY-MM-DD HH:MM:SS")."""
    df = pd.read_csv(file_path)

    # 🌟 CLEAN SENTINELS IMMEDIATELY ON LOAD
    sentinels = [9999.0, 9999.00, 99999.0, 99999.000, -9999.0, -99999.0]
    df = df.replace(sentinels, np.nan)

    t_col = 'timestamp_gui' if 'timestamp_gui' in df.columns else ('timestamp' if 'timestamp' in df.columns else df.columns[0])

    # Normalizer for space-formatted timestamps
    df[t_col] = df[t_col].astype(str).str.replace(r'(\d{2}) (\d{2}) (\d{2})$', r'\1:\2:\3', regex=True)
    df['dt'] = pd.to_datetime(df[t_col], errors='coerce', format='mixed')
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_localize(None)
    df['dt'] = df['dt'].dt.round('h')
    return df.dropna(subset=['dt'])

# --- 1. Load, Standardize, and Structure Datasets ---
try:
    print(f"🔄 Loading Master Datasets for {STATION_NAME} Ground Truth Sync...")

    # 1. Search for Test GFS Guidance Data
    df_g_path = os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{STATION_ID}_2026.csv')
    if not os.path.exists(df_g_path):
        df_g_path = os.path.join(MODELS_PATH, f'test_gfs_etss_data_{STATION_ID}_2026.csv')
    if not os.path.exists(df_g_path):
        df_g_path = f'test_gfs_etss_spatial_data_{STATION_ID}_2026.csv'

    # 2. Universal Search for Station Observation Data
    df_o_path = os.path.join(MODELS_PATH, f'obs_training_{STATION_ID}.csv')
    if not os.path.exists(df_o_path):
        df_o_path = os.path.join(MODELS_PATH, f'obs_{STATION_ID}_wl_only.csv')
    if not os.path.exists(df_o_path):
        possible_obs = [
            os.path.join(MODELS_PATH, f) for f in os.listdir(MODELS_PATH)
            if f.startswith(f'obs_{STATION_ID}') or f.startswith(f'obs_training_{STATION_ID}') or f.startswith(f'training_obs_{STATION_ID}')
        ]
        if possible_obs:
            df_o_path = possible_obs[0]

    print(f"   > Using GFS Guidance File: {df_g_path}")
    print(f"   > Using Observation File:   {df_o_path}")

    df_g = load_and_parse(df_g_path)
    df_o = load_and_parse(df_o_path)

    df_o = df_o.drop_duplicates(subset=['dt']).set_index('dt').sort_index()
    df_g['cycle_int'] = pd.to_numeric(df_g['cycle'], errors='coerce').fillna(0).astype(int)

    # Clean numeric columns in df_g via interpolation before block segmentation
    for col in ['etss_surge', 'etss_tide', 'etss_bias', 'BIAS']:
        if col in df_g.columns:
            df_g[col] = pd.to_numeric(df_g[col], errors='coerce').interpolate(method='linear').ffill().bfill()

    # 🌟 ROBUST ANCHOR MINING ON CLEANED DF_G
    df_g_work = df_g.copy().sort_values(['cycle_int', 'dt'])
    df_g_work['run_block'] = ((df_g_work['dt'].diff() != pd.Timedelta(hours=1)) |
                              (df_g_work['cycle_int'].diff() != 0)).cumsum()

    dt_start = pd.to_datetime(EVAL_START)
    dt_end = pd.to_datetime(EVAL_END)

    valid_anchors = []
    for b_id, block in df_g_work.groupby('run_block'):
        block_sorted = block.sort_values('dt')
        if len(block_sorted) >= N_FUTURE:
            for start_i in range(0, len(block_sorted) - N_FUTURE + 1, 6):
                c_code = block_sorted.iloc[start_i]['cycle_int']
                fc_dt = block_sorted.iloc[start_i]['dt']
                # FILTER ANCHORS WITHIN EVALUATION WINDOW
                if dt_start <= fc_dt <= dt_end:
                    valid_anchors.append((b_id, start_i, c_code, fc_dt))

    # Sort valid anchors chronologically by forecast timestamp
    valid_anchors = sorted(valid_anchors, key=lambda x: x[3])
    print(f"📅 Tracked {len(valid_anchors)} validation anchors within requested window ({EVAL_START} to {EVAL_END}).")

    # Deduplicate GFS timestamps chronologically for data_full alignment
    df_g_unique = df_g.sort_values(['dt', 'cycle_int']).drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()

    # Create unified database matrix shell
    data_full = pd.DataFrame(index=df_g_unique.index.union(df_o.index)).sort_index()

    # Ingest Observation Meteorology Features if present
    for col in ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust']:
        if col in df_o.columns:
            data_full[col] = pd.to_numeric(df_o[col], errors='coerce')

    # Tide Baseline Extraction
    if 'etss_tide' in df_g_unique.columns:
        data_full['etss_tide'] = df_g_unique['etss_tide']
    else:
        t_col = [c for c in df_g_unique.columns if 'tide' in c.lower()]
        data_full['etss_tide'] = df_g_unique[t_col[0]] if t_col else 0.0

    # Surge Extraction
    if 'etss_surge' in df_g_unique.columns:
        data_full['etss_surge'] = df_g_unique['etss_surge']
    elif 'SURGE' in df_g_unique.columns:
        data_full['etss_surge'] = df_g_unique['SURGE']
    else:
        data_full['etss_surge'] = 0.0

    # Water Level Ingestion
    if 'water_level' in df_o.columns:
        data_full['water_level'] = df_o['water_level'].interpolate(method='linear').ffill().bfill()
    else:
        data_full['water_level'] = np.nan
        data_full['water_level'] = data_full['water_level'].interpolate(method='linear').ffill().bfill()

    # Uncorrected Physics Baseline (ETSS_WL_raw = Tide + Surge)
    data_full['etss_water_level_raw'] = data_full['etss_tide'] + data_full['etss_surge']

    # Model Bias (BIAS = Obs WL - ETSS_WL_raw)
    if 'etss_bias' in df_g_unique.columns:
        data_full['etss_bias_gui'] = df_g_unique['etss_bias']
    elif 'BIAS' in df_g_unique.columns:
        data_full['etss_bias_gui'] = df_g_unique['BIAS']
    else:
        data_full['etss_bias_gui'] = data_full['water_level'] - data_full['etss_water_level_raw']

    raw_obs_surge = data_full['water_level'] - data_full['etss_tide']
    data_full['obs_surge_filtered'] = pd.Series(raw_obs_surge.values, index=raw_obs_surge.index)\
                                        .rolling(window=12, center=True, min_periods=1).mean().ffill().bfill()

    # Target Metric Definition
    if MODE == "BIAS":
        data_full['target_metric'] = data_full['etss_bias_gui']
    else:
        data_full['target_metric'] = data_full['obs_surge_filtered'] - data_full['etss_surge']

    data_full = data_full.interpolate(method='linear').ffill().bfill()
    print("✅ Hardened Chronological Data Alignment completed successfully.")
except Exception as e:
    print(f"❌ Plotter Data Mapping Alignment Error: {e}")
    sys.exit(1)

# --- 2. CACHED STITCHED MODEL STORAGE ---
print(f"🧠 Pre-loading Pure ETSS Stitched {MODE} CHRONO Models ({TARGET_SCALER} track) into static cache...")
cached_models = {}
cached_scalers = {}
lookbacks = [24, 48]
logics = ["PIVOT"]

for l_type in logics:
    for n_past in lookbacks:
        key = f"{l_type}_{n_past}_{TARGET_SCALER}"

        m_name = f"{STATION_NAME}_{N_FUTURE}_ETSS_{MODE}_PIVOT_ROBUST_{n_past}past_CHRONO"

        m_json_path = os.path.join(MODELS_PATH, f"{m_name}.json")
        m_h5_path = os.path.join(MODELS_PATH, f"{m_name}.weights.h5")
        if not os.path.exists(m_h5_path):
            m_h5_path = os.path.join(MODELS_PATH, f"{m_name}.h5")

        s_f_name = f"{STATION_NAME}_etss_{MODE.lower()}_pivot_f_{n_past}h_CHRONO.joblib"
        s_t_name = f"{STATION_NAME}_etss_{MODE.lower()}_pivot_t_{n_past}h_CHRONO.joblib"

        sf_path = os.path.join(MODELS_PATH, s_f_name)
        st_path = os.path.join(MODELS_PATH, s_t_name)

        try:
            if os.path.exists(m_json_path) and os.path.exists(m_h5_path) and os.path.exists(sf_path) and os.path.exists(st_path):
                with open(m_json_path, 'r') as f:
                    model = model_from_json(f.read())
                model.load_weights(m_h5_path)

                cached_models[key] = model
                cached_scalers[f"{key}_f"] = joblib.load(sf_path)
                cached_scalers[f"{key}_t"] = joblib.load(st_path)
                print(f"   > Loaded Stitched ETSS Bias Asset Matrix: {key} ({m_name})")
            else:
                print(f"⚠️ Warning: Missing model or scaler component files for {key}:")
                if not os.path.exists(m_json_path): print(f"       Missing JSON: {m_json_path}")
                if not os.path.exists(m_h5_path): print(f"       Missing Weights: {m_h5_path}")
                if not os.path.exists(sf_path): print(f"       Missing Feature Scaler: {sf_path}")
                if not os.path.exists(st_path): print(f"       Missing Target Scaler: {st_path}")
        except Exception as err:
            print(f"❌ Error loading assets for {key}: {err}")

print("✅ Global structure memory cache ready.")

# --- 3. Optimized Inference Engine ---
def generate_isolated_engine_blend(logic_type, fc_origin, op_gui, anchor_target):
    local_horizons = {}
    target_len = len(op_gui)

    for n_past in lookbacks:
        key = f"{logic_type.upper()}_{n_past}_{TARGET_SCALER}"

        if (key not in cached_models) or (f"{key}_f" not in cached_scalers) or (f"{key}_t" not in cached_scalers):
            local_horizons[n_past] = np.full(target_len, anchor_target)
            continue

        model = cached_models[key]
        sc_f = cached_scalers[f"{key}_f"]
        sc_t = cached_scalers[f"{key}_t"]

        # Feature 1: etss_surge
        raw_past_surge = data_full.loc[:fc_origin]['etss_surge'].copy().values.reshape(-1, 1)
        past_surge = raw_past_surge[-n_past:] if len(raw_past_surge) >= n_past else np.vstack([np.repeat(raw_past_surge[:1, :], n_past - len(raw_past_surge), axis=0), raw_past_surge])

        # Feature 2: target_metric (etss_bias)
        raw_past_target = data_full.loc[:fc_origin]['target_metric'].copy().values.reshape(-1, 1)
        past_target = raw_past_target[-n_past:] if len(raw_past_target) >= n_past else np.vstack([np.repeat(raw_past_target[:1, :], n_past - len(raw_past_target), axis=0), raw_past_target])

        past_block = np.hstack([past_surge, past_target])

        fut_etss_surge = op_gui['etss_surge'].copy().values.reshape(-1, 1) if 'etss_surge' in op_gui.columns else np.zeros((target_len, 1))
        fut_bias_pivot = np.full((target_len, 1), anchor_target)

        fut_sequence = np.hstack([fut_etss_surge, fut_bias_pivot])
        if len(fut_sequence) > N_FUTURE_TRAINED:
            fut_sequence = fut_sequence[:N_FUTURE_TRAINED]
        elif len(fut_sequence) < N_FUTURE_TRAINED:
            fut_sequence = np.vstack([fut_sequence, np.repeat(fut_sequence[-1:, :], N_FUTURE_TRAINED - len(fut_sequence), axis=0)])

        inp = np.ascontiguousarray(np.vstack([past_block, fut_sequence]))

        if hasattr(sc_f, 'transform'):
            scaled_inp = sc_f.transform(inp)
        elif TARGET_SCALER == "ROBUST" and hasattr(sc_f, 'center_') and hasattr(sc_f, 'scale_'):
            scaled_inp = (inp - np.atleast_1d(sc_f.center_)[0]) / np.atleast_1d(sc_f.scale_)[0]
        elif TARGET_SCALER == "MAXABS" and hasattr(sc_f, 'scale_'):
            scaled_inp = inp / np.atleast_1d(sc_f.scale_)[0]
        else:
            scaled_inp = (inp - np.mean(inp)) / (np.std(inp) + 1e-6)

        inp_tensor = tf.convert_to_tensor(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, -1), dtype=tf.float32)
        raw_pred = model(inp_tensor, training=False).numpy().flatten()
        aligned_pred = np.zeros(target_len)
        fill_len = min(len(raw_pred), target_len)
        aligned_pred[:fill_len] = raw_pred[:fill_len]

        if hasattr(sc_t, 'inverse_transform'):
            pred_final = sc_t.inverse_transform(aligned_pred.reshape(-1, 1)).flatten()
        elif TARGET_SCALER == "ROBUST" and hasattr(sc_t, 'center_') and hasattr(sc_t, 'scale_'):
            pred_final = (aligned_pred * np.atleast_1d(sc_t.scale_)[0]) + np.atleast_1d(sc_t.center_)[0]
        elif TARGET_SCALER == "MAXABS" and hasattr(sc_t, 'scale_'):
            pred_final = aligned_pred * np.atleast_1d(sc_t.scale_)[0]
        else:
            pred_final = aligned_pred

        local_horizons[n_past] = pred_final.flatten()[:target_len] + anchor_target

    timeline = np.arange(target_len)
    weight_24h = 1.0 / (1.0 + np.exp((timeline - 36) / 8))

    h24 = local_horizons.get(24, np.full(target_len, anchor_target))
    h48 = local_horizons.get(48, np.full(target_len, anchor_target))

    return ((h24.flatten()[:target_len] * weight_24h) + (h48.flatten()[:target_len] * (1.0 - weight_24h))).flatten()[:target_len]

# --- 4. Chronological Array Extraction Engine ---
print(f"📊 Running GRU array extraction sequentially across {len(valid_anchors)} chronological forecast anchors...")
time_axis_records = []

for idx, (b_id, start_i, c_code, fc_origin) in enumerate(valid_anchors):
    if idx % 30 == 0:
        print(f"   ⏳ Processing forecast block {b_id} anchor {idx}/{len(valid_anchors)} ({fc_origin})...")
        sys.stdout.flush()

    try:
        block = df_g_work[df_g_work['run_block'] == b_id].sort_values('dt')
        op_gui = block.iloc[start_i : start_i + N_FUTURE].copy()
        if len(op_gui) < N_FUTURE: continue

        op_gui = op_gui.set_index('dt')
        target_len = len(op_gui)

        # Nearest neighbor anchor lookup for 100% gap resistance
        if fc_origin in data_full.index:
            anchor_target = data_full.loc[fc_origin, 'target_metric']
        else:
            nearest_idx = data_full.index.get_indexer([fc_origin], method='nearest')[0]
            anchor_target = data_full.iloc[nearest_idx]['target_metric']

        if isinstance(anchor_target, pd.Series):
            anchor_target = anchor_target.iloc[0]

        piv_track = generate_isolated_engine_blend("PIVOT", fc_origin, op_gui, anchor_target).flatten()[:target_len]

        # Safe reindex with ffill/bfill for missing observation hours
        obs_slice = data_full.reindex(op_gui.index).ffill().bfill()

        etss_wl_vals = obs_slice['etss_water_level_raw'].values.flatten()[:target_len]
        etss_s_vals = obs_slice['etss_surge'].values.flatten()[:target_len]
        tide_vals = obs_slice['etss_tide'].values.flatten()[:target_len]

        recon_piv_wl = etss_wl_vals + piv_track
        recon_piv_surge = recon_piv_wl - tide_vals

        time_axis_records.append(pd.DataFrame({
            'obs_tide': tide_vals,
            'obs_water_level': obs_slice['water_level'].values.flatten()[:target_len],
            'obs_surge': obs_slice['obs_surge_filtered'].values.flatten()[:target_len],
            'etss_surge': etss_s_vals,
            'etss_water_level': etss_wl_vals,
            'pivot_surge': recon_piv_surge,
            'consensus_surge': recon_piv_surge,
            'pivot_wl': recon_piv_wl,
            'consensus_wl': recon_piv_wl
        }, index=op_gui.index[:target_len]))
    except Exception as inner_e:
        print(f"   ⚠️ Loop structural failure at step {idx}: {str(inner_e)}")
        sys.stdout.flush()
        continue

if not time_axis_records:
    print("❌ Error: Verification engine returned empty records for specified evaluation window. Cannot plot.")
    sys.exit(1)

print(f"✅ Extraction completed successfully. Merging records chronologically...")
master_plot_df = pd.concat(time_axis_records).groupby(level=0).mean().sort_index()

# 🌟 SLICE MASTER PLOT DATAFRAME EXACTLY TO EVALUATION WINDOW
master_plot_df = master_plot_df.loc[dt_start:dt_end].dropna().sort_index()

if len(master_plot_df) < 12:
    print("❌ Error: Insufficient extracted points within specified window to plot. Cannot proceed.")
    sys.exit(1)

# --- 5. Windowed Dual-Panel Plot + Side Panel Statistics Core ---
print(f"🎨 Rendering performance charts for custom window: {EVAL_START} to {EVAL_END}...")

obs_surge   = master_plot_df['obs_surge'].values
cons_surge  = master_plot_df['consensus_surge'].values
piv_surge   = master_plot_df['pivot_surge'].values
etss_surge  = master_plot_df['etss_surge'].values

rmse_cons_s = np.sqrt(np.mean((cons_surge - obs_surge) ** 2))
rmse_piv_s  = np.sqrt(np.mean((piv_surge - obs_surge) ** 2))
rmse_etss_s = np.sqrt(np.mean((etss_surge - obs_surge) ** 2))

obs_wl   = master_plot_df['obs_water_level'].values
cons_wl  = master_plot_df['consensus_wl'].values
piv_wl   = master_plot_df['pivot_wl'].values
etss_wl  = master_plot_df['etss_water_level'].values

rmse_cons_w = np.sqrt(np.mean((cons_wl - obs_wl) ** 2))
rmse_piv_w  = np.sqrt(np.mean((piv_wl - obs_wl) ** 2))
rmse_etss_w = np.sqrt(np.mean((etss_wl - obs_wl) ** 2))

bias_cons_w = np.mean(cons_wl - obs_wl)
bias_etss_w = np.mean(etss_wl - obs_wl)

skill_wl = ((rmse_etss_w - rmse_cons_w) / rmse_etss_w) * 100.0 if rmse_etss_w > 0 else 0.0

fig = plt.figure(figsize=(21, 12))

gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[0.75, 0.25], hspace=0.15, wspace=0.08)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax_stat = fig.add_subplot(gs[:, 1])
ax_stat.axis('off')

# --- PANEL 1: Total Water Level Space ---
ax1.plot(master_plot_df.index, master_plot_df['obs_water_level'], label='Observed Total Water (Truth)', color='#2c3e50', linewidth=2.5)
ax1.plot(master_plot_df.index, master_plot_df['consensus_wl'], label='AI Stitched Chrono Rolling Forecast', color='#d35400', linewidth=1.8)
ax1.plot(master_plot_df.index, master_plot_df['etss_water_level'], label='ETSS Physics Baseline WL', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)
ax1.plot(master_plot_df.index, master_plot_df['obs_tide'], label='Astronomical Tide Baseline', color='#7f8c8d', linestyle='-', linewidth=1.0, alpha=0.3)

ax1.set_title(
    f"Deep Learning Storm Surge Forecast Suite (ETSS Water Level Bias Correction): {STATION_NAME} [{dt_start.strftime('%Y-%m-%d %H:%M')} to {dt_end.strftime('%Y-%m-%d %H:%M')}]",
    fontsize=13,
    fontweight='bold',
    pad=12
)
ax1.set_ylabel("Total Water Level Scale (ft)", fontsize=10, labelpad=10)
ax1.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

# --- PANEL 2: Cleaned Meteorological Surge Residual Space ---
ax2.plot(master_plot_df.index, master_plot_df['obs_surge'], label='Pristine Observed Surge (Filtered)', color='#2c3e50', linewidth=2.5)
ax2.plot(master_plot_df.index, master_plot_df['consensus_surge'], label='AI Stitched Chrono Rolling Surge', color='#d35400', linewidth=1.8)
ax2.plot(master_plot_df.index, master_plot_df['etss_surge'], label='ETSS Physics Baseline Model', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)

ax2.set_xlabel("Timeline Partition (Hourly Increments)", fontsize=10, labelpad=10)
ax2.set_ylabel("Surge Elevation Scale (ft)", fontsize=10, labelpad=10)
ax2.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

fig.autofmt_xdate()

# --- SIDE PANEL: Validation Scorecard ---
stat_box_text = (
    f"    VALIDATION CARD\n"
    f"=========================\n"
    f"  Start: {dt_start.strftime('%Y-%m-%d %H:%M')}\n"
    f"  End  : {dt_end.strftime('%Y-%m-%d %H:%M')}\n"
    f"  Station Target: {STATION_ID}\n"
    f"  ETSS Dev Mode: {MODE}\n"
    f"  Forced Scaler: {TARGET_SCALER}\n\n"
    f"  [1] TOTAL WATER LEVEL (WL)\n"
    f"  -----------------------\n"
    f"  * AI STITCHED CONSENSUS:\n"
    f"    > RMSE : {rmse_cons_w:.3f} ft\n"
    f"    > Bias : {bias_cons_w:+.3f} ft\n"
    f"  * ETSS PHYSICS BASELINE:\n"
    f"    > RMSE : {rmse_etss_w:.3f} ft\n"
    f"    > Bias : {bias_etss_w:+.3f} ft\n\n"
    f"  [2] PURE SURGE RESIDUAL\n"
    f"  -----------------------\n"
    f"  * AI STITCHED CONSENSUS:\n"
    f"    > RMSE : {rmse_cons_s:.3f} ft\n"
    f"  * ETSS PHYSICS BASELINE:\n"
    f"    > RMSE : {rmse_etss_s:.3f} ft\n\n"
    f"  OPERATIONAL PERFORMANCE\n"
    f"  -----------------------\n"
    f"  AI WL Error Reduction:\n"
    f"    * {skill_wl:+.1f}% Improvement\n"
    f"      over Physics Baseline"
)

ax_stat.text(0.02, 0.98, stat_box_text, transform=ax_stat.transAxes,
             fontsize=9.0, verticalalignment='top', fontname='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#cfd8dc', linewidth=1.2))

start_tag = dt_start.strftime('%Y%m%d')
end_tag = dt_end.strftime('%Y%m%d')
plot_out = os.path.join(OUTPUT_DIR, f"etss_dev_{TARGET_SCALER}_{MODE.lower()}_wl_metrics_{STATION_ID}_{start_tag}_to_{end_tag}.png")

plt.savefig(plot_out, dpi=300, bbox_inches='tight')

plt.close(fig)
fig.clear()
ax1.clear()
ax2.clear()
ax_stat.clear()
del fig, ax1, ax2, ax_stat, master_plot_df
K.clear_session()
gc.collect()

print(f"💾 Summary chart compiled successfully: {plot_out}")
print("🎉 Complete error correction validation consensus dashboard rendered successfully.")
