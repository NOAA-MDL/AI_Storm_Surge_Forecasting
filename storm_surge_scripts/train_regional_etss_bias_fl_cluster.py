#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
########################################################################
# Regional Multi-Station ETSS Bias Pivot Model Trainer                 #
# Target Cohort: Florida West Coast Cluster (FL_WEST_COAST)            #
# Ingests obs_twl_<id>.csv + 4-Quadrant GFS spatial guidance matrices  #
########################################################################

import os, sys, glob, argparse, warnings
warnings.filterwarnings('ignore')

# Prioritize system CUDA drivers over Conda binaries
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np, pandas as pd, tensorflow as tf, joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Prevent TF from allocating entire GPU VRAM immediately
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Dynamic GPU Memory Growth Engaged.")
    except RuntimeError as e:
        print(f"⚠️ GPU Growth config failed: {e}")

# Base model directory
MODELS_PATH = '../storm_surge/'
if not os.path.exists(MODELS_PATH):
    MODELS_PATH = '/contrib/Mamoudou.Ba/storm_surge/'

ALPHA = 0.50
N_FUTURE = 12       # 12-hour chunk operational layout
GAP_THRESHOLD = 24.0
N_OFFSHORE_CELLS = 4
LOOKBACKS = [24, 48]

# Target Regional Cohort
FL_REGIONAL_STATIONS = [
    '8725520',  # Fort Myers, FL
    '8726520',  # St. Petersburg, FL
    '8726607',  # Old Port Tampa, FL
    '8726724'   # Clearwater Beach, FL
]
REGION_TAG = "FL_WEST_COAST"

# Define 4-Quadrant offshore feature columns
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

# =====================================================================
# 🌟 CUSTOM AGGRESSIVE WEIGHTED LOSS FUNCTION
# =====================================================================
def create_aggressive_weighted_loss(alpha):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        weights = tf.pow(tf.abs(y_true), 3) + 1.0
        mse = K.mean(weights * K.square(y_true - y_pred), axis=-1)

        y_true_diff = y_true[:, 1:] - y_true[:, :-1]
        y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        shape_loss = K.mean(K.square(y_true_diff - y_pred_diff), axis=-1)

        return alpha * mse + (1.0 - alpha) * shape_loss
    return loss

# =====================================================================
# 🌟 OPERATIONAL DUAL-TRACK CYCLE INGESTION ENGINE (ETSS BIAS PIVOT)
# =====================================================================
def create_operational_hybrid_sequences(
    group_df, n_past, n_future, target_col, scaler_p,
    obs_met_cols, gfs_met_cols
):
    X, y = [], []
    total_steps = n_past + n_future

    for i in range(len(group_df) - total_steps + 1):
        window = group_df.iloc[i : i + total_steps]
        pivot_value = window.iloc[n_past - 1][target_col]

        # 🌪️ WEATHER TRACK: History uses Obs; Horizon uses GFS guidance
        past_met = window.iloc[:n_past][obs_met_cols].values
        future_met = window.iloc[n_past:][gfs_met_cols].values
        if np.isnan(past_met).any() or np.isnan(future_met).any():
            continue

        hybrid_met = np.vstack([past_met, future_met])

        # 🌊 WATER TRACK: Observed surge for history; pivot value for unobserved horizon
        past_surge = window.iloc[:n_past][target_col].values.reshape(-1, 1)
        future_surge_lkv = np.full((n_future, 1), pivot_value)
        hybrid_surge = np.vstack([past_surge, future_surge_lkv])

        # Form combined feature array & scale
        combined = np.hstack([hybrid_met, hybrid_surge])
        X.append(scaler_p.transform(combined))

        actual_future_surge = window.iloc[n_past:][target_col].values
        y.append(actual_future_surge - pivot_value)

    return np.array(X), np.array(y)

# =====================================================================
# 🌟 ROBUST DATA PREPROCESSING (ALIGNED WITH WORKING PURE PIVOT)
# =====================================================================
def load_and_preprocess_station(station_id):
    obs_file = os.path.join(MODELS_PATH, f'obs_twl_{station_id}.csv')
    if not os.path.exists(obs_file):
        obs_file = f'obs_twl_{station_id}.csv'

    if not os.path.exists(obs_file):
        print(f"Missing required observation file: obs_twl_{station_id}.csv")
        return None

    gfs_candidates = [
        os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{station_id}_2026.csv'),
        os.path.join(MODELS_PATH, f'training_gfs_etss_spatial_data_{station_id}_2021_2025.csv'),
        os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{station_id}.csv')
    ]
    gfs_candidates.extend(glob.glob(os.path.join(MODELS_PATH, f'*{station_id}*spatial*.csv')))
    gfs_file = next((f for f in gfs_candidates if os.path.exists(f)), None)

    if not gfs_file:
        print(f"Missing GFS spatial file for station: {station_id}")
        return None

    print(f"Syncing obs_twl_{station_id}.csv with {os.path.basename(gfs_file)}...")
    df_obs = pd.read_csv(obs_file)
    t_obs_col = next((c for c in ['timestamp', 'timestamp_gui', 'dt', 'date_time', 'Date Time', 't'] if c in df_obs.columns), None)
    if not t_obs_col:
        print(f"No datetime column found in {obs_file}")
        return None
    df_obs['dt'] = pd.to_datetime(df_obs[t_obs_col], errors='coerce', format='mixed').dt.round('h')
    df_obs = df_obs.dropna(subset=['dt']).set_index('dt').sort_index()

    df_gfs = pd.read_csv(gfs_file)
    t_gfs_col = next((c for c in ['timestamp_gui', 'timestamp', 'dt', 'datetime', 'time', 'valid_time'] if c in df_gfs.columns), None)
    if not t_gfs_col:
        print(f"No datetime column found in {gfs_file}")
        return None
    df_gfs['dt'] = pd.to_datetime(df_gfs[t_gfs_col], errors='coerce', format='mixed').dt.round('h')
    df_gfs = df_gfs.dropna(subset=['dt']).set_index('dt').sort_index()

    # Identify water level column in obs_twl
    wl_col = next((c for c in ['water_level', 'Water Level', 'wl', 'v', 'observed_water_level_ft'] if c in df_obs.columns), None)
    if not wl_col:
        print(f"No valid water level column in {obs_file}")
        return None

    # Inner join solely on water_level to prevent duplicate column collisions
    data = df_gfs.join(df_obs[[wl_col]].rename(columns={wl_col: 'water_level'}), how='inner').sort_index()
    if data.empty:
        print(f"Station {station_id} has zero overlapping timestamps between OBS and GFS.")
        return None

    data = data.replace([99999.0, 99999.000, 99999, -99999.0, -99999, 9999.0, -9999.0], np.nan)

    # Build 4-Quadrant GFS columns
    for c in range(1, N_OFFSHORE_CELLS + 1):
        u_col, v_col, ws_col = f'c{c}_wind_u_gui', f'c{c}_wind_v_gui', f'c{c}_wind_speed_gui'
        if u_col in data.columns and v_col in data.columns:
            data[ws_col] = np.sqrt(pd.to_numeric(data[u_col], errors='coerce')**2 + pd.to_numeric(data[v_col], errors='coerce')**2)

    for col in GFS_4QUAD_COLS:
        if col not in data.columns:
            data[col] = 0.0

    raw_numeric_cols = [
        'water_level', 'etss_tide', 'etss_surge', 'etss_water_level', 'cycle',
        'air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_gust_gui'
    ] + GFS_4QUAD_COLS

    for col in raw_numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    if 'wind_gust_gui' not in data.columns and 'wind_speed_gui' in data.columns:
        data['wind_gust_gui'] = data['wind_speed_gui'] * 1.2

    for col in ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_gust_gui']:
        if col in data.columns:
            data[col] = data[col].interpolate(method='linear', limit=3)

    # Impute local station weather using GFS guidance if missing from obs_twl
    for p in ['air_temp', 'air_pressure', 'wind_speed', 'wind_gust']:
        gui_p = f"{p}_gui"
        if p not in data.columns:
            data[p] = data[gui_p] if gui_p in data.columns else 0.0
        else:
            data[p] = data[p].fillna(data[gui_p])

    if 'etss_surge' in data.columns:
        data['etss_surge_guidance'] = data['etss_surge']
    elif 'etss_water_level' in data.columns and 'etss_tide' in data.columns:
        data['etss_surge_guidance'] = data['etss_water_level'] - data['etss_tide']
    else:
        data['etss_surge_guidance'] = 0.0

    t_cols = [c for c in data.columns if 'tide' in c.lower()]
    if t_cols and 'etss_tide' not in data.columns:
        data['etss_tide'] = data[t_cols[0]]

    data['water_level'] = data['water_level'].interpolate(method='linear', limit=12).ffill().bfill()
    data[TARGET_COL] = data['water_level'] - data['etss_tide']

    if 'cycle' in data.columns:
        active_cycle_mask = data['cycle'] == (data.index.hour // 6) * 6
        data = data[active_cycle_mask].copy()

    data = data.ffill().bfill()
    return data

# =====================================================================
# 🌟 MAIN REGIONAL ETSS BIAS TRAINING PROTOCOL
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Station Regional ETSS Bias GRU Training")
    parser.add_argument('--region_tag', type=str, default=REGION_TAG, help="Regional cluster tag (default: FL_WEST_COAST)")
    parser.add_argument('--epochs', type=int, default=350, help="Max training epochs (default: 350)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"🏭 REGIONAL MULTI-STATION ETSS BIAS TRAINING: {args.region_tag}")
    print(f"   Cluster Stations : {FL_REGIONAL_STATIONS}")
    print(f"   Target Variable  : {TARGET_COL} (Observed Water Level - Harmonic Tide)")
    print("=" * 80)

    station_datasets = {}
    for sid in FL_REGIONAL_STATIONS:
        df_station = load_and_preprocess_station(sid)
        if df_station is not None and not df_station.empty:
            station_datasets[sid] = df_station

    if not station_datasets:
        print("❌ No valid station datasets could be loaded. Exiting.")
        sys.exit(1)

    for n_past in LOOKBACKS:
        MODEL_NAME = f"{args.region_tag}_{N_FUTURE}_ETSS_BIAS_PIVOT_ROBUST_{n_past}past_CHRONO"
        model_save_path = os.path.join(MODELS_PATH, f'{MODEL_NAME}.weights.h5')
        print(f"\n🚀 Training Regional ETSS Bias Configuration Block: {MODEL_NAME}")

        # Combine clean rows from all stations to fit a unified regional RobustScaler
        clean_frames = []
        for sid, df in station_datasets.items():
            df['storm_block_id'] = (df.index.to_series().diff() > pd.Timedelta(hours=GAP_THRESHOLD)).cumsum()
            clean_sub = df.dropna(subset=[TARGET_COL] + OBS_WEATHER_COLS + GFS_GUIDANCE_COLS)
            if not clean_sub.empty:
                clean_frames.append(clean_sub)

        if not clean_frames:
            print(f"⚠️ No clean data across region for {n_past}h lookback. Skipping.")
            continue

        pooled_clean_data = pd.concat(clean_frames, axis=0)

        # Fit single regional feature scaler and target scaler
        scaler_p = RobustScaler().fit(pooled_clean_data[OBS_WEATHER_COLS + [TARGET_COL]].values)
        scaler_t = RobustScaler().fit(pooled_clean_data[[TARGET_COL]].values)

        # Extract sequences per station & pool together
        all_X, all_y = [], []
        for sid, df in station_datasets.items():
            storm_groups = df.groupby('storm_block_id')
            stn_X, stn_y = [], []
            for name, grp in storm_groups:
                if len(grp) >= (n_past + N_FUTURE):
                    X_g, y_g = create_operational_hybrid_sequences(
                        grp, n_past, N_FUTURE, TARGET_COL, scaler_p,
                        OBS_WEATHER_COLS, GFS_GUIDANCE_COLS
                    )
                    if X_g.size > 0:
                        stn_X.append(X_g)
                        stn_y.append(y_g)

            if stn_X:
                stn_X_cat = np.concatenate(stn_X, axis=0)
                stn_y_cat = np.concatenate(stn_y, axis=0)
                all_X.append(stn_X_cat)
                all_y.append(stn_y_cat)
                print(f"   Station {sid}: Extracted {len(stn_X_cat)} sequences")

        if not all_X:
            print(f"⚠️ Insufficient sequences across all stations for {n_past}h. Skipping.")
            continue

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        print(f"📊 Pooled Regional Dataset Size: {X.shape[0]} sequences across {len(station_datasets)} stations")

        # Split Train/Test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

        # Standard Sequential GRU Architecture matching single-station setup
        model = Sequential([
            GRU(128, activation='tanh', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
            Dropout(0.2),
            GRU(64, activation='tanh', return_sequences=False),
            Dropout(0.2),
            Dense(N_FUTURE)
        ])

        baseline_loss_fn = create_aggressive_weighted_loss(alpha=ALPHA)
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        model.compile(optimizer=opt, loss=baseline_loss_fn)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        model.save_weights(model_save_path)

        # Save regional scalers matching generator lookup patterns
        s_f_name = f"{args.region_tag}_scaler_etss_bias_pivot_robust_{n_past}h_CHRONO.joblib"
        s_t_name = f"{args.region_tag}_scaler_etss_bias_pivot_target_robust_{n_past}h_CHRONO.joblib"

        joblib.dump(scaler_p, os.path.join(MODELS_PATH, s_f_name))
        joblib.dump(scaler_t, os.path.join(MODELS_PATH, s_t_name))

        with open(os.path.join(MODELS_PATH, f'{MODEL_NAME}.json'), "w") as f:
            f.write(model.to_json())

        print(f"✅ Exported Regional ETSS Bias Model and Scalers for {args.region_tag} ({n_past}h lookback)")
        K.clear_session()

    print("\n🏁 Master Regional ETSS Bias Training Protocol Completed Cleanly.")

if __name__ == '__main__':
    main()
