#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
########################################################################
# Operational 4-Quadrant GFS + ETSS Pure Guidance Weather Pivot Engine #
# Regional Multi-Station Training Protocol (DOY-Enhanced Model)        #
# Supports FL West Coast & Atlantic Hydrodynamic Shelf Clusters        #
########################################################################

import os, sys, glob, argparse, warnings
warnings.filterwarnings('ignore')

# Prioritize system CUDA drivers
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np, pandas as pd, tensorflow as tf, joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Prevent TF from taking all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Dynamic GPU Memory Growth Engaged.")
    except RuntimeError as e:
        print(f"GPU Growth config failed: {e}")

# Base model directory
MODELS_PATH = '../storm_surge/'
if not os.path.exists(MODELS_PATH):
    MODELS_PATH = '/contrib/Mamoudou.Ba/storm_surge/'

ALPHA = 0.50
N_FUTURE = 12       # 12-hour chunk rollout
GAP_THRESHOLD = 24.0
N_OFFSHORE_CELLS = 4
LOOKBACKS = [24, 48]

# Hydrodynamic & Bathymetric Regional Clusters
REGIONAL_CLUSTERS = {
    # Florida West Coast (Wide Shallow West Florida Shelf)
    '8725520': 'FL_WEST_COAST',          # Fort Myers, FL
    '8726520': 'FL_WEST_COAST',          # St Petersburg, FL
    '8726724': 'FL_WEST_COAST',          # Clearwater Beach, FL
    '8726607': 'FL_WEST_COAST',          # Old Port Tampa, FL

    # South Atlantic Bight (Broad Shallow Shelf)
    '8721604': 'SAB_GEORGIA_BIGHT',      # Trident Pier, FL
    '8720218': 'SAB_GEORGIA_BIGHT',      # Mayport, FL
    '8670870': 'SAB_GEORGIA_BIGHT',      # Fort Pulaski, GA
    '8665530': 'SAB_GEORGIA_BIGHT',      # Charleston, SC
    '8658120': 'SAB_GEORGIA_BIGHT',      # Wilmington, NC

    # Mid-Atlantic Bight (Open Sandy Shelf)
    '8651370': 'MAB_OPEN_COAST',         # Duck Pier, NC
    '8570283': 'MAB_OPEN_COAST',         # Ocean City, MD
    '8557380': 'MAB_OPEN_COAST',         # Lewes, DE
    '8534720': 'MAB_OPEN_COAST',         # Atlantic City, NJ
    '8531680': 'MAB_OPEN_COAST',         # Sandy Hook, NJ
    '8518750': 'MAB_OPEN_COAST',         # The Battery, NY

    # Mid-Atlantic Interior Estuaries (Chesapeake & Delaware Bays)
    '8638610': 'CHESAPEAKE_DELAWARE_BAY', # Sewells Point, VA
    '8575512': 'CHESAPEAKE_DELAWARE_BAY', # Annapolis, MD
    '8551910': 'CHESAPEAKE_DELAWARE_BAY', # Reedy Point, DE

    # Gulf of Maine (Deep Basin / Macro-Tidal)
    '8443970': 'GULF_OF_MAINE',          # Boston, MA
}

ATLANTIC_REGIONS = [
    'SAB_GEORGIA_BIGHT',
    'MAB_OPEN_COAST',
    'CHESAPEAKE_DELAWARE_BAY',
    'GULF_OF_MAINE'
]


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


def create_operational_hybrid_sequences(
    group_df, n_past, n_future, target_col, scaler_p,
    gfs_met_cols
):
    X, y = [], []
    total_steps = n_past + n_future

    for i in range(len(group_df) - total_steps + 1):
        window = group_df.iloc[i : i + total_steps]

        met_window = window[gfs_met_cols].values
        if np.isnan(met_window).any():
            continue

        pivot_value = window.iloc[n_past - 1][target_col]

        # Dynamic GFS weather + seasonal DOY sequence (27 features)
        hybrid_met = window[gfs_met_cols].values

        # Autoregressive surge track (1 target feature -> 28 total features)
        past_surge = window.iloc[:n_past][target_col].values.reshape(-1, 1)
        future_surge_lkv = np.full((n_future, 1), pivot_value)
        hybrid_surge = np.vstack([past_surge, future_surge_lkv])

        combined = np.hstack([hybrid_met, hybrid_surge])
        X.append(scaler_p.transform(combined))

        actual_future_surge = window.iloc[n_past:][target_col].values
        y.append(actual_future_surge - pivot_value)

    return np.array(X), np.array(y)


def load_and_preprocess_station(station_id):
    obs_candidates = [
        os.path.join(MODELS_PATH, f'obs_training_{station_id}.csv'),
        os.path.join(MODELS_PATH, f'obs_twl_{station_id}.csv'),
        f'obs_training_{station_id}.csv',
        f'obs_twl_{station_id}.csv'
    ]
    obs_file = next((f for f in obs_candidates if os.path.exists(f)), None)

    if not obs_file:
        print(f"Missing required observation file for station: {station_id}")
        return None, None

    # Prioritize multi-year training dataset (2021-2025) over 2026 test stream
    gfs_candidates = [
        os.path.join(MODELS_PATH, f'training_gfs_etss_spatial_data_{station_id}_2021_2025.csv'),
        f'training_gfs_etss_spatial_data_{station_id}_2021_2025.csv',
        os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{station_id}_2026.csv'),
        os.path.join(MODELS_PATH, f'test_gfs_etss_spatial_data_{station_id}.csv')
    ]
    gfs_candidates.extend(glob.glob(os.path.join(MODELS_PATH, f'*{station_id}*spatial*.csv')))
    gfs_file = next((f for f in gfs_candidates if os.path.exists(f)), None)

    if not gfs_file:
        print(f"Missing GFS spatial file for station: {station_id}")
        return None, None

    print(f"Syncing {os.path.basename(obs_file)} with {os.path.basename(gfs_file)}...")
    df_obs = pd.read_csv(obs_file)
    if len(df_obs.columns) == 2 and pd.notna(pd.to_datetime(df_obs.columns[0], errors='coerce')):
        df_obs = pd.read_csv(obs_file, header=None, names=['timestamp', 'water_level'])

    t_obs_col = next((c for c in ['timestamp', 'timestamp_gui', 'dt', 'date_time', 'Date Time', 't'] if c in df_obs.columns), None)
    if not t_obs_col:
        print(f"No datetime column found in {obs_file}")
        return None, None
    df_obs['dt'] = pd.to_datetime(df_obs[t_obs_col], errors='coerce', format='mixed').dt.round('h')
    df_obs = df_obs.dropna(subset=['dt']).set_index('dt').sort_index()

    df_gfs = pd.read_csv(gfs_file)
    t_gfs_col = next((c for c in ['timestamp_gui', 'timestamp', 'dt', 'datetime', 'time', 'valid_time'] if c in df_gfs.columns), None)
    if not t_gfs_col:
        print(f"No datetime column found in {gfs_file}")
        return None, None
    df_gfs['dt'] = pd.to_datetime(df_gfs[t_gfs_col], errors='coerce', format='mixed').dt.round('h')
    df_gfs = df_gfs.dropna(subset=['dt']).set_index('dt').sort_index()

    wl_col = next((c for c in ['water_level', 'Water Level', 'wl', 'twl', 'v', 'observed_water_level_ft'] if c in df_obs.columns), None)
    if not wl_col:
        print(f"No valid water level column in {obs_file}")
        return None, None

    # Inner join on matching hours
    data = df_gfs.join(df_obs[[wl_col]].rename(columns={wl_col: 'water_level'}), how='inner').sort_index()
    if data.empty:
        print(f"Station {station_id} has zero overlapping timestamps between OBS and GFS.")
        return None, None

    data = data.replace([99999.0, 99999.000, 99999, -99999.0, -99999, 9999.0, -9999.0], np.nan)

    # Build 4-Quadrant GFS columns (4 cells x 5 features = 20 columns)
    gfs_4quad_cols = []
    for c in range(1, N_OFFSHORE_CELLS + 1):
        gfs_4quad_cols.extend([f'c{c}_wind_u_gui', f'c{c}_wind_v_gui', f'c{c}_air_pressure_gui', f'c{c}_air_temp_gui'])
        u_col, v_col, ws_col = f'c{c}_wind_u_gui', f'c{c}_wind_v_gui', f'c{c}_wind_speed_gui'
        if u_col in data.columns and v_col in data.columns:
            data[ws_col] = np.sqrt(data[u_col]**2 + data[v_col]**2)
            gfs_4quad_cols.append(ws_col)

    for col in gfs_4quad_cols:
        if col not in data.columns:
            data[col] = 0.0

    raw_numeric_cols = [
        'water_level', 'etss_tide', 'etss_surge', 'etss_water_level', 'cycle',
        'air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_gust_gui'
    ] + gfs_4quad_cols

    for col in raw_numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    if 'wind_gust_gui' not in data.columns and 'wind_speed_gui' in data.columns:
        data['wind_gust_gui'] = data['wind_speed_gui'] * 1.2

    for col in ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_gust_gui']:
        if col in data.columns:
            data[col] = data[col].interpolate(method='linear', limit=3)

    if 'etss_surge' in data.columns:
        data['etss_surge_guidance'] = data['etss_surge']
    elif 'etss_water_level' in data.columns and 'etss_tide' in data.columns:
        data['etss_surge_guidance'] = data['etss_water_level'] - data['etss_tide']
    else:
        data['etss_surge_guidance'] = 0.0

    data['water_level'] = data['water_level'].interpolate(method='linear', limit=12).ffill().bfill()
    data['surge_residual'] = data['water_level'] - data['etss_tide']

    if 'cycle' in data.columns:
        active_cycle_mask = data['cycle'] == (data.index.hour // 6) * 6
        data = data[active_cycle_mask].copy()

    # Add cyclical Day-of-Year features to capture seasonal steric/current cycles
    doy = data.index.dayofyear
    data['doy_sin'] = np.sin(2.0 * np.pi * doy / 365.25)
    data['doy_cos'] = np.cos(2.0 * np.pi * doy / 365.25)

    data = data.ffill().bfill()
    gfs_guidance = [
        'air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_gust_gui',
        'etss_surge_guidance', 'doy_sin', 'doy_cos'
    ] + gfs_4quad_cols
    return data, gfs_guidance


def train_region(region_tag, epochs, batch_size):
    cluster_stations = [sid for sid, reg in REGIONAL_CLUSTERS.items() if reg == region_tag]
    if not cluster_stations:
        print(f"No stations configured for region_tag='{region_tag}'. Available: {sorted(set(REGIONAL_CLUSTERS.values()))}")
        return

    print("\n" + "=" * 80)
    print(f"MULTI-STATION REGIONAL PURE PIVOT + DOY TRAINING: {region_tag}")
    print(f"Cluster Stations ({len(cluster_stations)}): {cluster_stations}")
    print("=" * 80)

    station_datasets = {}
    gfs_guidance_cols = None

    for sid in cluster_stations:
        df_station, guidance_cols = load_and_preprocess_station(sid)
        if df_station is not None and not df_station.empty:
            station_datasets[sid] = df_station
            gfs_guidance_cols = guidance_cols

    if not station_datasets:
        print(f"No valid station datasets loaded for {region_tag}. Skipping.")
        return

    TARGET = 'surge_residual'

    for n_past in LOOKBACKS:
        MODEL_NAME = f"{region_tag}_DOY_{N_FUTURE}_GFS_WEATHER_4QUADRANT_GFS_WEATHER_OBS_ETSS_WL_PIVOT_ROBUST_{n_past}past_CHRONO"
        model_save_path = os.path.join(MODELS_PATH, f'{MODEL_NAME}.weights.h5')
        print(f"\nTraining Regional 4-Quadrant + DOY Configuration Block: {MODEL_NAME}")

        clean_frames = []
        for sid, df in station_datasets.items():
            df['storm_block_id'] = (df.index.to_series().diff() > pd.Timedelta(hours=GAP_THRESHOLD)).cumsum()
            clean_sub = df.dropna(subset=[TARGET] + gfs_guidance_cols)
            if not clean_sub.empty:
                clean_frames.append(clean_sub)

        if not clean_frames:
            print(f"No clean data across {region_tag} for {n_past}h lookback. Skipping.")
            continue

        pooled_clean_data = pd.concat(clean_frames, axis=0)

        scaler_p = RobustScaler().fit(pooled_clean_data[gfs_guidance_cols + [TARGET]].values)
        scaler_t = RobustScaler().fit(pooled_clean_data[[TARGET]].values)

        all_X, all_y = [], []
        for sid, df in station_datasets.items():
            storm_groups = df.groupby('storm_block_id')
            stn_X, stn_y = [], []
            for _, grp in storm_groups:
                if len(grp) >= (n_past + N_FUTURE):
                    X_g, y_g = create_operational_hybrid_sequences(
                        grp, n_past, N_FUTURE, TARGET, scaler_p, gfs_guidance_cols
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
            print(f"Insufficient sequences across all stations in {region_tag} for {n_past}h. Skipping.")
            continue

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        print(f"Pooled {region_tag} Dataset Size: {X.shape[0]} sequences | Input Shape: {X.shape[1:]} across {len(all_X)} stations")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

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
            EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
            ModelCheckpoint(model_save_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1)
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        model.save_weights(model_save_path)

        s_f_name = f"{region_tag}_doy_scaler_gfs_weather_4quadrant_gfs_weather_obs_etss_wl_pivot_robust_{n_past}h_CHRONO.joblib"
        s_t_name = f"{region_tag}_doy_scaler_gfs_weather_4quadrant_gfs_weather_obs_etss_wl_pivot_target_robust_{n_past}h_CHRONO.joblib"

        joblib.dump(scaler_p, os.path.join(MODELS_PATH, s_f_name))
        joblib.dump(scaler_t, os.path.join(MODELS_PATH, s_t_name))

        with open(os.path.join(MODELS_PATH, f'{MODEL_NAME}.json'), "w") as f:
            f.write(model.to_json())

        print(f"Exported Regional DOY Model and Scalers for {region_tag} ({n_past}h lookback)")
        K.clear_session()


def main():
    parser = argparse.ArgumentParser(description="Multi-Station Regional Pure Pivot + DOY GRU Training")
    parser.add_argument(
        '--region_tag',
        type=str,
        default='ALL_ATLANTIC',
        help="Target region cluster (SAB_GEORGIA_BIGHT, MAB_OPEN_COAST, CHESAPEAKE_DELAWARE_BAY, GULF_OF_MAINE, FL_WEST_COAST, or ALL_ATLANTIC)"
    )
    parser.add_argument('--epochs', type=int, default=350, help="Max training epochs (default: 350)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()

    target_regions = ATLANTIC_REGIONS if args.region_tag.upper() == 'ALL_ATLANTIC' else [args.region_tag.strip()]
    for reg in target_regions:
        train_region(reg, args.epochs, args.batch_size)

    print("\nRegional Stitched 4-Quadrant + DOY Training Completed Cleanly.")


if __name__ == '__main__':
    main()
