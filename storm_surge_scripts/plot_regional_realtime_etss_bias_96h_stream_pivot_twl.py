#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Operational Real-Time ETSS Bias / Surge Residual Regional Stream Visualizer.
Supports 102-Hour prospective forecast windows for regional multi-station clusters.
Plots:
  - Top Panel: Astronomical Tide, Raw ETSS TWL, 4-Quadrant GRU Corrected TWL, Pure GFS Weather Pivot TWL
  - Bottom Panel: ETSS Surge Guidance, Corrected Surge Forecast, Predicted Model Bias, Pure GFS Weather Pivot Surge
Guarantees full C0 seamless junctions across t0 on uniform MLLW datums.
'''

import os, sys, glob, argparse, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

BASE_PATH = '/contrib/Mamoudou.Ba/storm_surge'
STREAMS_DIR = os.path.join(BASE_PATH, 'realtime_forecast_streams')
PLOTS_DIR = os.path.join(BASE_PATH, 'verification_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Regional cluster mapping
REGIONAL_CLUSTERS = {
    '8725520': 'FL_WEST_COAST',  # Fort Myers, FL
    '8726520': 'FL_WEST_COAST',  # St. Petersburg, FL
    '8726607': 'FL_WEST_COAST',  # Old Port Tampa, FL
    '8726724': 'FL_WEST_COAST',  # Clearwater Beach, FL
}

def parse_args():
    parser = argparse.ArgumentParser(description="Operational Real-Time Regional Stream Visualizer (102h)")
    parser.add_argument('--station', type=str, default=None, help="Optional station ID to plot (e.g. 8726520)")
    parser.add_argument('--file', type=str, default=None, help="Optional direct stream CSV path")
    parser.add_argument('--horizon_hours', type=int, default=102, help="Forecast horizon hours to display (default: 102)")
    parser.add_argument('--lookback_hours', type=int, default=24, help="Hindcast hours to display (default: 24)")
    parser.add_argument('--all_stations', action='store_true', help="Plot all stations found in streams directory")
    return parser.parse_args()

def plot_stream(csv_path, lookback_hours=24, horizon_hours=102):
    df = pd.read_csv(csv_path)
    if 'dt' not in df.columns:
        print(f"   [SKIP] Missing 'dt' in {csv_path}")
        return

    df['dt'] = pd.to_datetime(df['dt'])
    df = df.sort_values('dt').reset_index(drop=True)

    station_id = str(df['station_id'].iloc[0]).strip() if 'station_id' in df.columns else "Station"
    station_name = str(df['station_name'].iloc[0]).strip() if 'station_name' in df.columns else station_id
    region_tag = REGIONAL_CLUSTERS.get(station_id, "SINGLE_STATION")

    # Locate forecast origin t0
    t0_rows = df[df['lead_hour'] == 0]
    if not t0_rows.empty:
        t0 = t0_rows['dt'].iloc[0]
    else:
        pos_lead = df[df['lead_hour'] > 0]
        t0 = pos_lead['dt'].iloc[0] - pd.Timedelta(hours=1) if not pos_lead.empty else df['dt'].iloc[len(df)//2]

    t_start = t0 - pd.Timedelta(hours=lookback_hours)
    t_end = min(t0 + pd.Timedelta(hours=horizon_hours), df['dt'].max())

    # Limit to hindcast lookback and specified prospective horizon
    view_df = df[(df['dt'] >= t_start) & (df['dt'] <= t_end)].copy()
    if view_df.empty:
        print(f"   [SKIP] Empty data window for {station_id}")
        return

    # Column resolution for Pure GFS Weather Pivot
    pure_gfs_col = next((c for c in ['pure_gfs_surge_ft', 'gfs_weather_pivot_surge_ft', 'pivot_surge_forecast_ft'] if c in view_df.columns), None)
    has_pure_gfs = pure_gfs_col is not None and view_df.loc[view_df['lead_hour'] > 0, pure_gfs_col].notna().any()

    # Column resolution for ETSS Corrected / GRU Bias model
    corrected_surge_col = next((c for c in ['etss_corrected_surge_ft', 'corrected_surge_ft'] if c in view_df.columns), None)
    corrected_twl_col = next((c for c in ['etss_corrected_twl_ft', 'corrected_twl_ft'] if c in view_df.columns), None)
    bias_col = next((c for c in ['predicted_etss_bias_ft', 'etss_bias_ft'] if c in view_df.columns), None)

    # Split into Hindcast (<= 0) and Prospective Forecast (>= 0 for C0 continuity)
    h_df = view_df[view_df['lead_hour'] <= 0].copy()
    f_df = view_df[view_df['lead_hour'] >= 0].copy()

    # Check for valid hindcast observations
    has_obs_wl = 'observed_water_level_ft' in h_df.columns and h_df['observed_water_level_ft'].notna().sum() > 2
    has_obs_res = 'observed_surge_residual_ft' in h_df.columns and h_df['observed_surge_residual_ft'].notna().sum() > 2

    # Plot styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.0]})
    fig.patch.set_facecolor('white')

    # -------------------------------------------------------------
    # TOP PANEL: Total Water Level (TWL) vs Astronomical Tide (MLLW)
    # -------------------------------------------------------------
    ax1.set_facecolor('white')

    # Hindcast Window shading
    ax1.axvspan(t_start, t0, color='lightgray', alpha=0.15, label=f'Hindcast Window (-{lookback_hours}h)')
    ax1.axvline(t0, color='firebrick', linestyle='--', linewidth=1.8, label='Forecast Origin ($t_0$)')

    # Astronomical Tide
    if 'etss_tide_ft' in view_df.columns:
        ax1.plot(view_df['dt'], view_df['etss_tide_ft'], color='#a6b8c7', linestyle=':', linewidth=1.5, label='Astronomical Tide')

    # Hindcast Observations
    if has_obs_wl:
        ax1.plot(h_df['dt'], h_df['observed_water_level_ft'], color='black', linewidth=2.0, marker='o', markersize=3, label='Observed Water Level (Hindcast)')

    # Prospective Forecasts
    if 'etss_water_level_raw_ft' in f_df.columns:
        ax1.plot(f_df['dt'], f_df['etss_water_level_raw_ft'], color='#d95f02', linestyle='--', linewidth=1.6, label='ETSS Guidance TWL (Raw)')

    if corrected_twl_col and corrected_twl_col in f_df.columns and f_df[corrected_twl_col].notna().any():
        ax1.plot(f_df['dt'], f_df[corrected_twl_col], color='#1b9e77', linewidth=2.2, label='4-Quadrant GRU Corrected TWL')

    if has_pure_gfs:
        pure_twl_col = 'pure_gfs_twl_ft' if 'pure_gfs_twl_ft' in f_df.columns and f_df['pure_gfs_twl_ft'].notna().any() else None
        gfs_twl = f_df[pure_twl_col] if pure_twl_col else f_df[pure_gfs_col] + f_df['etss_tide_ft']
        ax1.plot(f_df['dt'], gfs_twl, color='#7570b3', linestyle='-.', linewidth=1.8, label='Pure GFS Weather Pivot TWL')

    ax1.set_ylabel('Water Level (ft, MLLW)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', ncol=3, frameon=True, framealpha=0.9, fontsize=9)
    ax1.tick_params(labelsize=10)

    # -------------------------------------------------------------
    # BOTTOM PANEL: Surge Residual / Model Bias
    # -------------------------------------------------------------
    ax2.set_facecolor('white')
    ax2.axvspan(t_start, t0, color='lightgray', alpha=0.15)
    ax2.axvline(t0, color='firebrick', linestyle='--', linewidth=1.8)

    # Hindcast Observed Surge Residual
    if has_obs_res:
        ax2.plot(h_df['dt'], h_df['observed_surge_residual_ft'], color='black', linewidth=2.0, marker='o', markersize=3, label='Observed Residual (Hindcast)')

    # Prospective Surge Forecasts & Bias
    if 'etss_surge_ft' in f_df.columns:
        ax2.plot(f_df['dt'], f_df['etss_surge_ft'], color='#2b5c8f', linestyle='--', linewidth=1.6, label='ETSS Surge Guidance')

    if corrected_surge_col and corrected_surge_col in f_df.columns and f_df[corrected_surge_col].notna().any():
        ax2.plot(f_df['dt'], f_df[corrected_surge_col], color='#1f78b4', linewidth=2.2, label='Corrected Surge Forecast')

    if bias_col and bias_col in f_df.columns and f_df[bias_col].notna().any():
        ax2.plot(f_df['dt'], f_df[bias_col], color='#e31a1c', linewidth=1.8, label='Predicted Model Bias')

    if has_pure_gfs:
        ax2.plot(f_df['dt'], f_df[pure_gfs_col], color='#7570b3', linestyle='-.', linewidth=1.8, label='Pure GFS Weather Pivot Surge')

    ax2.axhline(0.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
    ax2.set_ylabel('Surge / Bias (ft)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', ncol=4, frameon=True, framealpha=0.9, fontsize=9)
    ax2.tick_params(labelsize=10)

    # Format Datetime X-Axis
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n00Z'))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax2.set_xlim(t_start, t_end)

    actual_horizon = int((t_end - t0).total_seconds() / 3600.0)
    t0_str = t0.strftime('%Y-%m-%d %H:%M:00 UTC')
    fig.suptitle(f"Operational {actual_horizon}-Hour Real-Time Prospective Guidance — {station_name} ({station_id})\nCluster: {region_tag} | Forecast Cycle Origin: {t0_str}",
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()
    fig.subplots_adjust(top=0.91, hspace=0.12)

    t0_tag = t0.strftime('%Y%m%d_%H%M')
    out_png = os.path.join(PLOTS_DIR, f"realtime_regional_forecast_plot_{station_id}_{t0_tag}_{actual_horizon}h_h{lookback_hours}.png")
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"   [PLOT] Saved: {out_png}")

def find_latest_stream_file(station_id):
    """
    Finds the latest stream file matching realtime_gfs_etss_96h_forecast_regional_{station_id}_*.csv
    or fallback to realtime_etss_bias_96h_forecast_{station_id}_*.csv.
    """
    patterns = [
        os.path.join(STREAMS_DIR, f"realtime_etss_bias_96h_forecast_regional_{station_id}_*.csv"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))

    if not matches:
        return None
    return sorted(matches)[-1]

def main():
    args = parse_args()

    if args.file:
        plot_stream(args.file, args.lookback_hours, args.horizon_hours)
        return

    # Determine stations to plot
    if args.station:
        stations_to_plot = [args.station.strip()]
    elif args.all_stations:
        all_files = glob.glob(os.path.join(STREAMS_DIR, "*.csv"))
        sids = set()
        for f in all_files:
            fname = os.path.basename(f)
            parts = fname.replace('.csv', '').split('_')
            # Look for 7-digit station IDs
            for p in parts:
                if len(p) == 7 and p.isdigit():
                    sids.add(p)
        stations_to_plot = sorted(list(sids))
    else:
        # Default: target Florida West Coast cluster
        stations_to_plot = list(REGIONAL_CLUSTERS.keys())
        print(f"Plotting Florida West Coast Regional Cluster: {stations_to_plot}")

    for sid in stations_to_plot:
        csv_path = find_latest_stream_file(sid)
        if csv_path:
            print(f"Plotting {sid} from: {os.path.basename(csv_path)}")
            plot_stream(csv_path, args.lookback_hours, args.horizon_hours)
        else:
            print(f"[!] No stream CSV found for station {sid} in {STREAMS_DIR}")

if __name__ == '__main__':
    main()
