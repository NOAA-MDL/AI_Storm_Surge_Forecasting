#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Operational Real-Time 96-Hour Stream Dashboard Plotter.
Skips/breaks lines across time gaps > 2 hours to avoid flat interpolation bridges.
Supports custom start and end dates for zoom or historical review.
'''

import os, sys, glob, argparse, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

STREAMS_DIR = '/contrib/Mamoudou.Ba/storm_surge/realtime_forecast_streams'
PLOTS_DIR = '/contrib/Mamoudou.Ba/storm_surge/verification_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot 96h stream cards")
    parser.add_argument('--station', type=str, default=None, help="Optional single station ID")
    parser.add_argument('--max_gap', type=float, default=2.0, help="Max gap in hours before breaking line (default: 2.0)")
    parser.add_argument('--start_date', type=str, default=None, help="Optional start date/time (e.g. '2026-02-17 00:00' or '2026-02-17')")
    parser.add_argument('--end_date', type=str, default=None, help="Optional end date/time (e.g. '2026-02-23 00:00' or '2026-02-23')")
    return parser.parse_args()

def plot_with_gaps(ax, dt_series, val_series, max_gap_hours=2.0, **plot_kwargs):
    """
    Plots lines but breaks them with NaNs whenever the gap between
    consecutive timestamps exceeds max_gap_hours.
    """
    if len(dt_series) == 0:
        return
    df = pd.DataFrame({'dt': pd.to_datetime(dt_series), 'val': val_series}).sort_values('dt').dropna(subset=['val'])
    if df.empty:
        return

    # Identify time deltas in hours
    diffs = df['dt'].diff().dt.total_seconds() / 3600.0
    gap_indices = np.where(diffs > max_gap_hours)[0]

    if len(gap_indices) == 0:
        ax.plot(df['dt'], df['val'], **plot_kwargs)
    else:
        # Split into distinct continuous segments
        start_idx = 0
        label_used = False
        for g_idx in gap_indices:
            seg = df.iloc[start_idx:g_idx]
            if not seg.empty:
                kwargs = dict(plot_kwargs)
                if label_used and 'label' in kwargs:
                    kwargs.pop('label')
                ax.plot(seg['dt'], seg['val'], **kwargs)
                label_used = True
            start_idx = g_idx

        # Last segment
        seg = df.iloc[start_idx:]
        if not seg.empty:
            kwargs = dict(plot_kwargs)
            if label_used and 'label' in kwargs:
                kwargs.pop('label')
            ax.plot(seg['dt'], seg['val'], **kwargs)

def plot_stream(csv_file, max_gap_hours=2.0, start_date=None, end_date=None):
    try:
        df = pd.read_csv(csv_file)
        df['dt'] = pd.to_datetime(df['dt'])
        df = df.sort_values('dt').reset_index(drop=True)
    except Exception as e:
        print(f"   [!] Failed to read {csv_file}: {e}")
        return False

    station_id = str(df['station_id'].iloc[0]).strip()
    station_name = str(df['station_name'].iloc[0]).strip() if 'station_name' in df.columns else station_id
    t0_str = str(df['forecast_origin'].iloc[0])
    t0 = pd.to_datetime(t0_str)

    # Determine plot window bounds
    t_start = pd.to_datetime(start_date) if start_date is not None else (t0 - pd.Timedelta(hours=48))
    t_end = pd.to_datetime(end_date) if end_date is not None else (t0 + pd.Timedelta(hours=96))

    # Filter data to plotting window
    df = df[(df['dt'] >= t_start) & (df['dt'] <= t_end)].copy()
    if df.empty:
        print(f"   [!] No data for {station_id} between {t_start} and {t_end}")
        return False

    df_hindcast = df[df['lead_hour'] <= 0].copy()
    df_forecast = df[df['lead_hour'] >= 0].copy()

    # Detect if constant / frozen guidance was injected across gaps
    if not df_forecast.empty and 'etss_tide_ft' in df_forecast.columns:
        tide_series = df_forecast['etss_tide_ft']
        repeats = (tide_series.diff() == 0) & (tide_series.shift(1).diff() == 0) & (tide_series.shift(2).diff() == 0)
        df_forecast.loc[repeats, ['etss_tide_ft', 'etss_water_level_raw_ft', 'etss_surge_ft']] = np.nan

    has_obs_wl = ('observed_water_level_ft' in df.columns and df_hindcast['observed_water_level_ft'].notna().sum() > 0)
    has_obs_res = ('observed_surge_residual_ft' in df.columns and df_hindcast['observed_surge_residual_ft'].notna().sum() > 0)

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.0]})

    # --- TOP PANEL: Total Water Level (TWL) ---
    if has_obs_wl:
        plot_with_gaps(ax1, df_hindcast['dt'], df_hindcast['observed_water_level_ft'], max_gap_hours,
                       color='black', linewidth=2.4, marker='o', markersize=3.5, label='Observed Water Level (Hindcast)')

    if not df_hindcast.empty and 'etss_tide_ft' in df_hindcast.columns:
        plot_with_gaps(ax1, df_hindcast['dt'], df_hindcast['etss_tide_ft'], max_gap_hours,
                       color='#7570b3', linestyle=':', linewidth=1.4, alpha=0.7)

    if not df_forecast.empty:
        if 'etss_water_level_raw_ft' in df_forecast.columns:
            plot_with_gaps(ax1, df_forecast['dt'], df_forecast['etss_water_level_raw_ft'], max_gap_hours,
                           color='#d95f02', linestyle='--', linewidth=1.8, label='ETSS Guidance TWL (Raw)')
        if 'etss_corrected_twl_ft' in df_forecast.columns:
            plot_with_gaps(ax1, df_forecast['dt'], df_forecast['etss_corrected_twl_ft'], max_gap_hours,
                           color='#1b9e77', linestyle='-', linewidth=2.4, label='4-Quadrant GRU Corrected TWL')
        if 'etss_tide_ft' in df_forecast.columns:
            plot_with_gaps(ax1, df_forecast['dt'], df_forecast['etss_tide_ft'], max_gap_hours,
                           color='#7570b3', linestyle=':', linewidth=1.4, alpha=0.85, label='Astronomical Tide')

    # Draw t0 line and hindcast span if within the visible range
    if t_start <= t0 <= t_end:
        ax1.axvline(t0, color='#e41a1c', linestyle='--', linewidth=2.0, alpha=0.9, label='Forecast Origin ($t_0$)')
    
    h_span_start = max(t_start, t0 - pd.Timedelta(hours=48))
    h_span_end = min(t_end, t0)
    if h_span_start < h_span_end:
        ax1.axvspan(h_span_start, h_span_end, color='#f0f0f0', alpha=0.5, label='Hindcast Window (-48h)')

    ax1.set_ylabel('Water Level (ft, MLLW)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9, ncol=2)

    # --- BOTTOM PANEL: Surge, Residual, and Bias ---
    if has_obs_res:
        plot_with_gaps(ax2, df_hindcast['dt'], df_hindcast['observed_surge_residual_ft'], max_gap_hours,
                       color='black', linewidth=2.2, marker='o', markersize=3.5, label='Observed Residual (Hindcast)')

    if not df_forecast.empty:
        if 'etss_surge_ft' in df_forecast.columns:
            plot_with_gaps(ax2, df_forecast['dt'], df_forecast['etss_surge_ft'], max_gap_hours,
                           color='#386cb0', linestyle='--', linewidth=1.8, label='ETSS Surge Guidance')
        if 'etss_corrected_surge_ft' in df_forecast.columns:
            plot_with_gaps(ax2, df_forecast['dt'], df_forecast['etss_corrected_surge_ft'], max_gap_hours,
                           color='#1f78b4', linestyle='-', linewidth=2.4, label='Corrected Surge Forecast')
        if 'predicted_etss_bias_ft' in df_forecast.columns:
            plot_with_gaps(ax2, df_forecast['dt'], df_forecast['predicted_etss_bias_ft'], max_gap_hours,
                           color='#e31a1c', linestyle='-', linewidth=2.0, label='Predicted Model Bias')

    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
    if t_start <= t0 <= t_end:
        ax2.axvline(t0, color='#e41a1c', linestyle='--', linewidth=2.0, alpha=0.9)
    if h_span_start < h_span_end:
        ax2.axvspan(h_span_start, h_span_end, color='#f0f0f0', alpha=0.5)

    ax2.set_ylabel('Surge / Bias (ft)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9, ncol=2)

    ax2.set_xlim(t_start, t_end)

    # Dynamic x-axis tick spacing depending on window width
    window_days = (t_end - t_start).total_seconds() / 86400.0
    if window_days <= 3:
        ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%HZ'))
        ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[3, 9, 15, 21]))
    elif window_days <= 8:
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%HZ'))
        ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    else:
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

    fig.suptitle(f"Operational 96-Hour Real-Time Prospective Guidance — {station_name} ({station_id})\nForecast Cycle Origin: {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    t0_tag = t0.strftime('%Y%m%d_%H%M')
    # Append date range tag if custom dates were specified
    range_suffix = ""
    if start_date or end_date:
        s_tag = t_start.strftime('%Y%m%d')
        e_tag = t_end.strftime('%Y%m%d')
        range_suffix = f"_{s_tag}_to_{e_tag}"

    plot_fname = f"realtime_stream_forecast_plot_{station_id}_{t0_tag}{range_suffix}.png"
    out_plot_path = os.path.join(PLOTS_DIR, plot_fname)
    plt.savefig(out_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"   ✓ Plotted dashboard ({t_start.strftime('%Y-%m-%d %H:%M')} to {t_end.strftime('%Y-%m-%d %H:%M')}) -> {plot_fname}")
    return True

def main():
    args = parse_args()
    print("=" * 80)
    print("OPERATIONAL 96-HOUR STREAM VISUALIZER (GAP-SKIPPING & DATE FILTER ENABLED)")
    print(f"Streams Directory : {STREAMS_DIR}")
    print(f"Max Allowed Gap   : {args.max_gap} hours")
    if args.start_date or args.end_date:
        print(f"Custom Date Range : {args.start_date} --> {args.end_date}")
    print("=" * 80)

    pattern = os.path.join(STREAMS_DIR, "realtime_etss_bias_96h_forecast_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if args.station:
        csv_files = [f for f in csv_files if f"_{args.station.strip()}_" in os.path.basename(f)]

    for f in csv_files:
        plot_stream(f, max_gap_hours=args.max_gap, start_date=args.start_date, end_date=args.end_date)

    print("=" * 80)
    print("Plotting complete.")

if __name__ == '__main__':
    main()
