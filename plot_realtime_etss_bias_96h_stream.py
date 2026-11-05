#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Operational Real-Time 96-Hour Stream Dashboard Plotter.
- Skips/breaks lines across time gaps > 2 hours to avoid flat interpolation bridges.
- Automatically selects the latest forecast cycle per station.
- Plots lead-in hindcast observations to the left of t0 alongside the 96h forecast rollout.
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
    parser.add_argument('--start_date', type=str, default=None, help="Optional start date/time (e.g. '2026-10-20 00:00')")
    parser.add_argument('--end_date', type=str, default=None, help="Optional end date/time (e.g. '2026-10-26 00:00')")
    parser.add_argument('--hindcast_hours', type=float, default=24.0, help="Hours of hindcast observations to show before t0 (default: 24.0 / 1 day)")
    parser.add_argument('--last_96h', action='store_true', help="Plot 96h forecast with lead-in hindcast on the left")
    parser.add_argument('--all_cycles', action='store_true', help="Plot every cycle file found instead of just the latest")
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

    diffs = df['dt'].diff().dt.total_seconds() / 3600.0
    gap_indices = np.where(diffs > max_gap_hours)[0]

    if len(gap_indices) == 0:
        ax.plot(df['dt'], df['val'], **plot_kwargs)
    else:
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

        seg = df.iloc[start_idx:]
        if not seg.empty:
            kwargs = dict(plot_kwargs)
            if label_used and 'label' in kwargs:
                kwargs.pop('label')
            ax.plot(seg['dt'], seg['val'], **kwargs)

def plot_stream(csv_file, max_gap_hours=2.0, start_date=None, end_date=None, last_96h=False, hindcast_hours=24.0):
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
    if last_96h:
        # Start 12-24h before t0 so hindcast is visible on the left; end at t0 + 96h
        t_start = max(df['dt'].min(), t0 - pd.Timedelta(hours=hindcast_hours))
        t_end = min(df['dt'].max(), t0 + pd.Timedelta(hours=96))
    else:
        t_start = pd.to_datetime(start_date) if start_date is not None else (t0 - pd.Timedelta(hours=hindcast_hours))
        t_end = pd.to_datetime(end_date) if end_date is not None else (t0 + pd.Timedelta(hours=96))

    # Slice DataFrame to the window
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

    # Draw t0 line and hindcast span
    if t_start <= t0 <= t_end:
        ax1.axvline(t0, color='#e41a1c', linestyle='--', linewidth=2.0, alpha=0.9, label='Forecast Origin ($t_0$)')

    h_span_start = max(t_start, t0 - pd.Timedelta(hours=hindcast_hours))
    h_span_end = min(t_end, t0)
    if h_span_start < h_span_end:
        ax1.axvspan(h_span_start, h_span_end, color='#f0f0f0', alpha=0.5, label=f'Hindcast Window (-{int(hindcast_hours)}h)')

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

    # Formatting tick intervals
    window_days = (t_end - t_start).total_seconds() / 86400.0
    if window_days <= 5.5:
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%HZ'))
        ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    else:
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%HZ'))
        ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))

    fig.suptitle(f"Operational 96-Hour Real-Time Prospective Guidance — {station_name} ({station_id})\nForecast Cycle Origin: {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    t0_tag = t0.strftime('%Y%m%d_%H%M')
    range_suffix = ""
    if last_96h:
        range_suffix = f"_last96h_h{int(hindcast_hours)}"
    elif start_date or end_date:
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
    print("OPERATIONAL 96-HOUR STREAM VISUALIZER (GAP-SKIPPING & HINDCAST LEAD-IN)")
    print(f"Streams Directory : {STREAMS_DIR}")
    print(f"Max Allowed Gap   : {args.max_gap} hours")
    print(f"Lead-in Hindcast  : {args.hindcast_hours} hours")
    if args.last_96h:
        print(f"Time Window Mode  : Last 96 Hours + {args.hindcast_hours}h Hindcast Lead-in")
    elif args.start_date or args.end_date:
        print(f"Custom Date Range : {args.start_date} --> {args.end_date}")
    else:
        print(f"Time Window Mode  : -{args.hindcast_hours}h hindcast to +96h forecast")
    print("=" * 80)

    pattern = os.path.join(STREAMS_DIR, "realtime_etss_bias_96h_forecast_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"❌ No forecast files found matching {pattern}")
        return

    if args.station:
        csv_files = [f for f in csv_files if f"_{args.station.strip()}_" in os.path.basename(f)]

    if not csv_files:
        print(f"❌ No forecast files found for station {args.station}")
        return

    if not args.all_cycles:
        latest_per_station = {}
        for f in csv_files:
            base = os.path.basename(f).replace('.csv', '')
            parts = base.split('_')
            stn = parts[5] if len(parts) >= 6 else 'default'
            latest_per_station[stn] = f

        csv_files_to_plot = list(latest_per_station.values())
        print(f"Targeting latest operational cycle for {len(csv_files_to_plot)} station(s):")
        for f in csv_files_to_plot:
            print(f"  -> {os.path.basename(f)}")
        print("-" * 80)
    else:
        csv_files_to_plot = csv_files

    for f in csv_files_to_plot:
        plot_stream(
            f,
            max_gap_hours=args.max_gap,
            start_date=args.start_date,
            end_date=args.end_date,
            last_96h=args.last_96h,
            hindcast_hours=args.hindcast_hours
        )

    print("=" * 80)
    print("Plotting complete.")

if __name__ == '__main__':
    main()
