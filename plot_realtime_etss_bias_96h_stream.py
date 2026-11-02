#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Operational Real-Time 96-Hour ETSS Bias & Surge Forecast Stream Visualizer (Batch Mode).
Plots multi-variable operational time-series for all available station streams:
  - Panel 1: Total Water Level (ETSS Raw vs Corrected TWL vs Astronomical Tide)
  - Panel 2: Surge & Bias (ETSS Guidance Surge vs Corrected Surge vs Bias Shift)
  - Panel 3: Meteorology Guidance (Wind Speed, Gusts, Pressure)
'''

import os, sys, glob, argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

STREAMS_DIR = '/contrib/Mamoudou.Ba/storm_surge/realtime_forecast_streams'
PLOTS_DIR = '/contrib/Mamoudou.Ba/storm_surge/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot 96-Hour Real-Time ETSS Bias Stream Dashboard across all stations")
    parser.add_argument('--station', type=str, default=None, help="Optional single station ID to plot (default: plots ALL available streams)")
    parser.add_argument('--file', type=str, default=None, help="Direct path to a specific stream CSV file")
    return parser.parse_args()

def plot_single_stream(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'dt' not in df.columns:
            print(f"   [!] Skipping empty or invalid file: {csv_path}")
            return False

        df['dt'] = pd.to_datetime(df['dt'])
        df = df.sort_values('dt')

        station_id = str(df['station_id'].iloc[0])
        station_name = str(df['station_name'].iloc[0])
        forecast_origin = str(df['forecast_origin'].iloc[0])

        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 11), sharex=True, gridspec_kw={'height_ratios': [2, 1.8, 1.2]})

        # --- PANEL 1: TOTAL WATER LEVEL ---
        if 'etss_water_level_raw_ft' in df.columns:
            ax1.plot(df['dt'], df['etss_water_level_raw_ft'], color='#d95f02', linestyle='--', linewidth=1.8, label='ETSS Guidance TWL (Raw)')
        if 'etss_corrected_twl_ft' in df.columns:
            ax1.plot(df['dt'], df['etss_corrected_twl_ft'], color='#1b9e77', linewidth=2.5, label='4-Quadrant GRU Corrected TWL')
        if 'etss_tide_ft' in df.columns:
            ax1.plot(df['dt'], df['etss_tide_ft'], color='#7570b3', linestyle=':', linewidth=1.5, alpha=0.8, label='Astronomical Tide')
        ax1.set_ylabel('Water Level (ft, MLLW)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Operational 96-Hour Real-Time Prospective Guidance — {station_name} ({station_id})\nForecast Cycle Origin: {forecast_origin} UTC', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- PANEL 2: SURGE & MODEL BIAS ---
        if 'etss_surge_ft' in df.columns:
            ax2.plot(df['dt'], df['etss_surge_ft'], color='#386cb0', linestyle='--', linewidth=1.8, label='ETSS Surge Guidance')
        if 'etss_corrected_surge_ft' in df.columns:
            ax2.plot(df['dt'], df['etss_corrected_surge_ft'], color='#2b83ba', linewidth=2.4, label='Corrected Surge Forecast')
        if 'predicted_etss_bias_ft' in df.columns:
            ax2.plot(df['dt'], df['predicted_etss_bias_ft'], color='#e41a1c', linewidth=2.0, label='Predicted Model Bias')
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
        ax2.set_ylabel('Surge / Bias (ft)', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', frameon=True, framealpha=0.9)
        ax2.grid(True, linestyle='--', alpha=0.5)

        # --- PANEL 3: METEOROLOGY ---
        color_wind = '#252525'
        color_gust = '#cb181d'
        if 'wind_speed_kts' in df.columns and df['wind_speed_kts'].notna().sum() > 0:
            ax3.plot(df['dt'], df['wind_speed_kts'], color=color_wind, linewidth=1.6, label='Wind Speed (kts)')
        if 'wind_gust_kts' in df.columns and df['wind_gust_kts'].notna().sum() > 0:
            ax3.plot(df['dt'], df['wind_gust_kts'], color=color_gust, linestyle='--', linewidth=1.4, label='Wind Gust (kts)')
        ax3.set_ylabel('Wind (kts)', fontsize=11, fontweight='bold', color=color_wind)
        ax3.set_ylim(bottom=0)

        if 'air_pressure_hPa' in df.columns and df['air_pressure_hPa'].notna().sum() > 0:
            ax3_pres = ax3.twinx()
            ax3_pres.plot(df['dt'], df['air_pressure_hPa'], color='#7f2704', linestyle='-.', linewidth=1.3, alpha=0.7, label='Pressure (hPa)')
            ax3_pres.set_ylabel('Pressure (hPa)', fontsize=10, color='#7f2704')
            ax3_pres.grid(False)

        ax3.set_xlabel('Valid Forecast Time (UTC)', fontsize=11, fontweight='bold')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %HZ'))
        ax3.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax3.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
        ax3.legend(loc='upper left', frameon=True, framealpha=0.9)
        ax3.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        out_png = os.path.join(PLOTS_DIR, f"realtime_stream_dashboard_{station_id}.png")
        fig.savefig(out_png, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"   [DONE] Saved plot: {out_png}")
        return True
    except Exception as e:
        print(f"   [!] Failed plotting {csv_path}: {e}")
        return False

def main():
    args = parse_args()

    if args.file and os.path.exists(args.file):
        files_to_plot = [args.file]
    elif args.station:
        matches = sorted(glob.glob(os.path.join(STREAMS_DIR, f"realtime_etss_bias_96h_forecast_{args.station}_*.csv")))
        if not matches:
            print(f"Error: No stream files found for station {args.station} in {STREAMS_DIR}")
            sys.exit(1)
        files_to_plot = [matches[-1]]
    else:
        # Find the latest stream file for each station in STREAMS_DIR
        all_csvs = sorted(glob.glob(os.path.join(STREAMS_DIR, "realtime_etss_bias_96h_forecast_*.csv")))
        if not all_csvs:
            print(f"Error: No stream CSV files found in {STREAMS_DIR}")
            sys.exit(1)

        station_latest = {}
        for path in all_csvs:
            sid = os.path.basename(path).split('_')[5]
            station_latest[sid] = path

        files_to_plot = list(station_latest.values())

    print(f"Found {len(files_to_plot)} stream file(s) to plot.")
    for idx, fpath in enumerate(files_to_plot, 1):
        print(f"[{idx}/{len(files_to_plot)}] Plotting {os.path.basename(fpath)}...")
        plot_single_stream(fpath)

    print("Batch plotting complete.")

if __name__ == '__main__':
    main()
