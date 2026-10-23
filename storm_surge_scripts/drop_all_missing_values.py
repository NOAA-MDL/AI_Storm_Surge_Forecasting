#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import glob, pandas as pd, numpy as np

sentinel_flags = [
    99999.0, 99999.000, 99999, -99999.0, -99999,
    9999.0, 9999, -9999.0, -9999,
    999.0, -999.0, 999,
    99.99, -99.99
]

for f in glob.glob('../storm_surge/training_gfs_etss_spatial_data_*.csv') + glob.glob('../storm_surge/training_data_*.csv'):
    try:
        df = pd.read_csv(f)
        initial_rows = len(df)
        df = df.replace(sentinel_flags, np.nan)
        df.to_csv(f, index=False)
        print(f'✅ Cleaned sentinels in: {f} (Rows: {initial_rows})')
    except Exception as e:
        print(f'❌ Error processing {f}: {e}')
"
