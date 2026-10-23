#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Developed with the assistance of the Gemini AI code assistant.
'''

# this code is provided by noaa-coops
# https://api.tidesandcurrents.noaa.gov/api/prod/
import noaa_coops as nc
# 1. IMPORT THE ERROR CLASS
from noaa_coops.station import COOPSAPIError

from datetime import datetime
import numpy as np
import os
# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pandas as pd

#----- Station information -----
#stn = nc.Station(8725520)  # Fort Myers FL
stn  = nc.Station(8726520) # St Petersburg FL
#stn  = nc.Station(8726607) #st_petersburg_fl Point (Norfolk), VA
#stn  = nc.Station(8727520) # clear_water_beatch FL
#stn  = nc.Station(8726607) # OLD Port Tampa FL
#stn  = nc.Station(8638610) # st_petersburg_fl Point
#stn  = nc.Station(8761724) # Grand Isle LA
#stn  = nc.Station(8665530) # Charleston SC
#stn  = nc.Station(8658120) # Wilmington NC
#stn  = nc.Station(8771450) # Galveston Pier, Texas
#stn  = nc.Station(8518750) # The Battery (New York), NY
#stn  = nc.Station(8531680) # Sandy Hook, NJ
#stn  = nc.Station(8575512) # Annapolis, MD
#stn  = nc.Station(9491090) # Nome, AK   (KeyError: 'stations': check station id)
#stn  = nc.Station(8443970) # Boston MA
#stn  = nc.Station(8551910) # Reedy Point, DE
#stn  = nc.Station(8720218) # Mayport, FL
#stn  = nc.Station(9415144) # Port Chicago, CA

#----- Output information -----
output_file_name = "../storm_surge/test_dataset_2022_st_petersburg_fl_for_water_level.csv"
if os.path.exists(output_file_name):
    cmd = "rm " + output_file_name
    os.system(cmd)

# Leap year function
def leap_year(y):
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    if y % 4 == 0:
        return True
    else:
        return False

# Date information
year = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', \
        '2011','2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', \
        '2020', '2021', '2022', '2023', '2024', '2025', '2026']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
day = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
         '25', '26','27', '28', '29', '30', '31']
number_of_day_in_a_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

#===== START ==================================================================

# Loop from 2012 (index 10) to 2021 (index 19)
# range(10, 20) runs index 10 up to (but not including) 20
for iyear in range(20, 24): 
    
    current_year_int = int(year[iyear])
    if leap_year(current_year_int):
        number_of_day_in_a_month[1] = 29 # Set February to 29
    else:
        number_of_day_in_a_month[1] = 28 # Set February to 28

    # Loop over the months in the year
    for imonth in range (0,12):
        beginDate = year[iyear] + month[imonth] + day[0]
        endDate = year[iyear] + month[imonth] + str(number_of_day_in_a_month[imonth])
        print(f"Processing: {beginDate} to {endDate}")

        try:
            print("  ...Get water_level", end='')
            df_water_levels = stn.get_data(begin_date=beginDate, end_date=endDate,
                      product="water_level", datum="MLLW", units="metric", time_zone="gmt")
            print("...Get air temp", end='')
            df_air_temp = stn.get_data(begin_date=beginDate, end_date=endDate,
                      product="air_temperature", units="metric", time_zone="gmt")
            print("...Get wind", end='')
            df_winds = stn.get_data(begin_date=beginDate, end_date=endDate,
                      product="wind", units="metric", time_zone="gmt")
            print("...Get air pressure")
            df_air_pressure = stn.get_data(begin_date=beginDate, end_date=endDate,
                      product="air_pressure", units="metric", time_zone="gmt")
        except (ValueError, COOPSAPIError) as e:
            print(f"Caught an API error: {e}, some obs not available in this period. Skipping.")
            continue

        # ----- START: FIX FOR RAW COLUMN NAMES -----
        try:
            # First, try to get the friendly names (e.g., 'water_level')
            wl = df_water_levels[['water_level']]
            at = df_air_temp[['air_temp']]
            ap = df_air_pressure[['air_pressure']]
            w = df_winds[['wind_speed', 'wind_direction', 'wind_gust']]
            
            print("  ...Using friendly column names.")

        except KeyError:
            # If friendly names fail, get the raw names (e.g., 'v', 's', 'g')
            # and *rename* them to the friendly names.
            try:
                print("  ...Friendly names failed. Trying raw API column names ('v', 's', 'g').")
                wl = df_water_levels[['v']].rename(columns={'v': 'water_level'})
                at = df_air_temp[['v']].rename(columns={'v': 'air_temp'})
                ap = df_air_pressure[['v']].rename(columns={'v': 'air_pressure'})
                
                # For wind: 's'=speed, 'd'=direction, 'g'=gust
                w = df_winds[['s', 'd', 'g']].rename(columns={
                    's': 'wind_speed',
                    'd': 'wind_direction',
                    'g': 'wind_gust'
                })
            
            except KeyError as e:
                # If this *also* fails, something is truly missing.
                print(f"  ...CRITICAL: Could not find raw column {e}. Skipping month.")
                print(f"  ...DEBUG: Available columns were:")
                print(f"  ...Water Levels: {df_water_levels.columns.to_list()}")
                print(f"  ...Air Temp: {df_air_temp.columns.to_list()}")
                print(f"  ...Winds: {df_winds.columns.to_list()}")
                print(f"  ...Air Pressure: {df_air_pressure.columns.to_list()}")
                continue
        # ----- END: FIX FOR RAW COLUMN NAMES -----
        

        # 2. Combine all DataFrames.
        df_month = pd.concat([wl, at, ap, w], axis=1)

        # 3. Filter for "top of the hour" records (where minute == 0)
        df_month_hourly = df_month[df_month.index.minute == 0]

        # 4. Drop any row that has ANY missing data (NaN values).
        # Added .copy() to prevent SettingWithCopyWarning
        df_month_complete = df_month_hourly.dropna().copy()

        # 5. Check if we have any data left after filtering and dropping
        if df_month_complete.empty:
            print("  ...No valid, complete hourly data found for this month.")
            continue

        # 6. Format the DataFrame for the CSV file
        
        # ----- START: THIS IS THE FIX -----
        # Reset the index, and explicitly name the new column 'timestamp'
        # The original index column was named 't', not 'index'.
        df_month_complete.reset_index(names='timestamp', inplace=True)
        # We no longer need the separate 'rename' line.
        
        # This access will now work correctly
        df_month_complete['timestamp'] = df_month_complete['timestamp'].dt.strftime('%Y-%m-%d %H %M %S')
        # ----- END: THIS IS THE FIX -----
        
        
        # 7. Reorder columns to match your desired output
        output_columns = ['timestamp', 'air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust', 'water_level']
        df_to_save = df_month_complete[output_columns]

        # 8. Save the current month to CSV
        write_header = not os.path.exists(output_file_name)
        if write_header:
            print(f"  ...Writing {len(df_to_save)} rows to new file with header.")
            df_to_save.to_csv(output_file_name, index=False, mode='a', float_format='%g')
        else:
            # Append to existing file without the header
            print(f"  ...Appending {len(df_to_save)} rows to existing file.")
            df_to_save.to_csv(output_file_name, index=False, mode='a', header=False, float_format='%g')
        
print("Data retrieval complete.")
