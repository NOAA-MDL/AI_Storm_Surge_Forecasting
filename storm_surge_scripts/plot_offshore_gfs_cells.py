# Developed with the assistance of the Gemini AI code assistant.
#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import os
# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import csv
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. Grid & Physics Helpers ---
PHI_MIN_DEG, LAMBDA_MIN_DEG = 2.0, 170.0
RES = 0.12

def compute_indices(lat, lon):
    """Computes (i, j) GFS grid indices for a given lat/lon."""
    lon_east = lon + 360.0 if lon < 0 else lon
    i_idx = int(round((lat - PHI_MIN_DEG) / RES))
    j_idx = int(round((lon_east - LAMBDA_MIN_DEG) / RES))
    return i_idx, j_idx

def indices_to_coords(i_idx, j_idx):
    """Reconstructs geographic (lat, lon) coordinates from GFS (i, j) indices."""
    lat = PHI_MIN_DEG + (i_idx * RES)
    lon_east = LAMBDA_MIN_DEG + (j_idx * RES)
    lon = lon_east - 360.0 if lon_east > 180.0 else lon_east
    return lat, lon

def get_seaward_offsets(lat, lon, station_id=""):
    """
    Determines 4 spatial sampling offset vectors (di_lat, dj_lon) covering 
    4 directional quadrants surrounding the gauge station.
    Grid resolution: 1 unit ~ 13.3 km.
    """
    # 🌟 Override 1: Lewes, DE (Quadrant Sampling: Upper Delaware Bay, Inland, South Coast, Deep Ocean)
    if station_id == '8557380' or (38.6 <= lat <= 39.0 and -75.3 <= lon <= -74.9):
        region = "Lewes, DE (Quadrant Fan: Bay, Inland, South Coast, Deep Ocean)"
        offsets = [( +2, -1 ), ( -2, -2 ), ( -6, 0 ), ( -2, +3 )]

    # 🌟 Override 2: Sewells Point, VA (Lower Bay, Inland, South Shelf, Open Atlantic)
    elif station_id == '8638610' or (36.8 <= lat <= 37.1 and -76.5 <= lon <= -76.1):
        region = "Sewells Point, VA (Quadrant Fan: Lower Bay, Inland, Shelf, Open Atlantic)"
        offsets = [( +1, -1 ), ( -2, -2 ), ( -2, +2 ), ( +2, +4 )]

    # 🌟 Override 3: Nome, AK (Norton Sound, Inland, SW Bering Sea, Deep West)
    elif station_id == '9468756' or (lat > 60.0 and lon < -160.0):
        region = "Nome, AK (Quadrant Fan: Norton Sound, Inland, SW Bering, West)"
        offsets = [( 0, -2 ), ( +2, +1 ), ( -2, -2 ), ( -3, -4 )]

    # 🌟 Override 4: Trident Pier / Port Canaveral, FL (Inland Lagoon, North Coast, South Coast, Deep Atlantic)
    elif station_id == '8721604' or (28.0 <= lat <= 28.8 and -80.8 <= lon <= -80.2):
        region = "Trident Pier, FL (Quadrant Fan: Lagoon, North Shelf, South Shelf, Deep Atlantic)"
        offsets = [( 0, -1 ), ( +2, +2 ), ( -2, +2 ), ( 0, +4 )]

    # 1. Texas / Western Gulf Coast (West of -93.5 W)
    elif lon < -93.5:
        region = "Western Gulf (4-Quadrant Arc: Inland, Bay, Coastal Shelf, Deep Gulf)"
        offsets = [( +1, -2 ), ( -2, -1 ), ( -2, +3 ), ( -4, +4 )]

    # 2. Northern & Central Gulf Coast (LA, MS, AL, FL Panhandle: -93.5 W to -84.5 W, Lat > 28.5 N)
    elif -93.5 <= lon <= -84.5 and lat > 28.5:
        region = "Northern Gulf (4-Quadrant Arc: Inland, West Bay, East Shelf, Deep South Gulf)"
        offsets = [( +2, 0 ), ( -1, -3 ), ( -1, +3 ), ( -4, 0 )]

    # 3. West Coast of Florida / Eastern Gulf (South of 28.5 N, West of -81.5 W)
    elif lon <= -81.5 and lat <= 28.5:
        region = "Eastern Gulf / West FL (4-Quadrant Arc: Inland, North Shelf, SW Offshore, Far SW Ocean)"
        offsets = [( +1, +1 ), ( +2, -2 ), ( -2, -2 ), ( -4, -4 )]

    # 4. Standard East Coast / Atlantic Basin (East of -81.5 W)
    else:
        region = "Atlantic Basin (4-Quadrant Arc: Inland, North Coast, South Coast, Deep Ocean)"
        offsets = [( 0, -2 ), ( +3, +1 ), ( -3, +1 ), ( 0, +4 )]

    return offsets, region

# --- 2. Load Configuration ---
station_conf_path = 'stations_config.csv'
if not os.path.exists(station_conf_path):
    station_conf_path = '../storm_surge/stations_config.csv'

stations = []
with open(station_conf_path, 'r') as conf_f:
    reader = csv.DictReader(conf_f)
    for row in reader:
        stations.append({
            'id': row['station_id'].strip(),
            'name': row['station_name'].strip().replace('"', ''),
            'lat': float(row['latitude']),
            'lon': float(row['longitude'])
        })

# --- 3. Map Generation Loop ---
output_dir = 'spatial_grid_maps'
os.makedirs(output_dir, exist_ok=True)

for station in stations:
    curr_id = station['id']
    curr_name = station['name']
    curr_lat = station['lat']
    curr_lon = station['lon']

    # Filter to specific stations if needed
    # if curr_id not in ["8557380", "8721604", "9468756"]: continue

    print(f"🗺️  Plotting 4-Quadrant Spatial Map for: {curr_name} ({curr_id})...")

    # Primary station index
    I_idx, J_idx = compute_indices(curr_lat, curr_lon)

    # Compute 4-quadrant offset indices
    seaward_offsets, coast_region = get_seaward_offsets(curr_lat, curr_lon, station_id=curr_id)

    # Reconstruct exact lat/lon for the 4 quadrant cells
    offshore_coords = []
    for c_idx, (di, dj) in enumerate(seaward_offsets, start=1):
        c_i, c_j = I_idx + di, J_idx + dj
        c_lat, c_lon = indices_to_coords(c_i, c_j)
        offshore_coords.append((c_idx, c_lat, c_lon))

    # --- Initialize Map Figure ---
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Map extent bounds (Zoomed to ~1.4 degrees around station to display all quadrants)
    zoom_margin = 1.2
    ax.set_extent([curr_lon - zoom_margin, curr_lon + zoom_margin,
                   curr_lat - zoom_margin, curr_lat + zoom_margin], crs=ccrs.PlateCarree())

    # Add high-resolution geographical features
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#e0e0e0', edgecolor='gray')
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#b3d8ff')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.2, color='black')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', color='gray')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=0.5, edgecolor='gray')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Plot Primary Station Location
    ax.scatter(curr_lon, curr_lat, color='red', marker='^', s=160, zorder=5,
               edgecolor='black', label=f"Station: {curr_id}")
    ax.text(curr_lon + 0.04, curr_lat + 0.04, f"{curr_name}
({curr_id})",
            fontsize=10, fontweight='bold', color='darkred', zorder=6,
            transform=ccrs.PlateCarree())

    # Plot 4 Quadrant Grid Cells
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ["Q1 / Point 1", "Q2 / Point 2", "Q3 / Point 3", "Q4 / Point 4"]

    for idx, (c_num, c_lat, c_lon) in enumerate(offshore_coords):
        # Draw Point as Star Marker
        ax.scatter(c_lon, c_lat, color=colors[idx], marker='*', s=220, zorder=5,
                   edgecolor='black', label=f"Cell C{c_num} ({labels[idx]})")

        # Draw connecting vector line from station to quadrant cell
        ax.plot([curr_lon, c_lon], [curr_lat, c_lat], color=colors[idx],
                linestyle=':', linewidth=1.5, zorder=4)

        # Label Cell
        ax.text(c_lon + 0.03, c_lat + 0.03, f"C{c_num}", fontsize=10,
                fontweight='bold', color=colors[idx], zorder=6,
                transform=ccrs.PlateCarree())

    # Title & Legend
    plt.title(f"4-Quadrant Spatial GFS Grid Sampling
Station: {curr_name} ({curr_id})
Sampling Pattern: {coast_region}",
              fontsize=11, fontweight='bold', pad=12)
    plt.legend(loc='lower right', framealpha=0.9, fontsize=9)

    # Save Output Figure
    save_path = os.path.join(output_dir, f"spatial_grid_map_{curr_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"
🎉 All 4-quadrant spatial maps generated successfully in directory: '{output_dir}/'")
