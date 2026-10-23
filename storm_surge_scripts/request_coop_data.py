#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import requests
import json

STATION_ID = "9468756"

print(f"🔍 DIAGNOSTIC INQUIRY FOR NOAA STATION {STATION_ID} (Nome, AK)\n" + "="*60)

# 1. Query Primary Station Metadata
meta_url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{STATION_ID}.json"
try:
    r = requests.get(meta_url, timeout=10)
    if r.status_code == 200:
        meta = r.json().get('stations', [{}])[0]
        print(f"✅ Station Name: {meta.get('name')}")
        print(f"📍 Location    : Lat {meta.get('lat')}, Lon {meta.get('lng')}")
        print(f"📡 Status      : {meta.get('type', 'N/A')}")
    else:
        print(f"⚠️ Metadata endpoint returned HTTP status: {r.status_code}")
except Exception as e:
    print(f"❌ Metadata request failed: {e}")

print("\n🧪 TESTING DIRECT PRODUCT STREAMS (July 2024 Test Window)...")

# 2. Test individual products directly on CO-OPS DataGetter
products = ["hourly_height", "water_level", "preliminary_water_level", "wind", "air_pressure"]
begin_date = "20240701"
end_date = "20240707"  # 1-week probe window

for prod in products:
    url = (
        f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        f"begin_date={begin_date}&end_date={end_date}&station={STATION_ID}&"
        f"product={prod}&datum=MLLW&units=metric&time_zone=gmt&format=json&application=web_services"
    )
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if "data" in data and len(data["data"]) > 0:
            print(f"  • Product '{prod}': ✅ DATA AVAILABLE ({len(data['data'])} records found)")
        elif "error" in data:
            print(f"  • Product '{prod}': ❌ API Message -> {data['error'].get('message')}")
        else:
            print(f"  • Product '{prod}': ⚠️ Empty response")
    except Exception as err:
        print(f"  • Product '{prod}': ❌ Connection Error ({err})")
