#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import requests

url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
params = {
    "begin_date": "20210101",
    "end_date": "20210131",
    "station": "8551910",  # Reedy Point, DE
    "product": "hourly_height",
    "datum": "MLLW",
    "units": "metric",
    "time_zone": "gmt",
    "format": "json",
    "application": "web_services"
}

r = requests.get(url, params=params).json()
print(r)
