"""Fetch historical hourly weather for each dataset session via Open-Meteo.

Open-Meteo archive API is free and requires no API key:
    https://archive-api.open-meteo.com/v1/archive

Coordinates and dates:
    NGSIM US-101 : 34.0522, -118.2437, 2005-06-16
    NGSIM I-80   : 37.8329, -122.2669, 2005-04-13
    MiTra T4-T9  : 45.4654, 9.1859, dates read from data/external/mitra_dates.json

Fetched variables: temperature_2m, precipitation, visibility, windspeed_10m, weathercode.
Output: data/external/weather.json keyed by dataset_source.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'external'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'weather.json'

MITRA_DATES_PATH = OUT_DIR / 'mitra_dates.json'

# Static session metadata.
NGSIM_SESSIONS = {
    'ngsim_us-101_0750am-0805am': (34.0522, -118.2437, '2005-06-16'),
    'ngsim_us-101_0805am-0820am': (34.0522, -118.2437, '2005-06-16'),
    'ngsim_us-101_0820am-0835am': (34.0522, -118.2437, '2005-06-16'),
    'ngsim_i-80_0400pm-0415pm': (37.8329, -122.2669, '2005-04-13'),
    'ngsim_i-80_0500pm-0515pm': (37.8329, -122.2669, '2005-04-13'),
    'ngsim_i-80_0515pm-0530pm': (37.8329, -122.2669, '2005-04-13'),
}

BASE_URL = 'https://archive-api.open-meteo.com/v1/archive'
HOURLY_VARS = ['temperature_2m', 'precipitation', 'visibility', 'windspeed_10m', 'weathercode']


def load_mitra_dates():
    if not MITRA_DATES_PATH.exists():
        raise FileNotFoundError(
            f'MiTra recording dates must be supplied in {MITRA_DATES_PATH}\n'
            'Create it as a JSON object, e.g.: {\n'
            '  "mitra_t4": {"date": "2018-10-15", "lat": 45.4654, "lon": 9.1859},\n'
            '  ...\n'
            '}'
        )
    with open(MITRA_DATES_PATH, encoding='utf-8') as f:
        return json.load(f)


def build_session_table():
    sessions = dict(NGSIM_SESSIONS)
    mitra_dates = load_mitra_dates()
    for key, meta in mitra_dates.items():
        sessions[key] = (meta['lat'], meta['lon'], meta['date'])
    return sessions


def fetch_weather(session_name: str, lat: float, lon: float, date: str):
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': date,
        'end_date': date,
        'hourly': ','.join(HOURLY_VARS),
        'timezone': 'auto',
    }
    query = '&'.join(f'{k}={v}' for k, v in params.items())
    url = f'{BASE_URL}?{query}'
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        print(f'[fetch_weather] HTTP {exc.code} for {session_name}: {exc.reason}', file=sys.stderr)
        raise

    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    records = []
    for i, t in enumerate(times):
        records.append({
            'time': t,
            'temperature_2m': hourly.get('temperature_2m', [])[i] if i < len(hourly.get('temperature_2m', [])) else None,
            'precipitation': hourly.get('precipitation', [])[i] if i < len(hourly.get('precipitation', [])) else None,
            'visibility': hourly.get('visibility', [])[i] if i < len(hourly.get('visibility', [])) else None,
            'windspeed_10m': hourly.get('windspeed_10m', [])[i] if i < len(hourly.get('windspeed_10m', [])) else None,
            'weathercode': hourly.get('weathercode', [])[i] if i < len(hourly.get('weathercode', [])) else None,
        })
    return records


def main():
    sessions = build_session_table()
    weather = {}
    for session_name, (lat, lon, date) in sessions.items():
        print(f'Fetching weather for {session_name} ({date}) ...', flush=True)
        try:
            weather[session_name] = fetch_weather(session_name, lat, lon, date)
        except Exception as exc:  # noqa: BLE001
            print(f'Failed for {session_name}: {exc}', file=sys.stderr)
            weather[session_name] = {'error': str(exc)}
        time.sleep(0.5)  # be polite to the free API

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(weather, f, indent=2)
    print(f'Weather data saved to {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
