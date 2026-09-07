"""Fetch real elevation and OSM road features for NGSIM and MiTra corridors.

Elevation: Open-Meteo Elevation API (free, no key)
    https://api.open-meteo.com/v1/elevation
Road features (maxspeed, lanes, nearest ramp): OpenStreetMap Overpass API (free, no key)
    https://overpass-api.de/api/interpreter

IMPORTANT LIMITATION:
    NGSIM's x_m/y_m are LOCAL frame coordinates (feet-converted Local_X/Local_Y),
    NOT georeferenced. There is no way to compute per-vehicle real elevation or
    OSM attributes for NGSIM. For NGSIM sources we fetch ONE static value at the
    known corridor reference point (documented in NGSIM metadata: US-101
    Hollywood Fwy at 34.0522,-118.2437; I-80 Emeryville at 37.8329,-122.2669)
    and apply it uniformly to the whole corridor.

    MiTra's x_m/y_m ARE real UTM Zone 32N easting/northing (Milan, Italy area).
    These are converted to lat/lon (via pyproj) per corridor centroid (median
    vehicle position) and used to fetch one real corridor-level value per
    MiTra session, for consistency with the NGSIM corridor-level values.

    In both cases, this produces ONE static map/road-context vector per
    dataset_source, not a per-position profile. Curvature is intentionally
    omitted (unreliable to estimate from a single static query); the A1
    map/elevation feature vector uses elevation, lane count, speed limit, and
    nearest-ramp distance only.

Output: data/external/map_features.json keyed by dataset_source:
  {
    "elevation_m": <float>,
    "num_lanes": <int or None>,
    "speed_limit_ms": <float or None>,
    "nearest_ramp_distance_m": <float or None>,
    "lat": <float>, "lon": <float>,
    "source": "corridor_reference_point" | "utm_centroid"
  }
"""
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from crest.datasets.crest_dataset import load_source_df

OUT_DIR = ROOT / 'data' / 'external'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'map_features.json'

ELEVATION_URL = 'https://api.open-meteo.com/v1/elevation'
OVERPASS_URL = 'https://overpass-api.de/api/interpreter'

# NGSIM corridor reference points (fixed; no per-vehicle geocoding possible).
NGSIM_CORRIDORS = {
    'ngsim_us-101_0750am-0805am': (34.0522, -118.2437),
    'ngsim_us-101_0805am-0820am': (34.0522, -118.2437),
    'ngsim_us-101_0820am-0835am': (34.0522, -118.2437),
    'ngsim_i-80_0400pm-0415pm': (37.8329, -122.2669),
    'ngsim_i-80_0500pm-0515pm': (37.8329, -122.2669),
    'ngsim_i-80_0515pm-0530pm': (37.8329, -122.2669),
}

MITRA_SOURCES = [f'mitra_t{i}' for i in range(4, 10)]

# MiTra x_m/y_m are UTM Zone 32N (WGS84) easting/northing.
_UTM_TO_WGS84 = Transformer.from_crs('EPSG:32632', 'EPSG:4326', always_xy=True)


def mitra_centroid_latlon(source: str) -> tuple:
    df = load_source_df(source)
    easting = float(df['x_m'].median())
    northing = float(df['y_m'].median())
    lon, lat = _UTM_TO_WGS84.transform(easting, northing)
    return lat, lon


def fetch_elevation(lat: float, lon: float) -> float:
    url = f'{ELEVATION_URL}?latitude={lat}&longitude={lon}'
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode('utf-8'))
    return float(data['elevation'][0])


def overpass_query(query: str) -> dict:
    encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
    req = urllib.request.Request(OVERPASS_URL, data=encoded)
    req.add_header('User-Agent', 'CREST-research/1.0 (academic; joyjit.roy.tech@gmail.com)')
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode('utf-8'))


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_road_attributes(lat: float, lon: float, radius_m: float = 200.0):
    """Query Overpass for the nearest motorway/trunk way and nearest ramp."""
    query = f"""
    [out:json][timeout:30];
    (
      way(around:{radius_m},{lat},{lon})["highway"~"motorway|trunk|primary"];
    );
    out tags center;
    """
    num_lanes, speed_limit_ms = None, None
    try:
        data = overpass_query(query)
        for el in data.get('elements', []):
            tags = el.get('tags', {})
            if num_lanes is None and 'lanes' in tags:
                try:
                    num_lanes = int(tags['lanes'])
                except ValueError:
                    pass
            if speed_limit_ms is None and 'maxspeed' in tags:
                raw = tags['maxspeed'].split()[0]
                try:
                    val = float(raw)
                    speed_limit_ms = val * 0.44704 if 'mph' in tags['maxspeed'] else val / 3.6
                except ValueError:
                    pass
    except Exception as exc:  # noqa: BLE001
        print(f'  Overpass road-attribute query failed: {exc}', file=sys.stderr)

    ramp_query = f"""
    [out:json][timeout:30];
    (
      way(around:2000,{lat},{lon})["highway"="motorway_link"];
    );
    out center;
    """
    nearest_ramp_distance_m = None
    try:
        data = overpass_query(ramp_query)
        best = None
        for el in data.get('elements', []):
            center = el.get('center')
            if not center:
                continue
            d = haversine_m(lat, lon, center['lat'], center['lon'])
            if best is None or d < best:
                best = d
        nearest_ramp_distance_m = best
    except Exception as exc:  # noqa: BLE001
        print(f'  Overpass ramp query failed: {exc}', file=sys.stderr)

    return num_lanes, speed_limit_ms, nearest_ramp_distance_m


def main():
    results = {}

    for source, (lat, lon) in NGSIM_CORRIDORS.items():
        print(f'Fetching {source} (corridor reference point) ...', flush=True)
        elevation_m = fetch_elevation(lat, lon)
        num_lanes, speed_limit_ms, ramp_dist = fetch_road_attributes(lat, lon)
        results[source] = {
            'elevation_m': elevation_m, 'num_lanes': num_lanes,
            'speed_limit_ms': speed_limit_ms, 'nearest_ramp_distance_m': ramp_dist,
            'lat': lat, 'lon': lon, 'source': 'corridor_reference_point',
        }
        time.sleep(1.0)

    for source in MITRA_SOURCES:
        print(f'Fetching {source} (UTM centroid) ...', flush=True)
        lat, lon = mitra_centroid_latlon(source)
        elevation_m = fetch_elevation(lat, lon)
        num_lanes, speed_limit_ms, ramp_dist = fetch_road_attributes(lat, lon)
        results[source] = {
            'elevation_m': elevation_m, 'num_lanes': num_lanes,
            'speed_limit_ms': speed_limit_ms, 'nearest_ramp_distance_m': ramp_dist,
            'lat': lat, 'lon': lon, 'source': 'utm_centroid',
        }
        time.sleep(1.0)

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
