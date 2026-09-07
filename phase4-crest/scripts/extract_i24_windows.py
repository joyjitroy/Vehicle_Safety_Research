#!/usr/bin/env python3
"""
I-24 MOTION — Windowed Trajectory Extractor (Tight Windows + Sampling)
Streams each large JSON file and saves only records within narrow target
time windows. Keeps 1-in-3 vehicles to limit output size.

Run:
    python -m pip install ijson
    python extract_i24_windows.py
"""

import json
import os
import sys
import random
from decimal import Decimal

try:
    import ijson
except ImportError:
    sys.exit("ijson not installed. Run:  python -m pip install ijson")

random.seed(42)

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = r"C:\Personal\EB1A\VehicleSafetyResearch\crest-paper-one\data\raw\I-24 MOTION"
OUT_DIR  = r"C:\Personal\EB1A\VehicleSafetyResearch\crest-paper-one\data\processed\i24_filtered"

# =============================================================================
# SETTINGS
# =============================================================================

WB        = -1    # westbound
BOTH      = None  # no direction filter
KEEP_RATE = 1/3   # keep 1 in 3 vehicles (random sample)
MIN_POINTS = 5    # minimum trajectory points after clipping

# =============================================================================
# TIME WINDOWS  (Unix timestamps; CST = UTC − 6)
#
# Nov 21 06:00 CST = 1669032000
# Nov 23 06:00 CST = 1669204800
# Dec 02 06:00 CST = 1669982400
# =============================================================================

DAYS = [
    {
        "name": "nov21_post1",
        "file": os.path.join(
            BASE_DIR, "11-21-2022",
            "637b023440527bf2daa5932f__post1",
            "637b023440527bf2daa5932f__post1.json"),
        # Crash: rear-end WB MM59.7 at 6:14 AM CST
        "windows": [
            (1669032600, 1669033800, "hazard", WB),   # 06:10–06:30 CST (around onset)
            (1669041600, 1669042500, "clean",  BOTH), # 09:00–09:15 CST (post-clearance)
        ],
    },
    {
        "name": "nov23_post3",
        "file": os.path.join(
            BASE_DIR, "11-23-2022",
            "637d8ea678f0cb97981425dd__post3",
            "637d8ea678f0cb97981425dd__post3.json"),
        # Crash: sideswipe WB MM59.2 at 7:35 AM CST
        "windows": [
            (1669209900, 1669211100, "hazard", WB),   # 07:25–07:45 CST (around onset)
            (1669204800, 1669205700, "clean",  BOTH), # 06:00–06:15 CST
        ],
    },
    {
        "name": "dec02_post12",
        "file": os.path.join(
            BASE_DIR, "12-02-2022",
            "63898d48d430891009401330__post12",
            "63898d48d430891009401330__post12.json"),
        # Crash: WB MM61.8 at 7:38 AM CST, cleared 8:20 AM CST
        "windows": [
            (1669987380, 1669988700, "hazard", WB),   # 07:23–07:45 CST (around onset)
            (1669982400, 1669983300, "clean",  BOTH), # 06:00–06:15 CST
        ],
    },
]

# =============================================================================
# HELPERS
# =============================================================================

def overlaps(t_start, t_end, win_start, win_end):
    return t_start <= win_end and t_end >= win_start


def clip_to_window(timestamps, x_pos, y_pos, win_start, win_end):
    return [
        (t, x, y)
        for t, x, y in zip(timestamps, x_pos, y_pos)
        if win_start <= t <= win_end
    ]

# =============================================================================
# MAIN
# =============================================================================

os.makedirs(OUT_DIR, exist_ok=True)
summary = []

for day in DAYS:
    name  = day["name"]
    fpath = day["file"]
    wins  = day["windows"]

    print(f"\n{'='*60}")
    print(f"Day : {name}")
    print(f"File: {fpath}")

    if not os.path.exists(fpath):
        print("  FILE NOT FOUND — skipping")
        continue

    buckets = {}
    for _, _, label, _ in wins:
        buckets.setdefault(label, [])

    global_start = min(w[0] for w in wins)
    global_end   = max(w[1] for w in wins)

    total_read = 0
    total_kept = 0
    total_skip_sample = 0

    with open(fpath, "rb") as f:
        for record in ijson.items(f, "item"):
            total_read += 1
            if total_read % 50000 == 0:
                print(f"  ... {total_read:,} read | {total_kept:,} kept | {total_skip_sample:,} sampled out")

            t_first = record.get("first_timestamp", 0)
            t_last  = record.get("last_timestamp",  0)

            if not overlaps(t_first, t_last, global_start, global_end):
                continue

            # Random sampling — drop 2 out of every 3 vehicles
            if random.random() > KEEP_RATE:
                total_skip_sample += 1
                continue

            direction  = record.get("direction", 0)
            timestamps = record.get("timestamp",  [])
            x_pos      = record.get("x_position", [])
            y_pos      = record.get("y_position", [])
            v_class    = record.get("coarse_vehicle_class", -1)
            length     = record.get("length", 0.0)
            oid        = record.get("_id", {}).get("$oid", "")

            for (win_start, win_end, label, dir_filter) in wins:
                if dir_filter is not None and direction != dir_filter:
                    continue
                if not overlaps(t_first, t_last, win_start, win_end):
                    continue

                clipped = clip_to_window(timestamps, x_pos, y_pos, win_start, win_end)
                if len(clipped) < MIN_POINTS:
                    continue

                ts_c, xs_c, ys_c = zip(*clipped)

                buckets[label].append({
                    "id":        oid,
                    "day":       name,
                    "label":     label,
                    "direction": int(direction),
                    "v_class":   int(v_class),
                    "length":    round(float(length), 3),
                    "timestamp": [float(t) for t in ts_c],
                    "x":         [float(x) for x in xs_c],
                    "y":         [float(y) for y in ys_c],
                })
                total_kept += 1
                break

    for label, recs in buckets.items():
        out_path = os.path.join(OUT_DIR, f"{name}_{label}.json")
        with open(out_path, "w") as fout:
            json.dump(recs, fout)
        size_kb = os.path.getsize(out_path) // 1024
        print(f"  [{label}] {len(recs):,} records → {out_path}  ({size_kb:,} KB)")
        summary.append((name, label, len(recs), out_path))

    print(f"  Total read: {total_read:,} | Kept: {total_kept:,} | Sampled out: {total_skip_sample:,}")

print(f"\n{'='*60}")
print("EXTRACTION SUMMARY")
print(f"{'='*60}")
for name, label, n, path in summary:
    print(f"  {name:25s}  {label:8s}  {n:6,} records")
print(f"\nAll files written to: {OUT_DIR}")
