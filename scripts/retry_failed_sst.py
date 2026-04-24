#!/usr/bin/env python3
"""Retry the days listed in data/sst/failed_days.log. Removes the log on success."""
import os, sys, zipfile, cdsapi

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "sst")
LOG  = os.path.join(BASE, "failed_days.log")

if not os.path.exists(LOG):
    sys.exit("no failed_days.log — nothing to retry")

# unique YYYY-MM-DD lines (tab-separated: first field is the date)
days = sorted({line.split("\t", 1)[0].strip()
               for line in open(LOG) if line.strip() and line[0].isdigit()})

print(f"retrying {len(days)} day(s): {', '.join(days)}")
c = cdsapi.Client()
still_failing = []

for ymd in days:
    y, m, d = ymd.split("-")
    dst = os.path.join(BASE, y, m, d)
    os.makedirs(dst, exist_ok=True)
    zip_path = os.path.join(dst, "download.zip")
    try:
        c.retrieve(
            "satellite-sea-surface-temperature-ensemble-product",
            {"variable": "all", "format": "zip", "day": d, "month": m, "year": y},
            zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dst)
        os.remove(zip_path)
        print(f"  [ok] {ymd}")
    except Exception as e:
        print(f"  [fail] {ymd}: {type(e).__name__}: {e}")
        still_failing.append(f"{ymd}\t{type(e).__name__}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

if still_failing:
    with open(LOG, "w") as f:
        f.write("\n".join(still_failing) + "\n")
    print(f"\n{len(still_failing)} still failing — kept in {LOG}")
else:
    os.remove(LOG)
    print("\nall recovered — removed failed_days.log")
