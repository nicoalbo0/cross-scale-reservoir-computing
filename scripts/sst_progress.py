#!/usr/bin/env python3
"""Print a one-line progress bar + ETA for the Copernicus SST download.

Rate is computed over a recent window (default last 1h of *activity*) and the
elapsed wall time since the most recent write is shown explicitly, so a dead
downloader cannot hide behind a stale average."""
import os, glob, time

BASE        = os.path.join(os.path.dirname(__file__), "..", "data", "sst")
TOTAL       = 12418          # 1982-01-01 .. 2015-12-31
IDLE_STALL  = 1800           # s — above this, mark STALLED
RATE_WINDOW = 3600           # s — compute rate over last N sec of activity

dirs = [d for y in sorted(os.listdir(BASE)) if y.isdigit()
        for d in glob.glob(os.path.join(BASE, y, "*", "*")) if os.path.isdir(d)]
done   = len(dirs)
mtimes = sorted(os.path.getmtime(d) for d in dirs)

size = 0
for d in dirs:
    for root, _, files in os.walk(d):
        for f in files:
            try: size += os.path.getsize(os.path.join(root, f))
            except OSError: pass
avg     = size / done if done else 0
est_tot = avg * TOTAL

def human(b):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024: return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}PB"

def dur(s):
    s = int(max(s, 0))
    d, s = divmod(s, 86400); h, s = divmod(s, 3600); m = s // 60
    return f"{d}d{h:02d}h{m:02d}m" if d else f"{h}h{m:02d}m"

now  = time.time()
idle = now - mtimes[-1] if mtimes else 0

# rate over last RATE_WINDOW sec of *activity* (end-anchored at last write)
recent = [m for m in mtimes if m >= mtimes[-1] - RATE_WINDOW] if mtimes else []
if len(recent) >= 2:
    rate = (len(recent) - 1) / (recent[-1] - recent[0])   # dirs/sec
else:
    rate = 0
eta = (TOTAL - done) / rate if rate > 0 else 0

pct    = 100 * done / TOTAL
fill   = int(40 * done / TOTAL)
bar    = "#" * fill + "-" * (40 - fill)
latest = os.path.relpath(max(dirs, key=os.path.getmtime), BASE) if dirs else "-"
state  = "STALLED" if idle > IDLE_STALL else "running"
eta_s  = "—" if state == "STALLED" or rate == 0 else dur(eta)

print(f"[{bar}] {pct:5.2f}%  {done}/{TOTAL}  {rate*60:4.1f}/min  "
      f"ETA {eta_s}  idle {dur(idle)} [{state}]  "
      f"disk {human(size)}/~{human(est_tot)}  latest={latest}")
