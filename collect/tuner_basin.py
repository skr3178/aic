#!/usr/bin/env python3
"""Aggregate InsertTuner logs -> capture-basin table (TRUE-seat rate vs injected offset).

  python3 tuner_basin.py [results_root]     # default /home/skr/aic_results/tuner

TRUE seat = seated (GT distance < 20 mm at rest) AND tug_held (survived the pull).
Prints the basin table; writes force_basin.md next to the results.
"""
import glob, json, math, sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "/home/skr/aic_results/tuner"
rows = []
for f in sorted(glob.glob(f"{root}/chunk_*/tuner_log.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if "error" not in r:
            rows.append(r)
if not rows:
    sys.exit(f"no trial records under {root}")

def lat_mm(r): return math.hypot(r["dx_mm"], r["dy_mm"])
def bucket_lat(r):
    l = lat_mm(r)
    return 0 if l < 1 else 2 * round(l / 2)
def bucket_yaw(r): return abs(round(r["dyaw_deg"]))

def true_seat(r): return bool(r.get("seated")) and bool(r.get("tug_held"))

cells = defaultdict(lambda: [0, 0, 0.0])     # (lat,yaw) -> [n, true_seats, peak_dF_sum]
for r in rows:
    c = cells[(bucket_lat(r), bucket_yaw(r))]
    c[0] += 1; c[1] += int(true_seat(r)); c[2] += r.get("peak_dF", 0.0)

lats = sorted({k[0] for k in cells}); yaws = sorted({k[1] for k in cells})
lines = ["# Force-insertion capture basin (InsertTuner)", "",
         f"trials: {len(rows)}  |  TRUE seat = seated AND survived tug test", "",
         "| lateral \\ yaw | " + " | ".join(f"{y}°" for y in yaws) + " |",
         "|---|" + "|".join("---" for _ in yaws) + "|"]
for l in lats:
    cols = []
    for y in yaws:
        n, s, fsum = cells.get((l, y), [0, 0, 0])
        cols.append(f"{s}/{n} ({100*s/n:.0f}%) F̄{fsum/n:.0f}N" if n else "—")
    lines.append(f"| {l} mm | " + " | ".join(cols) + " |")

# per-type + disagreement flags
byt = defaultdict(lambda: [0, 0])
flags = []
for r in rows:
    byt[r["type"]][0] += 1; byt[r["type"]][1] += int(true_seat(r))
    if r.get("seated") and r.get("tug_held") is False:
        flags.append(r["trial"])
lines += ["", "**Per type:** " + "  ".join(f"{t}: {s}/{n} ({100*s/n:.0f}%)" for t, (n, s) in sorted(byt.items())),
          f"**Seated-but-tug-failed (scorer-vs-sensor disagreement):** {flags or 'none'}",
          f"**Peak ΔF overall:** {max(r.get('peak_dF',0) for r in rows):.0f} N"]
out = "\n".join(lines)
print(out)
open(f"{root}/force_basin.md", "w").write(out + "\n")
print(f"\nwrote {root}/force_basin.md")
