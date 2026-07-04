#!/usr/bin/env python3
"""Generate the injected-offset schedule for InsertTuner.

  python3 tuner_gen_offsets.py smoke  -> tuner_offsets.json: 10x zero offset (rig check)
  python3 tuner_gen_offsets.py sweep  -> tuner_offsets.json: 100-trial capture-basin grid
        lateral radius {2,4,6,8,10,12,14} mm x yaw {0,3,6,8} deg, golden-angle directions,
        plus 10 zero-offset controls interleaved.
"""
import json, math, sys

GOLDEN = 2.399963

def smoke():
    return [[0.0, 0.0, 0.0] for _ in range(10)]

def sweep():
    out = [[0.0, 0.0, 0.0] for _ in range(10)]          # controls
    radii = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014]
    yaws = [0.0, math.radians(3), math.radians(6), math.radians(8)]
    i = 0
    for r in radii:
        for y in yaws:
            for rep in range(3):                        # 3 directions per (r, yaw)
                a = GOLDEN * i
                out.append([round(r * math.cos(a), 5), round(r * math.sin(a), 5),
                            round(y if i % 2 == 0 else -y, 5)])
                i += 1
    return out                                           # 10 + 84 = 94 trials

mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
sched = smoke() if mode == "smoke" else sweep()
json.dump(sched, open("tuner_offsets.json", "w"))
print(f"{mode}: wrote {len(sched)} offsets -> tuner_offsets.json")
