#!/usr/bin/env python3
"""Merge + finalize perception_v2 (M6). Scoring-independent so it works for partially-finished
chunks: acceptance = the CheatCode demo actually seated the plug (final plug->port < THRESH),
computed from the recorded GT poses, not per-chunk scoring.yaml.

Per episode: frames.jsonl -> frames.parquet (skip malformed/truncated lines; drop episodes with
too few valid frames = in-flight kills). Then index.parquet with type/rail/target + split.

Usage: python merge_v2.py --data ~/aic_data/perception_v2 [--thresh 0.006 --min_frames 30]
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

MANIFEST = "/home/skr/ws_aic/aic_local/collect/manifest.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/skr/aic_data/perception_v2")
    ap.add_argument("--thresh", type=float, default=0.006)   # 6 mm = "seated" (kept as metadata only)
    ap.add_argument("--min_frames", type=int, default=30)
    args = ap.parse_args()
    root = Path(args.data)
    man = {m["episode"]: m for m in json.load(open(MANIFEST))}   # rail_idx etc. live in the manifest
    rows_idx = []
    n_parq = n_skip = 0
    for epdir in sorted(root.glob("episode_*")):
        jl = epdir / "frames.jsonl"
        meta = json.load(open(epdir / "meta.json")) if (epdir / "meta.json").exists() else {}
        recs = []
        if jl.exists():
            for line in open(jl):
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                                   # truncated last line from a killed collector
        if len(recs) < args.min_frames:
            n_skip += 1
            print(f"  SKIP {epdir.name}: only {len(recs)} valid frames (in-flight/partial)")
            continue
        df = pd.DataFrame(recs)
        df.to_parquet(epdir / "frames.parquet")
        n_parq += 1
        # success = final plug reached the port
        try:
            plug = np.asarray(df["plug_base_pos"].tolist(), float)
            port = np.asarray(df["port_base_pos"].tolist(), float)
            final_mm = float(np.linalg.norm(plug[-1] - port[-1]) * 1000.0)
            min_mm = float(np.linalg.norm(plug - port, axis=1).min() * 1000.0)
        except Exception:
            final_mm = min_mm = float("nan")
        seated = bool(np.isfinite(min_mm) and min_mm < args.thresh * 1000.0)
        ep = int(epdir.name.split("_")[1])
        mm = man.get(ep, {})
        # PERCEPTION acceptance: the port-pose label is valid regardless of insertion success,
        # so keep every episode with enough valid frames. `seated` is retained as metadata only.
        rows_idx.append({"episode": ep, "type": mm.get("type", meta.get("type", "?")),
                         "rail_idx": mm.get("rail_idx", -1),
                         "target_module_name": mm.get("target_module_name", meta.get("target_module_name", "?")),
                         "port_name": mm.get("port_name", meta.get("port_name", "?")),
                         "frames": len(df), "final_mm": round(final_mm, 1), "min_mm": round(min_mm, 1),
                         "seated": seated, "split": "train"})
    idx = pd.DataFrame(rows_idx).sort_values("episode").reset_index(drop=True)
    idx.to_parquet(root / "index.parquet")
    idx.to_json(root / "index.json", orient="records", indent=1)

    kept = idx[idx["split"] == "train"]
    print(f"\n=== perception_v2 finalized ===")
    print(f"episodes kept (>= {args.min_frames} frames): {len(kept)}  (skipped partial: {n_skip})")
    print(f"  of which CheatCode seated (<{args.thresh*1000:.0f}mm, metadata only): {int(kept['seated'].sum())}")
    print(f"total frames: {int(kept['frames'].sum())}")
    print(f"types: {kept['type'].value_counts().to_dict()}")
    import collections
    cov = collections.Counter((r['type'], int(r['rail_idx'])) for _, r in kept.iterrows())
    print(f"per-rail coverage: {dict(sorted(cov.items()))}")

if __name__ == "__main__":
    main()
