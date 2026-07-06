#!/usr/bin/env python3
"""Offline consolidation of the parallel native-res collection.
In-container collector wrote PNG + frames.jsonl + meta.json (no parquet — pandas absent there).
Here (host pixi env) we: (1) frames.jsonl -> frames.parquet per episode, (2) build index.parquet
(all CheatCode episodes are successful expert demos -> split='train'), typed from meta.json.

Usage: python merge_native.py --data ~/aic_data/perception_native
"""
import argparse, json, os
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/skr/aic_data/perception_native")
    args = ap.parse_args()
    root = Path(args.data)
    epdirs = sorted(root.glob("episode_*"))
    rows_idx = []
    n_ok = 0
    for epdir in epdirs:
        jl = epdir / "frames.jsonl"
        pqf = epdir / "frames.parquet"
        meta = json.load(open(epdir / "meta.json")) if (epdir / "meta.json").exists() else {}
        if jl.exists():
            recs = [json.loads(l) for l in open(jl) if l.strip()]
            if recs:
                df = pd.DataFrame(recs)
                df.to_parquet(pqf)
                n_ok += 1
                nfr = len(df)
            else:
                nfr = 0
        else:
            nfr = pd.read_parquet(pqf).shape[0] if pqf.exists() else 0
        ep = int(str(epdir.name).split("_")[1])
        rows_idx.append({"episode": ep, "type": meta.get("type", "?"),
                         "trial": meta.get("trial", "?"), "frames": nfr, "split": "train"})
        print(f"  {epdir.name}: {nfr} frames  type={meta.get('type','?')}")
    idx = pd.DataFrame(rows_idx).sort_values("episode").reset_index(drop=True)
    idx.to_parquet(root / "index.parquet")
    idx.to_json(root / "index.json", orient="records", indent=1)
    print(f"\nwrote frames.parquet for {n_ok} episodes")
    print(f"index.parquet: {len(idx)} episodes, {idx['frames'].sum()} frames total, "
          f"types={idx['type'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
