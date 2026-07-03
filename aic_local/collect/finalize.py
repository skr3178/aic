#!/usr/bin/env python3
"""finalize.py — join collected episodes with CheatCode scores + apply the acceptance filter.

Reads each chunk's scoring.yaml (per-trial tier_3 = CheatCode success), pairs it with the
collected episode dir, and emits an index. Episodes CheatCode failed (tier_3 <= 0) are marked
'reject' and kept out of the perception training split (the real correctness guard).

Usage:
  aicrun python finalize.py --dataset ~/aic_data/perception_v1 --manifest manifest.json \
      --results ~/aic_results/collect
"""
import argparse, glob, json, os
import yaml


def load_scores(results):
    """chunk index -> {trial_name: tier3_score}"""
    out = {}
    for sy in glob.glob(os.path.join(os.path.expanduser(results), "chunk_*", "scoring.yaml")):
        k = int(os.path.basename(os.path.dirname(sy)).split("_")[1])
        try:
            d = yaml.safe_load(open(sy)) or {}
        except Exception:
            d = {}
        trials = {}
        for tname, tv in d.items():
            if not isinstance(tv, dict) or "tier_3" not in tv:
                continue
            trials[tname] = float(tv["tier_3"].get("score", 0.0))
        out[k] = trials
    return out


def n_frames(ep_dir):
    p = os.path.join(ep_dir, "frames.parquet")
    if os.path.exists(p):
        try:
            import pandas as pd
            return len(pd.read_parquet(p))
        except Exception:
            pass
    j = os.path.join(ep_dir, "frames.jsonl")
    if os.path.exists(j):
        return sum(1 for _ in open(j))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="~/aic_data/perception_v1")
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "manifest.json"))
    ap.add_argument("--results", default="~/aic_results/collect")
    args = ap.parse_args()

    dataset = os.path.expanduser(args.dataset)
    manifest = json.load(open(args.manifest))
    scores = load_scores(args.results)

    rows = []
    for m in manifest:
        ep_dir = os.path.join(dataset, f"episode_{m['episode']:04d}")
        collected = os.path.isdir(ep_dir)
        nf = n_frames(ep_dir) if collected else 0
        tier3 = scores.get(m["chunk"], {}).get(m["trial"])
        inserted = (tier3 is not None and tier3 >= 75.0)
        if tier3 is None:
            split = "unknown"
        elif tier3 > 0:
            split = "train"
        else:
            split = "reject"
        rows.append({
            "episode": m["episode"], "type": m["type"], "chunk": m["chunk"], "trial": m["trial"],
            "collected": collected, "n_frames": nf,
            "tier3": tier3, "inserted": inserted, "split": split,
            "port_frame": m["port_frame"], "plug_frame": m["plug_frame"],
            "port_name": m["port_name"], "target_module_name": m["target_module_name"],
            **{f"p_{k}": v for k, v in (m.get("params", {}) or {}).items() if not isinstance(v, dict)},
        })

    os.makedirs(dataset, exist_ok=True)
    with open(os.path.join(dataset, "index.json"), "w") as f:
        json.dump(rows, f, indent=2)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(os.path.join(dataset, "index.parquet"))
    except Exception as e:
        print("(parquet index skipped:", e, ")")

    n = len(rows)
    coll = sum(r["collected"] for r in rows)
    tr = sum(r["split"] == "train" for r in rows)
    rej = sum(r["split"] == "reject" for r in rows)
    unk = sum(r["split"] == "unknown" for r in rows)
    frames = sum(r["n_frames"] for r in rows)
    sfp_tr = sum(r["split"] == "train" and r["type"] == "sfp" for r in rows)
    print(f"episodes: {n} | collected: {coll} | frames: {frames}")
    print(f"split -> train: {tr} (sfp {sfp_tr}, sc {tr-sfp_tr}) | reject: {rej} | unknown: {unk}")
    print(f"index -> {os.path.join(dataset, 'index.parquet')} / index.json")


if __name__ == "__main__":
    main()
