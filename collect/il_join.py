#!/usr/bin/env python3
"""il_join.py — build the IL table for one episode.

The perception dataset (frames.parquet: images + GT pose + proprio) does NOT contain the expert
ACTION. The action lives in the engine's per-trial rosbag (`/aic_controller/pose_commands`,
type MotionUpdate). This joins them: for each image frame, pick the pose_command nearest in time
and append it as the action, giving a training-ready (3 images + proprio -> action) table.

Output: <episode>/il_frames.parquet  (= frames.parquet + action_{pos,quat,vel,mode,stamp,dt}).

Usage:
  aicrun python il_join.py --ep ~/aic_data/perception_v1/episode_0000 [--results ~/aic_results/collect]
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

TOPIC = "/aic_controller/pose_commands"


def read_commands(bag_dir):
    """Return list of (stamp, pos[3], quat[4], vel[6], mode) for every pose_command in the bag."""
    out = []
    for mf in sorted(glob.glob(os.path.join(bag_dir, "*.mcap"))):
        with open(mf, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, _chan, _msg, m in reader.iter_decoded_messages(topics=[TOPIC]):
                st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                p, q, v = m.pose.position, m.pose.orientation, m.velocity
                out.append((
                    st,
                    [p.x, p.y, p.z],
                    [q.x, q.y, q.z, q.w],
                    [v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z],
                    int(m.trajectory_generation_mode.mode),
                ))
    return out


def find_bag(results, chunk, trial):
    c = sorted(glob.glob(os.path.join(os.path.expanduser(results), f"chunk_{chunk}", f"bag_{trial}_*")))
    return c[0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--results", default="~/aic_results/collect")
    a = ap.parse_args()
    ep = os.path.expanduser(a.ep)
    meta = json.load(open(os.path.join(ep, "meta.json")))
    chunk, trial = meta["chunk"], meta["trial"]
    bag = find_bag(a.results, chunk, trial)
    if not bag:
        raise SystemExit(f"no bag for chunk {chunk} {trial} under {a.results}")
    print(f"ep {meta['episode']} ({meta['type']} {trial})  bag={os.path.basename(bag)}")

    cmds = read_commands(bag)
    print(f"  pose_commands in bag: {len(cmds)}")
    if not cmds:
        raise SystemExit("no pose_commands found (wrong topic / empty bag)")
    cmds.sort(key=lambda r: r[0])
    cs = np.array([r[0] for r in cmds])

    df = pd.read_parquet(os.path.join(ep, "frames.parquet"))
    fs = df["stamp"].to_numpy()
    # nearest command to each frame's (sim-time) stamp
    idx = np.clip(np.searchsorted(cs, fs), 1, len(cs) - 1)
    pick = np.where(np.abs(fs - cs[idx - 1]) <= np.abs(fs - cs[idx]), idx - 1, idx)
    dt = np.abs(fs - cs[pick])

    df["action_pos"] = [cmds[i][1] for i in pick]
    df["action_quat"] = [cmds[i][2] for i in pick]
    df["action_vel"] = [cmds[i][3] for i in pick]
    df["action_mode"] = [cmds[i][4] for i in pick]
    df["action_stamp"] = cs[pick]
    df["action_dt"] = dt
    out = os.path.join(ep, "il_frames.parquet")
    df.to_parquet(out)

    apos = np.array([r[1] for r in cmds])
    modes = sorted(set(r[4] for r in cmds))   # 0=UNSPECIFIED 1=VELOCITY 2=POSITION
    print(f"  frames: {len(df)} | align dt median={np.median(dt)*1000:.1f}ms max={dt.max()*1000:.1f}ms "
          f"({(dt<0.05).mean()*100:.0f}% within 50ms)")
    print(f"  action xyz range  x[{apos[:,0].min():.3f},{apos[:,0].max():.3f}]  "
          f"y[{apos[:,1].min():.3f},{apos[:,1].max():.3f}]  z[{apos[:,2].min():.3f},{apos[:,2].max():.3f}]  "
          f"command modes={modes}")
    print(f"  wrote {out}  (frames.parquet cols + action_pos/quat/vel/mode/stamp/dt)")


if __name__ == "__main__":
    main()
