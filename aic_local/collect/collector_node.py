#!/usr/bin/env python3
"""collector_node.py — pair /observations frames with ground-truth port/plug pose (TF).

Runs on the host in the pixi env (like CheatCode), while the eval container runs a chunk config
with ground_truth:=true and CheatCode drives the arm. Episode boundaries are detected by the
per-trial cable spawn/despawn (the plug TF frame appears when a trial starts, disappears on reset).

Per kept /observations frame it saves:
  - 3 downscaled wrist images (PNG) per camera
  - GT port / plug / port-entrance pose in base_link
  - GT port pose in each camera optical frame  (the perception label for pixels->pose)
  - proprio: joint positions, wrist wrench, TCP pose/vel/error (controller_state)
Written to <out>/episode_{NNNN}/{left,center,right}/*.png + frames.parquet + meta.json.

Usage (via aicrun, with sim time):
  aicrun python collector_node.py --chunk 0 --manifest manifest.json --out ~/aic_data/perception_v1 \
      --ros-args -p use_sim_time:=true
"""
import argparse, json, os, sys
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
import tf2_ros
from aic_model_interfaces.msg import Observation

CAMS = ["left", "center", "right"]


def _pose(tr):
    return ([tr.translation.x, tr.translation.y, tr.translation.z],
            [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w])


class Collector(Node):
    def __init__(self, args):
        super().__init__("aic_collector")
        self.out = os.path.expanduser(args.out)
        self.scale = args.scale
        self.keep_every = max(1, round(20.0 / args.hz))
        man = json.load(open(args.manifest))
        eps = [m for m in man if m["chunk"] == args.chunk]
        eps.sort(key=lambda m: int(m["trial"].split("_")[1]))
        self.eps = eps
        self.n = len(eps)
        self.tfbuf = tf2_ros.Buffer(cache_time=Duration(seconds=30))
        self.tfl = tf2_ros.TransformListener(self.tfbuf, self)
        self.ordinal = -1
        self.in_trial = False
        self.present = self.absent = 0
        self.frame_i = 0
        self.msg_i = 0
        self.rows = []
        self.meta_done = False
        self.create_subscription(Observation, "/observations", self.cb, 10)
        self.get_logger().info(f"collector: chunk {args.chunk}, expecting {self.n} episodes -> {self.out}")

    # ---- TF helpers ----
    def has(self, frame):
        """True only if `frame` is CURRENTLY published (fresh TF). TF frames aren't 'deleted' when
        the cable despawns — the last transform lingers in the buffer for cache_time — so check the
        transform's age, not mere availability, to catch the per-trial cable spawn/despawn."""
        try:
            tf = self.tfbuf.lookup_transform("base_link", frame, Time())
            age = (self.get_clock().now() - Time.from_msg(tf.header.stamp)).nanoseconds * 1e-9
            return age < 0.75
        except Exception:
            return False

    def look(self, target, source, stamp):
        for t in (stamp, Time()):
            try:
                return _pose(self.tfbuf.lookup_transform(target, source, t).transform)
            except Exception:
                continue
        return None

    # ---- episode lifecycle ----
    def cb(self, msg):
        if not self.in_trial:
            nxt = self.ordinal + 1
            if nxt < self.n and self.has(self.eps[nxt]["plug_frame"]):
                self.present += 1
                if self.present >= 3:
                    self.start(nxt)
            else:
                self.present = 0
            return
        ep = self.eps[self.ordinal]
        if not self.has(ep["plug_frame"]):
            self.absent += 1
            if self.absent >= 3:
                self.end()
            return
        self.absent = 0
        self.msg_i += 1
        if self.msg_i % self.keep_every == 0:
            self.record(msg, ep)

    def start(self, ordinal):
        self.ordinal = ordinal
        self.in_trial = True
        self.present = self.absent = self.frame_i = 0
        self.rows = []
        self.meta_done = False
        ep = self.eps[ordinal]
        self.ep_dir = os.path.join(self.out, f"episode_{ep['episode']:04d}")
        for c in CAMS:
            os.makedirs(os.path.join(self.ep_dir, c), exist_ok=True)
        self.jf = open(os.path.join(self.ep_dir, "frames.jsonl"), "w")   # crash-safe incremental log
        self.get_logger().info(f"START ep {ep['episode']} ({ep['type']} {ep['trial']}) plug={ep['plug_frame']}")

    def end(self):
        ep = self.eps[self.ordinal]
        try:
            self.jf.close()
        except Exception:
            pass
        try:
            import pandas as pd
            pd.DataFrame(self.rows).to_parquet(os.path.join(self.ep_dir, "frames.parquet"))
        except Exception as e:
            self.get_logger().warn(f"parquet failed ({e}); frames.jsonl retained")
        self.get_logger().info(f"END ep {ep['episode']}: {len(self.rows)} frames")
        self.in_trial = False
        self.present = self.absent = 0

    # ---- per-frame recording ----
    def img_np(self, im):
        a = np.frombuffer(im.data, np.uint8).reshape(im.height, im.width, -1)[:, :, :3]
        if self.scale != 1.0:
            a = cv2.resize(a, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return a, im.encoding

    def record(self, msg, ep):
        stamp = Time.from_msg(msg.left_image.header.stamp)
        row = {"frame": self.frame_i,
               "stamp": msg.left_image.header.stamp.sec + msg.left_image.header.stamp.nanosec * 1e-9}
        for c, im in zip(CAMS, (msg.left_image, msg.center_image, msg.right_image)):
            arr, enc = self.img_np(im)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if enc == "rgb8" else arr
            cv2.imwrite(os.path.join(self.ep_dir, c, f"{self.frame_i:04d}.png"), bgr)
        pf, plf, ef = ep["port_frame"], ep["plug_frame"], ep["port_entrance_frame"]
        for key, tgt, src in (("port_base", "base_link", pf),
                              ("plug_base", "base_link", plf),
                              ("entrance_base", "base_link", ef)):
            r = self.look(tgt, src, stamp)
            if r:
                row[key + "_pos"], row[key + "_quat"] = r[0], r[1]
        for c in CAMS:
            r = self.look(f"{c}_camera/optical", pf, stamp)
            if r:
                row[f"port_{c}_pos"], row[f"port_{c}_quat"] = r[0], r[1]
        js = msg.joint_states
        row["joint_pos"] = list(js.position)
        w = msg.wrist_wrench.wrench
        row["wrench"] = [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
        cs = msg.controller_state
        row["tcp_pose"] = [cs.tcp_pose.position.x, cs.tcp_pose.position.y, cs.tcp_pose.position.z,
                           cs.tcp_pose.orientation.x, cs.tcp_pose.orientation.y,
                           cs.tcp_pose.orientation.z, cs.tcp_pose.orientation.w]
        row["tcp_vel"] = [cs.tcp_velocity.linear.x, cs.tcp_velocity.linear.y, cs.tcp_velocity.linear.z,
                          cs.tcp_velocity.angular.x, cs.tcp_velocity.angular.y, cs.tcp_velocity.angular.z]
        row["tcp_error"] = list(cs.tcp_error)
        self.rows.append(row)
        self.jf.write(json.dumps(row, default=float) + "\n")   # crash-safe incremental mirror
        self.jf.flush()
        self.frame_i += 1
        if not self.meta_done:
            self.write_meta(msg, ep)
            self.meta_done = True

    def write_meta(self, msg, ep):
        def K(ci):
            return list(ci.k) if hasattr(ci, "k") else list(ci.K)
        meta = {
            "episode": ep["episode"], "type": ep["type"], "chunk": ep["chunk"], "trial": ep["trial"],
            "port_frame": ep["port_frame"], "plug_frame": ep["plug_frame"],
            "port_entrance_frame": ep["port_entrance_frame"],
            "params": ep.get("params", {}),
            "image_scale": self.scale, "hz": 20.0 / self.keep_every,
            "image_encoding": msg.left_image.encoding,
            "image_size": [int(msg.left_image.width * self.scale), int(msg.left_image.height * self.scale)],
            "intrinsics": {c: K(ci) for c, ci in zip(
                CAMS, (msg.left_camera_info, msg.center_camera_info, msg.right_camera_info))},
            "joint_names": list(msg.joint_states.name),
        }
        with open(os.path.join(self.ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="~/aic_data/perception_v1")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=0.25)
    args, _ = ap.parse_known_args()          # ignore --ros-args ...
    rclpy.init()
    node = Collector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.in_trial:                    # flush a partial episode on shutdown
            node.end()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
