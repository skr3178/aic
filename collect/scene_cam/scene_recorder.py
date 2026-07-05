#!/usr/bin/env python3
"""Host-side recorder for the scene_cam overlay: subscribe /scene_camera/image, dump PNGs.

Runs in the host pixi env exactly like the policy (same RMW path score.sh already proves).
Usage: python scene_recorder.py --out DIR [--every N] --ros-args -p use_sim_time:=true
"""
import argparse
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class SceneRecorder(Node):
    def __init__(self, out_dir, every):
        super().__init__("scene_recorder")
        self.out = out_dir
        self.every = every
        self.i = 0
        self.saved = 0
        os.makedirs(out_dir, exist_ok=True)
        self.create_subscription(Image, "/scene_camera/image", self.cb, 10)
        self.get_logger().info(f"recording /scene_camera/image -> {out_dir} (every {every})")

    def cb(self, msg):
        self.i += 1
        if (self.i - 1) % self.every:
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        cv2.imwrite(os.path.join(self.out, f"{self.saved:05d}.png"),
                    cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        self.saved += 1
        if self.saved % 50 == 1:
            self.get_logger().info(f"saved {self.saved} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1)
    args, _ = ap.parse_known_args()
    rclpy.init(args=sys.argv)
    rclpy.spin(SceneRecorder(args.out, args.every))


if __name__ == "__main__":
    main()
