#
#  GeoHome — DIAGNOSTIC. GeoReplayA (USE_GT_PORT) + return the arm to the pre-sweep HOME pose before the
#  approach loop. Tests the hypothesis that the wrist-raster SWEEP strands the arm in a bad config for the
#  high-yaw SC insertion (CheatCode reaches it only because it never sweeps / starts from home). GT port =
#  isolate execution. Run with ground_truth:=true.
#
import aic_example_policies.ros.PerceptionInsertYOLO as _yolo
_yolo.USE_GT_PORT = True   # feed the PERFECT GT port pose -> isolate execution

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion

from aic_example_policies.ros.PerceptionInsertGeo import PerceptionInsertGeo


class GeoHome(PerceptionInsertGeo):
    def _sweep_board(self, get_observation, move_robot):
        pts = super()._sweep_board(get_observation, move_robot)     # sweep + coarse lock (arm ends at raster pose)
        t = getattr(self, "_sweep_thome", None); q = getattr(self, "_sweep_qhome", None)
        if t is not None and q is not None:                          # return to the clean pre-sweep home pose
            pose = Pose(position=Point(x=float(t[0]), y=float(t[1]), z=float(t[2])),
                        orientation=Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3])))
            self.set_pose_target(move_robot=move_robot, pose=pose)
            for _ in range(40):
                self.sleep_for(0.05)
            self.get_logger().info(f"GeoHome: returned to home {np.round(t,3)} after sweep (pre-approach)")
        return pts
