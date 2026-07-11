#
#  GeoRealPlug — DIAGNOSTIC. GT port + use the REAL plug frame in calc_gripper_pose (exactly CheatCode's
#  inputs) instead of the modeled FK.T_GRASP[type] plug. Tests whether T_GRASP[sc]'s ORIENTATION is what
#  makes the commanded wrist pose unreachable on the high-yaw SC trial (CheatCode reaches it; we stall).
#  Run with ground_truth:=true.
#
import aic_example_policies.ros.PerceptionInsertYOLO as _yolo
_yolo.USE_GT_PORT = True

from rclpy.time import Time
from aic_example_policies.ros.PerceptionInsertGeo import PerceptionInsertGeo


class GeoRealPlug(PerceptionInsertGeo):
    def calc_gripper_pose(self, port_transform, plug_transform, *a, **kw):
        try:  # swap in the REAL plug link pose (like CheatCode) instead of FK.T_GRASP
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", f"{self._task.cable_name}/{self._task.plug_name}_link", Time())
            plug_transform = tf.transform
        except Exception as e:
            self.get_logger().warn(f"real plug TF unavailable, using modeled: {e}")
        return super().calc_gripper_pose(port_transform, plug_transform, *a, **kw)
