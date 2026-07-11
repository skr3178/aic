#
#  GeoReplayA — DIAGNOSTIC (not for scoring). PerceptionInsertGeo with USE_GT_PORT=True, i.e. the insertion
#  is fed the PERFECT GT port pose (position + orientation) instead of the perceived lock. Isolates the SC
#  trial arm-stall: if it STILL stalls ~53cm short with GT, the failure is execution/IK/reachability, not
#  perception. Run with ground_truth:=true.
#
import aic_example_policies.ros.PerceptionInsertYOLO as _yolo
_yolo.USE_GT_PORT = True   # feed the PERFECT GT port pose to the insertion (module-global, read in _perceive)

from aic_example_policies.ros.PerceptionInsertGeo import PerceptionInsertGeo


class GeoReplayA(PerceptionInsertGeo):
    pass
