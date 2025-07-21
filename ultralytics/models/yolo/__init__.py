# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo import classify, detect, obb, pose, segment, world, yoloe

from .model import YOLO, YOLOE, YOLOWorld

# MTL project: Register YOLOSegLandmarkMTL model for multi-task learning
from .mtl.mtl_model import YOLOSegLandmarkMTL

__all__ = "classify", "segment", "detect", "pose", "obb", "world", "yoloe", "YOLO", "YOLOWorld", "YOLOE"
