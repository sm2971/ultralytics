# MTL project: Multi-task model with YOLO backbone, segmentation head, and U2Net landmark head
import torch
import torch.nn as nn
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.universal_landmark_detection.model.networks.u2net import U2Net

class YOLOSegLandmarkMTL(nn.Module):
    """
    MTL model with YOLOv11 backbone, v11 segmentation head, and U2Net-based landmark detection head.
    Note: SegmentationModel is v11 when used with v11 configs (e.g., 'yolo11n-seg.yaml').
    """
    def __init__(self, yolo_cfg="yolo11n-seg.yaml", u2net_in_channels=3, u2net_out_channels=19, ch=3, nc=None, verbose=True):
        super().__init__()
        # Segmentation model (YOLOv11 backbone + seg head)
        self.seg_model = SegmentationModel(cfg=yolo_cfg, ch=ch, nc=nc, verbose=verbose)
        # Landmark detection head (U2Net)
        self.lm_head = U2Net(in_channels=u2net_in_channels, out_channels=u2net_out_channels)

    def forward(self, x):
        seg_out = self.seg_model(x)
        lm_out = self.lm_head(x)
        return {'segmentation': seg_out, 'landmarks': lm_out} 