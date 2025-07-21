# MTL project: Trainer for multi-task learning (MTL) with segmentation and landmark detection
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.utils.loss import MTLLoss
from ultralytics.nn.tasks import v8SegmentationLoss  # v8SegmentationLoss is used for v11 models as well
import torch.nn as nn

def get_actual_model(model):
    # Utility to get the actual model if wrapped and ensure it is a nn.Module
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        return model.model
    if isinstance(model, nn.Module):
        return model
    raise TypeError('MTLTrainer: model is not a valid nn.Module')

# Example landmark loss (MSE for coordinates or heatmaps)
def landmark_loss(pred, target):
    # Assume pred and target are both [B, N, 2] or heatmaps [B, N, H, W]
    return nn.MSELoss()(pred, target)

class MTLTrainer(BaseTrainer):
    def __init__(self, cfg, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # MTL project: seg_loss_fn and lm_loss_fn will be set after model is loaded
        self.seg_loss_fn = None
        self.lm_loss_fn = landmark_loss
        self.lf = None

    def setup_model(self):
        super().setup_model()
        # MTL project: Set up MTL-specific loss after model is loaded
        # Note: v8SegmentationLoss is used for v11 models as well; SegmentationModel is v11 when used with v11 configs
        model = get_actual_model(self.model)
        if hasattr(model, 'seg_model') and isinstance(model.seg_model, nn.Module):
            self.seg_loss_fn = v8SegmentationLoss(model.seg_model)
        else:
            raise AttributeError('MTLTrainer: model does not have seg_model attribute or it is not a nn.Module')
        self.lf = MTLLoss(self.seg_loss_fn, self.lm_loss_fn, seg_weight=1.0, lm_weight=1.0)

    def preprocess_batch(self, batch):
        # MTL project: Ensure batch contains both segmentation and landmark targets
        # batch['image'], batch['segmentation'], batch['landmarks'] expected
        return batch

    def compute_loss(self, preds, batch):
        # MTL project: Compute combined loss for both tasks
        if self.lf is None:
            raise RuntimeError('MTLTrainer: Loss function not initialized')
        return self.lf(preds, batch) 