# MTL project: Validator for multi-task learning (MTL) with segmentation and landmark detection
from ultralytics.engine.validator import BaseValidator
import torch

class MTLValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        # MTL project: Set up MTL-specific validation logic here as needed

    def update_metrics(self, preds, batch):
        # MTL project: Compute and store metrics for both segmentation and landmark detection
        seg_pred = preds['segmentation']
        lm_pred = preds['landmarks']
        seg_gt = batch['segmentation']
        lm_gt = batch['landmarks']
        # Example: IoU for segmentation, MSE for landmarks
        iou = self.compute_iou(seg_pred, seg_gt)
        mse = torch.nn.functional.mse_loss(lm_pred, lm_gt)
        if not hasattr(self, 'mtl_metrics'):
            self.mtl_metrics = {'iou': [], 'landmark_mse': []}
        self.mtl_metrics['iou'].append(iou)
        self.mtl_metrics['landmark_mse'].append(mse)

    def finalize_metrics(self):
        # MTL project: Aggregate and return metrics
        iou = torch.stack(self.mtl_metrics['iou']).mean().item() if hasattr(self, 'mtl_metrics') else 0.0
        mse = torch.stack(self.mtl_metrics['landmark_mse']).mean().item() if hasattr(self, 'mtl_metrics') else 0.0
        return {'segmentation_iou': iou, 'landmark_mse': mse}

    def compute_iou(self, pred, target):
        # MTL project: Compute mean IoU for segmentation masks
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        intersection = (pred * target).sum(dim=[1,2,3])
        union = ((pred + target) > 0).float().sum(dim=[1,2,3])
        iou = intersection / (union + 1e-6)
        return iou.mean() 