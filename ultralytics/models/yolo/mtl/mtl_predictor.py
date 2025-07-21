# MTL project: Predictor for multi-task learning (MTL) with segmentation and landmark detection
from ultralytics.engine.predictor import BasePredictor

class MTLPredictor(BasePredictor):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # MTL project: Set up MTL-specific prediction logic here as needed

    def postprocess(self, preds, img, orig_imgs):
        # MTL project: Format and return both segmentation and landmark outputs
        seg_out = preds['segmentation']
        lm_out = preds['landmarks']
        # Example: return a dict with both outputs
        return {
            'segmentation': seg_out,
            'landmarks': lm_out
        } 