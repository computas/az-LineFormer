import numpy as np
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .formatting import DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class CustomDefaultFormatBundle(DefaultFormatBundle):
    """Default formatting bundle.
        Add gt_event_points
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        super().__init__(self, img_to_float, pad_val)


    def __call__(self, results):
        super().__call__(self, results)

        # Add custom event point field
        if 'gt_event_points' in results:
            if results['gt_event_points']:
                results['gt_event_points'] = DC(
                    to_tensor(results['gt_event_points']),  # Convert to tensor
                    stack=False  # Disable stacking; each image may have a different number of points
                )
            else:
                results['gt_event_points'] = DC(
                    to_tensor([]).view(0, 2),  # Empty tensor with shape (0, 2)
                    stack=False
                )

        return results
    