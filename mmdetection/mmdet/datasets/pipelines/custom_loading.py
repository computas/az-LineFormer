from ..builder import PIPELINES
from .loading import LoadAnnotations


@PIPELINES.register_module()
class CustomLoadAnnotations(LoadAnnotations):
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 with_event_points=True,
                 poly2mask=True,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):

        super().__init__(with_bbox,
                 with_label,
                 with_mask,
                 with_seg,
                 poly2mask,
                 denorm_bbox,
                 file_client_args)
        
        self.with_event_points = with_event_points


    def _load_event_points(self, results):
        """Private function to load event points.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        ann_info = results['ann_info']
        results['gt_event_points'] = ann_info['event_points'].copy()

        return results
    

    def __call__(self, results):
        results = super().__call__(results)
        if self.with_event_points:
            results = self._load_event_points(results)
        return results

    