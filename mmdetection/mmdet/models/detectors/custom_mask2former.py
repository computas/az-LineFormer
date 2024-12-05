# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn.functional as F

from ..builder import DETECTORS, build_head
from .mask2former import Mask2Former
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CustomMask2Former(Mask2Former):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_."""

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 event_point_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        

        event_point_head_ = copy.deepcopy(event_point_head)
        self.event_point_head = build_head(event_point_head_)
    

    def forward_train(self,
                    img,
                    img_metas,
                    gt_bboxes,
                    gt_labels,
                    gt_masks,
                    gt_event_points,
                    gt_semantic_seg=None,
                    gt_bboxes_ignore=None,
                    **kargs):
                
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            gt_masks (list[BitmapMasks]): true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_semantic_seg (list[tensor]): semantic segmentation mask for
                images for panoptic segmentation.
                Defaults to None for instance segmentation.
            gt_bboxes_ignore (list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # add batch_input_shape in img_metas
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses, all_cls_scores, all_mask_preds = self.panoptic_head.forward_train(x, img_metas, gt_bboxes,
                                                                                    gt_labels, gt_masks,
                                                                                    gt_semantic_seg,
                                                                                    gt_bboxes_ignore)
        
        print(len(all_cls_scores))
        print(all_cls_scores[0])
        print()
        print(len(all_mask_preds))
        print(all_mask_preds[0])

        all_pan_results = []
        for mask_cls_result, mask_pred_result in zip(all_cls_scores, all_mask_preds):
            pan_result = self.panoptic_fusion_head.panoptic_postprocess(mask_cls_result, mask_pred_result)
            all_pan_results.append(pan_result)

        print()
        print(len(all_pan_results))
        print(all_pan_results[0])


        


        # Compute event point head losses
        event_point_losses = self.event_point_head.forward_train(all_mask_preds, gt_event_points)
        losses.update(event_point_losses)

        return losses    


    def preprocess_pan_results(self, pan_results, target_shape):
        # Resize masks to a fixed size (e.g., target_shape) if needed
        resized_masks = [F.interpolate(mask.unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False)
                        for mask in pan_results]
        return torch.stack(resized_masks)  # Shape: (N, C, H, W)
