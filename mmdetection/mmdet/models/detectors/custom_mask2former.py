# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from ..builder import DETECTORS
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
        
        self.event_point_head = event_point_head
    

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

        
        # Pass all_mask_preds through the custom event point prediction CNN
        event_point_coords = self.event_point_head(all_mask_preds)

        # Compute losses for event points (we'll create a custom loss function)
        event_loss = self.event_point_loss(event_point_coords, gt_event_points)

        losses.update(event_loss)  # Add event point loss to the final losses


        return losses
    

    def event_point_loss(self, pred_coords, true_coords):
        """
        Custom loss function for event point regression.
        Args:
            pred_coords (Tensor): Predicted coordinates of shape (batch_size, N_max, 2)
            true_coords (Tensor): Ground truth coordinates of shape (batch_size, N_max, 2)
        """
        # Assuming we want to compute MSE loss for the coordinates
        loss = F.mse_loss(pred_coords, true_coords, reduction='mean')
        return loss
    
