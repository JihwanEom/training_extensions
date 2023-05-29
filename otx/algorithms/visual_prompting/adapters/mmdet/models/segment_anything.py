"""OTX MaskRCNN Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmdet.models.builder import DETECTORS, build_loss

from otx.algorithms.common.utils.logger import get_logger
from mmcv.runner import BaseModule, auto_fp16
import torch.nn.functional as F
from mmdet.models.detectors.base import BaseDetector
from segment_anything import sam_model_registry
import torch.nn as nn

logger = get_logger()

# pylint: disable=too-many-locals, protected-access, unused-argument

ALPHA = 0.8
GAMMA = 2


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou

class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

@DETECTORS.register_module()
class SegmentAnything(BaseDetector):
    def __init__(self,
                 model_type='vit_b',
                 ckpt_path='sam_vit_b_01ec64.pth',
                 freeze_image_encoder=True,
                 freeze_prompt_encoder=True,
                 freeze_mask_decoder=False,
                 **kwargs):
        super().__init__()
        self.model = sam_model_registry[model_type](checkpoint=ckpt_path)
        focal_loss = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)
        # dice_loss = dict(
        #     type="DiceLoss",
        #     use_sigmoid=True,
        #     activate=True,
        #     reduction='mean',
        #     naive_dice=False,
        #     loss_weight=1.0,
        #     eps=1e-3
        # )
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        
        
        self.model.train()
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def extract_feat(self, img, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        _, _, H, W = img.shape
        image_embeddings = self.model.image_encoder(img)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, gt_bboxes):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
        return pred_masks, ious

    def simple_test(self, img, img_metas, **kwargs):
        """Test without augmentation."""
        gt_bboxes = kwargs.get("gt_bboxes", [])
        pred_masks, ious = self.extract_feat(img, gt_bboxes[0])
        return pred_masks, ious

    def aug_test(self, img, img_metas, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        gt_bboxes = kwargs.get("gt_bboxes", [])
        pred_masks, ious = self.extract_feat(img, gt_bboxes[0])
        return pred_masks, ious

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        pred_masks, ious = self.extract_feat(img, gt_bboxes)

        loss_focal = 0
        loss_dice = 0
        loss_iou = 0
        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        gt_masks = [i.to_tensor(float, self.model.device) for i in gt_masks]
        for pred, gt, iou in zip(pred_masks, gt_masks, ious):
            gt = gt.contiguous().float()
            batch_iou = calc_iou(pred, gt)
            loss_focal += self.focal_loss(pred, gt)
            loss_dice += self.dice_loss(pred, gt)
            loss_iou += F.mse_loss(iou, batch_iou, reduction='sum') / num_masks

        losses = dict(loss_focal=loss_focal*20,
                      loss_dice=loss_dice,
                      loss_iou=loss_iou)
        return losses

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        images = batch['image']
        bboxes = batch['bbox']

        pred_masks, ious = self.forward(images, bboxes)

        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        loss_focal = torch.tensor(0., device=self.model.device)
        loss_dice = torch.tensor(0., device=self.model.device)
        loss_iou = torch.tensor(0., device=self.model.device)

        focal_loss = FocalLoss()
        dice_loss = DiceLoss()
        gt_masks = batch['mask']
        for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, ious):
            gt_mask = gt_mask.float()
            batch_iou = calc_iou(pred_mask, gt_mask)
            loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
            loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
            loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks
        loss_total = 20. * loss_focal + loss_dice + loss_iou

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss_total)
        return loss_total

    def validation_step(self, batch, batch_idx):
        # this is the validation loop

        ious = AverageMeter()
        f1_scores = AverageMeter()

        images = batch['image']
        bboxes = batch['bbox']
        gt_masks = batch['mask']
        num_images = images.size(0)

        pred_masks, _ = self.forward(images, bboxes)

        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            batch_stats = smp.metrics.get_stats(
                pred_mask,
                gt_mask.int(),
                mode='binary',
                threshold=0.5,
            )
            batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
            batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
            ious.update(batch_iou, num_images)
            f1_scores.update(batch_f1, num_images)
        print(f"IoU: {batch_iou.item():.4f}, F1: {batch_f1.item():.4f}")
        result = dict(iou=ious.avg, f1_score=f1_scores.avg)
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
        return optimizer
