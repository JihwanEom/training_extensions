"""Module for defining hierarchical classification head for vision transformer models."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads import VisionTransformerClsHead


@HEADS.register_module()
class VisionTransformerHierarchicalClsHead(VisionTransformerClsHead):
    """Hierarchical head for vision transformer models.

    # TODO: update docstring, check performance
    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
        multilabel_loss (dict): Config of multi-label classification loss.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        multilabel_loss = kwargs.pop("multilabel_loss", dict(type="AsymmetricLoss", reduction="mean", loss_weight=1.0))
        loss = kwargs.pop("loss", dict(type="CrossEntropyLoss", use_sigmoid=False, reduction="mean", loss_weight=1.0))
        self.hierarchical_info = kwargs.pop("hierarchical_info", None)
        assert self.hierarchical_info
        super().__init__(loss=loss, *args, **kwargs)
        if self.hierarchical_info["num_multiclass_heads"] + self.hierarchical_info["num_multilabel_classes"] == 0:
            raise ValueError("Invalid classification heads configuration")
        self.compute_multilabel_loss = False
        if self.hierarchical_info["num_multilabel_classes"] > 0:
            self.compute_multilabel_loss = build_loss(multilabel_loss)

    def loss(self, cls_score, gt_label, multilabel=False, valid_label_mask=None):
        """Calculate loss for given cls_score/gt_label."""
        num_samples = len(cls_score)
        # compute loss
        if multilabel:
            gt_label = gt_label.type_as(cls_score)
            # map difficult examples to positive ones
            _gt_label = torch.abs(gt_label)

            loss = self.compute_multilabel_loss(
                cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples
            )
        else:
            loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)

        return loss

    def forward(self, x):
        """Forward fuction of VisionTransformerHierarchicalClsHead."""
        return self.simple_test(x)

    def forward_train(self, x, gt_label, **kwargs):
        """Forward_train fuction of VisionTransformerHierarchicalClsHead class."""
        img_metas = kwargs.get("img_metas", None)
        x = self.pre_logits(x)
        gt_label = gt_label.type_as(x)
        cls_score = self.layers.head(x)

        losses = dict(loss=0.0)
        for i in range(self.hierarchical_info["num_multiclass_heads"]):
            head_gt = gt_label[:, i]
            head_logits = cls_score[
                :,
                self.hierarchical_info["head_idx_to_logits_range"][str(i)][0] : self.hierarchical_info[
                    "head_idx_to_logits_range"
                ][str(i)][1],
            ]
            valid_mask = head_gt >= 0
            head_gt = head_gt[valid_mask].long()
            head_logits = head_logits[valid_mask, :]
            multiclass_loss = self.loss(head_logits, head_gt)
            losses["loss"] += multiclass_loss

        if self.hierarchical_info["num_multiclass_heads"] > 1:
            losses["loss"] /= self.hierarchical_info["num_multiclass_heads"]

        if self.compute_multilabel_loss:
            head_gt = gt_label[:, self.hierarchical_info["num_multiclass_heads"] :]
            head_logits = cls_score[:, self.hierarchical_info["num_single_label_classes"] :]
            valid_batch_mask = head_gt >= 0
            head_gt = head_gt[
                valid_batch_mask,
            ]
            head_logits = head_logits[
                valid_batch_mask,
            ]

            # multilabel_loss is assumed to perform no batch averaging
            if img_metas is not None:
                valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)[
                    :, self.hierarchical_info["num_single_label_classes"] :
                ]
                valid_label_mask = valid_label_mask.to(cls_score.device)
                valid_label_mask = valid_label_mask[valid_batch_mask]
            else:
                valid_label_mask = None
            multilabel_loss = self.loss(head_logits, head_gt, multilabel=True, valid_label_mask=valid_label_mask)
            losses["loss"] += multilabel_loss.mean()
        return losses

    def simple_test(self, x):
        """Test without augmentation."""
        x = self.pre_logits(x)
        cls_score = self.layers.head(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        multiclass_logits = []
        for i in range(self.hierarchical_info["num_multiclass_heads"]):
            multiclass_logit = cls_score[
                :,
                self.hierarchical_info["head_idx_to_logits_range"][str(i)][0] : self.hierarchical_info[
                    "head_idx_to_logits_range"
                ][str(i)][1],
            ]
            if not torch.onnx.is_in_onnx_export():
                multiclass_logit = torch.softmax(multiclass_logit, dim=1)
            multiclass_logits.append(multiclass_logit)
        multiclass_pred = torch.cat(multiclass_logits, dim=1) if multiclass_logits else None

        if self.compute_multilabel_loss:
            multilabel_logits = cls_score[:, self.hierarchical_info["num_single_label_classes"] :]
            if not torch.onnx.is_in_onnx_export():
                multilabel_pred = torch.sigmoid(multilabel_logits) if multilabel_logits is not None else None
            else:
                multilabel_pred = multilabel_logits
            if multiclass_pred is not None:
                pred = torch.cat([multiclass_pred, multilabel_pred], axis=1)
            else:
                pred = multilabel_pred
        else:
            pred = multiclass_pred

        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        """Get valid label with ignored_label mask."""
        valid_label_mask = []
        for meta in img_metas:
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask
