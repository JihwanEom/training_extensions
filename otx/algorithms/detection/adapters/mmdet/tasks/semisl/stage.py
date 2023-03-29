"""Stage for Semi-SL training with MMDET."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.tasks.stage import DetectionStage

logger = get_logger()


class SemiSLDetectionStage(DetectionStage):
    """Patch config to support semi supervised learning for object detection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_adapt_type = None
        self.task_adapt_op = "REPLACE"

    def configure_data(self, cfg, training, data_cfg):
        """Patch cfg.data."""
        super().configure_data(cfg, training, data_cfg)
        # Set unlabeled data hook
        if training:
            if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
                if len(cfg.data.unlabeled.get("pipeline", [])) == 0:
                    cfg.data.unlabeled.pipeline = cfg.data.train.pipeline.copy()
                self.configure_unlabeled_dataloader(cfg, self.distributed)

    def configure_task(self, cfg, training):
        """Patch config to support training algorithm."""
        logger.info(f"Semi-SL task config!!!!: training={training}")
        if "task_adapt" in cfg:
            self.task_adapt_type = cfg["task_adapt"].get("type", None)
            self.task_adapt_op = cfg["task_adapt"].get("op", "REPLACE")
            self.configure_classes(cfg)

            if self.data_classes != self.model_classes:
                self.configure_task_data_pipeline(cfg)
            # TODO[JAEGUK]: configure_anchor is not working
            if cfg["task_adapt"].get("use_mpa_anchor", False):
                self.configure_anchor(cfg)
            if self.task_adapt_type == "mpa":
                self.configure_bbox_head(cfg)
            else:
                src_data_cfg = self.get_data_cfg(cfg, "train")
                src_data_cfg.pop("old_new_indices", None)