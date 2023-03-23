"""Module for defining MMOVNeck for inference."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List

from mmcls.models.builder import NECKS

from otx.mpa.modules.ov.graph.parsers.cls.cls_base_parser import cls_base_parser
from otx.mpa.modules.ov.models.mmov_model import MMOVModel


@NECKS.register_module()
class MMOVNeck(MMOVModel):
    """Neck class for MMOV inference."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def parser(graph, **kwargs) -> Dict[str, List[str]]:
        """Parser function returns base_parser for given graph."""
        output = cls_base_parser(graph, "neck")
        if output is None:
            raise ValueError("Parser can not determine input and output of model. Please provide them explicitly")
        return output