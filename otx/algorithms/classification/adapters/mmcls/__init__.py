"""Adapters of classification - mmcls."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from .data import MPAClsDataset
from .data.twocrop_transform import ColorJitter, RandomAppliedTrans, TwoCropTransform
from .model.heads.selfsl_cls_head import SelfSLClsHead
from .model.losses.barlowtwins_loss import BarlowTwinsLoss
from .model.selfsl_classifier import SelfSLClassifier

__all__ = [
    "MPAClsDataset",
    "TwoCropTransform",
    "RandomAppliedTrans",
    "ColorJitter",
    "SelfSLClsHead",
    "BarlowTwinsLoss",
    "SelfSLClassifier"
]