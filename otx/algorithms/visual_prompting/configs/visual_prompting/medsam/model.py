"""Model configuration of EfficientNetB2B-MaskRCNN model for Instance-Seg Task."""

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

# pylint: disable=invalid-name

_base_ = [
    "../../../../../recipes/stages/instance-segmentation/incremental.py",
    # "../../../../common/adapters/mmcv/configs/backbones/efficientnet_b2b.yaml",
    # "../../base/models/detector.py",
]

task = "visual-prompting"

model = dict(
    type="SegmentAnything",  # Use CustomMaskRCNN for Incremental Learning
    neck=dict(),
    roi_head=dict(
    bbox_head=dict(type="dummyhead"),
    ),
    train_cfg=dict(
        ),
    test_cfg=dict(
    ),
)

load_from = "https://storage.openvinotoolkit.org/repositories/\
openvino_training_extensions/models/instance_segmentation/\
v2/efficientnet_b2b-mask_rcnn-576x576.pth"

evaluation = dict(interval=1, metric="mAP", save_best="mAP", iou_thr=[0.5])
# fp16 = dict(loss_scale=512.0)
