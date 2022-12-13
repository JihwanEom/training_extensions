"""EfficientNet-V2 for multi-class config."""

# pylint: disable=invalid-name

_base_ = "../base/models/efficientnet_v2.py"

model = dict(
    task="classification",
    type="SupConClassifier",
    backbone=dict(
        version="s_21k",
    ),
    head=dict(
        type="SupConClsHead",
        in_channels=-1,
        aux_mlp=dict(hid_channels=0, out_channels=1024),
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        aux_loss=dict(
            type="BarlowTwinsLoss",
            off_diag_penality=1.0 / 128.0,
            loss_weight=1.0,
        ),
    ),
)

fp16 = dict(loss_scale=512.0)