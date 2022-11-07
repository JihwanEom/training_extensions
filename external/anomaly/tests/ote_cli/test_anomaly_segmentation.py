"""Tests for anomaly segmentation with OTE CLI"""

# Copyright (C) 2021 Intel Corporation
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

import os
import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_sdk.entities.model_template import parse_model_template

from ote_cli.registry import Registry
from ote_cli.utils.tests import (
    create_venv,
    get_some_vars,
    ote_demo_deployment_testing,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_deployment_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_train_testing,
    ote_export_testing,
    pot_optimize_testing,
    pot_eval_testing,
    nncf_optimize_testing,
    nncf_export_testing,
    nncf_eval_testing,
    nncf_eval_openvino_testing,
)


args = {
    "--train-ann-file": "data/anomaly/segmentation/train.json",
    "--train-data-roots": "data/anomaly/shapes",
    "--val-ann-file": "data/anomaly/segmentation/val.json",
    "--val-data-roots": "data/anomaly/shapes",
    "--test-ann-files": "data/anomaly/segmentation/test.json",
    "--test-data-roots": "data/anomaly/shapes",
    "--input": "data/anomaly/shapes/test/hexagon",
    "train_params": [],
}

root = "/tmp/ote_cli/"
ote_dir = os.getcwd()

default_template = parse_model_template(
    os.path.join(
        "external/anomaly/configs", "segmentation", "padim", "template.yaml"
    )
)
templates = [default_template] * 100
templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]


class TestToolsAnomalySegmentation:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, _, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize('template', templates, ids=templates_ids)
    def test_ote_train(self, template):
        ote_train_testing(template, root, ote_dir, args)
