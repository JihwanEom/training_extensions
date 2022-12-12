"""Openvino Task of OTX Action Recognition."""

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

import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
from addict import Dict as ADDict
from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline

from otx.algorithms.classification.adapters.openvino import model_wrappers
from otx.algorithms.classification.configs import ClassificationConfig
from otx.api.entities.annotation import AnnotationSceneEntity
from otx.api.entities.datasets import DatasetEntity, DatasetItemEntity
from otx.api.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.exportable_code import demo
from otx.api.usecases.exportable_code.inference import BaseInferencer
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    ClassificationToAnnotationConverter,
    DetectionBoxToAnnotationConverter,
    IPredictionToAnnotationConverter,
)
from otx.api.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

try:
    from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
    from openvino.model_zoo.model_api.models import Model
except ImportError:
    import warnings

    warnings.warn("ModelAPI was not found.")

logger = logging.getLogger(__name__)

# TODO: refactoring to Sphinx style.
class ActionClsOpenVINOInferencer(BaseInferencer):
    """ActionClsOpenVINOInferencer class in OpenVINO task."""

    @check_input_parameters_type()
    def __init__(
        self,
        task_type: str,
        hparams: ClassificationConfig,
        label_schema: LabelSchemaEntity,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):  # pylint: disable=unused-argument, too-many-arguments
        """Inferencer implementation for OTXDetection using OpenVINO backend.

        :param model: Path to model to load, `.xml`, `.bin` or `.onnx` file.
        :param hparams: Hyper parameters that the model should use.
        :param num_requests: Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
        :param device: Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        """

        self.task_type = task_type
        self.label_schema = label_schema
        model_adapter = OpenvinoAdapter(
            create_core(), model_file, weight_file, device=device, max_num_requests=num_requests
        )
        self.configuration: Dict[Any, Any] = {}
        self.model = Model.create_model(self.task_type, model_adapter, self.configuration, preload=True)
        self.converter: IPredictionToAnnotationConverter
        if self.task_type == "ACTION_CLASSIFICATION":
            self.converter = ClassificationToAnnotationConverter(self.label_schema)
        else:
            self.converter = DetectionBoxToAnnotationConverter(self.label_schema)

    @check_input_parameters_type()
    def pre_process(self, image: DatasetItemEntity) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Pre-process function of OpenVINO Action Classification Inferencer."""
        return self.model.preprocess(image)

    @check_input_parameters_type()
    def post_process(self, prediction, metadata: Dict[str, Any]) -> Optional[AnnotationSceneEntity]:
        """Post-process function of OpenVINO Classification Inferencer."""

        prediction = self.model.postprocess(prediction, metadata)
        return self.converter.convert_to_annotation(prediction, metadata)

    @check_input_parameters_type()
    def predict(self, image: DatasetItemEntity) -> Tuple[AnnotationSceneEntity, np.ndarray, np.ndarray, Any]:
        """Predict function of OpenVINO Action Classification Inferencer."""
        data, metadata = self.pre_process(image)
        raw_predictions = self.forward(data)
        predictions = self.post_process(raw_predictions, metadata)
        return predictions

    # @check_input_parameters_type()
    def forward(self, image: Dict[str, DatasetItemEntity]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Action Classification Inferencer."""

        return self.model.infer_sync(image)


class OTXOpenVinoDataLoader(DataLoader):
    """DataLoader implementation for ActionClsOpenVINOTask."""

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def __init__(self, dataset: DatasetEntity, inferencer: BaseInferencer):
        super().__init__(config=None)
        self.dataset = dataset
        self.inferencer = inferencer

    @check_input_parameters_type()
    def __getitem__(self, index: int):
        """Get item from dataset."""
        image = self.dataset[index]
        annotation = self.dataset[index].annotation_scene
        inputs, metadata = self.inferencer.pre_process(image)
        return (index, annotation), inputs, metadata

    def __len__(self):
        """Get length of dataset."""
        return len(self.dataset)


class ActionClsOpenVINOTask(IDeploymentTask, IInferenceTask, IEvaluationTask, IOptimizationTask):
    """Task implementation for OTXActionCls using OpenVINO backend."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.hparams = self.task_environment.get_hyper_parameters(ClassificationConfig)
        self.model = self.task_environment.model
        self.task_type = self.task_environment.model_template.task_type.name
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> ActionClsOpenVINOInferencer:
        """load_inferencer function of ClassificationOpenVINOTask."""

        if self.model is None:
            raise RuntimeError("load_inferencer failed, model is None")

        return ActionClsOpenVINOInferencer(
            self.task_type,
            self.hparams,
            self.task_environment.label_schema,
            self.model.get_data("openvino.xml"),
            self.model.get_data("openvino.bin"),
        )

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None
    ) -> DatasetEntity:
        """Infer function of ClassificationOpenVINOTask."""

        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress  # type: ignore
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            predicted_scene = self.inferencer.predict(dataset_item)
            if self.task_type == "ACTION_CLASSIFICATION":
                dataset_item.append_labels(predicted_scene.annotations[0].get_labels())
            else:
                dataset_item.append_annotations(predicted_scene.annotations)
            update_progress_callback(int(i / dataset_size * 100))
        return dataset

    @check_input_parameters_type()
    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate function of ClassificationOpenVINOTask."""

        if evaluation_metric is not None:
            logger.warning(f"Requested to use {evaluation_metric} metric," "but parameter is ignored.")
        if self.task_type == "ACTION_CLASSIFICATION":
            output_resultset.performance = MetricsHelper.compute_accuracy(output_resultset).get_performance()
        elif self.task_type == "ACTION_DETECTION":
            output_resultset.performance = MetricsHelper.compute_f_measure(output_resultset).get_performance()

    @check_input_parameters_type()
    def deploy(self, output_model: ModelEntity) -> None:
        """Deploy function of ClassificationOpenVINOTask."""

        logger.info("Deploying the model")

        work_dir = os.path.dirname(demo.__file__)
        parameters = {}  # type: Dict[Any, Any]
        parameters["type_of_model"] = f"otx_{self.task_type.lower()}"
        parameters["converter_type"] = f"{self.task_type}"
        parameters["model_parameters"] = self.inferencer.configuration
        parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(self.task_environment.label_schema)

        if self.model is None:
            raise RuntimeError("deploy failed, model is None")

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            arch.writestr(os.path.join("model", "model.xml"), self.model.get_data("openvino.xml"))
            arch.writestr(os.path.join("model", "model.bin"), self.model.get_data("openvino.bin"))
            arch.writestr(os.path.join("model", "config.json"), json.dumps(parameters, ensure_ascii=False, indent=4))
            # model_wrappers files
            for root, _, files in os.walk(os.path.dirname(model_wrappers.__file__)):
                for file in files:
                    file_path = os.path.join(root, file)
                    arch.write(
                        file_path, os.path.join("python", "model_wrappers", file_path.split("model_wrappers/")[1])
                    )
            # other python files
            arch.write(os.path.join(work_dir, "requirements.txt"), os.path.join("python", "requirements.txt"))
            arch.write(os.path.join(work_dir, "LICENSE"), os.path.join("python", "LICENSE"))
            arch.write(os.path.join(work_dir, "README.md"), os.path.join("python", "README.md"))
            arch.write(os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py"))
        output_model.exportable_code = zip_buffer.getvalue()
        logger.info("Deploying completed")

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):  # pylint: disable=too-many-locals
        """Optimize function of ClassificationOpenVINOTask."""

        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVino models")

        data_loader = OTXOpenVinoDataLoader(dataset, self.inferencer)

        if self.model is None:
            raise RuntimeError("optimize failed, model is None")

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            model_config = ADDict({"model_name": "openvino_model", "model": xml_path, "weights": bin_path})

            model = load_model(model_config)

            if get_nodes_by_type(model, ["FakeQuantize"]):
                raise RuntimeError("Model is already optimized by POT")

        if optimization_parameters is not None:
            optimization_parameters.update_progress(10, None)

        engine_config = ADDict({"device": "CPU"})

        stat_subset_size = self.hparams.pot_parameters.stat_subset_size
        preset = self.hparams.pot_parameters.preset.name.lower()

        algorithms = [
            {
                "name": "DefaultQuantization",
                "params": {
                    "target_device": "ANY",
                    "preset": preset,
                    "stat_subset_size": min(stat_subset_size, len(data_loader)),
                    "shuffle_data": True,
                },
            }
        ]

        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

        pipeline = create_pipeline(algorithms, engine)

        compressed_model = pipeline.run(model)

        compress_model_weights(compressed_model)

        if optimization_parameters is not None:
            optimization_parameters.update_progress(90, None)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            with open(os.path.join(tempdir, "model.xml"), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())

        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))

        # set model attributes for quantized model
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()

        if optimization_parameters is not None:
            optimization_parameters.update_progress(100, None)
        logger.info("POT optimization completed")