"""Visual Prompting Task."""

# Copyright (C) 2023 Intel Corporation
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

import ctypes
import io
import os
import shutil
import subprocess  # nosec
import tempfile
import time
from collections import OrderedDict
from glob import glob
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.config import (
    get_visual_promtping_config,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets import (
    OTXVisualPromptingDataModule,
)
from otx.algorithms.visual_prompting.configs.base.configuration import (
    VisualPromptingBaseConfig,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.metrics import NullPerformance, Performance, ScoreMetric
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.callbacks import InferenceCallback

logger = get_logger()


# pylint: disable=too-many-instance-attributes
class InferenceTask(IInferenceTask, IEvaluationTask, IExportTask, IUnload):
    """Base Visual Prompting Task."""

    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None) -> None:
        """Train, Infer, Export, Optimize and Deploy an Visual Prompting Task.

        Args:
            task_environment (TaskEnvironment): OTX Task environment.
            output_path (Optional[str]): output path where task output are saved.
        """
        torch.backends.cudnn.enabled = True
        logger.info("Initializing the task environment.")
        self.task_environment = task_environment
        self.task_type = task_environment.model_template.task_type
        self.model_name = task_environment.model_template.name
        self.labels = task_environment.get_labels()

        template_file_path = task_environment.model_template.model_template_path
        self.base_dir = os.path.abspath(os.path.dirname(template_file_path))

        # Hyperparameters.
        self._work_dir_is_temp = False
        self.output_path = output_path
        if self.output_path is None:
            self.output_path = tempfile.mkdtemp(prefix="otx-visual_prompting")
            self._work_dir_is_temp = True
        self.config = self.get_config()

        # Set default model attributes.
        self.optimization_methods: List[OptimizationMethod] = []
        self.precision = [ModelPrecision.FP32]
        self.optimization_type = ModelOptimizationType.MO

        self.model = self.load_model(otx_model=task_environment.model)

        self.trainer: Trainer
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def get_config(self) -> Union[DictConfig, ListConfig]:
        """Get Visual Prompting Config from task environment.

        Returns:
            Union[DictConfig, ListConfig]: Visual Prompting config.
        """
        self.hyper_parameters: VisualPromptingBaseConfig = self.task_environment.get_hyper_parameters()
        config = get_visual_promtping_config(self.model_name, self.hyper_parameters, self.output_path)

        config.dataset.task = "visual_prompting"

        return config

    def load_model(self, otx_model: Optional[ModelEntity]) -> LightningModule:
        """Create and Load Visual Prompting Module.

        Currently, load model through `sam_model_registry` because there is only SAM.
        If other visual prompting model is added, loading model process must be changed.

        Args:
            otx_model (Optional[ModelEntity]): OTX Model from the task environment.

        Returns:
            LightningModule: Visual prompting model with/without weights.
        """
        def get_model(config: DictConfig, state_dict: Optional[OrderedDict] = None):
            if config.model.name == "SAM":
                from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models import (
                    SegmentAnything,
                )
                model = SegmentAnything(config=config, state_dict=state_dict)
            else:
                raise NotImplementedError((
                    f"Current selected model {config.model.name} is not implemented. "
                    f"Use SAM instead."
                ))
            return model
            
        if otx_model is None:
            model = get_model(config=self.config)
            logger.info(
                "No trained model in project yet. Created new model with '%s'",
                self.model_name,
            )
        else:
            buffer = io.BytesIO(otx_model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))
            from_lightning = False
            if model_data.get("config", None):
                from_lightning = True
                logger.info("Load pytorch lightning checkpoint.")
                if model_data["config"]["model"]["backbone"] != self.config["model"]["backbone"]:
                    logger.warning(
                        "Backbone of the model in the Task Environment is different from the one in the template. "
                        f"creating model with backbone={model_data['config']['model']['backbone']}"
                    )
                    self.config["model"]["backbone"] = model_data["config"]["model"]["backbone"]
            else:
                logger.info("Load pytorch checkpoint.")

            try:
                model = get_model(config=self.config, state_dict=model_data["model"] if from_lightning else model_data)
                logger.info("Complete to load model weights.")
            except BaseException as exception:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") from exception

        return model

    def cancel_training(self) -> None:
        """Cancel the training `after_batch_end`.

        This terminates the training; however validation is still performed.
        """
        logger.info("Cancel training requested.")
        self.trainer.should_stop = True

        # The runner periodically checks `.stop_training` file to ensure if cancellation is requested.
        cancel_training_file_path = os.path.join(self.config.project.path, ".stop_training")
        with open(file=cancel_training_file_path, mode="a", encoding="utf-8"):
            pass

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        """Perform inference on a dataset.

        Args:
            dataset (DatasetEntity): Dataset to infer.
            inference_parameters (InferenceParameters): Inference parameters.

        Returns:
            DatasetEntity: Output dataset with predictions.
        """
        logger.info("Performing inference on the validation set using the base torch model.")
        datamodule = OTXVisualPromptingDataModule(config=self.config.dataset, dataset=dataset)

        logger.info("Inference Configs '%s'", self.config)

        # Callbacks
        inference_callback = InferenceCallback(otx_dataset=dataset)
        callbacks = [TQDMProgressBar(), inference_callback]

        self.trainer = Trainer(**self.config.trainer, logger=False, callbacks=callbacks)
        self.trainer.predict(model=self.model, datamodule=datamodule)

        return inference_callback.otx_dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None) -> None:
        """Evaluate the performance on a result set.

        Args:
            output_resultset (ResultSetEntity): Result Set from which the performance is evaluated.
            evaluation_metric (Optional[str], optional): Evaluation metric. Defaults to None. Instead,
                metric is chosen depending on the task type.
        """
        metric = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metric.overall_dice.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def _export_to_onnx(self, onnx_path: str):
        """Export model to ONNX.

        Args:
             onnx_path (str): path to save ONNX file
        """
        raise NotImplementedError

    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = False,
    ) -> None:
        """Export model to OpenVINO IR.

        Args:
            export_type (ExportType): Export type should be ExportType.OPENVINO
            output_model (ModelEntity): The model entity in which to write the OpenVINO IR data
            precision (bool): Output model weights and inference precision
            dump_features (bool): Flag to return "feature_vector" and "saliency_map".

        Raises:
            Exception: If export_type is not ExportType.OPENVINO
        """
        raise NotImplementedError

    def model_info(self) -> Dict:
        """Return model info to save the model weights.

        Returns:
           Dict: Model info.
        """
        return {
            "model": self.model.state_dict(),
            "config": self.get_config(),
            "version": self.trainer.logger.version,
        }

    def save_model(self, output_model: ModelEntity) -> None:
        """Save the model after training is completed.

        Args:
            output_model (ModelEntity): Output model onto which the weights are saved.
        """
        logger.info("Saving the model weights.")
        model_info = self.model_info()
        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))
        self._set_metadata(output_model)

        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods

    def _set_metadata(self, output_model: ModelEntity):
        """"""
        if hasattr(self.model, "image_threshold"):
            output_model.set_data("image_threshold", self.model.image_threshold.value.cpu().numpy().tobytes())
        if hasattr(self.model, "pixel_threshold"):
            output_model.set_data("pixel_threshold", self.model.pixel_threshold.value.cpu().numpy().tobytes())
        if hasattr(self.model, "normalization_metrics"):
            output_model.set_data("min", self.model.normalization_metrics.state_dict()["min"].cpu().numpy().tobytes())
            output_model.set_data("max", self.model.normalization_metrics.state_dict()["max"].cpu().numpy().tobytes())
        else:
            logger.warning(
                "The model was not trained before saving. This will lead to incorrect normalization of the heatmaps."
            )

    @staticmethod
    def _is_docker() -> bool:
        """Check whether the task runs in docker container.

        Returns:
            bool: True if task runs in docker, False otherwise.
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Unload the task."""
        self.cleanup()

        if self._is_docker():
            logger.warning("Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            ctypes.string_at(0)

        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(
                "Done unloading. Torch is still occupying %f bytes of GPU memory",
                torch.cuda.memory_allocated(),
            )

    def cleanup(self) -> None:
        """Clean up work directory."""
        if self._work_dir_is_temp:
            self._delete_scratch_space()

    def _delete_scratch_space(self) -> None:
        """Remove model checkpoints and otx logs."""
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path, ignore_errors=False)
