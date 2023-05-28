"""Anomaly Classification Task."""

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

import ctypes
import io
import os
import shutil
import subprocess  # nosec
import tempfile
from glob import glob
from typing import Dict, List, Optional, Union
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from torch import optim, nn, utils, Tensor
import segmentation_models_pytorch as smp

import torch.nn.functional as F
import torch
from anomalib.models import AnomalyModule, get_model
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer

from otx.algorithms.anomaly.adapters.anomalib.callbacks import (
    AnomalyInferenceCallback,
    ProgressCallback,
)
# from otx.algorithms.anomaly.adapters.anomalib.config import get_anomalib_config
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.config import get_visual_promtping_config
from otx.algorithms.anomaly.adapters.anomalib.data import OTXAnomalyDataModule
from otx.algorithms.anomaly.adapters.anomalib.logger import get_logger
from otx.algorithms.anomaly.configs.base.configuration import BaseAnomalyConfig
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
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
torch.set_float32_matmul_precision('high')

ALPHA = 0.8
GAMMA = 2

logger = get_logger(__name__)

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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


# pylint: disable=too-many-instance-attributes
class InferenceTask(IInferenceTask, IEvaluationTask, IExportTask, IUnload):
    """Base Anomaly Task."""

    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None) -> None:
        """Train, Infer, Export, Optimize and Deploy an Anomaly Classification Task.

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
        if output_path is None:
            output_path = tempfile.mkdtemp(prefix="otx-anomalib")
            self._work_dir_is_temp = True
        self.project_path: str = output_path
        self.config = self.get_config()

        # Set default model attributes.
        self.optimization_methods: List[OptimizationMethod] = []
        self.precision = [ModelPrecision.FP32]
        self.optimization_type = ModelOptimizationType.MO

        # self.model = self.load_model(otx_model=task_environment.model)

        import pytorch_lightning as pl
        # define the LightningModule
        class SegmentAnything(pl.LightningModule):
            def __init__(self, model_type='vit_b', ckpt_path='sam_vit_b_01ec64.pth'):
                super().__init__()
                self.model = sam_model_registry[model_type](checkpoint=ckpt_path)
                self.model.train()
                # if self.cfg.model.freeze.image_encoder:
                if False:
                    for param in self.model.image_encoder.parameters():
                        param.requires_grad = False
                # if self.cfg.model.freeze.prompt_encoder:
                if False:
                    for param in self.model.prompt_encoder.parameters():
                        param.requires_grad = False
                # if self.cfg.model.freeze.mask_decoder:
                if True:
                    for param in self.model.mask_decoder.parameters():
                        param.requires_grad = False

            def forward(self, images, bboxes):
                _, _, H, W = images.shape
                image_embeddings = self.model.image_encoder(images)
                pred_masks = []
                ious = []
                for embedding, bbox in zip(image_embeddings, bboxes):
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

        self.model = SegmentAnything()
        self.trainer: Trainer

    def get_config(self) -> Union[DictConfig, ListConfig]:
        """Get Anomalib Config from task environment.

        Returns:
            Union[DictConfig, ListConfig]: Anomalib config.
        """
        self.hyper_parameters: BaseAnomalyConfig = self.task_environment.get_hyper_parameters()
        config = get_visual_promtping_config(task_name=self.model_name, otx_config=self.hyper_parameters)
        config.project.path = self.project_path

        config.dataset.task = "classification"

        return config

    def load_model(self, otx_model: Optional[ModelEntity]) -> AnomalyModule:
        """Create and Load Anomalib Module from OTX Model.

        This method checks if the task environment has a saved OTX Model,
        and creates one. If the OTX model already exists, it returns the
        the model with the saved weights.

        Args:
            otx_model (Optional[ModelEntity]): OTX Model from the
                task environment.

        Returns:
            AnomalyModule: Anomalib
                classification or segmentation model with/without weights.
        """
        if otx_model is None:
            model = get_model(config=self.config)
            logger.info(
                "No trained model in project yet. Created new model with '%s'",
                self.model_name,
            )
        else:
            buffer = io.BytesIO(otx_model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            if model_data["config"]["model"]["backbone"] != self.config["model"]["backbone"]:
                logger.warning(
                    "Backbone of the model in the Task Environment is different from the one in the template. "
                    f"creating model with backbone={model_data['config']['model']['backbone']}"
                )
                self.config["model"]["backbone"] = model_data["config"]["model"]["backbone"]
            try:
                model = get_model(config=self.config)
                model.load_state_dict(model_data["model"])
                logger.info("Loaded model weights from Task Environment")
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
        config = self.get_config()
        datamodule = OTXAnomalyDataModule(config=config, dataset=dataset, task_type=self.task_type)

        logger.info("Inference Configs '%s'", config)

        # Callbacks.
        progress = ProgressCallback(parameters=inference_parameters)
        inference = AnomalyInferenceCallback(dataset, self.labels, self.task_type)
        normalize = MinMaxNormalizationCallback()
        metrics_configuration = MetricsConfigurationCallback(
            task=config.dataset.task,
            image_metrics=config.metrics.image,
            pixel_metrics=config.metrics.get("pixel"),
        )
        post_processing_configuration = PostProcessingConfigurationCallback(
            normalization_method=NormalizationMethod.MIN_MAX,
            threshold_method=ThresholdMethod.ADAPTIVE,
            manual_image_threshold=config.metrics.threshold.manual_image,
            manual_pixel_threshold=config.metrics.threshold.manual_pixel,
        )
        callbacks = [progress, normalize, inference, metrics_configuration, post_processing_configuration]

        self.trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks)
        self.trainer.predict(model=self.model, datamodule=datamodule)
        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None) -> None:
        """Evaluate the performance on a result set.

        Args:
            output_resultset (ResultSetEntity): Result Set from which the performance is evaluated.
            evaluation_metric (Optional[str], optional): Evaluation metric. Defaults to None. Instead,
                metric is chosen depending on the task type.
        """
        metric: IPerformanceProvider
        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            metric = MetricsHelper.compute_f_measure(output_resultset)
        elif self.task_type == TaskType.ANOMALY_DETECTION:
            metric = MetricsHelper.compute_anomaly_detection_scores(output_resultset)
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            metric = MetricsHelper.compute_anomaly_segmentation_scores(output_resultset)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        output_resultset.performance = metric.get_performance()

        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            accuracy = MetricsHelper.compute_accuracy(output_resultset).get_performance()
            output_resultset.performance.dashboard_metrics.extend(accuracy.dashboard_metrics)

    def _export_to_onnx(self, onnx_path: str):
        """Export model to ONNX.

        Args:
             onnx_path (str): path to save ONNX file
        """
        height, width = self.config.model.input_size
        torch.onnx.export(
            model=self.model.model,
            args=torch.zeros((1, 3, height, width)).to(self.model.device),
            f=onnx_path,
            opset_version=11,
        )

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
        if dump_features:
            logger.warning(
                "Feature dumping is not implemented for the anomaly task."
                "The saliency maps and representation vector outputs will not be dumped in the exported model."
            )

        if export_type == ExportType.ONNX:
            output_model.model_format = ModelFormat.ONNX
            output_model.optimization_type = ModelOptimizationType.ONNX
            if precision == ModelPrecision.FP16:
                raise RuntimeError("Export to FP16 ONNX is not supported")
        elif export_type == ExportType.OPENVINO:
            output_model.model_format = ModelFormat.OPENVINO
            output_model.optimization_type = ModelOptimizationType.MO
        else:
            raise RuntimeError(f"not supported export type {export_type}")

        self.precision[0] = precision
        output_model.has_xai = dump_features

        # pylint: disable=no-member; need to refactor this
        logger.info("Exporting the OpenVINO model.")
        onnx_path = os.path.join(self.config.project.path, "onnx_model.onnx")
        self._export_to_onnx(onnx_path)

        if export_type == ExportType.ONNX:
            with open(onnx_path, "rb") as file:
                output_model.set_data("model.onnx", file.read())
        else:
            optimize_command = ["mo", "--input_model", onnx_path, "--output_dir", self.config.project.path]
            if precision == ModelPrecision.FP16:
                optimize_command.append("--compress_to_fp16")
            subprocess.run(optimize_command, check=True)
            bin_file = glob(os.path.join(self.config.project.path, "*.bin"))[0]
            xml_file = glob(os.path.join(self.config.project.path, "*.xml"))[0]
            with open(bin_file, "rb") as file:
                output_model.set_data("openvino.bin", file.read())
            with open(xml_file, "rb") as file:
                output_model.set_data("openvino.xml", file.read())

        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods

        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))
        self._set_metadata(output_model)

    def model_info(self) -> Dict:
        """Return model info to save the model weights.

        Returns:
           Dict: Model info.
        """
        return {
            "model": self.model.state_dict(),
            "config": self.get_config(),
            "VERSION": 1,
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

        if hasattr(self.model, "image_metrics"):
            f1_score = self.model.image_metrics.F1Score.compute().item()
            output_model.performance = Performance(score=ScoreMetric(name="F1 Score", value=f1_score))
        else:
            output_model.performance = NullPerformance()
        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods

    def _set_metadata(self, output_model: ModelEntity):
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
        path = "/proc/self/cgroup"
        is_in_docker = False
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as file:
                is_in_docker = is_in_docker or any("docker" in line for line in file)
        is_in_docker = is_in_docker or os.path.exists("/.dockerenv")
        return is_in_docker

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
        if self._work_dir_is_temp and os.path.exists(self.config.project.path):
            shutil.rmtree(self.config.project.path, ignore_errors=False)
