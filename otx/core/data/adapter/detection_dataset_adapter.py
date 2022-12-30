"""Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
from datumaro.components.annotation import AnnotationType

from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class DetectionDatasetAdapter(BaseDatasetAdapter):
    """Detection adapter inherited from BaseDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for object detection, and instance segmentation tasks
    """

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Detection."""
        # Prepare label information
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        dataset_items = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = Image(file_path=datumaro_item.media.path)
                    shapes = []
                    for ann in datumaro_item.annotations:
                        if ann.type == AnnotationType.polygon:
                            shapes.append(self._get_polygon_entity(ann, image.width, image.height))
                        if ann.type == AnnotationType.bbox:
                            shapes.append(self._get_normalized_bbox_entity(ann, image.width, image.height))

                    dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                    dataset_items.append(dataset_item)

        return DatasetEntity(items=dataset_items)