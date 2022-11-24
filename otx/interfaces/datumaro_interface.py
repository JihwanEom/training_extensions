"""Interface for Datumaro integration."""

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

# pylint: disable=too-many-nested-blocks, invalid-name

from enum import Enum, auto

import datumaro
from datumaro.components.dataset import Dataset as DatumaroDataset
from datumaro.components.annotation import Bbox as DatumaroBbox
from datumaro.components.annotation import Mask as DatumaroMask
from datumaro.plugins.transforms import MasksToPolygons
from datumaro.components.annotation import AnnotationType as DatumaroAnnotationType

from otx.api.entities.annotation import (Annotation, AnnotationSceneEntity, AnnotationSceneKind)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import (LabelGroup, LabelGroupType, LabelSchemaEntity)
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.subset import Subset
from otx.api.entities.model_template import TaskType

from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from mpa.utils.logger import get_logger
logger = get_logger()


class DatumaroHandler:
    """Handler to use Datumaro as a front-end dataset."""
    def __init__(self, task_type:str):
        self.task_type = task_type
        logger.info('[*] Task type: {}'.format(self.task_type))
        self.domain = task_type.domain
        self.sub_domain = []
    
    def import_dataset(
            self,
            train_data_roots: str,
            train_ann_files: str = None,
            val_data_roots: str = None,
            val_ann_files: str = None,
            test_data_roots: str = None,
            test_ann_files: str = None,
            unlabeled_data_roots: str = None,
            unlabeled_file_lists: float = None
        )-> DatumaroDataset:
        """ Import dataset by using Datumaro."""
        # Find self.data_type and task_type
        data_type_candidates = self._detect_dataset_format(path=train_data_roots)
        logger.info('[*] Data type candidates: {}'.format(data_type_candidates))
        self.data_type = self._select_data_type(data_type_candidates) 
        logger.info('[*] Selected data type: {}'.format(self.data_type))

        # Construct dataset for training, validation, unlabeled
        self.dataset = {}
        logger.info('[*] Importing Datasets...')
        datumaro_dataset = DatumaroDataset.import_from(train_data_roots, format=self.data_type)

        # Annotation type filtering
        # TODO: is there something better? 
        if DatumaroAnnotationType.mask in list(datumaro_dataset.categories().keys()):
            datumaro_dataset.categories().pop(DatumaroAnnotationType.mask)

        # Prepare subsets by using Datumaro dataset
        for k, v in datumaro_dataset.subsets().items():
            if 'train' in k or 'default' in k:
                self.dataset[Subset.TRAINING] = v
            elif 'val' in k:
                self.dataset[Subset.VALIDATION] = v
        
        # If validation is manually defined --> set the validation data according to user's input
        if val_data_roots is not None:
            val_data_candidates = self._detect_dataset_format(path=val_data_roots)
            val_data_type = self._select_data_type(val_data_candidates)
            assert self.data_type == val_data_type, "The data types of training and validation must be same, the type of train:{} val:{}".format(
               self.data_type, val_data_type 
            )
            self.dataset[Subset.VALIDATION] = DatumaroDataset.import_from(val_data_roots, format=val_data_type)

        if Subset.VALIDATION not in self.dataset.keys():
            #TODO: auto_split
            pass

        # If unlabeled data is defined --> Semi-SL enable?
        if unlabeled_data_roots is not None:
            self.dataset[Subset.UNLABELED] = DatumaroDataset.import_from(unlabeled_data_roots, format='image_dir')
            #TODO: enable to read unlabeled file lists
        
        return self.dataset

    def _select_data_type(self, candidates:list):
        #TODO: more better way for classification
        if 'imagenet' in candidates:
            data_type = 'imagenet'
        else:
            data_type = candidates[0]
        return data_type

    def _auto_split(self): ## To be implemented
        """ Automatic train/val split."""
        return

    def _detect_dataset_format(self, path: str) -> str:
        """ Detection dataset format (ImageNet, COCO, Cityscapes, ...). """
        return datumaro.Environment().detect_dataset(path=path) 
        
    def convert_to_otx_format(self, datumaro_dataset:dict) -> DatasetEntity:
        """ Convert Datumaro Datset to DatasetEntity(OTE_SDK)"""
        label_categories_list = datumaro_dataset[Subset.TRAINING].categories().get(DatumaroAnnotationType.label, None)
        category_items = label_categories_list.items

        # Check the label_groups to get the hierarchical information
        if hasattr(label_categories_list, 'label_groups'):
            label_group_items = label_categories_list.label_groups
        
        label_entities = [LabelEntity(name=class_name.name, domain=self.domain,
                            is_empty=False, id=ID(i)) for i, class_name in enumerate(category_items)]

        dataset_items = []
        for subset, subset_data in datumaro_dataset.items():
            for phase, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    print('[*] datumaro item: ', datumaro_item)
                    image = Image(file_path=datumaro_item.media.path)
                    if self.domain == Domain.CLASSIFICATION:
                        labels = [
                            ScoredLabel(
                                label= [label for label in label_entities if label.name == category_items[ann.label].name][0],
                                probability=1.0   
                            ) for ann in datumaro_item.annotations
                        ]
                        shapes = [Annotation(Rectangle.generate_full_box(), labels)]

                        # Multi-Label
                        label_schema = LabelSchemaEntity()

                        for label_group_item in label_group_items:
                            group_label_entity_list = []
                            for label in label_group_item.labels:
                                label_entity = [le for le in label_entities if le.name == label]
                                group_label_entity_list.append(label_entity[0])

                            label_schema.add_group(
                                LabelGroup(
                                    name=label_group_item.name,
                                    labels=group_label_entity_list,
                                    group_type=LabelGroupType.EXCLUSIVE
                                )
                            )
                        label_schema.add_group(self._generate_empty_label_entity())

                    elif self.domain == Domain.DETECTION:
                        shapes = []
                        for ann in datumaro_item.annotations:
                            if isinstance(ann, DatumaroBbox):
                                shapes.append(
                                    Annotation(
                                        Rectangle(
                                            x1=ann.points[0]/image.width, 
                                            y1=ann.points[1]/image.height,
                                            x2=ann.points[2]/image.width,
                                            y2=ann.points[3]/image.height),
                                        labels = [
                                            ScoredLabel(
                                                label=label_entities[ann.label]
                                            )
                                        ]
                                    )
                                )
                        label_schema = self._generate_label_schema(label_entities)
                    elif self.domain == Domain.SEGMENTATION:
                        shapes = []
                        for ann in datumaro_item.annotations:
                            if isinstance(ann, DatumaroMask):
                                if ann.label > 0:
                                    datumaro_polygons = MasksToPolygons.convert_mask(ann)
                                    for d_polygon in datumaro_polygons:
                                        shapes.append(
                                            Annotation(
                                                Polygon(points=[Point(x=d_polygon.points[i]/image.width,y=d_polygon.points[i+1]/image.height) for i in range(0,len(d_polygon.points),2)]),
                                                labels=[
                                                    ScoredLabel(
                                                        label=label_entities[d_polygon.label-1]
                                                    )
                                                ]
                                            )
                                        )
                        label_schema = self._generate_label_schema(label_entities)
                    else : #Video
                        raise NotImplementedError()
                    annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=shapes)
                    dataset_item = DatasetItemEntity(image, annotation_scene, subset=subset)
                    dataset_items.append(dataset_item)
        
        return DatasetEntity(items=dataset_items), label_schema

    def _generate_empty_label_entity(self):
        empty_label = LabelEntity(name="Empty label", is_empty=True, domain=self.domain)
        empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
        return empty_group
    

    def _generate_label_schema(self, label_entities:list):
        label_schema = LabelSchemaEntity()
        main_group = LabelGroup(
            name="labels",
            labels=label_entities,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        label_schema.add_group(main_group)
        return label_schema