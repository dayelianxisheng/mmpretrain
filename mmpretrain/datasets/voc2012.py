# Copyright (c) OpenMMLab. All rights reserved.
"""Pascal VOC 2012 multi-label classification dataset."""

import os
import xml.etree.ElementTree as ET
from typing import List, Optional

import torch

from mmengine import get_file_backend

from mmpretrain.registry import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class VOC2012(MultiLabelDataset):
    """Pascal VOC 2012 multi-label classification dataset.

    Reads images and annotations from a pre-downloaded VOCdevkit directory.
    If ``download=True`` and the dataset is not found, it will be downloaded
    automatically via torchvision.

    Expected directory structure::

        data_root/
        └── VOCdevkit/
            └── VOC2012/
                ├── Annotations/   (XML files)
                ├── ImageSets/
                │   └── Main/
                │       ├── train.txt
                │       └── val.txt
                └── JPEGImages/

    Args:
        data_root (str): Root directory containing ``VOCdevkit/``.
        split (str): Dataset split. Supports ``"train"`` and ``"val"``.
            Defaults to ``"train"``.
        download (bool): Download dataset if not found. Defaults to ``True``.
        **kwargs: Extra arguments forwarded to :class:`MultiLabelDataset`.
    """

    METAINFO = {
        'classes': (
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
        )
    }

    CLASS_TO_IDX = {
        'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13,
        'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17,
        'train': 18, 'tvmonitor': 19,
    }

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        download: bool = True,
        metainfo: Optional[dict] = None,
        **kwargs,
    ):
        assert split in ('train', 'val'), \
            f"split must be 'train' or 'val', got '{split}'"
        self.split = split

        voc_root = os.path.join(data_root, 'VOCdevkit', 'VOC2012')
        if download and not os.path.isdir(voc_root):
            from torchvision import datasets as tv_datasets
            tv_datasets.VOCDetection(
                root=data_root, year='2012',
                image_set=split, download=True,
            )

        self.backend = get_file_backend(data_root, enable_singleton=True)
        self._voc_root = voc_root

        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(img_path=os.path.join(
                'VOCdevkit', 'VOC2012', 'JPEGImages')),
            test_mode=(split == 'val'),
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load image paths and multi-label gt from VOC XML annotations."""
        split_file = os.path.join(
            self._voc_root, 'ImageSets', 'Main', f'{self.split}.txt')

        with open(split_file) as f:
            image_ids = [line.strip() for line in f if line.strip()]

        data_list = []
        for image_id in image_ids:
            img_path = self.backend.join_path(
                self.img_prefix, f'{image_id}.jpg')
            ann_path = os.path.join(
                self._voc_root, 'Annotations', f'{image_id}.xml')

            label_indices = self._parse_xml(ann_path)
            # Return torch tensor so each sample has its own storage,
            # avoiding collate issues with shared-storage numpy arrays.
            gt_label = torch.zeros(len(self.METAINFO['classes']), dtype=torch.int64)
            for i in label_indices:
                gt_label[i] = 1
            data_list.append(dict(
                img_path=img_path,
                gt_label=gt_label,
            ))

        return data_list

    def _parse_xml(self, ann_path: str) -> List[int]:
        """Parse a VOC annotation XML and return sorted class index list."""
        tree = ET.parse(ann_path)
        root = tree.getroot()

        labels = set()
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name in self.CLASS_TO_IDX:
                labels.add(self.CLASS_TO_IDX[name])
        return sorted(labels)

    def extra_repr(self) -> List[str]:
        return [f'Root of dataset: \t{self.data_root}']
