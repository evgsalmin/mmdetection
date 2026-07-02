# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class CocoMinecraftDataset(CocoDataset):
    """Dataset for COCO."""

    # Нам достаточно только переопределить метаинформаицю 
    # о классах и палитре цветов 
    METAINFO = {
        'classes': (
            'minecraft-mobs','bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
            'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
    ),
        'palette': [
            (220, 20, 60), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
            (255, 77, 255), (0, 226, 252)
    ]
}
