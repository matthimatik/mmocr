# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

from mmocr.datasets.ocr_dataset import OCRDataset
from mmocr.registry import DATASETS


@DATASETS.register_module()
class TrudiDataset(OCRDataset):
    """TRUDI dataset for text detection.

    The TRUDI build pipeline produces, for every split, a directory
    containing its own annotation file and images::

        <data_root>/train/textdet.json
        <data_root>/train/textdet_imgs/
        <data_root>/val/textdet.json
        <data_root>/val/textdet_imgs/
        <data_root>/test/textdet.json
        <data_root>/test/textdet_imgs/

    The ``img_path`` entries inside each annotation file are relative to
    the split directory. This dataset therefore appends ``split`` to the
    given ``data_root`` and points ``ann_file`` at the annotation file
    inside that split, hiding the per-split layout from the user.

    Args:
        split (str): The split to load. Options are 'train', 'val' and
            'test'. Defaults to 'train'.
        data_root (str, optional): The root directory containing the split
            directories. Defaults to 'data/trudi'.
        ann_file (str): The annotation file name inside the split
            directory. Defaults to 'textdet.json'.
        **kwargs: Remaining keyword arguments are passed to
            :class:`mmocr.datasets.OCRDataset`.
    """

    def __init__(self,
                 split: str = 'train',
                 data_root: Optional[str] = 'data/trudi',
                 ann_file: str = 'textdet.json',
                 **kwargs):
        split_data_root = osp.join(data_root, split)
        super().__init__(data_root=split_data_root,
                         ann_file=ann_file,
                         **kwargs)
