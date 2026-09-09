# Copyright (c) OpenMMLab. All rights reserved.
from .dataset_wrapper import ConcatDataset
from .icdar_dataset import IcdarDataset
from .ocr_dataset import OCRDataset
from .recog_lmdb_dataset import RecogLMDBDataset
from .recog_text_dataset import RecogTextDataset
from .samplers import *  # NOQA
from .transforms import *  # NOQA
from .trudi_dataset import TrudiDataset
from .wildreceipt_dataset import WildReceiptDataset

__all__ = [
    'IcdarDataset', 'OCRDataset', 'RecogLMDBDataset', 'RecogTextDataset',
    'TrudiDataset', 'WildReceiptDataset', 'ConcatDataset'
]
