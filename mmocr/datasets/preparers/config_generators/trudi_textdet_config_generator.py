# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from mmocr.registry import CFG_GENERATORS
from .textdet_config_generator import TextDetConfigGenerator


@CFG_GENERATORS.register_module()
class TrudiTextDetConfigGenerator(TextDetConfigGenerator):
    """Config generator for the TRUDI text detection dataset.

    Unlike other text detection datasets, TRUDI stores one annotation file
    per split directory, and the ``TrudiDataset`` resolves this layout via
    its ``split`` argument. This generator therefore emits a config that
    instantiates :class:`mmocr.datasets.TrudiDataset` instead of
    ``OCRDataset``.

    Args:
        data_root (str): The root path of the dataset.
        dataset_name (str): The name of the dataset.
        overwrite_cfg (bool): Whether to overwrite the dataset config file if
            it already exists. If False, config generator will not generate
            new config for datasets whose configs are already in base.
        train_anns (List[Dict], optional): A list of train annotation files.
            Defaults to ``[dict(ann_file='train/textdet.json')]``.
        val_anns (List[Dict], optional): A list of val annotation files.
            Defaults to ``[dict(ann_file='val/textdet.json')]``.
        test_anns (List[Dict], optional): A list of test annotation files.
            Defaults to ``[dict(ann_file='test/textdet.json')]``.
        config_path (str): Path to the configs. Defaults to 'configs/'.
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        overwrite_cfg: bool = False,
        train_anns: Optional[List[Dict]] = [
            dict(ann_file='train/textdet.json')
        ],
        val_anns: Optional[List[Dict]] = [
            dict(ann_file='val/textdet.json')
        ],
        test_anns: Optional[List[Dict]] = [
            dict(ann_file='test/textdet.json')
        ],
        config_path: str = 'configs/',
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset_name=dataset_name,
            overwrite_cfg=overwrite_cfg,
            train_anns=train_anns,
            val_anns=val_anns,
            test_anns=test_anns,
            config_path=config_path,
        )

    def _gen_dataset_config(self) -> str:
        """Generate a full dataset config that uses ``TrudiDataset``."""
        cfg = ''
        for split in ('train', 'val', 'test'):
            var_name = f'{self.dataset_name}_{self.task}_{split}'
            cfg += f'\n{var_name} = dict(\n'
            cfg += '    type=\'TrudiDataset\',\n'
            cfg += f'    split=\'{split}\',\n'
            data_root_name = f'{self.dataset_name}_{self.task}_data_root'
            cfg += f'    data_root={data_root_name},\n'
            if split == 'train':
                cfg += '    filter_cfg=dict(filter_empty_gt=True, min_size=32),\n'  # noqa: E501
            else:
                cfg += '    test_mode=True,\n'
            cfg += '    pipeline=None)\n'
        return cfg
