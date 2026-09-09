# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from mmocr.registry import CFG_GENERATORS
from .textspotting_config_generator import TextSpottingConfigGenerator


@CFG_GENERATORS.register_module()
class TrudiTextSpottingConfigGenerator(TextSpottingConfigGenerator):
    """Config generator for the TRUDI text spotting dataset.

    TRUDI stores one annotation file (``textspotting.json``) per split
    directory, resolved by the ``split`` argument of ``TrudiDataset``.
    Like the detection annotation, a spotting instance carries geometry plus
    a transcription, so the generated train config also filters empty
    ground truths.

    Args:
        data_root (str): The root path of the dataset.
        dataset_name (str): The name of the dataset.
        overwrite_cfg (bool): Whether to overwrite the dataset config file if
            it already exists. If False, config generator will not generate
            new config for datasets whose configs are already in base.
        train_anns (List[Dict], optional): A list of train annotation files.
            Defaults to ``[dict(ann_file='textspotting.json')]``.
        val_anns (List[Dict], optional): A list of val annotation files.
            Defaults to ``[dict(ann_file='textspotting.json')]``.
        test_anns (List[Dict], optional): A list of test annotation files.
            Defaults to ``[dict(ann_file='textspotting.json')]``.
        config_path (str): Path to the configs. Defaults to 'configs/'.
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        overwrite_cfg: bool = False,
        train_anns: Optional[List[Dict]] = None,
        val_anns: Optional[List[Dict]] = None,
        test_anns: Optional[List[Dict]] = None,
        config_path: str = 'configs/',
    ) -> None:
        if train_anns is None:
            train_anns = [dict(ann_file='textspotting.json')]
        if val_anns is None:
            val_anns = [dict(ann_file='textspotting.json')]
        if test_anns is None:
            test_anns = [dict(ann_file='textspotting.json')]
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
        for key_name, ann_dict in self.anns.items():
            cfg += f'\n{key_name} = dict(\n'
            cfg += '    type=\'TrudiDataset\',\n'
            cfg += f'    split=\'{ann_dict["split"]}\',\n'
            data_root_name = f'{self.dataset_name}_{self.task}_data_root'
            cfg += f'    data_root={data_root_name},\n'
            cfg += f'    ann_file=\'{ann_dict["ann_file"]}\',\n'
            if ann_dict['split'] == 'train':
                cfg += '    filter_cfg=dict(filter_empty_gt=True, min_size=32),\n'  # noqa: E501
            else:
                cfg += '    test_mode=True,\n'
            cfg += '    pipeline=None)\n'
        return cfg
