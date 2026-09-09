# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

from mmocr.datasets.preparers import TrudiTextDetConfigGenerator


class TestTrudiTextDetConfigGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()

    def test_trudi_textdet_config_generator(self):
        config_generator = TrudiTextDetConfigGenerator(
            data_root=self.root.name,
            dataset_name='trudi',
            config_path=self.root.name,
        )
        cfg_path = osp.join(self.root.name, 'textdet', '_base_', 'datasets',
                            'trudi.py')
        config_generator()
        self.assertTrue(osp.exists(cfg_path))
        with open(cfg_path, 'r') as f:
            lines = ''.join(f.readlines())

        self.assertEqual(
            lines, (f"trudi_textdet_data_root = '{self.root.name}'\n"
                    '\n'
                    'trudi_textdet_train = dict(\n'
                    "    type='TrudiDataset',\n"
                    "    split='train',\n"
                    '    data_root=trudi_textdet_data_root,\n'
                    '    filter_cfg=dict(filter_empty_gt=True, min_size=32),\n'
                    '    pipeline=None)\n'
                    '\n'
                    'trudi_textdet_val = dict(\n'
                    "    type='TrudiDataset',\n"
                    "    split='val',\n"
                    '    data_root=trudi_textdet_data_root,\n'
                    '    test_mode=True,\n'
                    '    pipeline=None)\n'
                    '\n'
                    'trudi_textdet_test = dict(\n'
                    "    type='TrudiDataset',\n"
                    "    split='test',\n"
                    '    data_root=trudi_textdet_data_root,\n'
                    '    test_mode=True,\n'
                    '    pipeline=None)\n'))
