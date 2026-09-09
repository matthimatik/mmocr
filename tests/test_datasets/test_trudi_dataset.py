# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import tempfile
from unittest import TestCase

from mmocr.registry import DATASETS
from mmocr.utils import register_all_modules


class TestTrudiDataset(TestCase):

    def setUp(self):
        register_all_modules()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = self.tmp_dir.name
        self._create_trudi_layout()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_trudi_layout(self):
        for split in ('train', 'val', 'test'):
            split_dir = osp.join(self.root, split)
            img_dir = osp.join(split_dir, 'textdet_imgs')
            os.makedirs(img_dir, exist_ok=True)
            ann = {
                'metainfo': {
                    'dataset_type': 'TextDetDataset',
                    'task_name': 'textdet',
                    'category': [{
                        'id': 0,
                        'name': 'text'
                    }]
                },
                'data_list': []
            }
            for idx in range(2):
                img_name = f'{split}_{idx}.jpg'
                open(osp.join(img_dir, img_name), 'w').close()
                ann['data_list'].append({
                    'img_path': f'textdet_imgs/{img_name}',
                    'height': 64,
                    'width': 64,
                    'instances': [{
                        'bbox': [10, 10, 30, 30],
                        'bbox_label': 0,
                        'polygon': [10, 10, 30, 10, 30, 30, 10, 30],
                        'text': 'TRUDI',
                        'ignore': False
                    }]
                })
            with open(osp.join(split_dir, 'textdet.json'), 'w') as f:
                json.dump(ann, f)

    def _build(self, split, **kwargs):
        cfg = dict(type='TrudiDataset', split=split, data_root=self.root)
        cfg.update(kwargs)
        return DATASETS.build(cfg)

    def test_trudi_dataset_split_resolution(self):
        for split, split_len in (('train', 2), ('val', 2), ('test', 2)):
            dataset = self._build(split)
            self.assertEqual(len(dataset), split_len)
            data_info = dataset.get_data_info(0)
            self.assertEqual(
                data_info['img_path'],
                osp.join(self.root, split, 'textdet_imgs', f'{split}_0.jpg'))

    def test_trudi_dataset_default_data_root(self):
        dataset = self._build('train', data_root=self.root)
        self.assertEqual(dataset.data_root, osp.join(self.root, 'train'))
