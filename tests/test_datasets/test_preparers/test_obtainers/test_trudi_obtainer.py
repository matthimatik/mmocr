# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
import unittest
from unittest import mock

from mmocr.datasets.preparers import TrudiObtainer
from mmocr.registry import DATA_OBTAINERS
from mmocr.utils import register_all_modules


class TestTrudiObtainer(unittest.TestCase):

    def setUp(self):
        register_all_modules()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = self.tmp_dir.name
        self.cache_path = osp.join(self.root, 'cache')
        self.data_root = osp.join(self.root, 'data', 'trudi')
        self.local_repo = osp.join(self.root, 'trudi_repo')
        self.target = 'v2/text_detection[type=id_iso6346]/mmocr'
        self._create_fake_repo()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_fake_repo(self):
        built_root = osp.join(self.local_repo, 'builds',
                              *self.target.split('/'))
        os.makedirs(built_root, exist_ok=True)
        for split in ('train', 'val', 'test'):
            split_dir = osp.join(built_root, split)
            os.makedirs(split_dir, exist_ok=True)
            open(osp.join(split_dir, 'textdet.json'), 'w').close()

    def _obtainer(self, download_images=False, **kwargs):
        local_repo = kwargs.pop('local_repo', self.local_repo)
        return TrudiObtainer(
            cache_path=self.cache_path,
            data_root=self.data_root,
            task='textdet',
            local_repo=local_repo,
            target=self.target,
            download_images=download_images,
            **kwargs)

    def test_registered(self):
        self.assertIn('TrudiObtainer', DATA_OBTAINERS.module_dict)

    @mock.patch('mmocr.datasets.preparers.obtainers.trudi_obtainer.'
                'subprocess.run')
    @mock.patch('mmocr.datasets.preparers.obtainers.trudi_obtainer.'
                'shutil.which', return_value='/usr/bin/uv')
    def test_obtain(self, _which, _run):
        obtainer = self._obtainer()
        obtainer()
        for split in ('train', 'val', 'test'):
            self.assertTrue(
                osp.exists(osp.join(self.data_root, split, 'textdet.json')))
        # obtainer is idempotent
        obtainer()
        self.assertEqual(_run.call_count, 2)

    @mock.patch('mmocr.datasets.preparers.obtainers.trudi_obtainer.'
                'subprocess.run')
    @mock.patch('mmocr.datasets.preparers.obtainers.trudi_obtainer.'
                'shutil.which', return_value='/usr/bin/uv')
    def test_already_prepared(self, _which, _run):
        obtainer = self._obtainer()
        obtainer()
        _run.reset_mock()
        obtainer()
        _run.assert_not_called()

    def test_missing_local_repo(self):
        with self.assertRaises(FileNotFoundError):
            self._obtainer(local_repo=osp.join(self.root, 'nope'))()
