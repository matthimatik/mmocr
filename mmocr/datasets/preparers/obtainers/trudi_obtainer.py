# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import subprocess
from typing import Optional

from mmengine import mkdir_or_exist

from mmocr.registry import DATA_OBTAINERS

_TASK_ANN_FILES = {
    'textdet': 'textdet.json',
    'textspotting': 'textspotting.json',
    'textrecog': 'textrecog.json',
}


@DATA_OBTAINERS.register_module()
class TrudiObtainer:
    """Obtain one of the TRUDI text datasets.

    The TRUDI (Transportation Unit Detection and Identification) dataset is
    distributed as a git repository of labelme annotations plus scripts that
    download the raw images and materialize derived task datasets, e.g. in
    MMOCR format. This obtainer therefore differs from e.g.
    :class:`NaiveDataObtainer` in that it does not download a single archive
    but instead drives the TRUDI toolchain:

    1. clone the TRUDI repository (or reuse an existing checkout, e.g. when
       ``local_repo`` is given).
    2. ``uv sync`` to set up the repository's virtual environment.
    3. download the images with ``uv run trudi download_images``.
    4. build the requested MMOCR dataset with
       ``uv run trudi build <target>``.
    5. publish the produced split directories (``train``, ``val``, ``test``)
       under ``data_root``.

    The obtainer is idempotent: if ``data_root`` already contains the
    annotation file for every split, it returns without re-running the
    pipeline, which makes it safe to declare for all three splits.

    Args:
        cache_path (str): The path to cache the cloned TRUDI repository.
        data_root (str): The root path of the dataset. It is usually set
            automatically and users do not need to set it manually in config
            file in most cases.
        task (str): The task of the dataset. It is usually set automatically
            and users do not need to set it manually in config file in most
            cases. Determines the annotation file name ('textdet.json',
            'textspotting.json' or 'textrecog.json').
        repo_url (str): The git URL of the TRUDI repository. Defaults to
            'git@github.com:matthimatik/trudi.git'.
        repo_dir (str, optional): Name of the directory the repository is
            cloned into below ``cache_path``. Defaults to 'trudi'.
        target (str): The build target DSL, e.g.
            'v2/text_detection[type=id_iso6346]/mmocr'. Defaults to
            'v2/text_detection[type=id_iso6346]/mmocr'.
        local_repo (str, optional): Path to an existing TRUDI repository
            checkout to use instead of cloning ``repo_url``. Defaults to None.
        download_images (bool): Whether to run ``trudi download_images``.
            Defaults to True.
        ann_file (str, optional): The name of the annotation file written
            per split. If not set, it is derived from ``task`` by
            :data:`_TASK_ANN_FILES`. Defaults to None.
    """

    def __init__(self,
                 cache_path: str,
                 data_root: str,
                 task: str,
                 repo_url: str = 'git@github.com:matthimatik/trudi.git',
                 repo_dir: Optional[str] = 'trudi',
                 target: str = 'v2/text_detection[type=id_iso6346]/mmocr',
                 local_repo: Optional[str] = None,
                 download_images: bool = True,
                 ann_file: Optional[str] = None) -> None:
        self.cache_path = cache_path
        self.data_root = data_root
        self.task = task
        self.repo_url = repo_url
        self.target = target
        self.local_repo = local_repo if local_repo is not None else ''
        self.download_images = download_images
        self.ann_file = ann_file or _TASK_ANN_FILES.get(task)
        if self.ann_file is None:
            raise ValueError(f'Unknown TRUDI task {task!r}; expected one of '
                             f'{sorted(_TASK_ANN_FILES)} or an explicit '
                             'ann_file.')
        self.repo_dir = (osp.join(cache_path, repo_dir) if repo_dir
                         else cache_path)
        mkdir_or_exist(self.cache_path)
        mkdir_or_exist(self.data_root)

    def __call__(self):
        if self._already_prepared():
            print(f'TRUDI dataset already prepared at {self.data_root}, '
                  'skipping.')
            return
        repo = self._obtain_repo()
        self._build(repo)
        self._publish(repo)

    def _already_prepared(self) -> bool:
        """Whether every split directory is already present."""
        return all(
            osp.exists(osp.join(self.data_root, split, self.ann_file))
            for split in ('train', 'val', 'test'))

    def _obtain_repo(self) -> str:
        """Return the TRUDI repository checkout, cloning it if needed."""
        if self.local_repo:
            if not osp.isdir(self.local_repo):
                raise FileNotFoundError(
                    f'local_repo {self.local_repo} does not exist.')
            return self.local_repo
        if not osp.isdir(self.repo_dir):
            print(f'Cloning TRUDI repository from {self.repo_url}...')
            self._run_cmd(['git', 'clone', self.repo_url, self.repo_dir],
                          cwd=osp.dirname(self.repo_dir))
        return self.repo_dir

    def _build(self, repo: str) -> None:
        """Sync dependencies, download images and build the dataset."""
        if shutil.which('uv') is None:
            raise RuntimeError(
                'uv is required to prepare the TRUDI dataset but it was not '
                'found on PATH.')
        print('Syncing TRUDI repository dependencies...')
        self._run_cmd(['uv', 'sync'], cwd=repo)
        if self.download_images:
            print('Downloading TRUDI images...')
            self._run_cmd(['uv', 'run', 'trudi', 'download_images'], cwd=repo)
        print(f'Building TRUDI target {self.target}...')
        self._run_cmd([
            'uv', 'run', 'trudi', 'build', self.target, '--force',
            '--out-dir', osp.join(repo, 'builds')
        ],
                      cwd=repo)

    def _publish(self, repo: str) -> None:
        """Copy the built split directories into ``data_root``."""
        built = osp.join(repo, 'builds', *self.target.split('/'))
        if not osp.isdir(built):
            raise FileNotFoundError(
                f'Expected build output at {built} but it does not exist.')
        shutil.copytree(built, self.data_root, dirs_exist_ok=True)

    @staticmethod
    def _run_cmd(cmd, cwd=None) -> None:
        """Run a command, raising on non-zero exit."""
        print(' '.join(cmd))
        subprocess.run(cmd, cwd=cwd, check=True)
