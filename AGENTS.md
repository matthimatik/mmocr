# AGENTS.md

Guidelines for AI coding agents working in this repository.

This is a fork of [open-mmlab/mmocr](https://github.com/open-mmlab/mmocr) (v1.0.1 base) with
first-class support for the TRUDI container-text dataset (dataset preparer, base configs,
DBNet++ text-detection config) and an ABCNet text-spotting finetune under `projects/ABCNet/`.

## Environment setup

Run this from the repository root:

```bash
bash setup_env.sh          # creates/repairs .venv (idempotent, ~3 GB download on first run)
source .venv/bin/activate
```

The script builds `.venv/` (Python 3.11, `uv`-managed) with the pinned stack and then verifies
it: imports, CUDA, and a `mmcv` deformable-conv forward on the GPU. **Read the pin comments in
`setup_env.sh` before changing any version** — numpy>=2 and opencv>=5 both break this stack,
and mmcv wheels for cu118 exist only on the OpenMMLab download index, not on PyPI.

Do not follow the install instructions in `README.md` or `docs/en/get_started/install.md`:
they are upstream docs and are stale (conda, Python 3.7/3.8, torch 1.10). `setup_env.sh` is the
source of truth for this fork.

Requirements: `uv` on PATH, an NVIDIA driver with CUDA 11.8 user-space support, and SSH access
to GitHub if you prepare the TRUDI dataset (its obtainer clones via `git@github.com:matthimatik/trudi.git`).

## Quick commands

Always run from the repository root; configs and the dataset preparer use relative paths
(`data/`, `dataset_zoo/`).

```bash
# Train (single GPU; use --launcher pytorch only with matching --nproc_per_node)
python tools/train.py configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_trudi.py \
    --work-dir work_dirs/dbnetpp_trudi

# Fine-tune DBNet++ on TRUDI id_texts from the ICDAR2015 checkpoint (results:
# configs/textdet/dbnetpp/README.md; or just run train_dbnetpp_trudi.sh)
python tools/train.py configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_200e_trudi.py \
    --work-dir work_dirs/dbnetpp_trudi_200e

# Fine-tune RobustScanner on TRUDI id_text crops from the ICDAR2015 checkpoint
# (results: configs/textrecog/robust_scanner/README.md; or train_robustscanner_trudi.sh)
python tools/train.py configs/textrecog/robust_scanner/robustscanner_resnet31_20e_trudi.py \
    --work-dir work_dirs/robustscanner_trudi_20e

# Fine-tune both models on the synthetic v1 dataset from the TRUDI checkpoints
# (synthetic-domain only — real TRUDI metrics drop; build data first with
# tools/dataset_converters/cta2024_to_mmocr.py; results in the READMEs; or
# train_dbnetpp_synth_v1.sh / train_robustscanner_synth_v1.sh)
python tools/train.py configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_30e_synth_v1.py \
    --work-dir work_dirs/dbnetpp_synth_v1_30e
python tools/train.py configs/textrecog/robust_scanner/robustscanner_resnet31_10e_synth_v1.py \
    --work-dir work_dirs/robustscanner_synth_v1_10e

# Evaluate
python tools/test.py configs/<config>.py work_dirs/<exp>/<ckpt>.pth

# Infer (downloads the config's pretrained weights on first run — needs network;
# point at local checkpoints with --det-weights/--rec-weights to avoid it)
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec CRNN --print-result

# Prepare a dataset (see dataset_zoo/<name>/ for the supported tasks)
python tools/dataset_converters/prepare_dataset.py <name> --task textdet

# Visualize a dataset pipeline
python tools/visualizations/browse_dataset.py \
    configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_200e_trudi.py \
    --phase train --mode pipeline --show-number 4 --not-show --output-dir /tmp/browse

# Tests
pytest tests -q
```

`mim` (openmim) is installed, and `mmocr/.mim` is populated by `setup_env.sh` (and by
`setup.py` on editable installs). However, because mmocr is installed as a **PEP 660 editable**
package, `mim train/test mmocr …` does **not** work here: mim resolves mmocr to a nonexistent
`site-packages/mmocr/tools/train.py` and exits 2 (and `mim list` omits mmocr). Use
`python tools/train.py` / `python tools/test.py` for mmocr; `mim` is fine for
`mim install mmcv|mmdet|mmengine`.

### Gotchas

- **GPU count / helper scripts**: this machine has 1 GPU. `train_dbnetpp_trudi.sh` is
  single-GPU and launches the TRUDI DBNet++ fine-tune; `train_trudi.sh` (ABCNet) and some
  scripts under `projects/` still assume `--nproc_per_node=2` **and hard-code the absolute
  path `/data/8hirsch/mmocr`** (interpreter, PYTHONPATH, config paths) — in a clone at
  another location they run the wrong tree or fail. Prefer the inline commands in Quick
  commands; if you do use them, override with `--nproc_per_node=1` and fix the paths first.
  Two ranks on one 24 GB GPU will OOM or crash.
- **`projects/` packages are not installed**: configs that import from `projects/…`
  (e.g. ABCNet) need the project dir on `PYTHONPATH`:
  `PYTHONPATH="$PWD:$PWD/projects/ABCNet" python tools/train.py projects/ABCNet/...`
  Without it `tools/train.py` fails fast with
  `ImportError: Failed to import custom modules from {'imports': ['abcnet'] …}` plus an
  explicit "You should set PYTHONPATH" hint; bare scripts that skip `init_default_scope`
  may instead show `KeyError: 'Xxx is not in the mmengine::model registry'`.
- **Registry scope**: build models with `mmocr.registry.MODELS` after
  `from mmocr.utils import register_all_modules; register_all_modules()`, not
  `mmengine.registry.MODELS` (this matters in bare scripts; Runner does it for you).
- **`.venv` and `data/` are untracked**: if the working directory is ever re-cloned, both are
  lost — re-run `bash setup_env.sh` and re-prepare datasets. Never let setup knowledge live
  only in an environment; update this file and `setup_env.sh` instead.
- `torch.distributed.launch` is deprecated; prefer `torchrun --nproc_per_node=<n>`.
- `pre-commit install` (hooks are configured in `.pre-commit-config.yaml` but not installed
  by `setup_env.sh`).

## Datasets

- Preparation configs live in `dataset_zoo/<name>/<task>.py`; output goes to `data/<name>/`.
- TRUDI: `python tools/dataset_converters/prepare_dataset.py trudi --task textdet` (also
  `textrecog`, `textspotting`). The obtainer reuses the local TRUDI checkout
  (`local_repo=/data/8hirsch/trudi` in `dataset_zoo/trudi/textdet.py`; set it to `None` to
  clone into `data/cache/` instead), runs `uv sync` + `uv run trudi download_images` (no-op
  when the images are already there) and `uv run trudi build …`, then copies the split dirs
  to `data/trudi/{train,val,test}/`. It is idempotent: existing prepared splits are skipped.
- The textdet and textrecog targets cover **all id_text types** (iso6346, en13044_1,
  registration_number, size_and_type_*); `textspotting` still uses `type=id_iso6346` only.
- Fine-tuned DBNet++ results and release-style weights (same `.pth` layout as the openmmlab
  checkpoints) are documented in `configs/textdet/dbnetpp/README.md`; artifacts live in
  `work_dirs/dbnetpp_trudi_200e/` (gitignored — retrain to regenerate). The same for the
  RobustScanner fine-tune in `configs/textrecog/robust_scanner/README.md`
  (`work_dirs/robustscanner_trudi_20e/`, also gitignored).
- Base configs for TRUDI textdet/textrecog/textspotting are in
  `configs/{textdet,textrecog,textspotting}/_base_/datasets/trudi.py` (`data/trudi`).
- The TRUDI dataset itself: <https://github.com/matthimatik/trudi>.
- Synthetic v1: `python tools/dataset_converters/cta2024_to_mmocr.py` converts the v1
  CTA2024 renders (`synthetic-data-generation/src/output/dataset`, ISO6346 ID + size/type
  texts) to `data/synth_v1/{train,val}/` in the TRUDI MMOCR layout (hard-linked scene
  images, padded recognition crops, aspect filter dropping diagonal-text quads from the
  recognition set only). Base configs: `configs/{textdet,textrecog}/_base_/datasets/synth_v1.py`.
  Pure synthetic-domain fine-tunes from the TRUDI checkpoints: the dbnetpp 30e and
  robust_scanner 10e `*_synth_v1.py` configs (results in the respective READMEs;
  `work_dirs/{dbnetpp_synth_v1_30e,robustscanner_synth_v1_10e}/`).

## Known test-suite state

Run `pytest tests -q` before claiming the environment is healthy. Two failures are **expected
and deliberate** (do not "fix" them without checking git history):

- `tests/test_utils/test_polygon_utils.py::TestPolygonUtils::test_crop_polygon` and
  `tests/test_datasets/test_transforms/test_adapters.py::TestMMDet2MMOCR::test_ocr2det2ocr` —
  caused by commit `0ae7d0d` ("Shorten back to 4 points"), which makes `crop_polygon()`
  return at most 4 points (`mmocr/utils/polygon_utils.py`). This truncation feeds the
  ABCNet bezier pipeline and the DBNet++ GT alignment work; it is intentional.
  `Resize._resize_polygons` (`clip_object_border=True`) calls `crop_polygon`, so any test
  expecting >4-point polygons after a resize will fail.

Everything else should pass. If a test fails only when the full suite runs (but passes
alone), suspect cross-test pollution in `tests/` before suspecting the environment.

## Layout pointers

- `mmocr/datasets/trudi_dataset.py`, `mmocr/datasets/preparers/obtainers/trudi_obtainer.py` —
  TRUDI dataset + preparer (fork additions; unit-tested in `tests/test_datasets/`).
- `projects/ABCNet/` — ABCNet text spotter (mmdet-based), TRUDI finetune configs under
  `projects/ABCNet/config/abcnet/`.
- `configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_trudi.py` — TRUDI DBNet++ config.
- `configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_200e_trudi.py` — TRUDI id_text
  fine-tune (ICDAR2015 init, early-stopped at epoch 73; results/weights in the dbnetpp
  README) — the run `train_dbnetpp_trudi.sh` reproduces.
- `setup_env.sh` — pinned environment bootstrap; keep it in sync with this file.
- Upstream contribution flow (pre-commit, CI) is described in `.github/CONTRIBUTING.md`.
