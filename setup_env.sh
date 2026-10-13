#!/usr/bin/env bash
# Recreate the verified development environment for this MMOCR fork in .venv/.
#
# Verified stack (do not "helpfully" upgrade these; each pin exists for a reason):
#   Python 3.11, torch 2.1.0+cu118, mmcv 2.1.0 (prebuilt cu118 wheel),
#   mmengine 0.10.x, mmdet 3.3.0, numpy 1.26.4, opencv-python 4.9.0.80
#
# Why the pins:
#   - mmcv 2.1.0: requirements/mminstall.txt allows >=2.0.0rc4,<2.2.0; PyPI only
#     ships sdists, so the prebuilt CUDA wheel must come from the OpenMMLab index.
#   - numpy 1.26.4: numpy>=2 breaks torch 2.1 and mmcv 2.1 (compiled against the
#     1.x ABI). Several deps pull numpy 2.x if left unpinned.
#   - opencv-python 4.9.0.80: opencv>=5 requires numpy>=2.
#
# Idempotent: safe to re-run; it re-resolves and repairs the environment in place.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required (https://docs.astral.sh/uv/). Install it first." >&2
    exit 1
fi

# The uv cache lives on a different filesystem than this repo, so hardlinking
# fails and slows everything down with copy fallbacks.
export UV_LINK_MODE=copy

VENV=".venv"
PY_VERSION="3.11"
TORCH="2.1.0+cu118"
TORCHVISION="0.16.0+cu118"
MMCV="2.1.0"
NUMPY="1.26.4"
OPENCV="4.9.0.80"
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"

if [ ! -x "$VENV/bin/python" ]; then
    echo "==> Creating $VENV (Python $PY_VERSION)"
    uv venv "$VENV" --python "$PY_VERSION"
else
    echo "==> Reusing existing $VENV"
fi

echo "==> Installing torch $TORCH / torchvision $TORCHVISION (cu118)"
uv pip install -p "$VENV" \
    --index-url "$PYTORCH_INDEX" \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    "torch==$TORCH" "torchvision==$TORCHVISION"

# Install numpy before mmcv/mmdet/mmocr so no resolver sneaks numpy 2.x in.
echo "==> Pinning numpy $NUMPY"
uv pip install -p "$VENV" "numpy==$NUMPY"

echo "==> Installing mmcv $MMCV (prebuilt cu118 wheel; not on PyPI as a wheel)"
uv pip install -p "$VENV" \
    --find-links "$MMCV_INDEX" \
    --index-strategy unsafe-best-match \
    "mmcv==$MMCV"

echo "==> Installing mmengine + mmdet 3.3.0"
uv pip install -p "$VENV" "mmengine>=0.7.0,<1.1.0" "mmdet==3.3.0"

echo "==> Installing openmim + test requirements"
uv pip install -p "$VENV" openmim -r requirements/tests.txt

echo "==> Installing mmocr (editable) with numpy/opencv pins"
uv pip install -p "$VENV" -e . "numpy==$NUMPY" "opencv-python==$OPENCV"

# `pip install -e .` creates mmocr/.mim via setup.py (editable_wheel), but make
# sure it exists regardless of installer behaviour: the `mim` CLI needs it.
"$VENV/bin/python" - <<'PY'
import os
import os.path as osp
import shutil

repo = osp.dirname(osp.abspath('setup.py'))
mim = osp.join(repo, 'mmocr', '.mim')
if osp.exists(osp.join(mim, 'tools')):
    raise SystemExit(0)
os.makedirs(mim, exist_ok=True)
for name in ('tools', 'configs', 'model-index.yml', 'dicts'):
    src = osp.join(repo, name)
    dst = osp.join(mim, name)
    if osp.isfile(dst) or osp.islink(dst):
        os.remove(dst)
    elif osp.isdir(dst):
        shutil.rmtree(dst)
    os.symlink(osp.relpath(src, mim), dst)
print('mmocr/.mim links ready')
PY

echo "==> Verifying environment"
"$VENV/bin/python" - <<'PY'
import sys

import cv2
import mmcv
import mmengine
import mmocr
import mmdet
import numpy
import torch
from mmcv.ops import DeformConv2d

errors = []


def check(label, ok, detail=''):
    line = f"    [{'ok' if ok else 'FAIL'}] {label}"
    if detail:
        line += f' ({detail})'
    print(line)
    if not ok:
        errors.append(label)


print('  Environment:')
check('python', sys.version_info[:2] == (3, 11), sys.version.split()[0])
check('torch', torch.__version__.startswith('2.1.0'), torch.__version__)
check('torchvision', True)
check('mmcv', mmcv.__version__ == '2.1.0', mmcv.__version__)
check('mmengine', True, mmengine.__version__)
check('mmdet', mmdet.__version__ == '3.3.0', mmdet.__version__)
check('mmocr', True, mmocr.__version__)
check('numpy', numpy.__version__ == '1.26.4', numpy.__version__)
check('opencv', cv2.__version__.startswith('4.9.0'), cv2.__version__)
check('mmocr/.mim', __import__('os').path.exists('mmocr/.mim'))

if torch.cuda.is_available():
    check('cuda available', True, torch.cuda.get_device_name(0))
    try:
        x = torch.randn(1, 8, 8, 8, device='cuda')
        out = DeformConv2d(8, 8, 3, padding=1).cuda()(
            x, torch.randn(1, 18, 8, 8, device='cuda'))
        check('mmcv DCN op on GPU', out.is_cuda and out.shape == (1, 8, 8, 8))
    except Exception as e:  # noqa: BLE001
        check('mmcv DCN op on GPU', False, type(e).__name__)
else:
    check('cuda available', False, 'no GPU visible; check nvidia-smi / drivers')

if errors:
    print(f'\nSETUP FINISHED WITH {len(errors)} FAILED CHECK(S): {errors}')
    raise SystemExit(1)
print('\nSetup OK. Activate with:  source .venv/bin/activate')
PY
