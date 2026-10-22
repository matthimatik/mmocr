#!/usr/bin/env bash
set -euo pipefail

cd /data/8hirsch/mmocr

PYTHONPATH=/data/8hirsch/mmocr \
  /data/8hirsch/mmocr/.venv/bin/python tools/train.py \
    configs/textrecog/robust_scanner/robustscanner_resnet31_10e_synth_v1.py \
    --work-dir work_dirs/robustscanner_synth_v1_10e
