#!/usr/bin/env bash
set -euo pipefail

cd /data/8hirsch/mmocr

PYTHONPATH=/data/8hirsch/mmocr \
  /data/8hirsch/mmocr/.venv/bin/python tools/train.py \
    configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_30e_synth_v1.py \
    --work-dir work_dirs/dbnetpp_synth_v1_30e
